"""Parsing, ellipsis expansion, and validation for binary einsum expressions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ParsedEquation:
    left_axes: tuple[str, ...]
    right_axes: tuple[str, ...]
    output_axes: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedNaryEquation:
    operand_axes: tuple[tuple[str, ...], ...]
    output_axes: tuple[str, ...]


def _alphabetic_labels(spec: str) -> tuple[str, ...]:
    return tuple(ch for ch in spec if ch.isalpha())


def _count_ellipsis_dims(operand_spec: str, total_ndim: int) -> int:
    n_explicit = len(_alphabetic_labels(operand_spec))
    if "..." not in operand_spec:
        if total_ndim != n_explicit:
            raise ValueError(
                f"Operand subscript {operand_spec!r} has {n_explicit} explicit labels "
                f"but tensor rank is {total_ndim}."
            )
        return 0
    ell = total_ndim - n_explicit
    if ell < 0:
        raise ValueError(
            f"Operand subscript {operand_spec!r} expects at least {n_explicit} dimensions "
            f"but tensor rank is {total_ndim}."
        )
    return ell


def _pick_unused_labels(n: int, used: set[str]) -> tuple[str, ...]:
    pool: list[str] = []
    for code in range(ord("A"), ord("Z") + 1):
        ch = chr(code)
        if ch not in used:
            pool.append(ch)
    for code in range(ord("a"), ord("z") + 1):
        ch = chr(code)
        if ch not in used:
            pool.append(ch)
    for code in range(0x03B1, 0x03C9 + 1):
        ch = chr(code)
        if ch not in used:
            pool.append(ch)
    if len(pool) < n:
        raise ValueError(
            "Too many implicit ellipsis axes; cannot assign distinct single-character labels."
        )
    return tuple(pool[:n])


def _expand_operand_spec(operand_spec: str, ell_labels: tuple[str, ...]) -> str:
    if "..." not in operand_spec:
        return operand_spec
    before, _, after = operand_spec.partition("...")
    explicit_before = "".join(ch for ch in before if ch.isalpha())
    explicit_after = "".join(ch for ch in after if ch.isalpha())
    return f"{explicit_before}{''.join(ell_labels)}{explicit_after}"


def _binary_operand_specs_before_arrow(equation: str) -> tuple[str, str]:
    """Left and right operand subscript strings (binary only), with or without ``->``."""
    eq = equation.replace(" ", "")
    operands_part = eq.split("->", 1)[0] if "->" in eq else eq
    if operands_part.count(",") != 1:
        raise ValueError("Einsum traces support only binary equations with two operands.")
    left_raw, right_raw = operands_part.split(",")
    return left_raw, right_raw


def _split_raw_equation(equation: str) -> tuple[str, str, str]:
    eq = equation.replace(" ", "")
    if eq.count("->") != 1:
        raise ValueError("Einsum traces require an explicit output using '->'.")
    operands_part, output_spec = eq.split("->", 1)
    if operands_part.count(",") != 1:
        raise ValueError("Einsum traces support only binary equations with two operands.")
    left_raw, right_raw = operands_part.split(",")
    return left_raw, right_raw, output_spec


def _implicit_binary_output_letters(left_raw: str, right_raw: str) -> str:
    """Output subscripts for implicit binary einsum (NumPy: non-repeated labels, sorted)."""
    letters: list[str] = []
    for spec in (left_raw, right_raw):
        letters.extend(ch for ch in spec if ch.isalpha())
    counts = Counter(letters)
    return "".join(sorted(ch for ch, n in counts.items() if n == 1))


def _implicit_nary_output_letters(operand_specs: tuple[str, ...]) -> str:
    """Implicit output for n operands: labels appearing exactly once, sorted (NumPy rule)."""
    letters: list[str] = []
    for spec in operand_specs:
        letters.extend(ch for ch in spec if ch.isalpha())
    counts = Counter(letters)
    return "".join(sorted(ch for ch, n in counts.items() if n == 1))


def canonicalize_binary_einsum_expression(
    expression: str,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> str:
    """Return explicit ``left,right->out`` for binary *expression* (no ``...`` in input)."""
    eq = expression.replace(" ", "")
    if "..." in eq:
        raise ValueError(
            "Einsum equations with ellipsis require an explicit output subscript using '->'."
        )
    if "->" in eq:
        return _validate_explicit_ranks(expression, left_shape, right_shape)
    left_raw, right_raw = _binary_operand_specs_before_arrow(eq)
    if len(left_raw) != len(left_shape) or len(right_raw) != len(right_shape):
        raise ValueError(
            "Subscripts do not match tensor ranks (counts include repeated labels per axis)."
        )
    out_spec = _implicit_binary_output_letters(left_raw, right_raw)
    explicit = f"{left_raw},{right_raw}->{out_spec}"
    zl = np.zeros(left_shape, dtype=np.float64)
    zr = np.zeros(right_shape, dtype=np.float64)
    try:
        z_implicit = np.einsum(eq, zl, zr, optimize=True)
        z_explicit = np.einsum(explicit, zl, zr, optimize=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid einsum equation {eq!r} for operand shapes.") from exc
    if z_implicit.shape != z_explicit.shape:
        raise ValueError(
            f"Implicit equation {eq!r} is inconsistent with inferred explicit form {explicit!r}."
        )
    return explicit


def _binary_explicit_string(
    expression: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> str:
    """Return validated explicit ``->`` equation for two operands (ellipsis allowed)."""
    eq = expression.replace(" ", "")
    if "..." in eq:
        if "->" not in eq:
            raise ValueError(
                "Einsum equations with ellipsis require an explicit output subscript using '->'."
            )
        return _expand_equation_explicit(expression, left_shape=left_shape, right_shape=right_shape)
    if "->" not in eq:
        return canonicalize_binary_einsum_expression(
            expression, left_shape=left_shape, right_shape=right_shape
        )
    return _validate_explicit_ranks(expression, left_shape, right_shape)


def _nary_explicit_string(
    expression: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> str:
    """Explicit ``->`` equation for unary / ternary+ (no ellipsis)."""
    eq = expression.replace(" ", "")
    if "..." in eq:
        raise ValueError(
            "Einsum traces with a single operand or more than two operands do not support "
            "ellipsis; use an explicit subscript list with '->'."
        )
    n = len(operand_shapes)
    operands_part = eq.split("->", 1)[0] if "->" in eq else eq
    raw_ops = tuple(operands_part.split(","))
    if len(raw_ops) != n:
        raise ValueError(
            f"Einsum equation has {len(raw_ops)} operand subscript group(s) "
            f"but received {n} operands."
        )
    for spec, shape in zip(raw_ops, operand_shapes, strict=True):
        if len(spec) != len(shape):
            raise ValueError(
                "Subscripts do not match tensor ranks (counts include repeated labels per axis)."
            )
    explicit = eq if "->" in eq else f"{operands_part}->{_implicit_nary_output_letters(raw_ops)}"
    zs = tuple(np.zeros(s, dtype=np.float64) for s in operand_shapes)
    eq_in = expression.replace(" ", "")
    try:
        z_implicit = np.einsum(eq_in, *zs, optimize=True)
        z_explicit = np.einsum(explicit.replace(" ", ""), *zs, optimize=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid einsum equation {eq_in!r} for operand shapes.") from exc
    if z_implicit.shape != z_explicit.shape:
        raise ValueError(
            f"Implicit equation {eq_in!r} is inconsistent with inferred explicit form {explicit!r}."
        )
    return explicit.replace(" ", "")


def _parse_nary_equation_explicit(explicit: str) -> _ParsedNaryEquation:
    equation = explicit.replace(" ", "")
    if "..." in equation:
        raise ValueError("Internal error: expand ellipsis before n-ary explicit parse.")
    if equation.count("->") != 1:
        raise ValueError("Einsum traces require exactly one '->' in the explicit equation.")
    operands_part, output_spec = equation.split("->", 1)
    specs = operands_part.split(",")
    operand_axes = tuple(tuple(spec) for spec in specs)
    output_axes = tuple(output_spec)
    for group in operand_axes:
        for ch in group:
            if not ch.isalpha():
                raise ValueError(
                    f"Unsupported einsum label {ch!r}; only single alphabetic labels are supported."
                )
    for ch in output_axes:
        if not ch.isalpha():
            raise ValueError(
                f"Unsupported einsum label {ch!r}; only single alphabetic labels are supported."
            )
    return _ParsedNaryEquation(operand_axes=operand_axes, output_axes=output_axes)


def parse_einsum_equation(
    expression: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> _ParsedNaryEquation:
    """Validate *expression* with NumPy for any arity and return explicit per-axis label tuples."""
    if len(operand_shapes) < 1:
        raise ValueError("einsum requires at least one operand shape.")
    if len(operand_shapes) == 2:
        explicit = _binary_explicit_string(expression, operand_shapes[0], operand_shapes[1])
    else:
        explicit = _nary_explicit_string(expression, operand_shapes)
    return _parse_nary_equation_explicit(explicit)


def nary_equation_canonical_string(parsed: _ParsedNaryEquation) -> str:
    """Serialize *parsed* as a single explicit ``subscripts->out`` string."""
    lhs = ",".join("".join(ax) for ax in parsed.operand_axes)
    rhs = "".join(parsed.output_axes)
    return f"{lhs}->{rhs}"


def _expand_equation_explicit(
    equation: str,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> str:
    left_raw, right_raw, output_spec = _split_raw_equation(equation)
    left_ndim = len(left_shape)
    right_ndim = len(right_shape)
    n_ell_left = _count_ellipsis_dims(left_raw, left_ndim)
    n_ell_right = _count_ellipsis_dims(right_raw, right_ndim)
    if n_ell_left != n_ell_right:
        raise ValueError(
            f"Ellipsis dimension count mismatch: left implies {n_ell_left}, "
            f"right implies {n_ell_right}."
        )
    n_ell = n_ell_left

    zl = np.zeros(left_shape, dtype=np.float64)
    zr = np.zeros(right_shape, dtype=np.float64)
    eq = equation.replace(" ", "")
    try:
        z_out = np.einsum(eq, zl, zr, optimize=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid einsum equation for given operand shapes: {eq!r}") from exc

    out_ndim = z_out.ndim
    if "..." in output_spec:
        out_explicit = len(_alphabetic_labels(output_spec))
        expected_out = n_ell + out_explicit
    else:
        expected_out = len(output_spec)
    if out_ndim != expected_out:
        raise ValueError(
            f"Output subscript implies rank {expected_out} but NumPy einsum yields rank {out_ndim}."
        )

    operands_part = eq.split("->")[0]
    used = set(_alphabetic_labels(operands_part + "->" + output_spec))
    ell_labels = _pick_unused_labels(n_ell, used)
    left_exp = _expand_operand_spec(left_raw, ell_labels)
    right_exp = _expand_operand_spec(right_raw, ell_labels)
    output_exp = _expand_operand_spec(output_spec, ell_labels)
    return f"{left_exp},{right_exp}->{output_exp}"


def parse_equation_for_shapes(
    expression: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> _ParsedEquation:
    """Validate *expression* with NumPy and return explicit per-axis label tuples."""
    explicit = _binary_explicit_string(expression, left_shape, right_shape)
    parsed = _parse_nary_equation_explicit(explicit)
    return _ParsedEquation(
        left_axes=parsed.operand_axes[0],
        right_axes=parsed.operand_axes[1],
        output_axes=parsed.output_axes,
    )


def _validate_explicit_ranks(
    expression: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> str:
    eq = expression.replace(" ", "")
    left_raw, right_raw, output_spec = _split_raw_equation(eq)
    if len(left_raw) != len(left_shape) or len(right_raw) != len(right_shape):
        raise ValueError(
            "Subscripts do not match tensor ranks (counts include repeated labels per axis)."
        )
    zl = np.zeros(left_shape, dtype=np.float64)
    zr = np.zeros(right_shape, dtype=np.float64)
    try:
        np.einsum(eq, zl, zr, optimize=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid einsum equation {eq!r} for operand shapes.") from exc
    return eq
