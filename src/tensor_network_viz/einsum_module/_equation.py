"""Parsing, ellipsis expansion, and validation for binary einsum expressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ParsedEquation:
    left_axes: tuple[str, ...]
    right_axes: tuple[str, ...]
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


def _split_raw_equation(equation: str) -> tuple[str, str, str]:
    eq = equation.replace(" ", "")
    if eq.count("->") != 1:
        raise ValueError("Einsum traces require an explicit output using '->'.")
    operands_part, output_spec = eq.split("->")
    if operands_part.count(",") != 1:
        raise ValueError("Einsum traces support only binary equations with two operands.")
    left_raw, right_raw = operands_part.split(",")
    return left_raw, right_raw, output_spec


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
    except Exception as exc:
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


def _parse_equation_explicit(expression: str) -> _ParsedEquation:
    equation = expression.replace(" ", "")
    if "..." in equation:
        raise ValueError("Internal error: expand ellipsis before explicit parse.")
    left_raw, right_raw, output_spec = _split_raw_equation(equation)
    left_axes = tuple(left_raw)
    right_axes = tuple(right_raw)
    output_axes = tuple(output_spec)

    for ch in left_axes + right_axes + output_axes:
        if not ch.isalpha():
            raise ValueError(
                f"Unsupported einsum label {ch!r}; only single alphabetic labels are supported."
            )

    return _ParsedEquation(
        left_axes=left_axes,
        right_axes=right_axes,
        output_axes=output_axes,
    )


def parse_equation_for_shapes(
    expression: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> _ParsedEquation:
    """Validate *expression* with NumPy and return explicit per-axis label tuples."""
    eq = expression.replace(" ", "")
    explicit = (
        _expand_equation_explicit(expression, left_shape=left_shape, right_shape=right_shape)
        if "..." in eq
        else _validate_explicit_ranks(expression, left_shape, right_shape)
    )
    return _parse_equation_explicit(explicit)


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
    except Exception as exc:
        raise ValueError(f"Invalid einsum equation {eq!r} for operand shapes.") from exc
    return eq


def _parse_equation(
    expression: str,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> _ParsedEquation:
    return parse_equation_for_shapes(expression, left_shape, right_shape)
