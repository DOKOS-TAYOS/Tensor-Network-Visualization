"""Naive dense contraction cost from a parsed einsum step and operand shapes."""

from __future__ import annotations

from .._core.graph import _ContractionStepMetrics
from ._equation import _ParsedNaryEquation


def metrics_for_parsed_step(
    parsed: _ParsedNaryEquation,
    operand_shapes: tuple[tuple[int, ...], ...],
    *,
    equation_snippet: str | None = None,
) -> _ContractionStepMetrics:
    """Return multiplicative cost C and MAC FLOPs F = 2C for one dense naive einsum evaluation.

    C is the product of d_ℓ over every index label ℓ that appears on any operand (each label
    once). This matches the total number of scalar multiply–add iterations in a single nested
    loop over all those indices. When a label appears with extent 1 on one operand and a larger
    extent on another (placeholder live tensors), the larger extent is kept—consistent with how
    traced graphs use ``(1,)*rank`` intermediates.
    """
    if len(parsed.operand_axes) != len(operand_shapes):
        raise ValueError("operand_axes length must match operand_shapes length.")
    dim_by_label: dict[str, int] = {}
    for axes, shape in zip(parsed.operand_axes, operand_shapes, strict=True):
        if len(axes) != len(shape):
            raise ValueError("Each operand must have one shape entry per axis label.")
        for ch, d in zip(axes, shape, strict=True):
            d_int = int(d)
            if d_int < 0:
                raise ValueError(f"Negative dimension for label {ch!r}.")
            prev = dim_by_label.get(ch)
            if prev is None:
                dim_by_label[ch] = d_int
            elif prev == d_int:
                continue
            elif prev == 1:
                dim_by_label[ch] = d_int
            elif d_int == 1:
                continue
            else:
                raise ValueError(f"Inconsistent dimension for label {ch!r}: {prev} vs {d_int}.")

    labels_in_step: set[str] = set()
    for axes in parsed.operand_axes:
        labels_in_step.update(axes)

    prod = 1
    for ch in labels_in_step:
        prod *= dim_by_label[ch]

    label_dims = tuple(sorted(dim_by_label.items(), key=lambda x: x[0]))
    flop_mac = prod * 2
    return _ContractionStepMetrics(
        label_dims=label_dims,
        multiplicative_cost=prod,
        flop_mac=flop_mac,
        equation_snippet=equation_snippet,
    )


def format_contraction_step_tooltip(m: _ContractionStepMetrics) -> str:
    """Multiline tooltip: index dimensions, C, and F = 2C (readable large integers)."""
    lines: list[str] = []
    if m.equation_snippet:
        lines.append(m.equation_snippet)
    if m.label_dims:
        parts = [f"  {ch}: {d}" for ch, d in m.label_dims]
        lines.append("Indices:\n" + "\n".join(parts))
    lines.append(f"C (product of dims): {_format_big_int(m.multiplicative_cost)}")
    lines.append(f"FLOPs (MAC, ≈2C): {_format_big_int(m.flop_mac)}")
    return "\n".join(lines)


def _format_big_int(n: int) -> str:
    if abs(n) < 10_000_000_000:
        return f"{n:,}"
    return f"{n:.6e}"
