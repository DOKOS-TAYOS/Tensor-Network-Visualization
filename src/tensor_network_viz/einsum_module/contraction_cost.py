"""Naive dense contraction cost and readable step details for parsed einsum steps."""

from __future__ import annotations

from textwrap import fill

from .._core.graph import _ContractionStepMetrics
from ._equation import _ParsedNaryEquation


def metrics_for_labeled_operands(
    *,
    operand_axes: tuple[tuple[str, ...], ...],
    operand_shapes: tuple[tuple[int, ...], ...],
    output_axes: tuple[str, ...],
    equation_snippet: str | None = None,
    operand_names: tuple[str, ...] | None = None,
) -> _ContractionStepMetrics:
    """Return naive dense cost metadata for one labeled contraction step."""
    if len(operand_axes) != len(operand_shapes):
        raise ValueError("operand_axes length must match operand_shapes length.")
    resolved_operand_names = tuple(operand_names or ())
    if resolved_operand_names and len(resolved_operand_names) != len(operand_shapes):
        raise ValueError("operand_names length must match operand_shapes length.")

    dim_by_label: dict[str, int] = {}
    for axes, shape in zip(operand_axes, operand_shapes, strict=True):
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

    label_order = _labels_in_first_occurrence_order_from_axes(operand_axes)
    label_dims = tuple((ch, dim_by_label[ch]) for ch in label_order)
    output_set = set(output_axes)
    contracted_labels = tuple(ch for ch in label_order if ch not in output_set)

    prod = 1
    for ch in label_order:
        prod *= dim_by_label[ch]
    flop_estimate = prod * len(operand_shapes)

    return _ContractionStepMetrics(
        label_dims=label_dims,
        multiplicative_cost=prod,
        flop_mac=flop_estimate,
        equation_snippet=equation_snippet,
        operand_names=resolved_operand_names,
        operand_shapes=tuple(tuple(int(dim) for dim in shape) for shape in operand_shapes),
        output_labels=tuple(output_axes),
        contracted_labels=contracted_labels,
        label_order=label_order,
    )


def metrics_for_parsed_step(
    parsed: _ParsedNaryEquation,
    operand_shapes: tuple[tuple[int, ...], ...],
    *,
    equation_snippet: str | None = None,
    operand_names: tuple[str, ...] | None = None,
) -> _ContractionStepMetrics:
    """Return naive dense cost metadata for one einsum contraction step."""
    return metrics_for_labeled_operands(
        operand_axes=parsed.operand_axes,
        operand_shapes=operand_shapes,
        output_axes=parsed.output_axes,
        equation_snippet=equation_snippet,
        operand_names=operand_names,
    )


def format_contraction_step_panel_text(m: _ContractionStepMetrics) -> str:
    """Readable multiline detail panel for the current contraction step."""
    lines: list[str] = []
    header = _contraction_header_text(m)
    if header:
        lines.append(_wrap_panel_line(header))
    if m.label_dims:
        label_parts = [f"{label}={_format_big_int(dim)}" for label, dim in m.label_dims]
        lines.append(_wrap_panel_line("Index sizes: " + ", ".join(label_parts)))
    if m.operand_shapes:
        shape_parts: list[str] = []
        for index, shape in enumerate(m.operand_shapes):
            shape_parts.append(f"{_operand_display_name(m, index)}={list(shape)}")
        lines.append(_wrap_panel_line("Tensor shapes: " + ", ".join(shape_parts)))
    lines.append(
        _wrap_panel_line(
            "Naive operations: "
            f"{_format_big_int(m.multiplicative_cost)} MACs "
            f"(\u2248{_format_big_int(m.flop_mac)} FLOPs)"
        )
    )
    complexity = _complexity_line(m)
    if complexity:
        lines.append(_wrap_panel_line(complexity))
    return "\n".join(lines)


def format_contraction_step_tooltip(m: _ContractionStepMetrics) -> str:
    """Backward-compatible alias for the contraction step detail formatter."""
    return format_contraction_step_panel_text(m)


def _labels_in_first_occurrence_order_from_axes(
    operand_axes: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for axes in operand_axes:
        for ch in axes:
            if ch in seen:
                continue
            seen.add(ch)
            ordered.append(ch)
    return tuple(ordered)


def _contraction_header_text(m: _ContractionStepMetrics) -> str | None:
    contracted = ", ".join(m.contracted_labels)
    if m.equation_snippet and contracted:
        return f"Contraction: {m.equation_snippet} (contracts: {contracted})"
    if m.equation_snippet:
        return f"Contraction: {m.equation_snippet}"
    if contracted:
        return f"Contraction indices: {contracted}"
    return None


def _operand_display_name(m: _ContractionStepMetrics, index: int) -> str:
    if index < len(m.operand_names) and m.operand_names[index]:
        return str(m.operand_names[index])
    return f"Operand {index}"


def _complexity_line(m: _ContractionStepMetrics) -> str | None:
    labels = m.label_order if m.label_order else tuple(label for label, _dim in m.label_dims)
    if not labels:
        return None
    symbolic_terms = " ".join(f"N_{label}" for label in labels)
    return f"Complexity: O({symbolic_terms})"


def _wrap_panel_line(text: str, width: int = 96) -> str:
    return fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        subsequent_indent="    ",
    )


def _format_big_int(n: int) -> str:
    if abs(n) < 10_000_000_000:
        return f"{n:,}"
    return f"{n:.6e}"
