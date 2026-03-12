"""Parsing and validation helpers for traced binary einsum expressions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ParsedEquation:
    left_axes: tuple[str, ...]
    right_axes: tuple[str, ...]
    output_axes: tuple[str, ...]


def _parse_equation(expression: str) -> _ParsedEquation:
    equation = expression.replace(" ", "")
    if "..." in equation:
        raise ValueError("Einsum ellipsis are not supported by the einsum trace backend.")
    if equation.count("->") != 1:
        raise ValueError("Einsum traces require an explicit output using '->'.")

    operands, output_spec = equation.split("->")
    if operands.count(",") != 1:
        raise ValueError("Einsum traces support only binary equations with two operands.")

    left_spec, right_spec = operands.split(",")
    left_axes = _parse_axis_spec(left_spec, role="left operand")
    right_axes = _parse_axis_spec(right_spec, role="right operand")
    output_axes = _parse_axis_spec(output_spec, role="output")

    left_set = set(left_axes)
    right_set = set(right_axes)
    both_operands = left_set & right_set
    allowed_output = left_set | right_set

    for label in output_axes:
        if label not in allowed_output:
            raise ValueError(f"Output label {label!r} must come from one of the operands.")
        if label in both_operands:
            raise ValueError(
                f"Output label {label!r} cannot come from both operands in the MVP einsum backend."
            )

    return _ParsedEquation(
        left_axes=left_axes,
        right_axes=right_axes,
        output_axes=output_axes,
    )


def _parse_axis_spec(spec: str, *, role: str) -> tuple[str, ...]:
    labels = tuple(spec)
    for label in labels:
        if not label.isalpha():
            raise ValueError(
                f"Unsupported einsum label {label!r} in {role}; "
                "only alphabetic labels are supported."
            )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{role.capitalize()} has repeated labels, which are not supported.")
    return labels
