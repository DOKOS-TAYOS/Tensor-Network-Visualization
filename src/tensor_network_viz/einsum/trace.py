"""Utilities for tracing binary ``torch.einsum`` contractions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._core.graph_utils import _is_unordered_collection


@dataclass(frozen=True)
class _ParsedEquation:
    left_axes: tuple[str, ...]
    right_axes: tuple[str, ...]
    output_axes: tuple[str, ...]


class pair_tensor(str):
    """Trace entry for one binary einsum contraction."""

    __slots__ = ("left_name", "right_name", "result_name", "metadata")

    left_name: str
    right_name: str
    result_name: str
    metadata: Mapping[str, Any] | None

    def __new__(
        cls,
        left_name: str,
        right_name: str,
        result_name: str,
        expression: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> pair_tensor:
        obj = super().__new__(cls, expression)
        obj.left_name = str(left_name)
        obj.right_name = str(right_name)
        obj.result_name = str(result_name)
        obj.metadata = None if metadata is None else dict(metadata)
        return obj

    @property
    def equation(self) -> str:
        return str(self)


def _normalize_trace(trace: Any) -> list[pair_tensor]:
    if isinstance(trace, (str, bytes, bytearray)):
        raise TypeError("Einsum traces must be an ordered iterable of pair_tensor objects.")
    if _is_unordered_collection(trace):
        raise TypeError(
            "Einsum traces must preserve order; unordered collections are not supported."
        )
    if isinstance(trace, dict):
        raise TypeError("Einsum traces must be an ordered iterable of pair_tensor objects.")
    if not isinstance(trace, Iterable):
        raise TypeError("Einsum traces must be an ordered iterable of pair_tensor objects.")

    try:
        items = list(trace)
    except TypeError as exc:
        raise TypeError("Einsum traces must be iterable.") from exc
    if not items:
        raise ValueError("The einsum trace does not contain any pair_tensor entries.")
    if not all(isinstance(item, pair_tensor) for item in items):
        raise TypeError("Einsum trace entries must be pair_tensor instances.")
    return items


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
                f"Unsupported einsum label {label!r} in {role}; only alphabetic labels "
                "are supported."
            )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{role.capitalize()} has repeated labels, which are not supported.")
    return labels
