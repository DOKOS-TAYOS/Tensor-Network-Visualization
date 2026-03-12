"""Private data structures used by the traced einsum implementation."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

_TensorState = Literal["available", "bound", "consumed"]


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


@dataclass(frozen=True)
class _PreparedOperand:
    tensor: Any
    tensor_id: int
    name: str
    record_exists: bool


@dataclass(frozen=True)
class _TrackedTensor:
    ref: weakref.ReferenceType[Any]
    name: str
    state: _TensorState


@dataclass(frozen=True)
class _PreparedCall:
    expression: str
    left: _PreparedOperand
    right: _PreparedOperand
    result_name: str
    backend: str
