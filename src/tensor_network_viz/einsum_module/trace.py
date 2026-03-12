"""Utilities for tracing binary ``torch.einsum`` contractions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .._core.graph_utils import _is_unordered_collection
from ._backend import _load_backend_einsum
from ._equation import _parse_equation
from ._trace_state import _TraceState
from ._trace_types import _PreparedCall, pair_tensor

__all__ = [
    "EinsumTrace",
    "einsum",
    "pair_tensor",
]


class EinsumTrace:
    """Stateful trace for automatically recorded binary einsum contractions."""

    def __init__(self) -> None:
        self._state = _TraceState()

    def __iter__(self) -> Iterator[pair_tensor]:
        return iter(self._state.pairs)

    def __len__(self) -> int:
        return len(self._state.pairs)

    def bind(self, name: str, tensor: Any) -> None:
        self._state.bind(name, tensor)

    def _prepare_call(
        self,
        expression: str,
        left: Any,
        right: Any,
        *,
        backend: str,
    ) -> _PreparedCall:
        return self._state.prepare_call(expression, left, right, backend=backend)

    def _commit_call(self, prepared: _PreparedCall, result: Any) -> None:
        self._state.commit_call(prepared, result)


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


def einsum(
    expression: Any,
    *operands: Any,
    trace: EinsumTrace | None = None,
    backend: str = "torch",
    **kwargs: Any,
) -> Any:
    """Execute einsum on the selected backend and optionally record the contraction."""
    backend_name = str(backend).lower()
    backend_fn = _load_backend_einsum(backend_name)

    if trace is None:
        return backend_fn(expression, *operands, **kwargs)

    if not isinstance(trace, EinsumTrace):
        raise TypeError("trace must be an EinsumTrace instance.")
    if "out" in kwargs:
        raise ValueError("Traced einsum does not support out= because it breaks result tracking.")
    if not isinstance(expression, str):
        raise TypeError("Traced einsum requires a string equation.")
    if len(operands) != 2:
        raise ValueError("Traced einsum currently supports exactly 2 operands.")

    _parse_equation(expression)
    prepared = trace._prepare_call(expression, operands[0], operands[1], backend=backend_name)
    result = backend_fn(expression, *operands, **kwargs)
    trace._commit_call(prepared, result)
    return result
