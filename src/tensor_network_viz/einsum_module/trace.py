"""Utilities for tracing binary ``torch.einsum`` contractions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .._core.graph_utils import _is_unordered_collection
from ._backend import _load_backend_einsum, _shape_tuple
from ._equation import nary_equation_canonical_string, parse_einsum_equation
from ._trace_state import _TraceState
from ._trace_types import _PreparedCall, einsum_trace_step, pair_tensor

__all__ = [
    "EinsumTrace",
    "einsum",
    "einsum_trace_step",
    "pair_tensor",
]


def _is_unsupported_out_keyword_error(exc: TypeError) -> bool:
    message = str(exc).lower()
    return (
        "unexpected keyword argument 'out'" in message
        or 'unexpected keyword argument "out"' in message
    )


def _validate_traced_out_argument(
    *,
    trace: EinsumTrace,
    operands: tuple[Any, ...],
    out_tensor: Any,
    expression: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> None:
    if trace._state.tensor_is_tracked(out_tensor):
        raise ValueError(
            "Traced einsum does not support out= pointing to a tensor already on this trace "
            "(in-place into a traced tensor is unsupported)."
        )
    if any(out_tensor is operand for operand in operands):
        raise ValueError("Traced einsum does not support out= reusing one of the current operands.")

    out_shape = _shape_tuple(out_tensor)
    if out_shape is None:
        raise TypeError("out= tensor must expose a shape.")
    zs = tuple(np.zeros(shape, dtype=np.float64) for shape in operand_shapes)
    eq_np = expression.replace(" ", "")
    try:
        expected_shape = np.einsum(eq_np, *zs, optimize=True).shape
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid einsum equation {eq_np!r} for operand shapes.") from exc
    if tuple(out_shape) != expected_shape:
        raise ValueError(
            f"out= shape {out_shape} does not match einsum result shape {expected_shape}."
        )


class EinsumTrace:
    """Stateful trace for automatically recorded einsum contractions (any arity)."""

    def __init__(self) -> None:
        self._state = _TraceState()

    def __iter__(self) -> Iterator[pair_tensor | einsum_trace_step]:
        return iter(self._state.pairs)

    def __len__(self) -> int:
        return len(self._state.pairs)

    def bind(self, name: str, tensor: Any) -> None:
        self._state.bind(name, tensor)

    def _prepare_call(
        self,
        expression: str,
        operands: tuple[Any, ...],
        *,
        backend: str,
    ) -> _PreparedCall:
        return self._state.prepare_call(expression, operands, backend=backend)

    def _commit_call(self, prepared: _PreparedCall, result: Any) -> None:
        self._state.commit_call(prepared, result)


def _normalize_trace(trace: Any) -> list[pair_tensor | einsum_trace_step]:
    if isinstance(trace, (str, bytes, bytearray)):
        raise TypeError(
            "Einsum traces must be an ordered iterable of pair_tensor or einsum_trace_step objects."
        )
    if _is_unordered_collection(trace):
        raise TypeError(
            "Einsum traces must preserve order; unordered collections are not supported."
        )
    if isinstance(trace, dict):
        raise TypeError(
            "Einsum traces must be an ordered iterable of pair_tensor or einsum_trace_step objects."
        )
    if not isinstance(trace, Iterable):
        raise TypeError(
            "Einsum traces must be an ordered iterable of pair_tensor or einsum_trace_step objects."
        )

    try:
        items = list(trace)
    except TypeError as exc:
        raise TypeError("Einsum traces must be iterable.") from exc
    if not items:
        raise ValueError("The einsum trace does not contain any trace entries.")
    if not all(isinstance(item, (pair_tensor, einsum_trace_step)) for item in items):
        raise TypeError("Einsum trace entries must be pair_tensor or einsum_trace_step instances.")
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
    if not isinstance(expression, str):
        raise TypeError("Traced einsum requires a string equation.")
    if len(operands) < 1:
        raise ValueError("Traced einsum requires at least one operand.")

    operand_shapes_list: list[tuple[int, ...]] = []
    for op in operands:
        sh = _shape_tuple(op)
        if sh is None:
            raise TypeError("Traced einsum operands must expose shape information.")
        operand_shapes_list.append(sh)
    operand_shapes = tuple(operand_shapes_list)

    out_kw = kwargs.get("out")
    parsed = parse_einsum_equation(expression, operand_shapes)
    canonical_eq = nary_equation_canonical_string(parsed)

    if out_kw is not None:
        _validate_traced_out_argument(
            trace=trace,
            operands=operands,
            out_tensor=out_kw,
            expression=expression,
            operand_shapes=operand_shapes,
        )

    prepared = trace._prepare_call(canonical_eq, operands, backend=backend_name)
    if backend_name == "torch" and "out" in kwargs:
        out_tensor = kwargs["out"]
        sub_kwargs = {k: v for k, v in kwargs.items() if k != "out"}
        try:
            result = backend_fn(expression, *operands, **kwargs)
        except TypeError as exc:
            if not _is_unsupported_out_keyword_error(exc):
                raise
            tmp = backend_fn(expression, *operands, **sub_kwargs)
            out_tensor.copy_(tmp)
            result = out_tensor
    else:
        result = backend_fn(expression, *operands, **kwargs)
    trace._commit_call(prepared, result)
    return result
