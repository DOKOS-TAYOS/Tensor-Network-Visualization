"""Utilities for tracing binary ``torch.einsum`` contractions."""

from __future__ import annotations

import importlib
import weakref
from collections.abc import Iterable, Iterator, Mapping
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
    state: str


@dataclass(frozen=True)
class _PreparedCall:
    expression: str
    left: _PreparedOperand
    right: _PreparedOperand
    result_name: str
    backend: str


class EinsumTrace:
    """Stateful trace for automatically recorded binary einsum contractions."""

    def __init__(self) -> None:
        self._pairs: list[pair_tensor] = []
        self._records: dict[int, _TrackedTensor] = {}
        self._pair_names: set[str] = set()
        self._reserved_names: set[str] = set()
        self._next_tensor_index = 0
        self._next_result_index = 0

    def __iter__(self) -> Iterator[pair_tensor]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def bind(self, name: str, tensor: Any) -> None:
        bound_name = str(name)
        if not bound_name:
            raise ValueError("Tensor binding names must be non-empty.")

        self._sweep_dead_records()
        tensor_id, record = self._find_record(tensor)
        self._ensure_name_is_available(bound_name, exclude_tensor_id=tensor_id)
        if record is None:
            self._set_record(
                tensor_id,
                _TrackedTensor(
                    ref=_make_weakref(tensor),
                    name=bound_name,
                    state="bound",
                ),
            )
            return
        if record.state != "bound":
            raise ValueError("Tensor has already been traced and cannot be rebound.")
        self._set_record(
            tensor_id,
            _TrackedTensor(
                ref=record.ref,
                name=bound_name,
                state="bound",
            ),
        )

    def _prepare_call(
        self,
        expression: str,
        left: Any,
        right: Any,
        *,
        backend: str,
    ) -> _PreparedCall:
        if left is right:
            raise ValueError("Traced einsum requires distinct operand objects.")

        self._sweep_dead_records()
        reserved_names = set(self._reserved_names)
        left_operand, next_tensor_index = self._resolve_operand(
            left,
            self._next_tensor_index,
            reserved_names=reserved_names,
        )
        right_operand, next_tensor_index = self._resolve_operand(
            right,
            next_tensor_index,
            reserved_names=reserved_names,
        )
        if left_operand.name == right_operand.name:
            raise ValueError("Traced einsum requires distinct operand names.")
        result_name = self._peek_name("r", self._next_result_index, reserved_names=reserved_names)
        return _PreparedCall(
            expression=expression,
            left=left_operand,
            right=right_operand,
            result_name=result_name,
            backend=backend,
        )

    def _commit_call(self, prepared: _PreparedCall, result: Any) -> None:
        for operand in (prepared.left, prepared.right):
            current = self._records.get(operand.tensor_id)
            if current is None:
                self._set_record(
                    operand.tensor_id,
                    _TrackedTensor(
                        ref=_make_weakref(operand.tensor),
                        name=operand.name,
                        state="consumed",
                    ),
                )
            else:
                self._set_record(
                    operand.tensor_id,
                    _TrackedTensor(
                        ref=current.ref,
                        name=current.name,
                        state="consumed",
                    ),
                )

        result_id, existing_result = self._find_record(result)
        if existing_result is not None:
            raise ValueError("Backend returned a tensor that is already tracked by this trace.")
        self._set_record(
            result_id,
            _TrackedTensor(
                ref=_make_weakref(result),
                name=prepared.result_name,
                state="available",
            ),
        )
        metadata = {
            "backend": prepared.backend,
            "left_dtype": _dtype_text(prepared.left.tensor),
            "left_shape": _shape_tuple(prepared.left.tensor),
            "result_dtype": _dtype_text(result),
            "result_shape": _shape_tuple(result),
            "right_dtype": _dtype_text(prepared.right.tensor),
            "right_shape": _shape_tuple(prepared.right.tensor),
        }
        self._pairs.append(
            pair_tensor(
                prepared.left.name,
                prepared.right.name,
                prepared.result_name,
                prepared.expression,
                metadata=metadata,
            )
        )
        self._reserve_pair_names(
            prepared.left.name,
            prepared.right.name,
            prepared.result_name,
        )

        new_tensor_count = sum(
            not operand.record_exists for operand in (prepared.left, prepared.right)
        )
        self._next_tensor_index += new_tensor_count
        self._next_result_index += 1

    def _resolve_operand(
        self,
        tensor: Any,
        next_tensor_index: int,
        *,
        reserved_names: set[str] | None = None,
    ) -> tuple[_PreparedOperand, int]:
        tensor_id, record = self._find_record(tensor)
        if record is None:
            name = self._peek_name("t", next_tensor_index, reserved_names=reserved_names)
            if reserved_names is not None:
                reserved_names.add(name)
            return (
                _PreparedOperand(
                    tensor=tensor,
                    tensor_id=tensor_id,
                    name=name,
                    record_exists=False,
                ),
                next_tensor_index + 1,
            )
        if record.state == "consumed":
            raise ValueError(f"Tensor {record.name!r} has already been consumed by this trace.")
        return (
            _PreparedOperand(
                tensor=tensor,
                tensor_id=tensor_id,
                name=record.name,
                record_exists=True,
            ),
            next_tensor_index,
        )

    def _find_record(self, tensor: Any) -> tuple[int, _TrackedTensor | None]:
        tensor_id = id(tensor)
        record = self._records.get(tensor_id)
        if record is None:
            return tensor_id, None
        current = record.ref()
        if current is not tensor:
            self._discard_record(tensor_id, record)
            return tensor_id, None
        return tensor_id, record

    def _peek_name(
        self,
        prefix: str,
        start_index: int,
        *,
        reserved_names: set[str] | None = None,
    ) -> str:
        in_use = self._reserved_names if reserved_names is None else reserved_names
        index = start_index
        while True:
            candidate = f"{prefix}{index}"
            if candidate not in in_use:
                return candidate
            index += 1

    def _ensure_name_is_available(self, name: str, *, exclude_tensor_id: int) -> None:
        self._sweep_dead_records()
        excluded_name: str | None = None
        if exclude_tensor_id in self._records:
            excluded_name = self._records[exclude_tensor_id].name
        if name in self._reserved_names and name != excluded_name:
            raise ValueError(f"Tensor name {name!r} is already in use by this trace.")

    def _set_record(self, tensor_id: int, record: _TrackedTensor) -> None:
        previous = self._records.get(tensor_id)
        if (
            previous is not None
            and previous.name != record.name
            and previous.name not in self._pair_names
        ):
            self._reserved_names.discard(previous.name)
        self._records[tensor_id] = record
        self._reserved_names.add(record.name)

    def _reserve_pair_names(self, *names: str) -> None:
        for name in names:
            self._pair_names.add(name)
            self._reserved_names.add(name)

    def _discard_record(self, tensor_id: int, record: _TrackedTensor | None = None) -> None:
        tracked = record if record is not None else self._records.get(tensor_id)
        if tracked is None:
            return
        self._records.pop(tensor_id, None)
        if tracked.name not in self._pair_names:
            self._reserved_names.discard(tracked.name)

    def _sweep_dead_records(self) -> None:
        for tensor_id, record in list(self._records.items()):
            if record.ref() is None:
                self._discard_record(tensor_id, record)


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


def _load_backend_einsum(backend: str) -> Any:
    if backend == "torch":
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError("torch is required for backend='torch'.") from exc
        return torch.einsum
    if backend == "numpy":
        try:
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ImportError("numpy is required for backend='numpy'.") from exc
        return np.einsum
    raise ValueError(f"Unsupported einsum backend: {backend}")


def _make_weakref(tensor: Any) -> weakref.ReferenceType[Any]:
    try:
        return weakref.ref(tensor)
    except TypeError as exc:
        raise TypeError("Traced einsum operands must support weak references.") from exc


def _shape_tuple(tensor: Any) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except TypeError:
        return None


def _dtype_text(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)
