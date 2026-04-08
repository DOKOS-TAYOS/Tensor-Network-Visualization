"""Mutable state container for automatically traced einsum calls."""

from __future__ import annotations

from typing import Any

from ._backend import _dtype_text, _make_weakref, _shape_tuple
from ._trace_types import (
    _PreparedCall,
    _PreparedOperand,
    _TrackedTensor,
    einsum_trace_step,
    pair_tensor,
)


class _TraceState:
    def __init__(self) -> None:
        self._pairs: list[pair_tensor | einsum_trace_step] = []
        self._records: dict[int, _TrackedTensor] = {}
        self._pair_names: set[str] = set()
        self._reserved_names: set[str] = set()
        self._next_tensor_index = 0
        self._next_result_index = 0

    @property
    def pairs(self) -> list[pair_tensor | einsum_trace_step]:
        return self._pairs

    def tensor_is_tracked(self, tensor: Any) -> bool:
        """True if *tensor* is already registered (bound, available, or consumed) on this trace."""
        self._sweep_dead_records()
        return self._find_record(tensor)[1] is not None

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

    def prepare_call(
        self,
        expression: str,
        operands: tuple[Any, ...],
        *,
        backend: str,
    ) -> _PreparedCall:
        if len(operands) < 1:
            raise ValueError("Traced einsum requires at least one operand.")
        if len({id(t) for t in operands}) != len(operands):
            raise ValueError("Traced einsum requires distinct operand objects.")

        self._sweep_dead_records()
        reserved_names = set(self._reserved_names)
        resolved: list[_PreparedOperand] = []
        next_tensor_index = self._next_tensor_index
        for tensor in operands:
            op, next_tensor_index = self._resolve_operand(
                tensor,
                next_tensor_index,
                reserved_names=reserved_names,
            )
            resolved.append(op)
        names = [op.name for op in resolved]
        if len(set(names)) != len(names):
            raise ValueError("Traced einsum requires distinct operand names.")
        result_name = self._peek_name("r", self._next_result_index, reserved_names=reserved_names)
        return _PreparedCall(
            expression=expression,
            operands=tuple(resolved),
            result_name=result_name,
            backend=backend,
        )

    def commit_call(self, prepared: _PreparedCall, result: Any) -> None:
        operand_updates = self._consumed_operand_updates(prepared)
        result_id, result_record = self._result_record_for_commit(prepared, result)
        trace_entry, reserved_names = self._trace_entry_for_commit(prepared, result)

        for tensor_id, record in operand_updates:
            self._set_record(tensor_id, record)
        self._set_record(result_id, result_record)
        self._pairs.append(trace_entry)
        self._reserve_pair_names(*reserved_names)

        new_tensor_count = sum(not op.record_exists for op in prepared.operands)
        self._next_tensor_index += new_tensor_count
        self._next_result_index += 1

    def _consumed_operand_updates(
        self,
        prepared: _PreparedCall,
    ) -> list[tuple[int, _TrackedTensor]]:
        updates: list[tuple[int, _TrackedTensor]] = []
        for operand in prepared.operands:
            current = self._records.get(operand.tensor_id)
            if current is None:
                updates.append(
                    (
                        operand.tensor_id,
                        _TrackedTensor(
                            ref=_make_weakref(operand.tensor),
                            name=operand.name,
                            state="consumed",
                        ),
                    )
                )
                continue
            updates.append(
                (
                    operand.tensor_id,
                    _TrackedTensor(
                        ref=current.ref,
                        name=current.name,
                        state="consumed",
                    ),
                )
            )
        return updates

    def _result_record_for_commit(
        self,
        prepared: _PreparedCall,
        result: Any,
    ) -> tuple[int, _TrackedTensor]:
        if any(result is operand.tensor for operand in prepared.operands):
            raise ValueError("Backend returned a tensor that is already tracked by this trace.")
        result_id, existing_result = self._find_record(result)
        if existing_result is not None:
            raise ValueError("Backend returned a tensor that is already tracked by this trace.")
        return (
            result_id,
            _TrackedTensor(
                ref=_make_weakref(result),
                name=prepared.result_name,
                state="available",
            ),
        )

    def _trace_entry_for_commit(
        self,
        prepared: _PreparedCall,
        result: Any,
    ) -> tuple[pair_tensor | einsum_trace_step, tuple[str, ...]]:
        n_op = len(prepared.operands)
        if n_op == 2:
            left, right = prepared.operands
            metadata = {
                "backend": prepared.backend,
                "left_dtype": _dtype_text(left.tensor),
                "left_shape": _shape_tuple(left.tensor),
                "result_dtype": _dtype_text(result),
                "result_shape": _shape_tuple(result),
                "right_dtype": _dtype_text(right.tensor),
                "right_shape": _shape_tuple(right.tensor),
            }
            return (
                pair_tensor(
                    left.name,
                    right.name,
                    prepared.result_name,
                    prepared.expression,
                    metadata=metadata,
                ),
                (left.name, right.name, prepared.result_name),
            )

        metadata = {
            "backend": prepared.backend,
            "operand_dtypes": tuple(_dtype_text(op.tensor) for op in prepared.operands),
            "operand_shapes": tuple(_shape_tuple(op.tensor) for op in prepared.operands),
            "result_dtype": _dtype_text(result),
            "result_shape": _shape_tuple(result),
        }
        return (
            einsum_trace_step(
                tuple(op.name for op in prepared.operands),
                prepared.result_name,
                prepared.expression,
                metadata=metadata,
            ),
            tuple(op.name for op in prepared.operands) + (prepared.result_name,),
        )

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
