"""Tensor extraction helpers for tensor-elements inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import tee
from typing import TYPE_CHECKING, Any

import numpy as np

from ._engine_specs import EngineName
from ._input_inspection import (
    _detect_tensor_engine_with_input,
    _is_tenpy_tensor,
    _is_unordered_collection,
)
from ._logging import package_logger
from ._tensor_elements_models import (
    NumericArray,
    TensorElementsSourceName,
    _PlaybackStepRecord,
    _TensorRecord,
)
from .exceptions import TensorDataError, TensorDataTypeError, UnsupportedEngineError

if TYPE_CHECKING:
    from .einsum_module._trace_types import einsum_trace_step, pair_tensor


class _DirectArrayDetectionError(Exception):
    """Internal marker for unexpected array-like failures during lightweight probing."""


def _detect_tensor_elements_engine(data: Any) -> tuple[EngineName, Any]:
    """Detect the tensor backend and preserve any prepared input wrapper."""
    return _detect_tensor_engine_with_input(data)


def parse_einsum_equation(
    equation: str,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> Any:
    from .einsum_module._equation import parse_einsum_equation as _parse_einsum_equation

    return _parse_einsum_equation(equation, operand_shapes)


def _has_exact_runtime_type(value: Any, *, module: str, qualname: str) -> bool:
    value_type = type(value)
    return value_type.__module__ == module and value_type.__qualname__ == qualname


def _is_einsum_trace_object(value: Any) -> bool:
    return _has_exact_runtime_type(
        value,
        module="tensor_network_viz.einsum_module.trace",
        qualname="EinsumTrace",
    )


def _is_einsum_pair_step(value: Any) -> bool:
    return _has_exact_runtime_type(
        value,
        module="tensor_network_viz.einsum_module._trace_types",
        qualname="pair_tensor",
    )


def _is_explicit_tenpy_network(value: Any) -> bool:
    return _has_exact_runtime_type(
        value,
        module="tensor_network_viz.tenpy.explicit",
        qualname="TenPyTensorNetwork",
    )


def _looks_like_backend_tensor_input(value: Any) -> bool:
    return (
        _is_einsum_trace_object(value)
        or _is_tenpy_tensor(value)
        or hasattr(value, "tensors")
        or hasattr(value, "leaf_nodes")
        or hasattr(value, "nodes")
        or hasattr(value, "axes_names")
        or (hasattr(value, "axis_names") and hasattr(value, "tensor"))
        or (hasattr(value, "inds") and hasattr(value, "data"))
    )


def _asarray_for_direct_tensor_detection(value: Any) -> NumericArray:
    """Probe array-like inputs while preserving broken ``__array__`` failures as type errors."""
    try:
        return _to_numpy_array(value)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise _DirectArrayDetectionError(str(exc)) from exc


def _is_direct_array_like_tensor(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    if _looks_like_backend_tensor_input(value):
        return False
    if not hasattr(value, "shape") and not hasattr(value, "__array__"):
        return False
    try:
        array = _asarray_for_direct_tensor_detection(value)
    except _DirectArrayDetectionError as exc:
        raise TypeError(str(exc)) from exc.__cause__
    except (TypeError, ValueError):
        return False
    return array.dtype != np.dtype("O")


def _direct_array_record_name(index: int, *, total: int) -> str:
    if total <= 1:
        return "Tensor"
    return f"Tensor {index + 1}"


def _extract_direct_array_records(data: Any) -> tuple[list[_TensorRecord] | None, Any]:
    if _is_direct_array_like_tensor(data):
        return (
            [
                _TensorRecord(
                    array=_to_numpy_array(data),
                    axis_names=(),
                    engine="numpy",
                    name=_direct_array_record_name(0, total=1),
                )
            ],
            data,
        )
    if isinstance(data, (str, bytes, bytearray, dict)) or _looks_like_backend_tensor_input(data):
        return None, data
    if not isinstance(data, Iterable):
        return None, data
    prepared_data = data
    try:
        iterator = iter(data)
        if iterator is data:
            probe_iter, prepared_data = tee(iterator)
            items = list(probe_iter)
        else:
            items = list(data)
    except TypeError:
        return None, data
    if not items or not all(_is_direct_array_like_tensor(item) for item in items):
        return None, prepared_data
    return (
        [
            _TensorRecord(
                array=_to_numpy_array(item),
                axis_names=(),
                engine="numpy",
                name=_direct_array_record_name(index, total=len(items)),
            )
            for index, item in enumerate(items)
        ],
        prepared_data,
    )


def _name_sort_key(item: Any) -> tuple[str, tuple[str, ...], str, str]:
    axis_attr = getattr(item, "axis_names", None)
    if axis_attr is None:
        axis_attr = getattr(item, "axes_names", None)
    if axis_attr is None:
        axis_attr = getattr(item, "inds", ())
    axis_names = tuple("" if value is None else str(value) for value in axis_attr)
    name = None if getattr(item, "name", None) is None else str(item.name)
    return (
        "" if name is None else name,
        axis_names,
        type(item).__module__,
        type(item).__qualname__,
    )


def _iter_attr_values(source: Any, attr_sources: tuple[str, ...]) -> list[Any]:
    collected: list[Any] = []
    for attr in attr_sources:
        if not hasattr(source, attr):
            continue
        raw = getattr(source, attr)
        if isinstance(raw, dict):
            collected.extend(raw.values())
            continue
        try:
            collected.extend(list(raw))
        except TypeError as exc:
            raise TypeError(f"Attribute {attr!r} must be iterable.") from exc
    return collected


def _collect_items(
    source: Any,
    *,
    attr_sources: tuple[str, ...],
    direct_predicate: Callable[[Any], bool],
) -> list[Any]:
    if direct_predicate(source):
        return [source]
    if isinstance(source, dict):
        raw_items = list(source.values())
        should_sort = False
        deduplicate = True
    elif attr_sources and any(hasattr(source, attr) for attr in attr_sources):
        raw_items = _iter_attr_values(source, attr_sources)
        should_sort = False
        deduplicate = True
    elif isinstance(source, (str, bytes, bytearray)):
        raise TypeError("Tensor collection input must be iterable.")
    elif isinstance(source, Iterable):
        raw_items = list(source)
        should_sort = _is_unordered_collection(source)
        deduplicate = False
    else:
        raise TypeError("Tensor collection input must be a supported tensor object or iterable.")

    if deduplicate:
        items: list[Any] = []
        seen: set[int] = set()
        for item in raw_items:
            if item is None:
                continue
            item_id = id(item)
            if item_id in seen:
                continue
            seen.add(item_id)
            items.append(item)
    else:
        items = [item for item in raw_items if item is not None]
    if should_sort:
        items.sort(key=_name_sort_key)
    return items


def _to_numpy_array(value: Any) -> NumericArray:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    array = np.asarray(current)
    if array.dtype == np.dtype("O"):
        raise TypeError("Tensor values must be numeric and convertible to a NumPy array.")
    return array


def _stringify_sequence(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple("" if value is None else str(value) for value in values)


def _extract_tensornetwork_records(data: Any) -> list[_TensorRecord]:
    items = _collect_items(
        data,
        attr_sources=("nodes",),
        direct_predicate=lambda item: hasattr(item, "axis_names") and hasattr(item, "tensor"),
    )
    records: list[_TensorRecord] = []
    for index, item in enumerate(items):
        if not hasattr(item, "tensor"):
            raise TypeError("TensorNetwork nodes must expose a 'tensor' attribute.")
        array = _to_numpy_array(item.tensor)
        axis_names = _stringify_sequence(getattr(item, "axis_names", ()))
        name = "" if getattr(item, "name", None) is None else str(item.name)
        records.append(
            _TensorRecord(
                array=array,
                axis_names=axis_names,
                engine="tensornetwork",
                name=name or f"T{index}",
            )
        )
    return records


def _extract_tensorkrowch_records(data: Any) -> list[_TensorRecord]:
    items = _collect_items(
        data,
        attr_sources=_tensorkrowch_attr_sources(data),
        direct_predicate=lambda item: hasattr(item, "axes_names"),
    )
    records: list[_TensorRecord] = []
    for index, item in enumerate(items):
        tensor = getattr(item, "tensor", None)
        if tensor is None:
            name = "" if getattr(item, "name", None) is None else str(item.name)
            raise ValueError(
                f"TensorKrowch tensor {name or index!r} does not have a materialized tensor value. "
                "show_tensor_elements requires materialized tensors, not shape-only nodes."
            )
        array = _to_numpy_array(tensor)
        axis_names = _stringify_sequence(getattr(item, "axes_names", ()))
        name = "" if getattr(item, "name", None) is None else str(item.name)
        records.append(
            _TensorRecord(
                array=array,
                axis_names=axis_names,
                engine="tensorkrowch",
                name=name or f"T{index}",
            )
        )
    return records


def _tensorkrowch_attr_sources(data: Any) -> tuple[str, ...]:
    leaf_nodes = getattr(data, "leaf_nodes", None)
    if isinstance(leaf_nodes, dict):
        if leaf_nodes:
            return ("leaf_nodes",)
    elif leaf_nodes:
        return ("leaf_nodes",)
    return ("leaf_nodes", "nodes")


def _extract_quimb_records(data: Any) -> list[_TensorRecord]:
    from .quimb.graph import _quimb_tensor_parsed, _tensors_sorted_with_meta

    if hasattr(data, "inds") and hasattr(data, "data"):
        parsed = _quimb_tensor_parsed(data)
        tensors = [data]
        meta_names = [(parsed[1], parsed[2])]
    else:
        tensors, meta_names = _tensors_sorted_with_meta(data)
    records: list[_TensorRecord] = []
    for index, tensor in enumerate(tensors):
        axis_names, name_opt = meta_names[index]
        records.append(
            _TensorRecord(
                array=_to_numpy_array(tensor.data),
                axis_names=axis_names,
                engine="quimb",
                name=(name_opt or f"T{index}"),
            )
        )
    return records


def _tenpy_tensor_name_pairs(data: Any) -> list[tuple[str, Any]]:
    if _is_explicit_tenpy_network(data):
        return [(str(name), tensor) for name, tensor in data.nodes]
    if _is_tenpy_tensor(data):
        return [("T0", data)]
    if callable(getattr(data, "get_W", None)):
        return [(f"W{index}", data.get_W(index)) for index in range(int(data.L))]
    if callable(getattr(data, "get_X", None)) and hasattr(data, "uMPS_GS"):
        geometry = data.uMPS_GS
        return [(f"X{index}", data.get_X(index)) for index in range(int(geometry.L))]
    if callable(getattr(data, "get_B", None)):
        return [(f"B{index}", data.get_B(index, form=None)) for index in range(int(data.L))]
    raise TypeError(f"Unsupported TeNPy tensor input: {type(data).__name__!r}.")


def _extract_tenpy_records(data: Any) -> list[_TensorRecord]:
    pairs = _tenpy_tensor_name_pairs(data)
    records: list[_TensorRecord] = []
    for name, tensor in pairs:
        to_ndarray = getattr(tensor, "to_ndarray", None)
        if not callable(to_ndarray):
            raise TypeError("TeNPy tensors must expose to_ndarray().")
        leg_labels = getattr(tensor, "get_leg_labels", None)
        if not callable(leg_labels):
            raise TypeError("TeNPy tensors must expose get_leg_labels().")
        records.append(
            _TensorRecord(
                array=_to_numpy_array(to_ndarray()),
                axis_names=_stringify_sequence(leg_labels()),
                engine="tenpy",
                name=name,
            )
        )
    return records


def _extract_einsum_records(data: Any) -> list[_TensorRecord]:
    if not _is_einsum_trace_object(data):
        raise TypeError(
            "show_tensor_elements only supports EinsumTrace objects with live tensor values. "
            "Manual pair_tensor/einsum_trace_step iterables describe contractions but do not carry "
            "tensor values."
        )

    state = data._state
    state._sweep_dead_records()
    records: list[_TensorRecord] = []
    sorted_records = sorted(state._records.values(), key=lambda tracked: tracked.name)
    for tracked in sorted_records:
        tensor = tracked.ref()
        if tensor is None:
            continue
        records.append(
            _TensorRecord(
                array=_to_numpy_array(tensor),
                axis_names=(),
                engine="einsum",
                name=str(tracked.name),
            )
        )
    if not records:
        raise ValueError(
            "The EinsumTrace does not hold any live tensor values to inspect. Keep references to "
            "the traced tensors before calling show_tensor_elements."
        )
    return records


def _operand_shapes_for_einsum_step(
    step: pair_tensor | einsum_trace_step,
) -> tuple[tuple[int, ...], ...] | None:
    metadata = step.metadata or {}
    operand_shapes = metadata.get("operand_shapes")
    if operand_shapes is not None:
        return tuple(tuple(int(dim) for dim in shape) for shape in operand_shapes)
    if _is_einsum_pair_step(step):
        left_shape = metadata.get("left_shape")
        right_shape = metadata.get("right_shape")
        if left_shape is not None and right_shape is not None:
            return (
                tuple(int(dim) for dim in left_shape),
                tuple(int(dim) for dim in right_shape),
            )
    return None


def _axis_names_for_einsum_step(step: pair_tensor | einsum_trace_step) -> tuple[str, ...]:
    operand_shapes = _operand_shapes_for_einsum_step(step)
    if operand_shapes is None:
        return ()
    try:
        parsed = parse_einsum_equation(str(step.equation), operand_shapes)
    except ValueError:
        return ()
    return tuple(str(axis_name) for axis_name in parsed.output_axes)


def _extract_einsum_playback_step_records(data: Any) -> tuple[_PlaybackStepRecord, ...]:
    if not _is_einsum_trace_object(data):
        raise TypeError("Playback tensor inspection only supports real EinsumTrace objects.")

    state = data._state
    state._sweep_dead_records()
    live_tensors_by_name: dict[str, Any] = {}
    for tracked in state._records.values():
        tensor = tracked.ref()
        if tensor is None:
            continue
        live_tensors_by_name[str(tracked.name)] = tensor

    step_records: list[_PlaybackStepRecord] = []
    for step in state.pairs:
        result_name = str(step.result_name)
        tensor = live_tensors_by_name.get(result_name)
        if tensor is None:
            step_records.append(
                _PlaybackStepRecord(
                    result_name=result_name,
                    record=None,
                )
            )
            continue
        step_records.append(
            _PlaybackStepRecord(
                result_name=result_name,
                record=_TensorRecord(
                    array=_to_numpy_array(tensor),
                    axis_names=_axis_names_for_einsum_step(step),
                    engine="einsum",
                    name=result_name,
                ),
            )
        )
    return tuple(step_records)


def _extract_tensorkrowch_playback_step_records(
    data: Any,
) -> tuple[_PlaybackStepRecord, ...] | None:
    if not hasattr(data, "resultant_nodes") or not hasattr(data, "leaf_nodes"):
        return None

    from .tensorkrowch._history import _recover_contraction_history

    recovered = _recover_contraction_history(data)
    if recovered is None or not recovered.step_result_nodes:
        return None

    step_records: list[_PlaybackStepRecord] = []
    for index, node in enumerate(recovered.step_result_nodes):
        tensor = getattr(node, "tensor", None)
        if tensor is None:
            return None
        name = "" if getattr(node, "name", None) is None else str(node.name)
        step_name = name or f"step_{index}"
        step_records.append(
            _PlaybackStepRecord(
                result_name=step_name,
                record=_TensorRecord(
                    array=_to_numpy_array(tensor),
                    axis_names=_stringify_sequence(getattr(node, "axes_names", ())),
                    engine="tensorkrowch",
                    name=step_name,
                ),
            )
        )
    return tuple(step_records)


def _extract_playback_step_records(data: Any) -> tuple[_PlaybackStepRecord, ...] | None:
    """Recover playback result tensors when the input carries contraction history."""
    if _is_einsum_trace_object(data):
        return _extract_einsum_playback_step_records(data)

    try:
        resolved_engine, prepared_input = _detect_tensor_elements_engine(data)
    except (TypeError, ValueError):
        return None

    if resolved_engine == "tensorkrowch":
        return _extract_tensorkrowch_playback_step_records(prepared_input)
    return None


def _extract_tensor_records(
    data: Any,
    *,
    engine: EngineName | None,
) -> tuple[TensorElementsSourceName, list[_TensorRecord]]:
    """Extract normalized tensor records from supported public inputs.

    Args:
        data: Tensor input passed to ``show_tensor_elements(...)``.
        engine: Optional explicit backend override.

    Returns:
        The resolved backend name and the normalized tensor records ready for rendering.

    Raises:
        UnsupportedEngineError: If ``engine`` names an unknown backend.
        TensorDataTypeError: If the backend input type is structurally incompatible.
        TensorDataError: If the input is valid in principle but exposes no usable tensors.
    """
    resolved_engine = engine
    prepared_input = data
    if resolved_engine is None:
        try:
            direct_array_records, prepared_input = _extract_direct_array_records(data)
        except TypeError as exc:
            raise TensorDataTypeError(str(exc)) from exc
        except ValueError as exc:
            raise TensorDataError(str(exc)) from exc
        if direct_array_records is not None:
            return "numpy", direct_array_records
        resolved_engine, prepared_input = _detect_tensor_elements_engine(prepared_input)

    package_logger.debug("Extracting tensor records with engine=%r.", resolved_engine)
    try:
        if resolved_engine == "tensornetwork":
            records = _extract_tensornetwork_records(prepared_input)
        elif resolved_engine == "tensorkrowch":
            records = _extract_tensorkrowch_records(prepared_input)
        elif resolved_engine == "quimb":
            records = _extract_quimb_records(prepared_input)
        elif resolved_engine == "tenpy":
            records = _extract_tenpy_records(prepared_input)
        elif resolved_engine == "einsum":
            records = _extract_einsum_records(prepared_input)
        else:
            raise UnsupportedEngineError(f"Unsupported tensor engine: {resolved_engine}")
    except UnsupportedEngineError:
        raise
    except TypeError as exc:
        raise TensorDataTypeError(str(exc)) from exc
    except ValueError as exc:
        raise TensorDataError(str(exc)) from exc

    if not records:
        raise TensorDataError("The input does not expose any tensors to visualize.")
    return resolved_engine, records


__all__ = [
    "parse_einsum_equation",
    "_PlaybackStepRecord",
    "_TensorRecord",
    "_detect_tensor_elements_engine",
    "_extract_einsum_playback_step_records",
    "_extract_playback_step_records",
    "_extract_tensor_records",
    "_to_numpy_array",
]
