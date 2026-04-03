from __future__ import annotations

from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import tee
from typing import Any, TypeAlias

import numpy as np

from ._engine_specs import EngineName
from .einsum_module.trace import EinsumTrace, einsum_trace_step, pair_tensor
from .quimb.graph import _quimb_tensor_parsed, _tensors_sorted_with_meta
from .tenpy.explicit import TenPyTensorNetwork

NumericArray: TypeAlias = np.ndarray[Any, Any]


@dataclass(frozen=True)
class _TensorRecord:
    array: NumericArray
    axis_names: tuple[str, ...]
    engine: EngineName
    name: str


@dataclass(frozen=True)
class _TensorStats:
    dtype_text: str
    element_count: int
    is_complex: bool
    shape: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class _MatrixMetadata:
    col_axes: tuple[int, ...]
    col_names: tuple[str, ...]
    original_shape: tuple[int, ...]
    row_axes: tuple[int, ...]
    row_names: tuple[str, ...]


def _first_non_none(items: Iterable[Any]) -> Any | None:
    for item in items:
        if item is not None:
            return item
    return None


def _peek_item(source: Any) -> tuple[Any | None, Any]:
    if isinstance(source, dict):
        return _first_non_none(source.values()), source
    if isinstance(source, (str, bytes, bytearray)) or not isinstance(source, Iterable):
        return None, source

    iterator = iter(source)
    if iterator is source:
        probe_iter, runtime_iter = tee(iterator)
        return _first_non_none(probe_iter), runtime_iter
    return _first_non_none(iterator), source


def _is_unordered_collection(value: Any) -> bool:
    return isinstance(value, AbstractSet) and not isinstance(value, (str, bytes, bytearray))


def _is_tenpy_tensor(value: Any) -> bool:
    return callable(getattr(value, "get_leg_labels", None)) and callable(
        getattr(value, "to_ndarray", None)
    )


def _is_tenpy_like(value: Any) -> bool:
    return (
        isinstance(value, TenPyTensorNetwork)
        or callable(getattr(value, "get_W", None))
        or (callable(getattr(value, "get_X", None)) and hasattr(value, "uMPS_GS"))
        or callable(getattr(value, "get_B", None))
        or _is_tenpy_tensor(value)
    )


def _detect_engine_from_sample(sample_item: Any | None) -> EngineName | None:
    if isinstance(sample_item, (pair_tensor, einsum_trace_step)):
        return "einsum"
    if _is_tenpy_like(sample_item):
        return "tenpy"
    if hasattr(sample_item, "data") and hasattr(sample_item, "inds"):
        return "quimb"
    if hasattr(sample_item, "axes_names"):
        return "tensorkrowch"
    if hasattr(sample_item, "axis_names") and hasattr(sample_item, "tensor"):
        return "tensornetwork"
    return None


def _detect_tensor_elements_engine(data: Any) -> tuple[EngineName, Any]:
    if isinstance(data, EinsumTrace):
        return "einsum", data
    if _is_tenpy_like(data):
        return "tenpy", data

    direct_engine = _detect_engine_from_sample(data)
    if direct_engine is not None:
        return direct_engine, data

    if hasattr(data, "tensors"):
        tensor_sample, _ = _peek_item(data.tensors)
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            return "quimb", data

    if hasattr(data, "leaf_nodes"):
        sample, _ = _peek_item(data.leaf_nodes)
        if sample is None or hasattr(sample, "axes_names"):
            return "tensorkrowch", data
    if hasattr(data, "nodes"):
        sample, _ = _peek_item(data.nodes)
        if sample is None or hasattr(sample, "axes_names"):
            return "tensorkrowch", data
        if sample is None or hasattr(sample, "axis_names"):
            return "tensornetwork", data

    sample_item, sampled_input = _peek_item(data)
    detected_engine = _detect_engine_from_sample(sample_item)
    if detected_engine is not None:
        return detected_engine, sampled_input

    raise ValueError(
        "Could not infer tensor engine from input of type "
        f"{type(data).__name__!r}. Pass engine= explicitly or provide a supported tensor input."
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
    elif attr_sources and any(hasattr(source, attr) for attr in attr_sources):
        raw_items = _iter_attr_values(source, attr_sources)
        should_sort = False
    elif isinstance(source, (str, bytes, bytearray)):
        raise TypeError("Tensor collection input must be iterable.")
    elif isinstance(source, Iterable):
        raw_items = list(source)
        should_sort = _is_unordered_collection(source)
    else:
        raise TypeError("Tensor collection input must be a supported tensor object or iterable.")

    unique: list[Any] = []
    seen: set[int] = set()
    for item in raw_items:
        if item is None:
            continue
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)
    if should_sort:
        unique.sort(key=_name_sort_key)
    return unique


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
        attr_sources=("leaf_nodes", "nodes"),
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


def _extract_quimb_records(data: Any) -> list[_TensorRecord]:
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
    if isinstance(data, TenPyTensorNetwork):
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
    if not isinstance(data, EinsumTrace):
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


def _extract_tensor_records(
    data: Any,
    *,
    engine: EngineName | None,
) -> tuple[EngineName, list[_TensorRecord]]:
    resolved_engine = engine
    prepared_input = data
    if resolved_engine is None:
        resolved_engine, prepared_input = _detect_tensor_elements_engine(data)

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
        raise ValueError(f"Unsupported tensor engine: {resolved_engine}")

    if not records:
        raise ValueError("The input does not expose any tensors to visualize.")
    return resolved_engine, records


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def _build_stats(record: _TensorRecord) -> _TensorStats:
    array = np.asarray(record.array)
    flat = np.ravel(array)
    shape = tuple(int(dimension) for dimension in array.shape)
    element_count = int(array.size)
    axis_text = ", ".join(record.axis_names) if record.axis_names else "-"
    lines = [
        f"name: {record.name}",
        f"engine: {record.engine}",
        f"shape: {shape}",
        f"ndim: {array.ndim}",
        f"elements: {element_count}",
        f"dtype: {array.dtype}",
        f"axes: {axis_text}",
    ]
    if flat.size == 0:
        lines.append("stats: empty tensor")
        return _TensorStats(
            dtype_text=str(array.dtype),
            element_count=element_count,
            is_complex=bool(np.iscomplexobj(array)),
            shape=shape,
            text="\n".join(lines),
        )

    if np.iscomplexobj(array):
        magnitude = np.abs(flat)
        lines.append(
            "magnitude: "
            f"min={_format_float(float(np.nanmin(magnitude)))}, "
            f"max={_format_float(float(np.nanmax(magnitude)))}, "
            f"mean={_format_float(float(np.nanmean(magnitude)))}, "
            f"std={_format_float(float(np.nanstd(magnitude)))}"
        )
        lines.append(
            "real range: "
            f"{_format_float(float(np.nanmin(np.real(flat))))} .. "
            f"{_format_float(float(np.nanmax(np.real(flat))))}"
        )
        lines.append(
            "imag range: "
            f"{_format_float(float(np.nanmin(np.imag(flat))))} .. "
            f"{_format_float(float(np.nanmax(np.imag(flat))))}"
        )
    else:
        values = np.real(flat)
        lines.append(
            "stats: "
            f"min={_format_float(float(np.nanmin(values)))}, "
            f"max={_format_float(float(np.nanmax(values)))}, "
            f"mean={_format_float(float(np.nanmean(values)))}, "
            f"std={_format_float(float(np.nanstd(values)))}"
        )

    return _TensorStats(
        dtype_text=str(array.dtype),
        element_count=element_count,
        is_complex=bool(np.iscomplexobj(array)),
        shape=shape,
        text="\n".join(lines),
    )


__all__ = [
    "NumericArray",
    "_MatrixMetadata",
    "_TensorRecord",
    "_TensorStats",
    "_build_stats",
    "_extract_tensor_records",
]
