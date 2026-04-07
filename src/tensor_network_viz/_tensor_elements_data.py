"""Normalization and analysis helpers for tensor-elements rendering."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
from math import inf, log
from typing import Any, TypeAlias

import numpy as np

from ._engine_specs import EngineName
from ._input_inspection import (
    _detect_tensor_engine_with_input,
    _is_tenpy_tensor,
    _is_unordered_collection,
)
from ._logging import package_logger
from .einsum_module._equation import parse_einsum_equation
from .einsum_module.trace import EinsumTrace, einsum_trace_step, pair_tensor
from .exceptions import TensorDataError, TensorDataTypeError, UnsupportedEngineError
from .quimb.graph import _quimb_tensor_parsed, _tensors_sorted_with_meta
from .tenpy.explicit import TenPyTensorNetwork
from .tensor_elements_config import TensorAxisSelector, TensorElementsConfig
from .tensorkrowch._history import _recover_contraction_history

NumericArray: TypeAlias = np.ndarray[Any, Any]


@dataclass(frozen=True)
class _TensorRecord:
    """Normalized tensor entry extracted from a supported backend.

    Attributes:
        array: Concrete NumPy array used by the rendering pipeline.
        axis_names: Axis labels associated with ``array``.
        engine: Backend that produced the tensor.
        name: Stable display name shown in controls and summaries.
    """

    array: NumericArray
    axis_names: tuple[str, ...]
    engine: EngineName
    name: str


@dataclass(frozen=True)
class _TensorStats:
    """Human-readable summary statistics for one tensor.

    Attributes:
        dtype_text: Display-ready dtype description.
        element_count: Number of scalar entries in the tensor.
        is_complex: Whether the tensor contains complex values.
        shape: Tensor shape used in the summary.
        text: Multiline textual summary rendered in ``data`` mode.
    """

    dtype_text: str
    element_count: int
    is_complex: bool
    shape: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class _MatrixMetadata:
    """Metadata describing how a tensor was matrixized for inspection views.

    Attributes:
        col_axes: Axis indices grouped into the matrix columns.
        col_names: Display names for the column axes.
        original_shape: Shape of the source tensor before reshaping.
        row_axes: Axis indices grouped into the matrix rows.
        row_names: Display names for the row axes.
    """

    col_axes: tuple[int, ...]
    col_names: tuple[str, ...]
    original_shape: tuple[int, ...]
    row_axes: tuple[int, ...]
    row_names: tuple[str, ...]


@dataclass(frozen=True)
class _SpectralAnalysis:
    """Derived singular-value and eigenvalue information for one tensor.

    Attributes:
        analysis_shape: Shape of the matrix actually analyzed after optional downsampling.
        col_names: Column-axis names used in the matrix view.
        eigenvalues: Eigenvalues when the analysis matrix is square, else ``None``.
        issue: Reason why the spectral analysis is unavailable, if any.
        matrix_shape: Shape of the full matrixized tensor before reduction.
        row_names: Row-axis names used in the matrix view.
        singular_values: Singular values for the analysis matrix, if available.
        used_reduced_matrix: Whether downsampling changed the matrix before analysis.
    """

    analysis_shape: tuple[int, int]
    col_names: tuple[str, ...]
    eigenvalues: NumericArray | None
    issue: str | None
    matrix_shape: tuple[int, int]
    row_names: tuple[str, ...]
    singular_values: NumericArray | None
    used_reduced_matrix: bool


@dataclass(frozen=True)
class _PlaybackStepRecord:
    """Tensor payload associated with one playback step result.

    Attributes:
        result_name: Name of the result tensor produced by the step.
        record: Normalized tensor data for the result, when available.
    """

    result_name: str
    record: _TensorRecord | None


_EinsumPlaybackStepRecord = _PlaybackStepRecord


def _detect_tensor_elements_engine(data: Any) -> tuple[EngineName, Any]:
    """Detect the tensor backend and preserve any prepared input wrapper."""
    return _detect_tensor_engine_with_input(data)


def _normalize_axis_selector(
    selector: TensorAxisSelector,
    *,
    axis_names: tuple[str, ...],
    ndim: int,
) -> int:
    """Resolve one row/column selector to a concrete axis index.

    Args:
        selector: Axis name or integer index supplied by the user.
        axis_names: Known axis names for the tensor.
        ndim: Tensor rank.

    Returns:
        The resolved non-negative axis index.

    Raises:
        ValueError: If the axis name is unknown or the index is out of bounds.
    """
    if isinstance(selector, str):
        if not axis_names:
            raise ValueError("Axis names are not available for this tensor.")
        try:
            return axis_names.index(selector)
        except ValueError as exc:
            raise ValueError(
                f"Unknown axis name {selector!r}. Available axes are {axis_names!r}."
            ) from exc
    index = int(selector)
    if index < 0:
        index += ndim
    if index < 0 or index >= ndim:
        raise ValueError(f"Axis index {selector!r} is out of bounds for ndim={ndim}.")
    return index


def _normalize_axis_group(
    selectors: tuple[TensorAxisSelector, ...] | None,
    *,
    axis_names: tuple[str, ...],
    ndim: int,
) -> tuple[int, ...] | None:
    """Resolve a user-provided row/column selector tuple to axis indices.

    Args:
        selectors: User-provided axis selectors or ``None``.
        axis_names: Known axis names for the tensor.
        ndim: Tensor rank.

    Returns:
        The resolved axis indices, or ``None`` when the selector group was omitted.

    Raises:
        ValueError: If any selector is invalid or the group contains duplicates.
    """
    if selectors is None:
        return None
    resolved = tuple(
        _normalize_axis_selector(selector, axis_names=axis_names, ndim=ndim)
        for selector in selectors
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError("row_axes and col_axes must not contain duplicate axes.")
    return resolved


def _axis_names_for_shape(axis_names: tuple[str, ...], *, ndim: int) -> tuple[str, ...]:
    if len(axis_names) >= ndim:
        return axis_names[:ndim]
    generated = tuple(f"axis{index}" for index in range(len(axis_names), ndim))
    return axis_names + generated


def _balanced_axes_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose a deterministic row-axis partition for matrixized tensor views."""
    ndim = len(shape)
    if ndim == 0:
        return ()
    if ndim == 1:
        return (0,)

    if ndim > 16:
        selected: list[int] = [0]
        row_log = log(float(shape[0]))
        col_log = sum(log(float(dim)) for dim in shape[1:])
        for index in range(1, ndim):
            dim_log = log(float(shape[index]))
            if row_log <= col_log:
                selected.append(index)
                row_log += dim_log
                col_log -= dim_log
        if len(selected) == ndim:
            return tuple(selected[:-1])
        return tuple(selected)

    total_log = sum(log(float(dim)) for dim in shape)
    best_axes: tuple[int, ...] | None = None
    best_score = inf
    best_rank_gap = inf
    for size in range(1, ndim):
        for combo in combinations(range(1, ndim), size - 1):
            candidate = (0,) + combo
            row_log = sum(log(float(shape[index])) for index in candidate)
            col_log = total_log - row_log
            score = abs(row_log - col_log)
            rank_gap = abs(len(candidate) - (ndim - len(candidate)))
            if (
                score < best_score
                or (np.isclose(score, best_score) and rank_gap < best_rank_gap)
                or (
                    np.isclose(score, best_score)
                    and rank_gap == best_rank_gap
                    and (best_axes is None or candidate < best_axes)
                )
            ):
                best_axes = candidate
                best_score = score
                best_rank_gap = rank_gap
    assert best_axes is not None
    return best_axes


def _resolve_matrix_axes(
    *,
    shape: tuple[int, ...],
    row_axes: tuple[TensorAxisSelector, ...] | None,
    col_axes: tuple[TensorAxisSelector, ...] | None,
    axis_names: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve the row/column axis split used by matrixized inspection modes.

    Args:
        shape: Shape of the source tensor.
        row_axes: Optional user-selected row axes.
        col_axes: Optional user-selected column axes.
        axis_names: Known axis names for the tensor.

    Returns:
        Two tuples containing the resolved row-axis and column-axis indices.

    Raises:
        ValueError: If the explicit row/column groups overlap or do not cover the tensor.
    """
    ndim = len(shape)
    if ndim == 0:
        return (), ()

    resolved_rows = _normalize_axis_group(row_axes, axis_names=axis_names, ndim=ndim)
    resolved_cols = _normalize_axis_group(col_axes, axis_names=axis_names, ndim=ndim)

    if resolved_rows is None and resolved_cols is None:
        resolved_rows = _balanced_axes_for_shape(shape)
        resolved_cols = tuple(index for index in range(ndim) if index not in resolved_rows)
        return resolved_rows, resolved_cols

    if resolved_rows is None:
        assert resolved_cols is not None
        resolved_rows = tuple(index for index in range(ndim) if index not in resolved_cols)
    if resolved_cols is None:
        assert resolved_rows is not None
        resolved_cols = tuple(index for index in range(ndim) if index not in resolved_rows)

    if set(resolved_rows).intersection(resolved_cols):
        raise ValueError("row_axes and col_axes must be disjoint.")
    if set(resolved_rows) | set(resolved_cols) != set(range(ndim)):
        raise ValueError("row_axes and col_axes must cover every axis exactly once.")
    return resolved_rows, resolved_cols


def _matrixize_tensor(
    array: NumericArray,
    *,
    axis_names: tuple[str, ...],
    row_axes: tuple[TensorAxisSelector, ...] | None,
    col_axes: tuple[TensorAxisSelector, ...] | None,
) -> tuple[NumericArray, _MatrixMetadata]:
    """Reshape a tensor into the matrix used by heatmap and spectral views.

    Args:
        array: Tensor values to reshape.
        axis_names: Axis labels associated with ``array``.
        row_axes: Optional axes assigned to matrix rows.
        col_axes: Optional axes assigned to matrix columns.

    Returns:
        The matrixized tensor together with metadata describing the reshape.
    """
    shape = tuple(int(dimension) for dimension in array.shape)
    resolved_axis_names = _axis_names_for_shape(axis_names, ndim=len(shape))
    resolved_rows, resolved_cols = _resolve_matrix_axes(
        shape=shape,
        row_axes=row_axes,
        col_axes=col_axes,
        axis_names=resolved_axis_names,
    )
    if array.ndim == 0:
        matrix = np.asarray(array).reshape(1, 1)
    else:
        order = resolved_rows + resolved_cols
        transposed = np.transpose(array, axes=order) if order else array
        row_size = int(np.prod([shape[index] for index in resolved_rows], dtype=int)) or 1
        col_size = int(np.prod([shape[index] for index in resolved_cols], dtype=int)) or 1
        matrix = np.reshape(transposed, (row_size, col_size))
    metadata = _MatrixMetadata(
        col_axes=resolved_cols,
        col_names=tuple(resolved_axis_names[index] for index in resolved_cols),
        original_shape=shape,
        row_axes=resolved_rows,
        row_names=tuple(resolved_axis_names[index] for index in resolved_rows),
    )
    return matrix, metadata


def _downsample_axis_mean(array: NumericArray, *, axis: int, target_size: int) -> NumericArray:
    source = np.asarray(array)
    length = int(source.shape[axis])
    if length <= target_size:
        return source

    edges = np.linspace(0, length, target_size + 1, dtype=int)
    moved = np.moveaxis(source, axis, 0)
    contiguous = np.ascontiguousarray(moved)
    slices: list[NumericArray] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        safe_stop = min(max(stop, start + 1), length)
        slices.append(np.nanmean(contiguous[start:safe_stop], axis=0, keepdims=True))
    reduced = np.concatenate(slices, axis=0)
    return np.moveaxis(reduced, 0, axis)


def _downsample_matrix(matrix: NumericArray, *, max_shape: tuple[int, int]) -> NumericArray:
    max_rows, max_cols = max_shape
    reduced = _downsample_axis_mean(np.asarray(matrix), axis=0, target_size=int(max_rows))
    reduced = _downsample_axis_mean(reduced, axis=1, target_size=int(max_cols))
    return reduced


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


def _operand_shapes_for_einsum_step(
    step: pair_tensor | einsum_trace_step,
) -> tuple[tuple[int, ...], ...] | None:
    metadata = step.metadata or {}
    operand_shapes = metadata.get("operand_shapes")
    if operand_shapes is not None:
        return tuple(tuple(int(dim) for dim in shape) for shape in operand_shapes)
    if isinstance(step, pair_tensor):
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
    except Exception:
        return ()
    return tuple(str(axis_name) for axis_name in parsed.output_axes)


def _extract_einsum_playback_step_records(data: Any) -> tuple[_PlaybackStepRecord, ...]:
    if not isinstance(data, EinsumTrace):
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
    if isinstance(data, EinsumTrace):
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
) -> tuple[EngineName, list[_TensorRecord]]:
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
        resolved_engine, prepared_input = _detect_tensor_elements_engine(data)

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


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def _format_scalar(value: complex | float) -> str:
    if np.iscomplexobj(value):
        complex_value = complex(value)
        real_text = _format_float(float(np.real(complex_value)))
        imag_sign = "+" if float(np.imag(complex_value)) >= 0.0 else ""
        imag_text = _format_float(float(np.imag(complex_value)))
        return f"{real_text}{imag_sign}{imag_text}j"
    return _format_float(float(np.real(value)))


def _resolved_axis_names(record: _TensorRecord) -> tuple[str, ...]:
    axis_names: list[str] = []
    for axis_index in range(np.asarray(record.array).ndim):
        if axis_index < len(record.axis_names) and record.axis_names[axis_index]:
            axis_names.append(record.axis_names[axis_index])
        else:
            axis_names.append(f"axis{axis_index}")
    return tuple(axis_names)


def _format_range(values: NumericArray) -> str:
    flat = np.ravel(np.asarray(values, dtype=float))
    non_nan = flat[~np.isnan(flat)]
    if non_nan.size == 0:
        return "nan .. nan"
    return f"{_format_float(float(np.min(non_nan)))} .. {_format_float(float(np.max(non_nan)))}"


def _summary_metric_array(array: NumericArray) -> NumericArray:
    return np.abs(array) if np.iscomplexobj(array) else np.real(array)


def _build_axis_summary_lines(record: _TensorRecord) -> list[str]:
    array = np.asarray(record.array)
    shape = tuple(int(dimension) for dimension in array.shape)
    if not shape:
        return ["axis summary:", "- scalar tensor"]

    metrics = np.asarray(_summary_metric_array(array), dtype=float)
    axis_names = _resolved_axis_names(record)
    mean_label = "mean|x|" if np.iscomplexobj(array) else "mean"
    norm_label = "norm|x|" if np.iscomplexobj(array) else "norm"
    lines = ["axis summary:"]
    for axis_index, (axis_name, axis_size) in enumerate(zip(axis_names, shape, strict=True)):
        other_axes = tuple(index for index in range(array.ndim) if index != axis_index)
        if other_axes:
            marginal_mean = np.nanmean(metrics, axis=other_axes)
            marginal_norm = np.sqrt(np.nansum(np.square(metrics), axis=other_axes))
        else:
            marginal_mean = metrics
            marginal_norm = np.abs(metrics)
        lines.append(
            f"- {axis_name} (size={axis_size}): "
            f"{mean_label}={_format_range(np.asarray(marginal_mean))}, "
            f"{norm_label}={_format_range(np.asarray(marginal_norm))}"
        )
    return lines


def _topk_sort_key(magnitude: float, flat_index: int) -> tuple[float, int]:
    sortable_magnitude = float(magnitude)
    if np.isnan(sortable_magnitude):
        sortable_magnitude = float("-inf")
    return (-sortable_magnitude, flat_index)


def _build_topk_lines(record: _TensorRecord, *, count: int) -> list[str]:
    array = np.asarray(record.array)
    requested_count = min(int(count), int(array.size))
    lines = [f"top {int(count)} by magnitude:"]
    if requested_count <= 0:
        lines.append("- tensor is empty")
        return lines

    magnitudes = np.ravel(np.abs(array))
    values = np.ravel(array)
    axis_names = _resolved_axis_names(record)
    ranked_indices = sorted(
        range(int(magnitudes.size)),
        key=lambda flat_index: _topk_sort_key(float(magnitudes[flat_index]), flat_index),
    )
    for rank, flat_index in enumerate(ranked_indices[:requested_count], start=1):
        coordinates = np.unravel_index(flat_index, array.shape) if array.shape else ()
        if coordinates:
            coordinate_text = ", ".join(
                f"{axis_name}={coordinate}"
                for axis_name, coordinate in zip(axis_names, coordinates, strict=True)
            )
        else:
            coordinate_text = "scalar"
        lines.append(
            f"{rank}. |x|={_format_float(float(magnitudes[flat_index]))}, "
            f"value={_format_scalar(values[flat_index])}, at ({coordinate_text})"
        )
    return lines


def _spectral_analysis_for_record(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
) -> _SpectralAnalysis:
    """Compute spectral diagnostics for the matrixized view of one tensor.

    Args:
        record: Tensor selected for inspection.
        config: Active tensor-elements configuration.

    Returns:
        A spectral-analysis bundle containing singular values, optional eigenvalues, and
        any issue that prevented the analysis from running.
    """
    matrix, metadata = _matrixize_tensor(
        np.asarray(record.array),
        axis_names=record.axis_names,
        row_axes=config.row_axes,
        col_axes=config.col_axes,
    )
    matrix_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    reduced_matrix = _downsample_matrix(matrix, max_shape=config.max_matrix_shape)
    analysis_shape = (int(reduced_matrix.shape[0]), int(reduced_matrix.shape[1]))
    used_reduced_matrix = analysis_shape != matrix_shape

    if not np.all(np.isfinite(matrix)):
        return _SpectralAnalysis(
            analysis_shape=analysis_shape,
            col_names=metadata.col_names,
            eigenvalues=None,
            issue="matrix contains NaN or Inf values",
            matrix_shape=matrix_shape,
            row_names=metadata.row_names,
            singular_values=None,
            used_reduced_matrix=used_reduced_matrix,
        )

    try:
        singular_values = np.asarray(np.linalg.svd(reduced_matrix, compute_uv=False), dtype=float)
        eigenvalues = None
        if analysis_shape[0] == analysis_shape[1]:
            eigenvalues = np.asarray(np.linalg.eigvals(reduced_matrix))
    except np.linalg.LinAlgError as exc:
        return _SpectralAnalysis(
            analysis_shape=analysis_shape,
            col_names=metadata.col_names,
            eigenvalues=None,
            issue=f"linear algebra failed ({exc.__class__.__name__})",
            matrix_shape=matrix_shape,
            row_names=metadata.row_names,
            singular_values=None,
            used_reduced_matrix=used_reduced_matrix,
        )

    return _SpectralAnalysis(
        analysis_shape=analysis_shape,
        col_names=metadata.col_names,
        eigenvalues=eigenvalues,
        issue=None,
        matrix_shape=matrix_shape,
        row_names=metadata.row_names,
        singular_values=singular_values,
        used_reduced_matrix=used_reduced_matrix,
    )


def _format_name_list(names: tuple[str, ...]) -> str:
    if not names:
        return "-"
    return ", ".join(names)


def _format_percent(value: float) -> str:
    return f"{_format_float(100.0 * value)}%"


def _topk_singular_value_lines(values: NumericArray, *, count: int) -> list[str]:
    requested_count = min(int(count), int(values.size))
    lines = [f"top {int(count)} singular values:"]
    for rank, singular_value in enumerate(values[:requested_count], start=1):
        lines.append(f"{rank}. sigma{rank}={_format_float(float(singular_value))}")
    return lines


def _topk_eigenvalue_lines(values: NumericArray, *, count: int) -> list[str]:
    requested_count = min(int(count), int(values.size))
    lines = [f"top {int(count)} eigenvalues by magnitude:"]
    ranked = sorted(
        np.ravel(values).tolist(),
        key=lambda value: (-float(np.abs(value)), float(np.real(value)), float(np.imag(value))),
    )
    for rank, eigenvalue in enumerate(ranked[:requested_count], start=1):
        lines.append(
            f"{rank}. |lambda|={_format_float(float(np.abs(eigenvalue)))}, "
            f"lambda={_format_scalar(eigenvalue)}"
        )
    return lines


def _build_spectral_summary_lines(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    topk_count: int,
) -> list[str]:
    analysis = _spectral_analysis_for_record(record, config=config)
    lines = [
        "spectral analysis:",
        f"- matrixized shape: {analysis.matrix_shape}",
        f"- matrix rows: {_format_name_list(analysis.row_names)}",
        f"- matrix cols: {_format_name_list(analysis.col_names)}",
    ]

    if analysis.used_reduced_matrix:
        lines.append(
            "- analysis matrix: "
            f"reduced to {analysis.analysis_shape} via max_matrix_shape={config.max_matrix_shape}"
        )
    else:
        lines.append("- analysis matrix: full matrix")

    if analysis.issue is not None:
        lines.append(f"- spectral analysis unavailable: {analysis.issue}")
        return lines

    assert analysis.singular_values is not None
    singular_values = np.asarray(analysis.singular_values, dtype=float)
    if singular_values.size == 0:
        lines.append("- spectral analysis unavailable: empty singular spectrum")
        return lines

    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    positive_singular_values = singular_values[singular_values > 0.0]
    if positive_singular_values.size == 0:
        condition_number = float("inf") if sigma_max > 0.0 else 1.0
    else:
        condition_number = float(sigma_max / np.min(positive_singular_values))
    frobenius_sq = float(np.sum(np.square(singular_values)))
    stable_rank = float(frobenius_sq / (sigma_max * sigma_max)) if sigma_max > 0.0 else 0.0
    energy_numerator = float(
        np.sum(np.square(singular_values[: min(int(topk_count), singular_values.size)]))
    )
    energy_ratio = float(energy_numerator / frobenius_sq) if frobenius_sq > 0.0 else 1.0

    lines.extend(
        [
            f"- singular-value range: {_format_float(sigma_min)} .. {_format_float(sigma_max)}",
            f"- condition number: {_format_float(condition_number)}",
            f"- stable rank: {_format_float(stable_rank)}",
            f"- top {int(topk_count)} energy: {_format_percent(energy_ratio)}",
        ]
    )
    lines.extend(
        f"- {line}" for line in _topk_singular_value_lines(singular_values, count=topk_count)
    )

    if analysis.eigenvalues is None:
        lines.append("- eigenvalues: not available for non-square matrices")
        return lines

    eigenvalues = np.asarray(analysis.eigenvalues)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    lines.append(f"- spectral radius: {_format_float(spectral_radius)}")
    lines.extend(f"- {line}" for line in _topk_eigenvalue_lines(eigenvalues, count=topk_count))
    return lines


def _build_data_summary_text(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    topk_count: int,
) -> str:
    """Assemble the multiline textual summary used by ``data`` mode."""
    sections = [
        _build_stats(record).text,
        "\n".join(_build_axis_summary_lines(record)),
        "\n".join(_build_topk_lines(record, count=topk_count)),
    ]
    return "\n\n".join(sections)


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
    "_EinsumPlaybackStepRecord",
    "_MatrixMetadata",
    "_PlaybackStepRecord",
    "_SpectralAnalysis",
    "_TensorRecord",
    "_TensorStats",
    "_build_axis_summary_lines",
    "_build_data_summary_text",
    "_build_stats",
    "_build_topk_lines",
    "_downsample_matrix",
    "_extract_playback_step_records",
    "_extract_einsum_playback_step_records",
    "_extract_tensor_records",
    "_matrixize_tensor",
    "_resolve_matrix_axes",
    "_spectral_analysis_for_record",
]
