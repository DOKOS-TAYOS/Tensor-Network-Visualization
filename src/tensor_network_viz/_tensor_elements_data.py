"""Normalization and analysis helpers for tensor-elements rendering."""

from __future__ import annotations

from itertools import combinations
from math import inf, log

import numpy as np

from ._tensor_elements_inputs import (
    _extract_einsum_playback_step_records,
    _extract_playback_step_records,
    _extract_tensor_records,
    _is_direct_array_like_tensor,
    parse_einsum_equation,
)
from ._tensor_elements_models import (
    NumericArray,
    _MatrixMetadata,
    _PlaybackStepRecord,
    _ResolvedTensorAnalysis,
    _SpectralAnalysis,
    _TensorRecord,
    _TensorStats,
)
from .tensor_elements_config import TensorAnalysisConfig, TensorAxisSelector, TensorElementsConfig


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


def _resolve_axis_selector_with_fallback(
    selector: TensorAxisSelector | None,
    *,
    axis_names: tuple[str, ...],
    ndim: int,
    fallback_index: int | None,
    fallback: bool,
) -> tuple[int | None, bool]:
    if selector is None:
        return fallback_index, False
    try:
        return _normalize_axis_selector(selector, axis_names=axis_names, ndim=ndim), False
    except ValueError:
        if not fallback:
            raise
        return fallback_index, True


def _resolve_axis_group_with_fallback(
    selectors: tuple[TensorAxisSelector, ...] | None,
    *,
    axis_names: tuple[str, ...],
    ndim: int,
    fallback_axes: tuple[int, ...],
    fallback: bool,
) -> tuple[int, ...]:
    if selectors is None:
        return fallback_axes
    try:
        resolved = _normalize_axis_group(selectors, axis_names=axis_names, ndim=ndim)
    except ValueError:
        if not fallback:
            raise
        return fallback_axes
    return () if resolved is None else resolved


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
    if any(int(dimension) <= 0 for dimension in shape):
        return (0,)

    def dimension_log(dimension: int) -> float:
        return log(float(dimension))

    if ndim > 16:
        selected: list[int] = [0]
        row_log = dimension_log(shape[0])
        col_log = sum(dimension_log(dim) for dim in shape[1:])
        for index in range(1, ndim):
            dim_log = dimension_log(shape[index])
            if row_log <= col_log:
                selected.append(index)
                row_log += dim_log
                col_log -= dim_log
        if len(selected) == ndim:
            return tuple(selected[:-1])
        return tuple(selected)

    total_log = sum(dimension_log(dim) for dim in shape)
    best_axes: tuple[int, ...] | None = None
    best_score = inf
    best_rank_gap = inf
    for size in range(1, ndim):
        for combo in combinations(range(1, ndim), size - 1):
            candidate = (0,) + combo
            row_log = sum(dimension_log(shape[index]) for index in candidate)
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


def _axis_size_product(shape: tuple[int, ...], axes: tuple[int, ...]) -> int:
    """Return the size product for grouped axes, preserving zero-sized dimensions."""
    if not axes:
        return 1
    return int(np.prod([shape[index] for index in axes], dtype=int))


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
        row_size = _axis_size_product(shape, resolved_rows)
        col_size = _axis_size_product(shape, resolved_cols)
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
    if int(target_size) <= 0:
        raise ValueError("max_matrix_shape must contain exactly two positive integers.")
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


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def _non_nan_values(values: NumericArray) -> NumericArray:
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def _safe_min_max_mean_std(values: NumericArray) -> tuple[float, float, float, float]:
    non_nan = _non_nan_values(values)
    if non_nan.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        return (
            float(np.min(non_nan)),
            float(np.max(non_nan)),
            float(np.mean(non_nan)),
            float(np.std(non_nan)),
        )


def _safe_nanmean_axis(values: NumericArray, *, axis: int | tuple[int, ...]) -> NumericArray:
    array = np.asarray(values, dtype=float)
    valid_mask = ~np.isnan(array)
    counts = np.sum(valid_mask, axis=axis)
    totals = np.sum(np.where(valid_mask, array, 0.0), axis=axis, dtype=float)
    result = np.full(np.shape(totals), np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _safe_nan_norm_axis(values: NumericArray, *, axis: int | tuple[int, ...]) -> NumericArray:
    array = np.asarray(values, dtype=float)
    valid_mask = ~np.isnan(array)
    squared = np.square(np.where(valid_mask, array, 0.0))
    totals = np.sum(squared, axis=axis, dtype=float)
    counts = np.sum(valid_mask, axis=axis)
    norms = np.sqrt(totals)
    return np.where(counts > 0, norms, np.nan)


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


def _resolve_tensor_analysis(
    record: _TensorRecord,
    *,
    analysis: TensorAnalysisConfig | None,
    mode: str,
    fallback: bool = False,
) -> _ResolvedTensorAnalysis:
    """Resolve analytical selectors for one tensor, optionally falling back per tensor."""
    array = np.asarray(record.array)
    original_axis_names = _resolved_axis_names(record)
    explicit_slice_requested = analysis is not None and analysis.slice_axis is not None
    slice_active = bool(array.ndim > 0 and (mode == "slice" or explicit_slice_requested))

    slice_axis: int | None = None
    slice_axis_name: str | None = None
    slice_axis_size = 1
    slice_index = 0
    post_slice_axis_names = original_axis_names
    if slice_active:
        slice_axis, slice_used_fallback = _resolve_axis_selector_with_fallback(
            None if analysis is None else analysis.slice_axis,
            axis_names=original_axis_names,
            ndim=array.ndim,
            fallback_index=0,
            fallback=fallback,
        )
        assert slice_axis is not None
        slice_axis_name = original_axis_names[slice_axis]
        slice_axis_size = int(array.shape[slice_axis])
        requested_slice_index = (
            0 if analysis is None or slice_used_fallback else int(analysis.slice_index)
        )
        slice_index = min(requested_slice_index, max(slice_axis_size - 1, 0))
        post_slice_axis_names = tuple(
            name for axis_index, name in enumerate(original_axis_names) if axis_index != slice_axis
        )

    post_slice_ndim = len(post_slice_axis_names)
    default_reduce_axes = tuple(range(2, post_slice_ndim))
    reduce_axes = _resolve_axis_group_with_fallback(
        None if analysis is None or mode != "reduce" else analysis.reduce_axes,
        axis_names=post_slice_axis_names,
        ndim=post_slice_ndim,
        fallback_axes=default_reduce_axes,
        fallback=fallback,
    )
    reduce_axis_names = tuple(post_slice_axis_names[axis_index] for axis_index in reduce_axes)

    default_profile_axis = None if post_slice_ndim == 0 else 0
    profile_axis, _profile_used_fallback = _resolve_axis_selector_with_fallback(
        None if analysis is None or mode != "profiles" else analysis.profile_axis,
        axis_names=post_slice_axis_names,
        ndim=post_slice_ndim,
        fallback_index=default_profile_axis,
        fallback=fallback,
    )
    profile_axis_name = None if profile_axis is None else post_slice_axis_names[profile_axis]

    return _ResolvedTensorAnalysis(
        original_axis_names=original_axis_names,
        post_slice_axis_names=post_slice_axis_names,
        profile_axis=profile_axis,
        profile_axis_name=profile_axis_name,
        profile_method="mean" if analysis is None else analysis.profile_method,
        reduce_axes=reduce_axes,
        reduce_axis_names=reduce_axis_names,
        reduce_method="mean" if analysis is None else analysis.reduce_method,
        slice_active=slice_active,
        slice_axis=slice_axis,
        slice_axis_name=slice_axis_name,
        slice_axis_size=slice_axis_size,
        slice_index=slice_index,
    )


def _analysis_config_from_resolved(
    analysis: _ResolvedTensorAnalysis,
) -> TensorAnalysisConfig:
    """Build a normalized public analysis config from the resolved per-tensor state."""
    return TensorAnalysisConfig(
        slice_axis=analysis.slice_axis_name if analysis.slice_active else None,
        slice_index=analysis.slice_index,
        reduce_axes=analysis.reduce_axis_names or None,
        reduce_method=analysis.reduce_method,
        profile_axis=analysis.profile_axis_name,
        profile_method=analysis.profile_method,
    )


def _selector_names_for_record(
    selectors: tuple[TensorAxisSelector, ...] | None,
    *,
    record: _TensorRecord,
) -> tuple[str, ...] | None:
    if selectors is None:
        return None
    axis_names = _resolved_axis_names(record)
    resolved = tuple(
        axis_names[_normalize_axis_selector(selector, axis_names=axis_names, ndim=len(axis_names))]
        for selector in selectors
    )
    return resolved or None


def _project_selector_names(
    axis_names: tuple[str, ...],
    selector_names: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if selector_names is None:
        return None
    projected = tuple(name for name in selector_names if name in axis_names)
    return projected or None


def _matrixize_record(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    selector_source: _TensorRecord | None = None,
) -> tuple[NumericArray, _MatrixMetadata]:
    """Matrixize one tensor record while preserving name-based row/column selectors."""
    source_record = record if selector_source is None else selector_source
    row_selector_names = _selector_names_for_record(config.row_axes, record=source_record)
    col_selector_names = _selector_names_for_record(config.col_axes, record=source_record)
    resolved_axis_names = _resolved_axis_names(record)
    return _matrixize_tensor(
        np.asarray(record.array),
        axis_names=resolved_axis_names,
        row_axes=_project_selector_names(resolved_axis_names, row_selector_names),
        col_axes=_project_selector_names(resolved_axis_names, col_selector_names),
    )


def _slice_tensor_record(
    record: _TensorRecord,
    *,
    analysis: _ResolvedTensorAnalysis,
) -> _TensorRecord:
    """Apply the resolved slice operation to one tensor record."""
    resolved_axis_names = _resolved_axis_names(record)
    if not analysis.slice_active or analysis.slice_axis is None:
        return _TensorRecord(
            array=np.asarray(record.array),
            axis_names=resolved_axis_names,
            engine=record.engine,
            name=record.name,
        )
    sliced = np.take(
        np.asarray(record.array),
        indices=int(analysis.slice_index),
        axis=int(analysis.slice_axis),
    )
    sliced_axis_names = tuple(
        name
        for axis_index, name in enumerate(resolved_axis_names)
        if axis_index != analysis.slice_axis
    )
    return _TensorRecord(
        array=np.asarray(sliced),
        axis_names=sliced_axis_names,
        engine=record.engine,
        name=record.name,
    )


def _reduce_tensor_record(
    record: _TensorRecord,
    *,
    analysis: _ResolvedTensorAnalysis,
) -> _TensorRecord:
    """Apply the resolved reduction axes and method to one tensor record."""
    sliced_record = _slice_tensor_record(record, analysis=analysis)
    if not analysis.reduce_axes:
        return sliced_record
    metric_array = np.asarray(_summary_metric_array(np.asarray(sliced_record.array)), dtype=float)
    reducer = _safe_nanmean_axis if analysis.reduce_method == "mean" else _safe_nan_norm_axis
    reduce_axis: int | tuple[int, ...]
    reduce_axis = (
        analysis.reduce_axes[0] if len(analysis.reduce_axes) == 1 else analysis.reduce_axes
    )
    reduced = reducer(metric_array, axis=reduce_axis)
    remaining_names = tuple(
        name
        for axis_index, name in enumerate(_resolved_axis_names(sliced_record))
        if axis_index not in set(analysis.reduce_axes)
    )
    return _TensorRecord(
        array=np.asarray(reduced),
        axis_names=remaining_names,
        engine=sliced_record.engine,
        name=sliced_record.name,
    )


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
            marginal_mean = _safe_nanmean_axis(metrics, axis=other_axes)
            marginal_norm = _safe_nan_norm_axis(metrics, axis=other_axes)
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


def _normalized_topk_magnitudes(magnitudes: NumericArray) -> NumericArray:
    normalized = np.asarray(magnitudes, dtype=float).reshape(-1).copy()
    normalized[np.isnan(normalized)] = float("-inf")
    return normalized


def _topk_flat_indices(magnitudes: NumericArray, *, count: int) -> NumericArray:
    normalized = _normalized_topk_magnitudes(magnitudes)
    size = int(normalized.size)
    requested_count = min(max(int(count), 0), size)
    if requested_count <= 0:
        return np.asarray([], dtype=np.intp)
    if requested_count >= size:
        ranked = sorted(
            range(size),
            key=lambda flat_index: _topk_sort_key(float(normalized[flat_index]), flat_index),
        )
        return np.asarray(ranked, dtype=np.intp)

    partition_index = size - requested_count
    partitioned = np.argpartition(normalized, partition_index)
    threshold = float(normalized[int(partitioned[partition_index])])
    higher = np.flatnonzero(normalized > threshold)
    higher_ranked = np.asarray(
        sorted(
            (int(flat_index) for flat_index in higher),
            key=lambda flat_index: _topk_sort_key(float(normalized[flat_index]), flat_index),
        ),
        dtype=np.intp,
    )
    remaining = requested_count - int(higher_ranked.size)
    if remaining <= 0:
        return higher_ranked[:requested_count]
    equal = np.flatnonzero(normalized == threshold)
    selected_equal = np.asarray(equal[:remaining], dtype=np.intp)
    if higher_ranked.size == 0:
        return selected_equal
    return np.concatenate((higher_ranked, selected_equal))


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
    ranked_indices = _topk_flat_indices(magnitudes, count=requested_count)
    for rank, flat_index_raw in enumerate(ranked_indices, start=1):
        flat_index = int(flat_index_raw)
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
        min_mag, max_mag, mean_mag, std_mag = _safe_min_max_mean_std(magnitude)
        real_min, real_max, _, _ = _safe_min_max_mean_std(np.real(flat))
        imag_min, imag_max, _, _ = _safe_min_max_mean_std(np.imag(flat))
        lines.append(
            "magnitude: "
            f"min={_format_float(min_mag)}, "
            f"max={_format_float(max_mag)}, "
            f"mean={_format_float(mean_mag)}, "
            f"std={_format_float(std_mag)}"
        )
        lines.append(f"real range: {_format_float(real_min)} .. {_format_float(real_max)}")
        lines.append(f"imag range: {_format_float(imag_min)} .. {_format_float(imag_max)}")
    else:
        values = np.real(flat)
        value_min, value_max, value_mean, value_std = _safe_min_max_mean_std(values)
        lines.append(
            "stats: "
            f"min={_format_float(value_min)}, "
            f"max={_format_float(value_max)}, "
            f"mean={_format_float(value_mean)}, "
            f"std={_format_float(value_std)}"
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
    "_PlaybackStepRecord",
    "_ResolvedTensorAnalysis",
    "_SpectralAnalysis",
    "_TensorRecord",
    "_TensorStats",
    "_analysis_config_from_resolved",
    "_build_axis_summary_lines",
    "_build_data_summary_text",
    "_build_stats",
    "_build_topk_lines",
    "_downsample_matrix",
    "_extract_playback_step_records",
    "_extract_einsum_playback_step_records",
    "_extract_tensor_records",
    "_is_direct_array_like_tensor",
    "_matrixize_record",
    "_matrixize_tensor",
    "_reduce_tensor_record",
    "_resolve_tensor_analysis",
    "_resolve_matrix_axes",
    "_resolved_axis_names",
    "_slice_tensor_record",
    "_spectral_analysis_for_record",
    "parse_einsum_equation",
]
