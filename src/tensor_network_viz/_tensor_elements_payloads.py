from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import inf, log
from typing import Literal, TypeAlias

import numpy as np

from ._tensor_elements_data import (
    NumericArray,
    _build_data_summary_text,
    _MatrixMetadata,
    _TensorRecord,
)
from .tensor_elements_config import TensorAxisSelector, TensorElementsConfig, TensorElementsMode

TensorElementsGroup: TypeAlias = Literal["basic", "complex", "diagnostic"]
_ArrayTransform: TypeAlias = Callable[[NumericArray], NumericArray]
_PayloadBuilder: TypeAlias = Callable[
    [_TensorRecord, TensorElementsConfig],
    "_TensorElementsPayload",
]
_HeatmapBuilder: TypeAlias = Callable[[NumericArray, TensorElementsConfig], "_HeatmapComputation"]

_GROUP_MODES: dict[TensorElementsGroup, tuple[str, ...]] = {
    "basic": ("elements", "magnitude", "log_magnitude", "distribution", "data"),
    "complex": ("real", "imag", "phase"),
    "diagnostic": ("sign", "signed_value", "sparsity", "nan_inf"),
}
_MODE_TO_GROUP: dict[str, TensorElementsGroup] = {
    mode: group for group, modes in _GROUP_MODES.items() for mode in modes
}


@dataclass(frozen=True)
class _HeatmapPayload:
    matrix: NumericArray
    metadata: _MatrixMetadata
    mode_label: str
    colorbar_label: str
    style_key: str
    color_limits: tuple[float, float] | None = None
    outlier_mask: NumericArray | None = None


@dataclass(frozen=True)
class _HistogramPayload:
    values: NumericArray
    xlabel: str
    mode_label: str


@dataclass(frozen=True)
class _TextSummaryPayload:
    text: str
    mode_label: str = "data"


_TensorElementsPayload: TypeAlias = _HeatmapPayload | _HistogramPayload | _TextSummaryPayload


@dataclass(frozen=True)
class _HeatmapComputation:
    matrix: NumericArray
    mode_label: str
    colorbar_label: str
    style_key: str
    post_downsample: _ArrayTransform | None = None


def _normalize_axis_selector(
    selector: TensorAxisSelector,
    *,
    axis_names: tuple[str, ...],
    ndim: int,
) -> int:
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


def _mode_group(mode: str) -> TensorElementsGroup:
    if mode not in _MODE_TO_GROUP:
        raise ValueError(f"Unknown tensor-elements mode: {mode!r}.")
    return _MODE_TO_GROUP[mode]


def _group_modes(group: TensorElementsGroup) -> tuple[str, ...]:
    return _GROUP_MODES[group]


def _mode_supported_for_array(array: NumericArray, mode: str) -> bool:
    is_complex = bool(np.iscomplexobj(array))
    if mode in (
        "elements",
        "magnitude",
        "log_magnitude",
        "distribution",
        "data",
        "real",
        "sign",
        "signed_value",
        "sparsity",
        "nan_inf",
    ):
        return True
    if mode in ("imag", "phase"):
        return is_complex
    return False


def _valid_group_modes_for_record(
    record: _TensorRecord,
    group: TensorElementsGroup,
) -> tuple[str, ...]:
    return tuple(
        mode for mode in _group_modes(group) if _mode_supported_for_array(record.array, mode)
    )


def _validate_mode_for_record(record: _TensorRecord, mode: str) -> None:
    if _mode_supported_for_array(record.array, mode):
        return
    raise ValueError(
        f"Mode {mode!r} is not available for tensor {record.name!r}. "
        "Complex-only views require a tensor with complex values."
    )


def _resolve_group_mode_for_record(
    record: _TensorRecord,
    *,
    group: TensorElementsGroup,
    preferred_mode: str | None,
) -> tuple[TensorElementsGroup, str]:
    modes_in_group = _valid_group_modes_for_record(record, group)
    if preferred_mode in modes_in_group:
        return group, str(preferred_mode)

    for mode in modes_in_group:
        if _mode_supported_for_array(record.array, mode):
            return group, mode

    for mode in _group_modes("basic"):
        if _mode_supported_for_array(record.array, mode):
            return "basic", mode

    raise ValueError(f"Tensor {record.name!r} does not expose any supported visualization modes.")


def _resolve_mode(array: NumericArray, requested_mode: TensorElementsMode | str) -> str:
    if requested_mode != "auto":
        return str(requested_mode)
    return "magnitude" if np.iscomplexobj(array) else "elements"


def _distribution_values(array: NumericArray, *, max_samples: int) -> NumericArray:
    values = np.abs(np.ravel(array)) if np.iscomplexobj(array) else np.real(np.ravel(array))
    if int(values.size) <= max_samples:
        return np.asarray(values)
    step = max(int(values.size // max_samples), 1)
    return np.asarray(values[::step][:max_samples])


def _prepare_heatmap_matrix(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
) -> tuple[NumericArray, _MatrixMetadata]:
    return _matrixize_tensor(
        np.asarray(record.array),
        axis_names=record.axis_names,
        row_axes=config.row_axes,
        col_axes=config.col_axes,
    )


def _real_component(matrix: NumericArray) -> NumericArray:
    return np.ascontiguousarray(np.real(matrix))


def _imag_component(matrix: NumericArray) -> NumericArray:
    return np.ascontiguousarray(np.imag(matrix))


def _phase_postprocess(matrix: NumericArray) -> NumericArray:
    return np.angle(matrix)


def _elements_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    if np.iscomplexobj(matrix):
        return _HeatmapComputation(
            matrix=_real_component(matrix),
            mode_label="elements (real)",
            colorbar_label="real(value)",
            style_key="elements_complex",
        )
    return _HeatmapComputation(
        matrix=np.asarray(matrix, dtype=float),
        mode_label="elements",
        colorbar_label="value",
        style_key="elements",
    )


def _magnitude_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    return _HeatmapComputation(
        matrix=np.abs(matrix),
        mode_label="magnitude",
        colorbar_label="magnitude",
        style_key="magnitude",
    )


def _log_magnitude_heatmap(
    matrix: NumericArray,
    config: TensorElementsConfig,
) -> _HeatmapComputation:
    magnitude = np.abs(matrix)
    floored = np.maximum(magnitude, float(config.log_magnitude_floor))
    return _HeatmapComputation(
        matrix=np.log10(floored),
        mode_label="log magnitude",
        colorbar_label="log10(magnitude)",
        style_key="log_magnitude",
    )


def _real_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    return _HeatmapComputation(
        matrix=_real_component(matrix),
        mode_label="real",
        colorbar_label="real(value)",
        style_key="real",
    )


def _imag_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    return _HeatmapComputation(
        matrix=_imag_component(matrix),
        mode_label="imag",
        colorbar_label="imag(value)",
        style_key="imag",
    )


def _phase_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    return _HeatmapComputation(
        matrix=np.ascontiguousarray(np.exp(1j * np.angle(matrix))),
        mode_label="phase",
        colorbar_label="phase (rad)",
        style_key="phase",
        post_downsample=_phase_postprocess,
    )


def _sign_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    display_matrix = np.sign(_real_component(matrix) if np.iscomplexobj(matrix) else matrix)
    return _HeatmapComputation(
        matrix=np.asarray(display_matrix),
        mode_label="sign (real)" if np.iscomplexobj(matrix) else "sign",
        colorbar_label="sign(real)" if np.iscomplexobj(matrix) else "sign",
        style_key="sign",
    )


def _signed_value_heatmap(
    matrix: NumericArray,
    _config: TensorElementsConfig,
) -> _HeatmapComputation:
    display_matrix = (
        _real_component(matrix)
        if np.iscomplexobj(matrix)
        else np.asarray(
            matrix,
            dtype=float,
        )
    )
    return _HeatmapComputation(
        matrix=display_matrix,
        mode_label="signed value (real)" if np.iscomplexobj(matrix) else "signed value",
        colorbar_label="signed real(value)" if np.iscomplexobj(matrix) else "signed value",
        style_key="signed_value",
    )


def _sparsity_heatmap(matrix: NumericArray, config: TensorElementsConfig) -> _HeatmapComputation:
    display_matrix = (np.abs(matrix) <= float(config.zero_threshold)).astype(float)
    return _HeatmapComputation(
        matrix=np.asarray(display_matrix),
        mode_label="sparsity",
        colorbar_label="near-zero mask",
        style_key="sparsity",
    )


def _nan_inf_heatmap(matrix: NumericArray, _config: TensorElementsConfig) -> _HeatmapComputation:
    real_part = np.real(matrix)
    imag_part = np.imag(matrix) if np.iscomplexobj(matrix) else np.zeros_like(real_part)
    display_matrix = np.zeros(np.asarray(real_part).shape, dtype=float)
    negative_inf = np.isneginf(real_part) | np.isneginf(imag_part)
    positive_inf = np.isposinf(real_part) | np.isposinf(imag_part)
    nan_mask = np.isnan(real_part) | np.isnan(imag_part)
    display_matrix[negative_inf] = 3.0
    display_matrix[positive_inf] = 2.0
    display_matrix[nan_mask] = 1.0
    return _HeatmapComputation(
        matrix=np.asarray(display_matrix),
        mode_label="nan/inf",
        colorbar_label="value state",
        style_key="nan_inf",
    )


_HEATMAP_DEFINITIONS: dict[str, _HeatmapBuilder] = {
    "elements": _elements_heatmap,
    "imag": _imag_heatmap,
    "log_magnitude": _log_magnitude_heatmap,
    "magnitude": _magnitude_heatmap,
    "nan_inf": _nan_inf_heatmap,
    "phase": _phase_heatmap,
    "real": _real_heatmap,
    "sign": _sign_heatmap,
    "sparsity": _sparsity_heatmap,
    "signed_value": _signed_value_heatmap,
}


def _build_heatmap_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
    *,
    mode: str,
) -> _HeatmapPayload:
    matrix, metadata = _prepare_heatmap_matrix(record, config=config)
    computation = _HEATMAP_DEFINITIONS[mode](matrix, config)
    reduced = _downsample_matrix(computation.matrix, max_shape=config.max_matrix_shape)
    if computation.post_downsample is not None:
        reduced = computation.post_downsample(reduced)
    return _HeatmapPayload(
        matrix=np.asarray(reduced),
        metadata=metadata,
        mode_label=computation.mode_label,
        colorbar_label=computation.colorbar_label,
        style_key=computation.style_key,
    )


def _build_distribution_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HistogramPayload:
    values = _distribution_values(record.array, max_samples=int(config.histogram_max_samples))
    return _HistogramPayload(
        values=values,
        xlabel="magnitude" if np.iscomplexobj(record.array) else "value",
        mode_label="distribution (magnitude)" if np.iscomplexobj(record.array) else "distribution",
    )


def _build_text_summary_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _TextSummaryPayload:
    return _TextSummaryPayload(
        text=_build_data_summary_text(record, topk_count=int(config.topk_count))
    )


def _build_elements_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="elements")


def _build_magnitude_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="magnitude")


def _build_log_magnitude_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="log_magnitude")


def _build_real_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="real")


def _build_imag_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="imag")


def _build_phase_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="phase")


def _build_sign_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="sign")


def _build_signed_value_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="signed_value")


def _build_sparsity_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="sparsity")


def _build_nan_inf_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapPayload:
    return _build_heatmap_payload(record, config, mode="nan_inf")


_MODE_PAYLOAD_BUILDERS: dict[str, _PayloadBuilder] = {
    "data": _build_text_summary_payload,
    "distribution": _build_distribution_payload,
    "elements": _build_elements_payload,
    "imag": _build_imag_payload,
    "log_magnitude": _build_log_magnitude_payload,
    "magnitude": _build_magnitude_payload,
    "nan_inf": _build_nan_inf_payload,
    "phase": _build_phase_payload,
    "real": _build_real_payload,
    "sign": _build_sign_payload,
    "sparsity": _build_sparsity_payload,
    "signed_value": _build_signed_value_payload,
}


def _prepare_mode_payload(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    mode: TensorElementsMode | str,
) -> tuple[str, _TensorElementsPayload]:
    resolved_mode = _resolve_mode(record.array, mode)
    _validate_mode_for_record(record, resolved_mode)
    payload = _MODE_PAYLOAD_BUILDERS[resolved_mode](record, config)
    return resolved_mode, payload


__all__ = [
    "TensorElementsGroup",
    "_HeatmapPayload",
    "_HistogramPayload",
    "_TensorElementsPayload",
    "_TextSummaryPayload",
    "_downsample_matrix",
    "_group_modes",
    "_matrixize_tensor",
    "_mode_group",
    "_prepare_mode_payload",
    "_resolve_group_mode_for_record",
    "_resolve_matrix_axes",
    "_valid_group_modes_for_record",
]
