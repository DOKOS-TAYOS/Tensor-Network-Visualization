"""Payload builders for tensor-elements groups, modes, and spectral views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np

from ._tensor_elements_models import _SpectralAnalysis
from ._tensor_elements_data import (
    NumericArray,
    _analysis_config_from_resolved,
    _build_data_summary_text,
    _downsample_matrix,
    _matrixize_record,
    _MatrixMetadata,
    _reduce_tensor_record,
    _resolve_tensor_analysis,
    _resolved_axis_names,
    _safe_nan_norm_axis,
    _safe_nanmean_axis,
    _slice_tensor_record,
    _spectral_analysis_for_record,
    _summary_metric_array,
    _TensorRecord,
)
from .tensor_elements_config import TensorElementsConfig, TensorElementsMode

TensorElementsGroup: TypeAlias = Literal["basic", "complex", "diagnostic", "analysis"]
_ArrayTransform: TypeAlias = Callable[[NumericArray], NumericArray]
_PayloadBuilder: TypeAlias = Callable[
    [_TensorRecord, TensorElementsConfig],
    "_TensorElementsPayload",
]
_HeatmapBuilder: TypeAlias = Callable[[NumericArray, TensorElementsConfig], "_HeatmapComputation"]

_GROUP_MODES: dict[TensorElementsGroup, tuple[str, ...]] = {
    "basic": ("elements", "magnitude", "log_magnitude", "distribution", "data"),
    "complex": ("real", "imag", "phase"),
    "diagnostic": (
        "sign",
        "signed_value",
        "sparsity",
        "nan_inf",
        "singular_values",
        "eigen_real",
        "eigen_imag",
    ),
    "analysis": ("slice", "reduce", "profiles"),
}
_MODE_TO_GROUP: dict[str, TensorElementsGroup] = {
    mode: group for group, modes in _GROUP_MODES.items() for mode in modes
}


@dataclass(frozen=True)
class _HeatmapPayload:
    """Prepared payload for heatmap-like tensor-elements modes.

    Attributes:
        matrix: Matrix data ready to pass to the renderer.
        metadata: Matrixization metadata used for axis labels.
        mode_label: Human-readable mode title.
        colorbar_label: Label displayed next to the colorbar.
        style_key: Renderer style selector for the mode.
        color_limits: Optional explicit colorbar limits.
        outlier_mask: Optional mask used to highlight outliers.
    """

    matrix: NumericArray
    metadata: _MatrixMetadata
    mode_label: str
    colorbar_label: str
    style_key: str
    color_limits: tuple[float, float] | None = None
    outlier_mask: NumericArray | None = None


@dataclass(frozen=True)
class _HistogramPayload:
    """Prepared payload for the value-distribution histogram view."""

    values: NumericArray
    xlabel: str
    mode_label: str


@dataclass(frozen=True)
class _SeriesPayload:
    """Prepared payload for line-plot spectral diagnostics."""

    mode_label: str
    x_values: NumericArray
    xlabel: str
    y_values: NumericArray
    ylabel: str
    yscale: Literal["linear", "log"] = "linear"
    overlay_x_values: NumericArray | None = None
    overlay_y_values: NumericArray | None = None
    overlay_color: str | None = None


@dataclass(frozen=True)
class _TextSummaryPayload:
    """Prepared payload for the textual ``data`` summary view."""

    text: str
    mode_label: str = "data"


_TensorElementsPayload: TypeAlias = (
    _HeatmapPayload | _HistogramPayload | _SeriesPayload | _TextSummaryPayload
)
_HeatmapLikePayload: TypeAlias = _HeatmapPayload | _TextSummaryPayload


@dataclass(frozen=True)
class _HeatmapComputation:
    """Intermediate description of a heatmap transformation before downsampling."""

    matrix: NumericArray
    mode_label: str
    colorbar_label: str
    style_key: str
    post_downsample: _ArrayTransform | None = None


def _mode_group(mode: str) -> TensorElementsGroup:
    """Return the control-group name that owns a concrete tensor-elements mode."""
    if mode not in _MODE_TO_GROUP:
        raise ValueError(f"Unknown tensor-elements mode: {mode!r}.")
    return _MODE_TO_GROUP[mode]


def _group_modes(group: TensorElementsGroup) -> tuple[str, ...]:
    return _GROUP_MODES[group]


def _spectral_mode_flags(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
) -> tuple[bool, bool]:
    """Report whether spectral views are finite-safe and square-safe for one tensor."""
    matrix, _ = _prepare_heatmap_matrix(record, config=config)
    reduced_matrix = _downsample_matrix(matrix, max_shape=config.max_matrix_shape)
    is_finite = bool(np.all(np.isfinite(matrix)))
    is_square = bool(reduced_matrix.shape[0] == reduced_matrix.shape[1])
    return is_finite, is_square


def _mode_supported_for_record(
    record: _TensorRecord,
    mode: str,
    *,
    config: TensorElementsConfig,
) -> bool:
    """Check whether a concrete mode is available for the selected tensor."""
    is_complex = bool(np.iscomplexobj(record.array))
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
        "slice",
        "reduce",
        "profiles",
    ):
        return True
    if mode in ("singular_values", "eigen_real", "eigen_imag"):
        is_finite, is_square = _spectral_mode_flags(record, config=config)
        if mode == "singular_values":
            return is_finite
        return is_finite and is_square
    if mode in ("imag", "phase"):
        return is_complex
    return False


def _valid_group_modes_for_record(
    record: _TensorRecord,
    group: TensorElementsGroup,
    *,
    config: TensorElementsConfig,
) -> tuple[str, ...]:
    return tuple(
        mode
        for mode in _group_modes(group)
        if _mode_supported_for_record(record, mode, config=config)
    )


def _validate_mode_for_record(
    record: _TensorRecord,
    mode: str,
    *,
    config: TensorElementsConfig,
) -> None:
    """Raise a user-facing error when a requested mode is invalid for one tensor.

    Args:
        record: Tensor selected for inspection.
        mode: Concrete mode name requested by the UI or configuration.
        config: Active tensor-elements configuration.

    Raises:
        ValueError: If the requested mode is incompatible with the tensor data.
    """
    if _mode_supported_for_record(record, mode, config=config):
        return
    if mode == "singular_values":
        raise ValueError(
            f"Mode {mode!r} is not available for tensor {record.name!r}. "
            "Spectral analysis requires finite tensor values."
        )
    if mode in ("eigen_real", "eigen_imag"):
        is_finite, is_square = _spectral_mode_flags(record, config=config)
        if not is_finite:
            raise ValueError(
                f"Mode {mode!r} is not available for tensor {record.name!r}. "
                "Eigenvalue views require finite tensor values."
            )
        if not is_square:
            raise ValueError(
                f"Mode {mode!r} is not available for tensor {record.name!r}. "
                "Eigenvalue views require a square analysis matrix."
            )
        raise ValueError(
            f"Mode {mode!r} is not available for tensor {record.name!r}. "
            "Eigenvalue views are not supported for this tensor."
        )
    raise ValueError(
        f"Mode {mode!r} is not available for tensor {record.name!r}. "
        "Complex-only views require a tensor with complex values."
    )


def _resolve_group_mode_for_record(
    record: _TensorRecord,
    *,
    group: TensorElementsGroup,
    preferred_mode: str | None,
    config: TensorElementsConfig,
) -> tuple[TensorElementsGroup, str]:
    """Resolve a valid ``(group, mode)`` pair for the selected tensor.

    The preferred mode is kept when possible. Otherwise the function falls back to the
    first valid mode in the requested group, and finally to a valid mode in ``basic``.
    """
    modes_in_group = _valid_group_modes_for_record(record, group, config=config)
    if preferred_mode in modes_in_group:
        return group, str(preferred_mode)

    for mode in modes_in_group:
        if _mode_supported_for_record(record, mode, config=config):
            return group, mode

    for mode in _group_modes("basic"):
        if _mode_supported_for_record(record, mode, config=config):
            return "basic", mode

    raise ValueError(f"Tensor {record.name!r} does not expose any supported visualization modes.")


def _resolve_mode(array: NumericArray, requested_mode: TensorElementsMode | str) -> str:
    if requested_mode != "auto":
        return str(requested_mode)
    return "magnitude" if np.iscomplexobj(array) else "elements"


def _distribution_values(array: NumericArray, *, max_samples: int) -> NumericArray:
    """Extract finite scalar values for the histogram view, with deterministic sampling."""
    values = np.abs(np.ravel(array)) if np.iscomplexobj(array) else np.real(np.ravel(array))
    values = np.asarray(values)[np.isfinite(values)]
    if int(values.size) <= max_samples:
        return np.asarray(values)
    step = max(int(values.size // max_samples), 1)
    return np.asarray(values[::step][:max_samples])


def _prepare_heatmap_matrix(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    selector_source: _TensorRecord | None = None,
) -> tuple[NumericArray, _MatrixMetadata]:
    """Matrixize one tensor using the active row/column axis configuration."""
    return _matrixize_record(
        record,
        config=config,
        selector_source=selector_source,
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
    selector_source: _TensorRecord | None = None,
) -> _HeatmapLikePayload:
    """Build the renderer payload for one heatmap-like tensor-elements mode."""
    matrix, metadata = _prepare_heatmap_matrix(
        record,
        config=config,
        selector_source=selector_source,
    )
    computation = _HEATMAP_DEFINITIONS[mode](matrix, config)
    if np.asarray(computation.matrix).size == 0:
        return _TextSummaryPayload(
            text=(
                "empty tensor\n\n"
                f"shape: {metadata.original_shape}\n"
                f"mode: {computation.mode_label}\n"
                f"rows: {', '.join(metadata.row_names) if metadata.row_names else '-'}\n"
                f"cols: {', '.join(metadata.col_names) if metadata.col_names else '-'}"
            ),
            mode_label=f"{computation.mode_label} (empty tensor)",
        )
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
    """Build the histogram payload used by ``distribution`` mode."""
    values = _distribution_values(record.array, max_samples=int(config.histogram_max_samples))
    return _HistogramPayload(
        values=values,
        xlabel="magnitude" if np.iscomplexobj(record.array) else "value",
        mode_label="distribution (magnitude)" if np.iscomplexobj(record.array) else "distribution",
    )


def _build_singular_values_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _SeriesPayload:
    analysis = _spectral_analysis_for_record(record, config=config)
    return _build_singular_values_payload_from_analysis(
        record,
        config,
        analysis=analysis,
    )


def _build_singular_values_payload_from_analysis(
    record: _TensorRecord,
    config: TensorElementsConfig,
    *,
    analysis: _SpectralAnalysis,
) -> _SeriesPayload:
    """Build the line-plot payload for the singular-value spectrum.

    Raises:
        ValueError: If the tensor cannot produce a finite singular-value analysis.
    """
    if analysis.issue is not None or analysis.singular_values is None:
        raise ValueError(
            f"Mode 'singular_values' is not available for tensor {record.name!r}. "
            "Spectral analysis requires finite tensor values."
        )

    singular_values = np.asarray(analysis.singular_values, dtype=float)
    zero_threshold = float(config.zero_threshold)
    visual_floor = float(max(config.zero_threshold, config.log_magnitude_floor))
    zero_mask = singular_values <= zero_threshold
    display_values = np.asarray(singular_values, dtype=float).copy()
    display_values[zero_mask] = visual_floor
    mode_label = "singular values"
    if analysis.used_reduced_matrix:
        mode_label += f" (reduced {analysis.analysis_shape})"
    x_values = np.arange(1, int(singular_values.size) + 1, dtype=float)
    overlay_x_values: NumericArray | None = None
    overlay_y_values: NumericArray | None = None
    if np.any(zero_mask):
        overlay_x_values = np.asarray(x_values[zero_mask], dtype=float)
        overlay_y_values = np.full(int(np.count_nonzero(zero_mask)), visual_floor, dtype=float)
    return _SeriesPayload(
        mode_label=mode_label,
        x_values=x_values,
        xlabel="index",
        y_values=display_values,
        ylabel="singular value",
        yscale="log",
        overlay_x_values=overlay_x_values,
        overlay_y_values=overlay_y_values,
        overlay_color="#7F1D1D",
    )


def _ranked_eigenvalues(values: NumericArray) -> NumericArray:
    ranked = sorted(
        np.ravel(values).tolist(),
        key=lambda value: (-float(np.abs(value)), float(np.real(value)), float(np.imag(value))),
    )
    return np.asarray(ranked)


def _build_eigen_component_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
    *,
    component: Literal["real", "imag"],
) -> _SeriesPayload:
    analysis = _spectral_analysis_for_record(record, config=config)
    return _build_eigen_component_payload_from_analysis(
        record,
        config,
        component=component,
        analysis=analysis,
    )


def _build_eigen_component_payload_from_analysis(
    record: _TensorRecord,
    config: TensorElementsConfig,
    *,
    component: Literal["real", "imag"],
    analysis: _SpectralAnalysis,
) -> _SeriesPayload:
    """Build the real or imaginary eigenvalue payload for one tensor.

    Raises:
        ValueError: If the tensor does not expose a finite square analysis matrix.
    """
    mode_name = f"eigen_{component}"
    if analysis.issue is not None:
        raise ValueError(
            f"Mode {mode_name!r} is not available for tensor {record.name!r}. "
            "Eigenvalue views require finite tensor values."
        )
    if analysis.eigenvalues is None:
        raise ValueError(
            f"Mode {mode_name!r} is not available for tensor {record.name!r}. "
            "Eigenvalue views require a square analysis matrix."
        )

    eigenvalues = _ranked_eigenvalues(np.asarray(analysis.eigenvalues))
    if component == "real":
        mode_label = "eigenvalues (real)"
        y_values = np.real(eigenvalues)
        ylabel = "Re(lambda)"
    else:
        mode_label = "eigenvalues (imag)"
        y_values = np.imag(eigenvalues)
        ylabel = "Im(lambda)"
    if analysis.used_reduced_matrix:
        mode_label += f" (reduced {analysis.analysis_shape})"
    return _SeriesPayload(
        mode_label=mode_label,
        x_values=np.arange(1, int(eigenvalues.size) + 1, dtype=float),
        xlabel="index",
        y_values=np.asarray(y_values, dtype=float),
        ylabel=ylabel,
    )


def _build_spectral_payload_from_analysis(
    record: _TensorRecord,
    config: TensorElementsConfig,
    *,
    mode: Literal["singular_values", "eigen_real", "eigen_imag"],
    analysis: _SpectralAnalysis,
) -> _SeriesPayload:
    if mode == "singular_values":
        return _build_singular_values_payload_from_analysis(
            record,
            config,
            analysis=analysis,
        )
    if mode == "eigen_real":
        return _build_eigen_component_payload_from_analysis(
            record,
            config,
            component="real",
            analysis=analysis,
        )
    return _build_eigen_component_payload_from_analysis(
        record,
        config,
        component="imag",
        analysis=analysis,
    )


def _build_text_summary_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _TextSummaryPayload:
    return _TextSummaryPayload(
        text=_build_data_summary_text(
            record,
            config=config,
            topk_count=int(config.topk_count),
        )
    )


def _slice_context_text(record: _TensorRecord, *, mode: str, config: TensorElementsConfig) -> str:
    analysis = _resolve_tensor_analysis(record, analysis=config.analysis, mode=mode)
    if not analysis.slice_active or analysis.slice_axis_name is None:
        return "scalar"
    return f"{analysis.slice_axis_name}={analysis.slice_index}"


def _build_slice_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    analysis = _resolve_tensor_analysis(record, analysis=config.analysis, mode="slice")
    sliced_record = _slice_tensor_record(record, analysis=analysis)
    base_mode = "magnitude" if np.iscomplexobj(sliced_record.array) else "elements"
    payload = _build_heatmap_payload(
        sliced_record,
        replace(config, analysis=_analysis_config_from_resolved(analysis)),
        mode=base_mode,
        selector_source=record,
    )
    return replace(
        payload, mode_label=f"slice ({_slice_context_text(record, mode='slice', config=config)})"
    )


def _build_reduce_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    analysis = _resolve_tensor_analysis(record, analysis=config.analysis, mode="reduce")
    reduced_record = _reduce_tensor_record(record, analysis=analysis)
    base_mode = "magnitude" if np.iscomplexobj(reduced_record.array) else "elements"
    payload = _build_heatmap_payload(
        reduced_record,
        replace(config, analysis=_analysis_config_from_resolved(analysis)),
        mode=base_mode,
        selector_source=record,
    )
    reduced_axes_text = ", ".join(analysis.reduce_axis_names) if analysis.reduce_axis_names else "-"
    mode_label = f"reduce ({analysis.reduce_method} over {reduced_axes_text})"
    if analysis.slice_active:
        mode_label += f" after {analysis.slice_axis_name}={analysis.slice_index}"
    return replace(payload, mode_label=mode_label)


def _build_profiles_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _SeriesPayload:
    analysis = _resolve_tensor_analysis(record, analysis=config.analysis, mode="profiles")
    sliced_record = _slice_tensor_record(record, analysis=analysis)
    metric_array = np.asarray(_summary_metric_array(np.asarray(sliced_record.array)), dtype=float)
    if analysis.profile_axis is None:
        y_values = np.asarray(
            [float(metric_array.reshape(-1)[0]) if metric_array.size else np.nan],
            dtype=float,
        )
        x_values = np.asarray([0.0], dtype=float)
        axis_label = "scalar"
    else:
        other_axes = tuple(
            axis_index
            for axis_index in range(metric_array.ndim)
            if axis_index != analysis.profile_axis
        )
        if other_axes:
            reducer = (
                _safe_nanmean_axis if analysis.profile_method == "mean" else _safe_nan_norm_axis
            )
            reduce_axis: int | tuple[int, ...]
            reduce_axis = other_axes[0] if len(other_axes) == 1 else other_axes
            profile_values = reducer(metric_array, axis=reduce_axis)
        else:
            profile_values = metric_array
        y_values = np.ravel(np.asarray(profile_values, dtype=float))
        x_values = np.arange(int(y_values.size), dtype=float)
        axis_label = analysis.profile_axis_name or _resolved_axis_names(sliced_record)[0]
    is_complex = bool(np.iscomplexobj(record.array))
    if analysis.profile_method == "mean":
        ylabel = "mean(|x|)" if is_complex else "mean"
    else:
        ylabel = "norm(|x|)" if is_complex else "norm"
    mode_label = f"profiles ({analysis.profile_method} along {axis_label})"
    if analysis.slice_active:
        mode_label += f" after {analysis.slice_axis_name}={analysis.slice_index}"
    return _SeriesPayload(
        mode_label=mode_label,
        x_values=x_values,
        xlabel=axis_label,
        y_values=y_values,
        ylabel=ylabel,
    )


def _build_elements_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="elements")


def _build_magnitude_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="magnitude")


def _build_log_magnitude_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="log_magnitude")


def _build_real_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="real")


def _build_imag_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="imag")


def _build_phase_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="phase")


def _build_sign_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="sign")


def _build_signed_value_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="signed_value")


def _build_sparsity_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="sparsity")


def _build_nan_inf_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _HeatmapLikePayload:
    return _build_heatmap_payload(record, config, mode="nan_inf")


def _build_singular_values_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _SeriesPayload:
    return _build_singular_values_payload(record, config)


def _build_eigen_real_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _SeriesPayload:
    return _build_eigen_component_payload(record, config, component="real")


def _build_eigen_imag_mode_payload(
    record: _TensorRecord,
    config: TensorElementsConfig,
) -> _SeriesPayload:
    return _build_eigen_component_payload(record, config, component="imag")


_MODE_PAYLOAD_BUILDERS: dict[str, _PayloadBuilder] = {
    "data": _build_text_summary_payload,
    "distribution": _build_distribution_payload,
    "eigen_imag": _build_eigen_imag_mode_payload,
    "eigen_real": _build_eigen_real_mode_payload,
    "elements": _build_elements_payload,
    "imag": _build_imag_payload,
    "log_magnitude": _build_log_magnitude_payload,
    "magnitude": _build_magnitude_payload,
    "nan_inf": _build_nan_inf_payload,
    "phase": _build_phase_payload,
    "profiles": _build_profiles_mode_payload,
    "real": _build_real_payload,
    "reduce": _build_reduce_mode_payload,
    "slice": _build_slice_mode_payload,
    "sign": _build_sign_payload,
    "sparsity": _build_sparsity_payload,
    "singular_values": _build_singular_values_mode_payload,
    "signed_value": _build_signed_value_payload,
}


def _prepare_mode_payload(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    mode: TensorElementsMode | str,
) -> tuple[str, _TensorElementsPayload]:
    """Resolve and build the payload for one requested tensor-elements mode.

    Args:
        record: Tensor selected for inspection.
        config: Active tensor-elements configuration.
        mode: Requested mode name or ``"auto"``.

    Returns:
        The resolved mode name together with the renderer payload.

    Raises:
        ValueError: If the resolved mode is not available for the selected tensor.
    """
    resolved_mode = _resolve_mode(record.array, mode)
    _validate_mode_for_record(record, resolved_mode, config=config)
    payload = _MODE_PAYLOAD_BUILDERS[resolved_mode](record, config)
    return resolved_mode, payload


__all__ = [
    "TensorElementsGroup",
    "_HeatmapPayload",
    "_HistogramPayload",
    "_SeriesPayload",
    "_TensorElementsPayload",
    "_TextSummaryPayload",
    "_group_modes",
    "_mode_group",
    "_prepare_mode_payload",
    "_resolve_group_mode_for_record",
    "_valid_group_modes_for_record",
]
