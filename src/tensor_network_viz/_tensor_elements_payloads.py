"""Payload builders for tensor-elements groups, modes, and spectral views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from ._tensor_elements_data import (
    NumericArray,
    _build_data_summary_text,
    _downsample_matrix,
    _matrixize_tensor,
    _MatrixMetadata,
    _spectral_analysis_for_record,
    _TensorRecord,
)
from .tensor_elements_config import TensorElementsConfig, TensorElementsMode

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
    "diagnostic": (
        "sign",
        "signed_value",
        "sparsity",
        "nan_inf",
        "singular_values",
        "eigen_real",
        "eigen_imag",
    ),
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
) -> tuple[NumericArray, _MatrixMetadata]:
    """Matrixize one tensor using the active row/column axis configuration."""
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
    """Build the renderer payload for one heatmap-like tensor-elements mode."""
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
    """Build the line-plot payload for the singular-value spectrum.

    Raises:
        ValueError: If the tensor cannot produce a finite singular-value analysis.
    """
    analysis = _spectral_analysis_for_record(record, config=config)
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
    """Build the real or imaginary eigenvalue payload for one tensor.

    Raises:
        ValueError: If the tensor does not expose a finite square analysis matrix.
    """
    analysis = _spectral_analysis_for_record(record, config=config)
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
    "real": _build_real_payload,
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
