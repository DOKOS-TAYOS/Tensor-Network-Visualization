from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap

from ._tensor_elements_support import (
    _HeatmapPayload,
    _HistogramPayload,
    _SeriesPayload,
    _TensorElementsPayload,
    _TensorRecord,
    _TextSummaryPayload,
)
from .tensor_elements_config import TensorElementsConfig

_OUTLIER_EDGE_COLOR: Final[str] = "#F8FAFC"
_DATA_TEXT_BOX: Final[dict[str, Any]] = {
    "boxstyle": "round,pad=0.45",
    "facecolor": "#F8FAFC",
    "edgecolor": "#CBD5E1",
}
_CONTINUOUS_STYLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "elements",
        "elements_complex",
        "imag",
        "log_magnitude",
        "magnitude",
        "real",
        "signed_value",
    }
)


@dataclass
class _RenderedTensorPanel:
    base_position: tuple[float, float, float, float]
    main_ax: Axes
    colorbar: Any | None = None


class _PanelLike(Protocol):
    base_position: tuple[float, float, float, float]
    main_ax: Axes
    colorbar: Any | None


def _supports_dynamic_scaling(style_key: str) -> bool:
    return style_key in _CONTINUOUS_STYLE_KEYS


def _finite_display_values(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values = np.asarray(matrix, dtype=float)
    return values[np.isfinite(values)]


def _normalize_color_limits(low: float, high: float) -> tuple[float, float]:
    if np.isclose(low, high):
        pad = max(abs(low) * 0.01, 1e-12)
        return float(low - pad), float(high + pad)
    return float(low), float(high)


def _derive_color_limits(
    matrices: list[np.ndarray[Any, Any]],
    *,
    style_key: str,
    robust_percentiles: tuple[float, float] | None,
) -> tuple[float, float] | None:
    finite_values = [value for matrix in matrices for value in _finite_display_values(matrix)]
    if not finite_values:
        return None
    values = np.asarray(finite_values, dtype=float)
    if robust_percentiles is None:
        low = float(np.min(values))
        high = float(np.max(values))
    else:
        low, high = np.percentile(values, robust_percentiles)
        low = float(low)
        high = float(high)
    if style_key == "signed_value":
        bound = max(abs(low), abs(high), 1e-12)
        return -bound, bound
    return _normalize_color_limits(low, high)


def _compute_outlier_mask(
    matrix: np.ndarray[Any, Any],
    *,
    threshold: float,
) -> np.ndarray[Any, Any] | None:
    values = np.asarray(matrix, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return None
    median = float(np.median(finite_values))
    deviations = np.abs(finite_values - median)
    mad = float(np.median(deviations))
    if mad <= 0.0:
        return None
    modified_z_scores = np.zeros(values.shape, dtype=float)
    modified_z_scores[finite_mask] = 0.6745 * (values[finite_mask] - median) / mad
    outlier_mask = finite_mask & (np.abs(modified_z_scores) > float(threshold))
    if not np.any(outlier_mask):
        return None
    return outlier_mask


def _axis_label_text(prefix: str, labels: tuple[str, ...]) -> str:
    if not labels:
        return f"{prefix}: -"
    return f"{prefix}: {', '.join(labels)}"


def _remove_colorbar(panel: _PanelLike) -> None:
    if panel.colorbar is None:
        return
    panel.colorbar.remove()
    panel.colorbar = None


def _heatmap_style_kwargs(
    *,
    matrix: np.ndarray[Any, Any],
    style_key: str,
    color_limits: tuple[float, float] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "aspect": "auto",
        "interpolation": "nearest",
        "cmap": "viridis",
    }
    if style_key == "phase":
        kwargs["cmap"] = "twilight"
        kwargs["vmin"] = -np.pi
        kwargs["vmax"] = np.pi
    elif style_key == "sign":
        cmap = ListedColormap(("#B91C1C", "#E2E8F0", "#0369A1"))
        kwargs["cmap"] = cmap
        kwargs["norm"] = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), cmap.N)
    elif style_key == "sparsity":
        cmap = ListedColormap(("#0F172A", "#F8FAFC"))
        kwargs["cmap"] = cmap
        kwargs["norm"] = BoundaryNorm((-0.5, 0.5, 1.5), cmap.N)
    elif style_key == "nan_inf":
        cmap = ListedColormap(("#0F766E", "#D97706", "#7C3AED", "#B91C1C"))
        kwargs["cmap"] = cmap
        kwargs["norm"] = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), cmap.N)
    elif style_key == "log_magnitude":
        kwargs["cmap"] = "magma"
    elif style_key == "signed_value":
        kwargs["cmap"] = "RdBu_r"
        if color_limits is None:
            finite_values = matrix[np.isfinite(matrix)]
            bound = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
            bound = max(bound, 1e-12)
            kwargs["vmin"] = -bound
            kwargs["vmax"] = bound
    if color_limits is not None and "norm" not in kwargs:
        kwargs["vmin"], kwargs["vmax"] = color_limits
    return kwargs


def _render_panel(
    panel: _PanelLike,
    *,
    config: TensorElementsConfig,
    record: _TensorRecord,
    payload: _TensorElementsPayload,
) -> None:
    _remove_colorbar(panel)
    panel.main_ax.clear()
    panel.main_ax.set_position(panel.base_position)

    if isinstance(payload, _TextSummaryPayload):
        panel.main_ax.axis("off")
        panel.main_ax.set_title(f"{record.name} [{record.engine}] - {payload.mode_label}")
        panel.main_ax.text(
            0.02,
            0.98,
            payload.text,
            ha="left",
            va="top",
            family="monospace",
            fontsize=9.2,
            linespacing=1.35,
            transform=panel.main_ax.transAxes,
            bbox=_DATA_TEXT_BOX,
        )
        return

    panel.main_ax.set_axis_on()
    if isinstance(payload, _HistogramPayload):
        values = np.asarray(payload.values)
        panel.main_ax.hist(
            values,
            bins=int(config.histogram_bins),
            color="#0369A1",
            edgecolor="#0F172A",
            alpha=0.85,
        )
        panel.main_ax.set_xlabel(payload.xlabel)
        panel.main_ax.set_ylabel("count")
        panel.main_ax.set_title(f"{record.name} [{record.engine}] - {payload.mode_label}")
        return

    if isinstance(payload, _SeriesPayload):
        panel.main_ax.plot(
            np.asarray(payload.x_values, dtype=float),
            np.asarray(payload.y_values, dtype=float),
            color="#0369A1",
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
        if payload.overlay_x_values is not None and payload.overlay_y_values is not None:
            overlay_x = np.asarray(payload.overlay_x_values, dtype=float)
            overlay_y = np.asarray(payload.overlay_y_values, dtype=float)
            if overlay_x.size and overlay_y.size:
                panel.main_ax.scatter(
                    overlay_x,
                    overlay_y,
                    s=40.0,
                    color=payload.overlay_color or "#7F1D1D",
                    zorder=3,
                )
        panel.main_ax.set_xlabel(payload.xlabel)
        panel.main_ax.set_ylabel(payload.ylabel)
        panel.main_ax.set_yscale(payload.yscale)
        panel.main_ax.grid(True, alpha=0.25)
        panel.main_ax.set_title(f"{record.name} [{record.engine}] - {payload.mode_label}")
        return

    assert isinstance(payload, _HeatmapPayload)
    matrix = np.asarray(payload.matrix)
    metadata = payload.metadata
    style_key = payload.style_key
    image = panel.main_ax.imshow(
        matrix,
        **_heatmap_style_kwargs(
            matrix=matrix,
            style_key=style_key,
            color_limits=payload.color_limits,
        ),
    )
    panel.main_ax.set_ylabel(_axis_label_text("rows", metadata.row_names))
    panel.main_ax.set_xlabel(_axis_label_text("cols", metadata.col_names))
    panel.main_ax.set_title(f"{record.name} [{record.engine}] - {payload.mode_label}")
    if payload.outlier_mask is not None and np.any(payload.outlier_mask):
        y_coords, x_coords = np.nonzero(payload.outlier_mask)
        panel.main_ax.scatter(
            x_coords,
            y_coords,
            s=36.0,
            marker="x",
            color=_OUTLIER_EDGE_COLOR,
            linewidths=1.5,
            zorder=3,
        )
    panel.colorbar = panel.main_ax.figure.colorbar(
        image,
        ax=panel.main_ax,
        fraction=0.055,
        pad=0.03,
    )
    if style_key == "sign":
        panel.colorbar.set_ticks([-1.0, 0.0, 1.0])
        panel.colorbar.set_ticklabels(["-1", "0", "+1"])
    elif style_key == "sparsity":
        panel.colorbar.set_ticks([0.0, 1.0])
        panel.colorbar.set_ticklabels(["dense", "zero-ish"])
    elif style_key == "nan_inf":
        panel.colorbar.set_ticks([0.0, 1.0, 2.0, 3.0])
        panel.colorbar.set_ticklabels(["finite", "NaN", "+Inf", "-Inf"])
    panel.colorbar.ax.set_ylabel(payload.colorbar_label, rotation=90)


__all__ = [
    "_RenderedTensorPanel",
    "_compute_outlier_mask",
    "_derive_color_limits",
    "_remove_colorbar",
    "_render_panel",
    "_supports_dynamic_scaling",
]
