from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons, Slider

from ._tensor_elements_support import (
    _extract_tensor_records,
    _group_modes,
    _HeatmapPayload,
    _HistogramPayload,
    _mode_group,
    _prepare_mode_payload,
    _resolve_group_mode_for_record,
    _TensorElementsPayload,
    _TensorRecord,
    _TextSummaryPayload,
    _valid_group_modes_for_record,
)
from ._typing import root_figure
from ._ui_utils import _reserve_figure_bottom
from .config import EngineName
from .tensor_elements_config import TensorElementsConfig, TensorElementsMode
from .viewer import _show_figure

TensorElementsGroup = Literal["basic", "complex", "diagnostic"]

_GROUP_OPTIONS: Final[tuple[TensorElementsGroup, ...]] = ("basic", "complex", "diagnostic")
_GROUP_SELECTOR_BOUNDS: Final[tuple[float, float, float, float]] = (0.02, 0.048, 0.12, 0.12)
_MODE_SELECTOR_BOUNDS: Final[tuple[float, float, float, float]] = (0.16, 0.028, 0.18, 0.18)
_TENSOR_SLIDER_BOUNDS: Final[tuple[float, float, float, float]] = (0.42, 0.058, 0.38, 0.03)
_TENSOR_ELEMENTS_CONTROLS_BOTTOM: Final[float] = 0.24
_INTERACTIVE_LABEL_PROPS: Final[dict[str, Sequence[Any]]] = {"fontsize": [9.5]}
_INTERACTIVE_RADIO_PROPS: Final[dict[str, float]] = {"s": 38.0, "linewidth": 0.9}
_SLIDER_ACTIVE_COLOR: Final[str] = "#0369A1"
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


@dataclass
class _TensorPayloadCacheEntry:
    payloads: dict[str, _TensorElementsPayload] = field(default_factory=dict)


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
    matrices: Sequence[np.ndarray[Any, Any]],
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


class _TensorElementsFigureController:
    def __init__(
        self,
        *,
        config: TensorElementsConfig,
        figure: Figure,
        panel: _RenderedTensorPanel,
        records: list[_TensorRecord],
        allow_interactive_fallback: bool,
        initial_payload_cache: dict[int, _TensorPayloadCacheEntry] | None = None,
    ) -> None:
        self._config = config
        self._figure = figure
        self._panel = panel
        self._records = records
        self._allow_interactive_fallback = allow_interactive_fallback
        self._payload_cache = {} if initial_payload_cache is None else dict(initial_payload_cache)
        self._shared_color_scale_cache: dict[str, tuple[float, float] | None] = {}
        self._tensor_index = 0
        self._group: TensorElementsGroup
        self._mode: str
        self._group_radio_ax: Axes | None = None
        self._group_radio: RadioButtons | None = None
        self._mode_radio_ax: Axes | None = None
        self._mode_radio: RadioButtons | None = None
        self._slider_ax: Axes | None = None
        self._slider: Slider | None = None
        self._group_callback_guard = False
        self._mode_callback_guard = False
        self._slider_callback_guard = False
        self._initialize_selection()

    def _current_record(self) -> _TensorRecord:
        return self._records[self._tensor_index]

    def _payload_cache_for_index(self, index: int) -> _TensorPayloadCacheEntry:
        if index not in self._payload_cache:
            self._payload_cache[index] = _TensorPayloadCacheEntry()
        return self._payload_cache[index]

    def _payload_for_index_mode(self, index: int, mode: str) -> tuple[str, _TensorElementsPayload]:
        record = self._records[index]
        cache_entry = self._payload_cache_for_index(index)
        if mode in cache_entry.payloads:
            return mode, cache_entry.payloads[mode]
        resolved_mode, payload = _prepare_mode_payload(
            record,
            config=self._config,
            mode=mode,
        )
        cache_entry.payloads[resolved_mode] = payload
        return resolved_mode, payload

    def _initial_requested_mode(self) -> str:
        if self._config.mode == "auto":
            return "magnitude" if np.iscomplexobj(self._current_record().array) else "elements"
        return self._config.mode

    def _initialize_selection(self) -> None:
        requested_mode = self._initial_requested_mode()
        requested_group = _mode_group(requested_mode)
        if self._allow_interactive_fallback:
            self._group, self._mode = _resolve_group_mode_for_record(
                self._current_record(),
                group=requested_group,
                preferred_mode=requested_mode,
            )
            return
        self._group = requested_group
        self._mode = requested_mode

    def initialize(self, *, show_controls: bool) -> None:
        if show_controls:
            _reserve_figure_bottom(self._figure, _TENSOR_ELEMENTS_CONTROLS_BOTTOM)
            self._group_radio_ax = self._figure.add_axes(_GROUP_SELECTOR_BOUNDS)
            self._group_radio = RadioButtons(
                self._group_radio_ax,
                _GROUP_OPTIONS,
                active=_GROUP_OPTIONS.index(self._group),
                label_props=_INTERACTIVE_LABEL_PROPS,
                radio_props=_INTERACTIVE_RADIO_PROPS,
            )
            self._group_radio.on_clicked(self._on_group_clicked)
            self._rebuild_mode_radio()

            if len(self._records) > 1:
                self._slider_ax = self._figure.add_axes(_TENSOR_SLIDER_BOUNDS)
                self._slider = Slider(
                    self._slider_ax,
                    "Tensor",
                    0.0,
                    float(len(self._records) - 1),
                    valinit=0.0,
                    valstep=1,
                    color=_SLIDER_ACTIVE_COLOR,
                )
                self._slider.on_changed(self._on_slider_changed)

        self._panel.base_position = self._panel.main_ax.get_position().bounds
        self._sync_slider_label()
        self._render_current(redraw=False)
        self._figure._tensor_network_viz_tensor_elements_controls = self  # type: ignore[attr-defined]

    def _current_group_modes(self) -> tuple[str, ...]:
        if not self._allow_interactive_fallback:
            return _group_modes(self._group)
        return _valid_group_modes_for_record(self._current_record(), self._group)

    def _rebuild_mode_radio(self) -> None:
        if self._mode_radio_ax is not None:
            self._mode_radio_ax.remove()
        self._mode_radio_ax = self._figure.add_axes(_MODE_SELECTOR_BOUNDS)
        mode_options = self._current_group_modes()
        active_index = mode_options.index(self._mode)
        self._mode_radio = RadioButtons(
            self._mode_radio_ax,
            mode_options,
            active=active_index,
            label_props=_INTERACTIVE_LABEL_PROPS,
            radio_props=_INTERACTIVE_RADIO_PROPS,
        )
        self._mode_radio.on_clicked(self._on_mode_clicked)

    def _sync_group_radio_active(self) -> None:
        if self._group_radio is None:
            return
        value_selected = getattr(self._group_radio, "value_selected", None)
        if value_selected == self._group:
            return
        try:
            self._group_callback_guard = True
            self._group_radio.set_active(_GROUP_OPTIONS.index(self._group))
        finally:
            self._group_callback_guard = False

    def _sync_mode_radio_active(self) -> None:
        if self._mode_radio is None:
            return
        mode_options = self._current_group_modes()
        current_labels = tuple(text.get_text() for text in self._mode_radio.labels)
        value_selected = getattr(self._mode_radio, "value_selected", None)
        if value_selected == self._mode and current_labels == mode_options:
            return
        self._rebuild_mode_radio()

    def _sync_slider_label(self) -> None:
        if self._slider is None:
            return
        record = self._current_record()
        self._slider.valtext.set_text(
            f"{self._tensor_index + 1}/{len(self._records)}: {record.name}"
        )

    def _on_group_clicked(self, label: str | None) -> None:
        if label is None or self._group_callback_guard:
            return
        self.set_group(label)

    def _on_mode_clicked(self, label: str | None) -> None:
        if label is None or self._mode_callback_guard:
            return
        self.set_mode(label)

    def _on_slider_changed(self, value: float) -> None:
        if self._slider_callback_guard:
            return
        self.set_tensor_index(int(round(value)))

    def _render_current(self, *, redraw: bool) -> None:
        resolved_mode, payload = self._payload_for_current()
        _render_panel(
            self._panel,
            config=self._config,
            record=self._current_record(),
            payload=payload,
        )
        self._mode = resolved_mode
        self._sync_group_radio_active()
        self._sync_mode_radio_active()
        self._sync_slider_label()
        if redraw:
            self._figure.canvas.draw_idle()

    def _payload_for_current(self) -> tuple[str, _TensorElementsPayload]:
        resolved_mode, payload = self._payload_for_index_mode(self._tensor_index, self._mode)
        if isinstance(payload, _HeatmapPayload):
            payload = self._finalize_heatmap_payload(resolved_mode, payload)
        return resolved_mode, payload

    def _shared_color_limits_for_mode(self, mode: str, style_key: str) -> tuple[float, float] | None:
        if mode in self._shared_color_scale_cache:
            return self._shared_color_scale_cache[mode]
        matrices: list[np.ndarray[Any, Any]] = []
        for index in range(len(self._records)):
            try:
                _, payload = self._payload_for_index_mode(index, mode)
            except ValueError:
                continue
            if not isinstance(payload, _HeatmapPayload):
                continue
            if not _supports_dynamic_scaling(payload.style_key):
                continue
            matrices.append(np.asarray(payload.matrix, dtype=float))
        limits = _derive_color_limits(
            matrices,
            style_key=style_key,
            robust_percentiles=self._config.robust_percentiles,
        )
        self._shared_color_scale_cache[mode] = limits
        return limits

    def _finalize_heatmap_payload(self, mode: str, payload: _HeatmapPayload) -> _HeatmapPayload:
        color_limits: tuple[float, float] | None = None
        outlier_mask: np.ndarray[Any, Any] | None = None
        matrix = np.asarray(payload.matrix, dtype=float)
        if _supports_dynamic_scaling(payload.style_key):
            if self._config.shared_color_scale:
                color_limits = self._shared_color_limits_for_mode(mode, payload.style_key)
            elif self._config.robust_percentiles is not None:
                color_limits = _derive_color_limits(
                    [matrix],
                    style_key=payload.style_key,
                    robust_percentiles=self._config.robust_percentiles,
                )
            if self._config.highlight_outliers:
                outlier_mask = _compute_outlier_mask(
                    matrix,
                    threshold=float(self._config.outlier_zscore),
                )
        return replace(payload, color_limits=color_limits, outlier_mask=outlier_mask)

    def set_group(self, group: TensorElementsGroup | str, *, redraw: bool = True) -> None:
        resolved_group = str(group)
        self._group, self._mode = _resolve_group_mode_for_record(
            self._current_record(),
            group=resolved_group,  # type: ignore[arg-type]
            preferred_mode=self._mode if _mode_group(self._mode) == resolved_group else None,
        )
        self._render_current(redraw=redraw)

    def set_mode(self, mode: TensorElementsMode | str, *, redraw: bool = True) -> None:
        resolved_group = _mode_group(str(mode))
        if self._allow_interactive_fallback:
            self._group, self._mode = _resolve_group_mode_for_record(
                self._current_record(),
                group=resolved_group,
                preferred_mode=str(mode),
            )
        else:
            self._group = resolved_group
            self._mode = str(mode)
        self._render_current(redraw=redraw)

    def set_tensor_index(self, index: int, *, redraw: bool = True) -> None:
        clamped = int(np.clip(index, 0, len(self._records) - 1))
        self._tensor_index = clamped
        if self._allow_interactive_fallback:
            self._group, self._mode = _resolve_group_mode_for_record(
                self._current_record(),
                group=self._group,
                preferred_mode=self._mode,
            )
        if (
            self._slider is not None
            and not self._slider_callback_guard
            and float(self._slider.val) != float(clamped)
        ):
            try:
                self._slider_callback_guard = True
                self._slider.set_val(float(clamped))
            finally:
                self._slider_callback_guard = False
        self._render_current(redraw=redraw)


def _axis_label_text(prefix: str, labels: tuple[str, ...]) -> str:
    if not labels:
        return f"{prefix}: -"
    return f"{prefix}: {', '.join(labels)}"


def _remove_colorbar(panel: _RenderedTensorPanel) -> None:
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
    panel: _RenderedTensorPanel,
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


def _build_single_external_axis(ax: Axes) -> tuple[Figure, Axes]:
    figure = root_figure(ax.figure)
    return figure, ax


def _build_internal_axis(*, config: TensorElementsConfig) -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=config.figsize)
    return root_figure(figure), ax


def show_tensor_elements(
    data: Any,
    *,
    engine: EngineName | None = None,
    config: TensorElementsConfig | None = None,
    ax: Axes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Render tensor values in a single Matplotlib view with optional controls."""
    style = config or TensorElementsConfig()
    _, records = _extract_tensor_records(data, engine=engine)
    initial_payload_cache: dict[int, _TensorPayloadCacheEntry] = {}
    if style.mode != "auto" and (not show_controls or len(records) == 1):
        resolved_mode, payload = _prepare_mode_payload(records[0], config=style, mode=style.mode)
        initial_payload_cache[0] = _TensorPayloadCacheEntry(payloads={resolved_mode: payload})
    if ax is not None and len(records) != 1:
        raise ValueError("An explicit ax is only supported when visualizing a single tensor.")
    if ax is not None and show_controls and len(root_figure(ax.figure).axes) > 1:
        raise ValueError(
            "show_controls=True with an external ax is only supported when the target figure "
            "contains a single axes."
        )

    if ax is None:
        figure, main_ax = _build_internal_axis(config=style)
    else:
        figure, main_ax = _build_single_external_axis(ax)

    panel = _RenderedTensorPanel(
        base_position=main_ax.get_position().bounds,
        main_ax=main_ax,
    )
    controller = _TensorElementsFigureController(
        config=style,
        figure=figure,
        panel=panel,
        records=records,
        allow_interactive_fallback=show_controls,
        initial_payload_cache=initial_payload_cache,
    )
    controller.initialize(show_controls=show_controls)

    if show:
        _show_figure(figure)
    return root_figure(figure), main_ax


__all__ = ["show_tensor_elements"]
