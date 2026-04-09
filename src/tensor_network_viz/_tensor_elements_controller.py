from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

from ._matplotlib_state import set_tensor_elements_controls
from ._tensor_elements_data import _analysis_config_from_resolved, _resolve_tensor_analysis
from ._tensor_elements_rendering import (
    _compute_outlier_mask,
    _derive_color_limits,
    _remove_colorbar,
    _render_panel,
    _RenderedTensorPanel,
    _supports_dynamic_scaling,
)
from ._tensor_elements_support import (
    _group_modes,
    _HeatmapPayload,
    _mode_group,
    _resolve_group_mode_for_record,
    _TensorElementsPayload,
    _TensorRecord,
    _valid_group_modes_for_record,
)
from ._ui_utils import _reserve_figure_bottom, _style_control_tray_axes
from ._widgets import _SafeSlider
from .tensor_elements_config import TensorAnalysisConfig, TensorElementsConfig, TensorElementsMode

TensorElementsGroup = Literal["basic", "complex", "diagnostic", "analysis"]
PrepareModePayloadFn = Callable[
    [_TensorRecord, TensorElementsConfig, str],
    tuple[str, _TensorElementsPayload],
]

_ANALYSIS_METHOD_OPTIONS: Final[tuple[str, str]] = ("mean", "norm")
_ANALYSIS_MODES: Final[frozenset[str]] = frozenset({"slice", "reduce", "profiles"})
_GROUP_OPTIONS: Final[tuple[TensorElementsGroup, ...]] = (
    "basic",
    "complex",
    "diagnostic",
    "analysis",
)
_GROUP_SELECTOR_BOUNDS: Final[tuple[float, float, float, float]] = (0.02, 0.04, 0.15, 0.145)
_MODE_SELECTOR_BOUNDS: Final[tuple[float, float, float, float]] = (0.175, 0.028, 0.21, 0.16)
_TENSOR_SLIDER_BOUNDS: Final[tuple[float, float, float, float]] = (0.48, 0.01, 0.38, 0.055)
_ANALYSIS_AXIS_BOUNDS: Final[tuple[float, float, float, float]] = (0.392, 0.078, 0.205, 0.155)
_ANALYSIS_CHECK_BOUNDS: Final[tuple[float, float, float, float]] = (0.392, 0.075, 0.215, 0.165)
_ANALYSIS_METHOD_BOUNDS: Final[tuple[float, float, float, float]] = (0.61, 0.122, 0.13, 0.105)
_ANALYSIS_SLIDER_BOUNDS: Final[tuple[float, float, float, float]] = (0.66, 0.135, 0.3, 0.05)
_TENSOR_ELEMENTS_CONTROLS_BOTTOM: Final[float] = 0.31
_INTERACTIVE_LABEL_PROPS: Final[dict[str, Sequence[Any]]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: Final[dict[str, float]] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: Final[dict[str, float]] = {"s": 34.0, "linewidth": 1.0}
_INTERACTIVE_RADIO_PROPS: Final[dict[str, float]] = {"s": 38.0, "linewidth": 0.9}
_ANALYSIS_SLIDER_LABEL_X: Final[float] = -0.05
_SLIDER_ACTIVE_COLOR: Final[str] = "#0369A1"
_PLACEHOLDER_TEXT_BOX: Final[dict[str, Any]] = {
    "boxstyle": "round,pad=0.45",
    "facecolor": "#F8FAFC",
    "edgecolor": "#CBD5E1",
}


@dataclass(frozen=True)
class _TensorElementsControlsLayout:
    controls_bottom: float
    group_selector_bounds: tuple[float, float, float, float]
    mode_selector_bounds: tuple[float, float, float, float]
    tensor_slider_bounds: tuple[float, float, float, float]
    tensor_slider_label_x: float
    analysis_axis_bounds: tuple[float, float, float, float]
    analysis_check_bounds: tuple[float, float, float, float]
    analysis_method_bounds: tuple[float, float, float, float]
    analysis_slider_bounds: tuple[float, float, float, float]


_DEFAULT_TENSOR_ELEMENTS_CONTROLS_LAYOUT: Final[_TensorElementsControlsLayout] = (
    _TensorElementsControlsLayout(
        controls_bottom=_TENSOR_ELEMENTS_CONTROLS_BOTTOM,
        group_selector_bounds=_GROUP_SELECTOR_BOUNDS,
        mode_selector_bounds=_MODE_SELECTOR_BOUNDS,
        tensor_slider_bounds=_TENSOR_SLIDER_BOUNDS,
        tensor_slider_label_x=-0.075,
        analysis_axis_bounds=_ANALYSIS_AXIS_BOUNDS,
        analysis_check_bounds=_ANALYSIS_CHECK_BOUNDS,
        analysis_method_bounds=_ANALYSIS_METHOD_BOUNDS,
        analysis_slider_bounds=_ANALYSIS_SLIDER_BOUNDS,
    )
)


@dataclass
class _TensorPayloadCacheEntry:
    payloads: dict[str, _TensorElementsPayload] = field(default_factory=dict)


class _TensorElementsFigureController:
    def __init__(
        self,
        *,
        config: TensorElementsConfig,
        controls_layout: _TensorElementsControlsLayout | None,
        figure: Figure,
        panel: _RenderedTensorPanel,
        records: list[_TensorRecord],
        allow_interactive_fallback: bool,
        prepare_mode_payload: PrepareModePayloadFn,
        initial_payload_cache: dict[int, _TensorPayloadCacheEntry] | None = None,
    ) -> None:
        self._config = config
        self._controls_layout = (
            _DEFAULT_TENSOR_ELEMENTS_CONTROLS_LAYOUT if controls_layout is None else controls_layout
        )
        self._figure = figure
        self._panel = panel
        self._records = records
        self._allow_interactive_fallback = allow_interactive_fallback
        self._prepare_mode_payload = prepare_mode_payload
        self._payload_cache = {} if initial_payload_cache is None else dict(initial_payload_cache)
        self._shared_color_scale_cache: dict[str, tuple[float, float] | None] = {}
        self._analysis = config.analysis or TensorAnalysisConfig()
        self._tensor_index = 0
        self._group: TensorElementsGroup
        self._mode: str
        self._group_radio_ax: Axes | None = None
        self._group_radio: RadioButtons | None = None
        self._mode_radio_ax: Axes | None = None
        self._mode_radio: RadioButtons | None = None
        self._analysis_axis_ax: Axes | None = None
        self._analysis_axis_radio: RadioButtons | None = None
        self._analysis_check_ax: Axes | None = None
        self._analysis_checkbuttons: CheckButtons | None = None
        self._analysis_method_ax: Axes | None = None
        self._analysis_method_radio: RadioButtons | None = None
        self._analysis_slider_ax: Axes | None = None
        self._analysis_slider: Slider | None = None
        self._slider_ax: Axes | None = None
        self._slider: Slider | None = None
        self._analysis_axis_callback_guard = False
        self._analysis_check_callback_guard = False
        self._analysis_method_callback_guard = False
        self._analysis_slider_callback_guard = False
        self._group_callback_guard = False
        self._mode_callback_guard = False
        self._slider_callback_guard = False
        self._initialize_selection()

    @property
    def selected_mode(self) -> str:
        return self._mode

    def _current_record(self) -> _TensorRecord:
        return self._records[self._tensor_index]

    def _reset_payload_caches(self) -> None:
        self._payload_cache.clear()
        self._shared_color_scale_cache.clear()

    def _set_group_mode_for_record(
        self,
        record: _TensorRecord,
        *,
        group: TensorElementsGroup,
        preferred_mode: str | None,
    ) -> None:
        self._group, self._mode = _resolve_group_mode_for_record(
            record,
            group=group,
            preferred_mode=preferred_mode,
            config=self._config,
        )

    def _sync_controls_after_render(self, *, rebuild_analysis_controls: bool = True) -> None:
        self._sync_group_radio_active()
        self._sync_mode_radio_active()
        if rebuild_analysis_controls:
            self._rebuild_analysis_controls()
        self._sync_slider_label()

    def _maybe_redraw(self, *, redraw: bool) -> None:
        if redraw:
            self._figure.canvas.draw_idle()

    def _sync_slider_value(self, index: int) -> None:
        if (
            self._slider is None
            or self._slider_callback_guard
            or float(self._slider.val) == float(index)
        ):
            return
        try:
            self._slider_callback_guard = True
            self._slider.set_val(float(index))
        finally:
            self._slider_callback_guard = False

    def _payload_cache_for_index(self, index: int) -> _TensorPayloadCacheEntry:
        if index not in self._payload_cache:
            self._payload_cache[index] = _TensorPayloadCacheEntry()
        return self._payload_cache[index]

    def _payload_for_index_mode(self, index: int, mode: str) -> tuple[str, _TensorElementsPayload]:
        record = self._records[index]
        cache_entry = self._payload_cache_for_index(index)
        if mode in cache_entry.payloads:
            return mode, cache_entry.payloads[mode]
        active_config = self._config
        if mode in _ANALYSIS_MODES:
            resolved_analysis = _resolve_tensor_analysis(
                record,
                analysis=self._analysis,
                mode=mode,
                fallback=self._allow_interactive_fallback,
            )
            active_config = replace(
                self._config,
                analysis=_analysis_config_from_resolved(resolved_analysis),
            )
        resolved_mode, payload = self._prepare_mode_payload(record, active_config, mode)
        cache_entry.payloads[resolved_mode] = payload
        return resolved_mode, payload

    def _initial_requested_mode(self) -> str:
        if self._config.mode == "auto":
            return "magnitude" if np.iscomplexobj(self._current_record().array) else "elements"
        return self._config.mode

    def _initialize_selection(self) -> None:
        requested_mode = self._initial_requested_mode()
        requested_group = cast(TensorElementsGroup, _mode_group(requested_mode))
        if self._allow_interactive_fallback:
            self._set_group_mode_for_record(
                self._current_record(),
                group=requested_group,
                preferred_mode=requested_mode,
            )
            return
        self._group = requested_group
        self._mode = requested_mode

    def initialize(self, *, show_controls: bool) -> None:
        if show_controls:
            _reserve_figure_bottom(self._figure, self._controls_layout.controls_bottom)
            self._group_radio_ax = self._figure.add_axes(
                self._controls_layout.group_selector_bounds
            )
            _style_control_tray_axes(self._group_radio_ax)
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
                self._slider_ax = self._figure.add_axes(self._controls_layout.tensor_slider_bounds)
                _style_control_tray_axes(self._slider_ax)
                self._slider = _SafeSlider(
                    self._slider_ax,
                    "Tensor",
                    0.0,
                    float(len(self._records) - 1),
                    valinit=0.0,
                    valstep=1,
                    color=_SLIDER_ACTIVE_COLOR,
                )
                self._slider.label.set_x(self._controls_layout.tensor_slider_label_x)
                self._slider.on_changed(self._on_slider_changed)

        self._panel.base_position = self._panel.main_ax.get_position().bounds
        self._sync_slider_label()
        self._render_current(redraw=False)
        set_tensor_elements_controls(self._figure, self)

    def _current_group_modes(self) -> tuple[str, ...]:
        if not self._allow_interactive_fallback:
            return _group_modes(self._group)
        return _valid_group_modes_for_record(
            self._current_record(),
            self._group,
            config=self._config,
        )

    def _rebuild_mode_radio(self) -> None:
        if self._mode_radio_ax is not None:
            self._mode_radio_ax.remove()
        self._mode_radio_ax = self._figure.add_axes(self._controls_layout.mode_selector_bounds)
        _style_control_tray_axes(self._mode_radio_ax)
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

    def _clear_analysis_controls(self) -> None:
        for widget in (
            self._analysis_axis_radio,
            self._analysis_checkbuttons,
            self._analysis_method_radio,
            self._analysis_slider,
        ):
            if widget is not None and hasattr(widget, "disconnect_events"):
                widget.disconnect_events()
        for attr_name in (
            "_analysis_axis_ax",
            "_analysis_check_ax",
            "_analysis_method_ax",
            "_analysis_slider_ax",
        ):
            axes = getattr(self, attr_name)
            if axes is not None:
                axes.remove()
                setattr(self, attr_name, None)
        self._analysis_axis_radio = None
        self._analysis_checkbuttons = None
        self._analysis_method_radio = None
        self._analysis_slider = None

    def _rebuild_analysis_controls(self) -> None:
        self._clear_analysis_controls()
        if self._mode not in _ANALYSIS_MODES:
            return
        analysis = _resolve_tensor_analysis(
            self._current_record(),
            analysis=self._analysis,
            mode=self._mode,
            fallback=self._allow_interactive_fallback,
        )
        if self._mode == "slice":
            if analysis.original_axis_names:
                self._analysis_axis_ax = self._figure.add_axes(
                    self._controls_layout.analysis_axis_bounds
                )
                _style_control_tray_axes(self._analysis_axis_ax)
                active_index = int(analysis.slice_axis or 0)
                self._analysis_axis_radio = RadioButtons(
                    self._analysis_axis_ax,
                    analysis.original_axis_names,
                    active=active_index,
                    label_props=_INTERACTIVE_LABEL_PROPS,
                    radio_props=_INTERACTIVE_RADIO_PROPS,
                )
                self._analysis_axis_radio.on_clicked(self._on_analysis_axis_clicked)
            self._analysis_slider_ax = self._figure.add_axes(
                self._controls_layout.analysis_slider_bounds
            )
            _style_control_tray_axes(self._analysis_slider_ax)
            self._analysis_slider = _SafeSlider(
                self._analysis_slider_ax,
                "Slice",
                0.0,
                float(max(analysis.slice_axis_size - 1, 0)),
                valinit=float(analysis.slice_index),
                valstep=1,
                color=_SLIDER_ACTIVE_COLOR,
            )
            self._analysis_slider.label.set_x(_ANALYSIS_SLIDER_LABEL_X)
            self._analysis_slider.on_changed(self._on_analysis_slider_changed)
            return

        if self._mode == "reduce":
            if analysis.post_slice_axis_names:
                self._analysis_check_ax = self._figure.add_axes(
                    self._controls_layout.analysis_check_bounds
                )
                _style_control_tray_axes(self._analysis_check_ax)
                self._analysis_checkbuttons = CheckButtons(
                    self._analysis_check_ax,
                    analysis.post_slice_axis_names,
                    [
                        axis_name in analysis.reduce_axis_names
                        for axis_name in analysis.post_slice_axis_names
                    ],
                    label_props=_INTERACTIVE_LABEL_PROPS,
                    frame_props=_INTERACTIVE_CHECK_FRAME_PROPS,
                    check_props=_INTERACTIVE_CHECK_MARK_PROPS,
                )
                self._analysis_checkbuttons.on_clicked(self._on_analysis_check_clicked)
            self._analysis_method_ax = self._figure.add_axes(
                self._controls_layout.analysis_method_bounds
            )
            _style_control_tray_axes(self._analysis_method_ax)
            self._analysis_method_radio = RadioButtons(
                self._analysis_method_ax,
                _ANALYSIS_METHOD_OPTIONS,
                active=_ANALYSIS_METHOD_OPTIONS.index(analysis.reduce_method),
                label_props=_INTERACTIVE_LABEL_PROPS,
                radio_props=_INTERACTIVE_RADIO_PROPS,
            )
            self._analysis_method_radio.on_clicked(self._on_analysis_method_clicked)
            return

        if analysis.post_slice_axis_names:
            self._analysis_axis_ax = self._figure.add_axes(
                self._controls_layout.analysis_axis_bounds
            )
            _style_control_tray_axes(self._analysis_axis_ax)
            active_index = int(analysis.profile_axis or 0)
            self._analysis_axis_radio = RadioButtons(
                self._analysis_axis_ax,
                analysis.post_slice_axis_names,
                active=active_index,
                label_props=_INTERACTIVE_LABEL_PROPS,
                radio_props=_INTERACTIVE_RADIO_PROPS,
            )
            self._analysis_axis_radio.on_clicked(self._on_analysis_axis_clicked)
        self._analysis_method_ax = self._figure.add_axes(
            self._controls_layout.analysis_method_bounds
        )
        _style_control_tray_axes(self._analysis_method_ax)
        self._analysis_method_radio = RadioButtons(
            self._analysis_method_ax,
            _ANALYSIS_METHOD_OPTIONS,
            active=_ANALYSIS_METHOD_OPTIONS.index(analysis.profile_method),
            label_props=_INTERACTIVE_LABEL_PROPS,
            radio_props=_INTERACTIVE_RADIO_PROPS,
        )
        self._analysis_method_radio.on_clicked(self._on_analysis_method_clicked)

    def _clear_analysis_caches(self) -> None:
        self._reset_payload_caches()

    def _set_analysis(
        self,
        analysis: TensorAnalysisConfig,
        *,
        redraw: bool = True,
        rebuild_analysis_controls: bool = True,
    ) -> None:
        self._analysis = analysis
        self._clear_analysis_caches()
        self._render_current(redraw=redraw, rebuild_analysis_controls=rebuild_analysis_controls)

    def _on_analysis_axis_clicked(self, label: str | None) -> None:
        if label is None or self._analysis_axis_callback_guard:
            return
        if self._mode == "slice":
            self._set_analysis(replace(self._analysis, slice_axis=str(label), slice_index=0))
            return
        if self._mode == "profiles":
            self._set_analysis(replace(self._analysis, profile_axis=str(label)))

    def _on_analysis_check_clicked(self, _label: str | None) -> None:
        if self._analysis_check_callback_guard or self._analysis_checkbuttons is None:
            return
        labels = tuple(text.get_text() for text in self._analysis_checkbuttons.labels)
        statuses = tuple(bool(value) for value in self._analysis_checkbuttons.get_status())
        selected_axes = tuple(
            label for label, enabled in zip(labels, statuses, strict=True) if enabled
        )
        self._set_analysis(replace(self._analysis, reduce_axes=selected_axes))

    def _on_analysis_method_clicked(self, label: str | None) -> None:
        if label is None or self._analysis_method_callback_guard:
            return
        if self._mode == "reduce":
            self._set_analysis(
                replace(self._analysis, reduce_method=cast(Literal["mean", "norm"], label))
            )
            return
        if self._mode == "profiles":
            self._set_analysis(
                replace(self._analysis, profile_method=cast(Literal["mean", "norm"], label))
            )

    def _on_analysis_slider_changed(self, value: float) -> None:
        if self._analysis_slider_callback_guard:
            return
        self._set_analysis(
            replace(self._analysis, slice_index=int(round(value))),
            rebuild_analysis_controls=False,
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

    def _render_current(self, *, redraw: bool, rebuild_analysis_controls: bool = True) -> None:
        resolved_mode, payload = self._payload_for_current()
        _render_panel(
            self._panel,
            config=self._config,
            record=self._current_record(),
            payload=payload,
        )
        self._mode = resolved_mode
        self._sync_controls_after_render(rebuild_analysis_controls=rebuild_analysis_controls)
        self._maybe_redraw(redraw=redraw)

    def _payload_for_current(self) -> tuple[str, _TensorElementsPayload]:
        resolved_mode, payload = self._payload_for_index_mode(self._tensor_index, self._mode)
        if isinstance(payload, _HeatmapPayload):
            payload = self._finalize_heatmap_payload(resolved_mode, payload)
        return resolved_mode, payload

    def _shared_color_limits_for_mode(
        self, mode: str, style_key: str
    ) -> tuple[float, float] | None:
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
        resolved_group = cast(TensorElementsGroup, str(group))
        self._set_group_mode_for_record(
            self._current_record(),
            group=resolved_group,
            preferred_mode=self._mode if _mode_group(self._mode) == resolved_group else None,
        )
        self._render_current(redraw=redraw)

    def set_mode(self, mode: TensorElementsMode | str, *, redraw: bool = True) -> None:
        resolved_group = cast(TensorElementsGroup, _mode_group(str(mode)))
        if self._allow_interactive_fallback:
            self._set_group_mode_for_record(
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
            self._set_group_mode_for_record(
                self._current_record(),
                group=self._group,
                preferred_mode=self._mode,
            )
        self._sync_slider_value(clamped)
        self._render_current(redraw=redraw)

    def set_single_record(self, record: _TensorRecord, *, redraw: bool = True) -> None:
        if len(self._records) != 1:
            raise ValueError("set_single_record is only supported for single-record controllers.")
        self._records = [record]
        self._tensor_index = 0
        self._reset_payload_caches()
        if self._allow_interactive_fallback:
            self._set_group_mode_for_record(
                record,
                group=self._group,
                preferred_mode=self._mode,
            )
        self._render_current(redraw=redraw)

    def render_placeholder(
        self,
        text: str,
        *,
        title: str = "Tensor inspector",
        redraw: bool = True,
    ) -> None:
        _remove_colorbar(self._panel)
        self._panel.main_ax.clear()
        self._panel.main_ax.set_position(self._panel.base_position)
        self._panel.main_ax.axis("off")
        self._panel.main_ax.set_title(title)
        self._panel.main_ax.text(
            0.02,
            0.98,
            text,
            ha="left",
            va="top",
            family="monospace",
            fontsize=10.0,
            linespacing=1.35,
            transform=self._panel.main_ax.transAxes,
            bbox=_PLACEHOLDER_TEXT_BOX,
        )
        if redraw:
            self._figure.canvas.draw_idle()


__all__ = [
    "_RenderedTensorPanel",
    "_TensorElementsControlsLayout",
    "_TensorElementsFigureController",
    "_TensorPayloadCacheEntry",
]
