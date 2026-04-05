from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, RadioButtons
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core.draw.scene_state import _InteractiveSceneState
from ._interactive_scene import (
    _apply_scene_hover_state,
    _ensure_edge_label_artists,
    _ensure_scene_label_descriptors,
    _ensure_tensor_label_artists,
    _scene_from_axes,
    _set_artist_visible,
)
from ._matplotlib_state import (
    get_contraction_controls,
    set_active_axes,
    set_interactive_controls,
)
from ._registry import _get_plotters
from ._tensor_elements_data import (
    _EinsumPlaybackStepRecord,
    _extract_einsum_playback_step_records,
)
from ._tensor_elements_support import _TensorRecord
from ._typing import root_figure
from ._ui_utils import _set_axes_visible, _set_figure_bottom_reserved
from .config import EngineName, PlotConfig, ViewName
from .contraction_viewer import _MAIN_FIGURE_BOTTOM_RESERVED, _PLAYBACK_DETAILS_TOP
from .einsum_module.trace import EinsumTrace
from .tensor_elements import _show_tensor_records
from .tensor_elements_config import TensorElementsConfig

RenderedAxes = Axes | Axes3D

# Menu column: fixed bottom = tallest stack (inspector + scheme). Without playback, checkboxes/radio
# stay as low as when the bottom row exists. Top aligned with cost-details top.
_VIEW_SELECTOR_LEFT: float = 0.213
_VIEW_SELECTOR_WIDTH: float = 0.053
_VIEW_SELECTOR_HEIGHT: float = 0.063
# Manual axes positions: 2D extends slightly below *base*, 3D starts higher (base + lift).
_INTERACTIVE_2D_BOTTOM_EXTRA: float = 0.022
_INTERACTIVE_3D_BOTTOM_LIFT: float = 0.084
_BASE_INTERACTIVE_HEIGHT: float = 0.09
_SCHEME_INSPECTOR_INTERACTIVE_HEIGHT: float = 0.172
_INTERACTIVE_MENU_COLUMN_HEIGHT: float = _SCHEME_INSPECTOR_INTERACTIVE_HEIGHT
_INTERACTIVE_MENU_COLUMN_BOTTOM: float = _PLAYBACK_DETAILS_TOP - _INTERACTIVE_MENU_COLUMN_HEIGHT
_INTERACTIVE_CHECKBOX_AXES_BOUNDS: tuple[float, float, float, float] = (
    0.02,
    _INTERACTIVE_MENU_COLUMN_BOTTOM,
    0.19,
    _INTERACTIVE_MENU_COLUMN_HEIGHT,
)
# When Scheme is off, main axes bottom (not tied to menu column bottom after unifying menus).
_SCHEME_OFF_FIGURE_BOTTOM_PAD: float = 0.02
_MAIN_FIGURE_BOTTOM_SCHEME_OFF: float = (
    _PLAYBACK_DETAILS_TOP - _BASE_INTERACTIVE_HEIGHT + _SCHEME_OFF_FIGURE_BOTTOM_PAD
)
_BASE_TOGGLE_LABELS: tuple[str, str, str] = ("Hover", "Tensor labels", "Edge labels")
_SCHEME_TOGGLE_LABELS: tuple[str, str, str] = ("Scheme", "Playback", "Costs")
_TENSOR_INSPECTOR_LABEL: str = "Tensor inspector"
_INTERACTIVE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}
_INTERACTIVE_RADIO_PROPS: dict[str, float] = {"s": 38.0, "linewidth": 0.9}
_CONTROL_TRAY_FACE: tuple[float, float, float] = (0.97, 0.97, 0.99)
_CONTROL_TRAY_FRAME: tuple[float, float, float] = (0.78, 0.78, 0.82)


def _style_interactive_control_axes(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_navigate(False)
    ax.patch.set_facecolor(_CONTROL_TRAY_FACE)
    ax.patch.set_alpha(0.88)
    ax.patch.set_edgecolor(_CONTROL_TRAY_FRAME)
    ax.patch.set_linewidth(0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(_CONTROL_TRAY_FRAME)


@dataclass
class _ViewCache:
    ax: RenderedAxes | None = None
    scene: _InteractiveSceneState | None = None


class _LinkedTensorInspectorController:
    def __init__(
        self,
        *,
        trace: EinsumTrace,
        on_closed: Callable[[], None],
    ) -> None:
        self._step_records: tuple[_EinsumPlaybackStepRecord, ...] = (
            _extract_einsum_playback_step_records(trace)
        )
        self._on_closed = on_closed
        self._config = TensorElementsConfig()
        self._enabled: bool = False
        self._viewer: Any = None
        self._figure: Figure | None = None
        self._elements_controller: Any = None
        self._saved_mode: str | None = None
        self._closing_programmatically: bool = False
        self._close_cid: int | None = None

    def bind_viewer(self, viewer: Any) -> None:
        if self._viewer is viewer:
            if self._enabled and self._viewer is not None:
                self._viewer.add_step_changed_callback(
                    self._sync_to_step,
                    call_immediately=True,
                )
            return
        if self._viewer is not None:
            self._viewer.remove_step_changed_callback(self._sync_to_step)
        self._viewer = viewer
        if self._viewer is not None:
            self._viewer.add_step_changed_callback(
                self._sync_to_step,
                call_immediately=self._enabled,
            )

    def set_enabled(self, enabled: bool) -> None:
        target = bool(enabled)
        if target == self._enabled:
            if target and self._viewer is not None:
                self._sync_to_step(int(self._viewer.current_step))
            return
        self._enabled = target
        if not target:
            self._close_figure()
            return
        self._ensure_figure()
        if self._viewer is not None:
            self._sync_to_step(int(self._viewer.current_step))
        else:
            self._render_placeholder("No contraction selected yet.")

    def close_from_owner(self) -> None:
        self._enabled = False
        if self._viewer is not None:
            self._viewer.remove_step_changed_callback(self._sync_to_step)
            self._viewer = None
        self._close_figure()

    def _placeholder_record(self) -> _TensorRecord:
        return _TensorRecord(
            array=np.zeros((1, 1), dtype=float),
            axis_names=(),
            engine="einsum",
            name="Tensor inspector",
        )

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        initial_step = int(self._viewer.current_step) if self._viewer is not None else 0
        record = self._record_for_step(initial_step)
        if record is None:
            record = self._placeholder_record()
        figure, _ax, controller = _show_tensor_records(
            [record],
            config=self._config,
            ax=None,
            show_controls=True,
            show=False,
        )
        self._figure = figure
        self._elements_controller = controller
        if self._saved_mode is not None:
            with suppress(ValueError):
                controller.set_mode(self._saved_mode, redraw=False)
        self._close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)

    def _record_for_step(self, step: int) -> _TensorRecord | None:
        if step <= 0:
            return None
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            return None
        return self._step_records[index].record

    def _render_placeholder(self, text: str) -> None:
        if self._elements_controller is None:
            return
        self._elements_controller.render_placeholder(text)

    def _sync_to_step(self, step: int) -> None:
        if not self._enabled:
            return
        self._ensure_figure()
        if self._elements_controller is None:
            return
        if step <= 0:
            self._render_placeholder("No contraction selected yet.")
            return
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        step_record = self._step_records[index]
        if step_record.record is None:
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        self._elements_controller.set_single_record(step_record.record)
        self._saved_mode = str(self._elements_controller.selected_mode)

    def _close_figure(self) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        figure = self._figure
        if figure is None:
            self._elements_controller = None
            self._close_cid = None
            return
        if self._close_cid is not None:
            figure.canvas.mpl_disconnect(self._close_cid)
        self._closing_programmatically = True
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        plt.close(figure)
        self._closing_programmatically = False

    def _on_figure_closed(self, _event: Any) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        if self._closing_programmatically:
            return
        self._enabled = False
        self._on_closed()


def _interactive_checkbox_bounds(
    *,
    include_scheme_toggles: bool,
    include_tensor_inspector: bool,
) -> tuple[float, float, float, float]:
    _ = include_scheme_toggles, include_tensor_inspector
    return _INTERACTIVE_CHECKBOX_AXES_BOUNDS


class _InteractiveTensorFigureController:
    def __init__(
        self,
        *,
        network: Any,
        engine: EngineName,
        config: PlotConfig,
        initial_view: ViewName,
        initial_ax: RenderedAxes | None,
    ) -> None:
        self.network = network
        self.engine = engine
        self.config = config
        self.current_view: ViewName = initial_view
        self.hover_on: bool = bool(config.hover_labels)
        self.tensor_labels_on: bool = bool(config.show_tensor_labels)
        self.edge_labels_on: bool = bool(config.show_index_labels)
        self.scheme_on: bool = bool(config.show_contraction_scheme)
        self.playback_on: bool = bool(config.contraction_playback)
        self.cost_hover_on: bool = bool(config.contraction_scheme_cost_hover)
        self.tensor_inspector_available: bool = isinstance(network, EinsumTrace)
        self.tensor_inspector_on: bool = bool(
            config.contraction_tensor_inspector and self.tensor_inspector_available
        )
        if self.cost_hover_on or self.tensor_inspector_on:
            self.scheme_on = True
            self.playback_on = True
        elif self.playback_on:
            self.scheme_on = True
        self._initial_ax = initial_ax
        self._external_ax = initial_ax is not None
        self._plot_2d, self._plot_3d = _get_plotters(engine)
        self._view_caches: dict[ViewName, _ViewCache] = {
            "2d": _ViewCache(),
            "3d": _ViewCache(),
        }
        self._radio_ax: Axes | None = None
        self._radio: RadioButtons | None = None
        self._check_ax: Axes | None = None
        self._checkbuttons: CheckButtons | None = None
        self._callback_guard: bool = False
        self.figure: Figure | None = None
        self._tensor_inspector: _LinkedTensorInspectorController | None = None
        self._figure_close_cid: int | None = None
        if self.tensor_inspector_available:
            self._tensor_inspector = _LinkedTensorInspectorController(
                trace=cast(EinsumTrace, network),
                on_closed=self._on_tensor_inspector_closed,
            )

    @property
    def current_scene(self) -> _InteractiveSceneState:
        scene = self._view_caches[self.current_view].scene
        assert scene is not None
        return scene

    def initialize(self) -> tuple[Figure, RenderedAxes]:
        figure, ax = self._build_view(self.current_view, ax=self._initial_ax)
        self.figure = figure
        if self._view_caches[self.current_view].scene is None:
            return figure, ax
        self._build_controls()
        self._apply_scene_state(self.current_scene)
        set_interactive_controls(figure, self)
        set_active_axes(figure, ax)
        figure._tensor_network_viz_tensor_inspector = self._tensor_inspector  # type: ignore[attr-defined]
        self._figure_close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)
        return figure, ax

    def _base_config(self) -> PlotConfig:
        return replace(
            self.config,
            hover_labels=False,
            show_tensor_labels=False,
            show_index_labels=False,
            show_contraction_scheme=False,
            contraction_playback=False,
            contraction_scheme_cost_hover=False,
            contraction_tensor_inspector=False,
        )

    def _render_view(
        self,
        view: ViewName,
        *,
        ax: RenderedAxes | None,
    ) -> tuple[Figure, RenderedAxes]:
        plotter = self._plot_2d if view == "2d" else self._plot_3d
        fig, rendered_ax = plotter(
            self.network,
            ax=ax,
            config=self._base_config(),
            _build_contraction_controls=True,
            _contraction_controls_build_ui=False,
            _register_contraction_controls_on_figure=False,
        )
        return fig, rendered_ax

    def _build_view(
        self,
        view: ViewName,
        *,
        ax: RenderedAxes | None,
    ) -> tuple[Figure, RenderedAxes]:
        cache = self._view_caches[view]
        if cache.ax is not None and cache.scene is not None:
            return root_figure(cache.ax.figure), cache.ax
        fig, rendered_ax = self._render_view(view, ax=ax)
        scene = _scene_from_axes(rendered_ax)
        cache.ax = rendered_ax
        cache.scene = scene
        if scene is not None:
            scene.contraction_controls = get_contraction_controls(rendered_ax)
            _ensure_scene_label_descriptors(scene)
        return fig, rendered_ax

    def _shared_data_axes_top(self) -> float:
        ax3 = self._view_caches["3d"].ax
        if ax3 is not None:
            p = ax3.get_position()
            return float(p.y0 + p.height)
        ax2 = self._view_caches["2d"].ax
        if ax2 is not None:
            p = ax2.get_position()
            return float(p.y0 + p.height)
        return 0.9

    def _interactive_scheme_chrome_on(self) -> bool:
        return self.current_scene.contraction_controls is not None and self.scheme_on

    def _interactive_main_axes_bottom(self) -> float:
        return float(
            _MAIN_FIGURE_BOTTOM_RESERVED
            if self._interactive_scheme_chrome_on()
            else _MAIN_FIGURE_BOTTOM_SCHEME_OFF
        )

    def _figure_bottom_margin(self) -> float:
        base = self._interactive_main_axes_bottom()
        lows: list[float] = []
        if self._view_caches["2d"].ax is not None:
            lows.append(base - float(_INTERACTIVE_2D_BOTTOM_EXTRA))
        if self._view_caches["3d"].ax is not None:
            lows.append(base + float(_INTERACTIVE_3D_BOTTOM_LIFT))
        return min(lows) if lows else base

    def _apply_interactive_figure_layout(self) -> None:
        if self.figure is None or self._external_ax:
            return
        _set_figure_bottom_reserved(self.figure, self._figure_bottom_margin())
        self._sync_data_axes_vertical_layout()

    def _sync_data_axes_vertical_layout(self) -> None:
        if self.figure is None or self._external_ax:
            return
        base = self._interactive_main_axes_bottom()
        top = self._shared_data_axes_top()
        ax2 = self._view_caches["2d"].ax
        ax3 = self._view_caches["3d"].ax
        if ax2 is not None:
            bottom_2d = base - float(_INTERACTIVE_2D_BOTTOM_EXTRA)
            pos = ax2.get_position()
            height = max(top - bottom_2d, 0.08)
            ax2.set_position((pos.x0, bottom_2d, pos.width, height))
        if ax3 is not None:
            bottom_3d = base + float(_INTERACTIVE_3D_BOTTOM_LIFT)
            pos = ax3.get_position()
            height = max(top - bottom_3d, 0.08)
            ax3.set_position((pos.x0, bottom_3d, pos.width, height))

    def _build_controls(self) -> None:
        assert self.figure is not None
        labels = list(_BASE_TOGGLE_LABELS)
        has_scheme_toggles = self.current_scene.contraction_controls is not None
        has_tensor_inspector = bool(has_scheme_toggles and self.tensor_inspector_available)
        if has_scheme_toggles:
            labels.extend(_SCHEME_TOGGLE_LABELS)
        if has_tensor_inspector:
            labels.append(_TENSOR_INSPECTOR_LABEL)
        cb_bounds = _interactive_checkbox_bounds(
            include_scheme_toggles=has_scheme_toggles,
            include_tensor_inspector=has_tensor_inspector,
        )
        cb_bottom = float(cb_bounds[1])
        check_ax = self.figure.add_axes(cb_bounds)
        _style_interactive_control_axes(check_ax)
        self._check_ax = check_ax
        if not self._external_ax:
            radio_bounds: tuple[float, float, float, float] = (
                _VIEW_SELECTOR_LEFT,
                cb_bottom,
                _VIEW_SELECTOR_WIDTH,
                _VIEW_SELECTOR_HEIGHT,
            )
            radio_ax = self.figure.add_axes(radio_bounds)
            _style_interactive_control_axes(radio_ax)
            self._radio_ax = radio_ax
            active_index = 0 if self.current_view == "2d" else 1
            self._radio = RadioButtons(
                radio_ax,
                ("2d", "3d"),
                active=active_index,
                label_props=_INTERACTIVE_LABEL_PROPS,
                radio_props=_INTERACTIVE_RADIO_PROPS,
            )
            self._radio.on_clicked(self._on_view_clicked)
        statuses = [
            self.hover_on,
            self.tensor_labels_on,
            self.edge_labels_on,
        ]
        if has_scheme_toggles:
            statuses.extend([self.scheme_on, self.playback_on, self.cost_hover_on])
        if has_tensor_inspector:
            statuses.append(self.tensor_inspector_on)
        self._checkbuttons = CheckButtons(
            check_ax,
            labels,
            statuses,
            label_props=_INTERACTIVE_LABEL_PROPS,
            frame_props=_INTERACTIVE_CHECK_FRAME_PROPS,
            check_props=_INTERACTIVE_CHECK_MARK_PROPS,
        )
        self._checkbuttons.on_clicked(self._on_toggle_clicked)

    def _sync_checkbuttons(self) -> None:
        if self._checkbuttons is None:
            return
        desired = [self.hover_on, self.tensor_labels_on, self.edge_labels_on]
        if len(self._checkbuttons.labels) > len(_BASE_TOGGLE_LABELS):
            desired.extend([self.scheme_on, self.playback_on, self.cost_hover_on])
            if self.tensor_inspector_available and len(self._checkbuttons.labels) > 6:
                desired.append(self.tensor_inspector_on)
        current = [bool(value) for value in self._checkbuttons.get_status()]
        self._callback_guard = True
        try:
            for index, value in enumerate(desired):
                if index < len(current) and current[index] != value:
                    self._checkbuttons.set_active(index, state=value)
        finally:
            self._callback_guard = False

    def _on_view_clicked(self, label: str | None) -> None:
        if self._callback_guard or label is None:
            return
        self.set_view(cast(ViewName, label))

    def _on_toggle_clicked(self, _label: str | None) -> None:
        if self._callback_guard or self._checkbuttons is None:
            return
        status = [bool(value) for value in self._checkbuttons.get_status()]
        self.hover_on = status[0]
        self.tensor_labels_on = status[1]
        self.edge_labels_on = status[2]
        if len(status) >= 6:
            self.scheme_on = status[3]
            self.playback_on = status[4]
            self.cost_hover_on = status[5]
        if len(status) >= 7 and self.tensor_inspector_available:
            self.tensor_inspector_on = status[6]
        self._apply_scene_state(self.current_scene)

    def _deactivate_non_current_views(self) -> None:
        for view_name, cache in self._view_caches.items():
            if cache.ax is None:
                continue
            is_current = view_name == self.current_view
            _set_axes_visible(cache.ax, is_current)
            scene = cache.scene
            if scene is None or scene.contraction_controls is None:
                continue
            viewer = scene.contraction_controls._viewer
            if viewer is not None and not is_current:
                viewer.pause()
                viewer.set_playback_widgets_visible(False)

    def _on_tensor_inspector_closed(self) -> None:
        self.tensor_inspector_on = False
        self._sync_checkbuttons()
        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def _on_figure_closed(self, _event: Any) -> None:
        if self._figure_close_cid is not None and self.figure is not None:
            self.figure.canvas.mpl_disconnect(self._figure_close_cid)
            self._figure_close_cid = None
        if self._tensor_inspector is not None:
            self._tensor_inspector.close_from_owner()

    def _apply_scene_state(self, scene: _InteractiveSceneState) -> None:
        if self.tensor_labels_on:
            _ensure_tensor_label_artists(scene)
        for artist in scene.tensor_label_artists:
            _set_artist_visible(artist, self.tensor_labels_on)
        if self.edge_labels_on:
            _ensure_edge_label_artists(scene)
        for artist in scene.edge_label_artists:
            _set_artist_visible(artist, self.edge_labels_on)

        if self.cost_hover_on or self.tensor_inspector_on:
            self.scheme_on = True
            self.playback_on = True
        elif self.playback_on:
            self.scheme_on = True

        controls = scene.contraction_controls
        if controls is not None:
            controls.set_states(
                scheme_on=self.scheme_on,
                playback_on=self.playback_on,
                cost_hover_on=self.cost_hover_on,
            )
            self.scheme_on = bool(controls.scheme_on)
            self.playback_on = bool(controls.playback_on)
            self.cost_hover_on = bool(controls.cost_hover_on)
            if self._tensor_inspector is not None:
                self._tensor_inspector.bind_viewer(controls._viewer)
        if self._tensor_inspector is not None:
            self._tensor_inspector.set_enabled(self.tensor_inspector_on)
        _apply_scene_hover_state(scene, hover_on=self.hover_on)
        self._sync_checkbuttons()
        if not self._external_ax:
            self._apply_interactive_figure_layout()
        scene.ax.figure.canvas.draw_idle()

    def set_view(self, view: ViewName) -> None:
        if view == self.current_view:
            return
        if self._radio is not None and self._radio.value_selected != view:
            self._callback_guard = True
            try:
                self._radio.set_active(0 if view == "2d" else 1)
            finally:
                self._callback_guard = False
        target_ax: RenderedAxes | None = None
        cache = self._view_caches[view]
        needs_new_axes = cache.ax is None or cache.scene is None
        if self.figure is not None and not self._external_ax and needs_new_axes:
            if view == "3d":
                target_ax = cast(RenderedAxes, self.figure.add_subplot(111, projection="3d"))
            else:
                target_ax = cast(RenderedAxes, self.figure.add_subplot(111))
        fig, ax = self._build_view(view, ax=target_ax)
        self.current_view = view
        self._deactivate_non_current_views()
        self._apply_scene_state(self.current_scene)
        set_active_axes(fig, ax)

    def set_hover_enabled(self, enabled: bool) -> None:
        self.hover_on = bool(enabled)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)

    def set_tensor_labels_enabled(self, enabled: bool) -> None:
        self.tensor_labels_on = bool(enabled)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)

    def set_edge_labels_enabled(self, enabled: bool) -> None:
        self.edge_labels_on = bool(enabled)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)


def show_tensor_network_interactive(
    network: Any,
    *,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig,
    ax: RenderedAxes | None,
) -> tuple[Figure, RenderedAxes]:
    controller = _InteractiveTensorFigureController(
        network=network,
        engine=engine,
        config=config,
        initial_view=view,
        initial_ax=ax,
    )
    return controller.initialize()


__all__ = [
    "show_tensor_network_interactive",
]
