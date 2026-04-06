from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, RadioButtons
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._core.draw.scene_state import _InteractiveSceneState
from .._interactive_scene import (
    _apply_scene_hover_state,
    _ensure_edge_label_artists,
    _ensure_scene_label_descriptors,
    _ensure_tensor_label_artists,
    _node_mode_from_show_nodes,
    _scene_from_axes,
    _set_artist_visible,
    _set_scene_node_mode,
)
from .._logging import package_logger
from .._matplotlib_state import (
    get_contraction_controls,
    set_active_axes,
    set_interactive_controls,
)
from .._registry import _get_plotters
from .._tensor_elements_data import (
    _extract_playback_step_records,
    _PlaybackStepRecord,
)
from .._typing import root_figure
from .._ui_utils import _set_axes_visible, _set_figure_bottom_reserved, _style_control_tray_axes
from ..config import EngineName, PlotConfig, ViewName
from ..contraction_viewer import _MAIN_FIGURE_BOTTOM_RESERVED, _PLAYBACK_DETAILS_TOP
from .state import InteractiveViewCache
from .tensor_inspector import _LinkedTensorInspectorController

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
_BASE_TOGGLE_LABELS: tuple[str, str, str, str] = (
    "Hover",
    "Nodes",
    "Tensor labels",
    "Edge labels",
)
_SCHEME_TOGGLE_LABELS: tuple[str, str] = ("Scheme", "Costs")
_TENSOR_INSPECTOR_LABEL: str = "Tensor inspector"
_INTERACTIVE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}
_INTERACTIVE_RADIO_PROPS: dict[str, float] = {"s": 38.0, "linewidth": 0.9}
_TOGGLE_INDEX_HOVER: int = 0
_TOGGLE_INDEX_NODES: int = 1
_TOGGLE_INDEX_TENSOR_LABELS: int = 2
_TOGGLE_INDEX_EDGE_LABELS: int = 3


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
        package_logger.debug(
            "Initializing interactive tensor figure controller engine=%s initial_view=%s.",
            engine,
            initial_view,
        )
        self.current_view: ViewName = initial_view
        self.hover_on: bool = bool(config.hover_labels)
        self.nodes_on: bool = bool(config.show_nodes)
        self.tensor_labels_on: bool = bool(config.show_tensor_labels)
        self.edge_labels_on: bool = bool(config.show_index_labels)
        self.scheme_on: bool = bool(config.show_contraction_scheme)
        self.cost_hover_on: bool = bool(config.contraction_scheme_cost_hover)
        self._playback_step_records = _extract_playback_step_records(network)
        self.tensor_inspector_available: bool = self._playback_step_records is not None
        self.tensor_inspector_on: bool = bool(
            config.contraction_tensor_inspector and self.tensor_inspector_available
        )
        if self.cost_hover_on or self.tensor_inspector_on:
            self.scheme_on = True
        self._initial_ax = initial_ax
        self._external_ax = initial_ax is not None
        self._plot_2d, self._plot_3d = _get_plotters(engine)
        self._view_caches: dict[ViewName, InteractiveViewCache] = {
            "2d": InteractiveViewCache(view="2d"),
            "3d": InteractiveViewCache(view="3d"),
        }
        self._radio_ax: Axes | None = None
        self._radio: RadioButtons | None = None
        self._check_ax: Axes | None = None
        self._checkbuttons: CheckButtons | None = None
        self._callback_guard: bool = False
        self.figure: Figure | None = None
        self._tensor_inspector: _LinkedTensorInspectorController | None = None
        self._figure_close_cid: int | None = None
        self._initialized: bool = False
        if self.tensor_inspector_available:
            self._tensor_inspector = _LinkedTensorInspectorController(
                step_records=cast(tuple[_PlaybackStepRecord, ...], self._playback_step_records),
                placeholder_engine=engine,
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
        self._initialized = True
        set_interactive_controls(figure, self)
        set_active_axes(figure, ax)
        figure._tensor_network_viz_tensor_inspector = self._tensor_inspector  # type: ignore[attr-defined]
        self._figure_close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)
        return figure, ax

    def _base_config(self) -> PlotConfig:
        return replace(
            self.config,
            show_nodes=self.nodes_on,
            hover_labels=False,
            show_tensor_labels=False,
            show_index_labels=False,
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
            contraction_tensor_inspector=False,
        )

    def _scene_requires_node_mode_rerender(self, scene: _InteractiveSceneState) -> bool:
        desired_mode = _node_mode_from_show_nodes(self.nodes_on)
        return (
            scene.dimensions == 2
            and scene.active_node_mode != desired_mode
            and any(edge.kind == "dangling" for edge in scene.graph.edges)
        )

    def _rerender_cached_view(self, view: ViewName) -> _InteractiveSceneState:
        cache = self._view_caches[view]
        assert cache.ax is not None
        fig, rendered_ax = self._render_view(view, ax=cache.ax)
        scene = _scene_from_axes(rendered_ax)
        assert scene is not None
        cache.ax = rendered_ax
        cache.scene = scene
        scene.contraction_controls = get_contraction_controls(rendered_ax)
        _ensure_scene_label_descriptors(scene)
        set_active_axes(fig, rendered_ax)
        return scene

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
        _style_control_tray_axes(check_ax)
        self._check_ax = check_ax
        if not self._external_ax:
            radio_bounds: tuple[float, float, float, float] = (
                _VIEW_SELECTOR_LEFT,
                cb_bottom,
                _VIEW_SELECTOR_WIDTH,
                _VIEW_SELECTOR_HEIGHT,
            )
            radio_ax = self.figure.add_axes(radio_bounds)
            _style_control_tray_axes(radio_ax)
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
            self.nodes_on,
            self.tensor_labels_on,
            self.edge_labels_on,
        ]
        if has_scheme_toggles:
            statuses.extend([self.scheme_on, self.cost_hover_on])
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
        desired = [self.hover_on, self.nodes_on, self.tensor_labels_on, self.edge_labels_on]
        if len(self._checkbuttons.labels) > len(_BASE_TOGGLE_LABELS):
            desired.extend([self.scheme_on, self.cost_hover_on])
            if self.tensor_inspector_available and len(self._checkbuttons.labels) > 5:
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
        self.hover_on = status[_TOGGLE_INDEX_HOVER]
        self.nodes_on = status[_TOGGLE_INDEX_NODES]
        self.tensor_labels_on = status[_TOGGLE_INDEX_TENSOR_LABELS]
        self.edge_labels_on = status[_TOGGLE_INDEX_EDGE_LABELS]
        scheme_index = len(_BASE_TOGGLE_LABELS)
        if len(status) >= scheme_index + 2:
            self.scheme_on = status[scheme_index]
            self.cost_hover_on = status[scheme_index + 1]
        if len(status) >= scheme_index + 3 and self.tensor_inspector_available:
            self.tensor_inspector_on = status[scheme_index + 2]
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
        if self._scene_requires_node_mode_rerender(scene):
            scene = self._rerender_cached_view(self.current_view)
        _set_scene_node_mode(scene, mode=_node_mode_from_show_nodes(self.nodes_on))
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

        controls = scene.contraction_controls
        if controls is not None:
            controls.set_states(
                scheme_on=self.scheme_on,
                cost_hover_on=self.cost_hover_on,
            )
            self.scheme_on = bool(controls.scheme_on)
            self.cost_hover_on = bool(controls.cost_hover_on)
            if self._tensor_inspector is not None:
                self._tensor_inspector.bind_viewer(controls._viewer)
        if self._tensor_inspector is not None:
            reveal_inspector = bool(
                self._initialized
                and self.tensor_inspector_on
                and not self._tensor_inspector.is_enabled
            )
            self._tensor_inspector.set_enabled(
                self.tensor_inspector_on,
                reveal=reveal_inspector,
            )
        _apply_scene_hover_state(scene, hover_on=self.hover_on)
        self._sync_checkbuttons()
        if not self._external_ax:
            self._apply_interactive_figure_layout()
        scene.ax.figure.canvas.draw_idle()

    def set_view(self, view: ViewName) -> None:
        if view == self.current_view:
            return
        package_logger.debug(
            "Switching interactive tensor view from %s to %s.",
            self.current_view,
            view,
        )
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
        package_logger.debug("Interactive hover enabled=%s.", self.hover_on)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)

    def set_nodes_enabled(self, enabled: bool) -> None:
        self.nodes_on = bool(enabled)
        package_logger.debug("Interactive node visibility enabled=%s.", self.nodes_on)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)

    def set_tensor_labels_enabled(self, enabled: bool) -> None:
        self.tensor_labels_on = bool(enabled)
        package_logger.debug("Interactive tensor labels enabled=%s.", self.tensor_labels_on)
        self._sync_checkbuttons()
        self._apply_scene_state(self.current_scene)

    def set_edge_labels_enabled(self, enabled: bool) -> None:
        self.edge_labels_on = bool(enabled)
        package_logger.debug("Interactive edge labels enabled=%s.", self.edge_labels_on)
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
