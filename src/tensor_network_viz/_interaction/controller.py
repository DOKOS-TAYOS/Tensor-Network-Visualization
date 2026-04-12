from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._core.draw.scene_state import _InteractiveSceneState
from .._core.draw.tensors import _tensor_disk_radius_px
from .._input_inspection import _merge_grid_positions_into_config
from .._interactive_scene import (
    _apply_scene_hover_state,
    _bring_scene_label_artists_to_front,
    _ensure_edge_label_artists,
    _ensure_tensor_label_artists,
    _node_mode_from_show_nodes,
    _set_artist_visible,
    _set_scene_node_mode,
)
from .._logging import package_logger
from .._matplotlib_state import (
    request_canvas_redraw,
    set_active_axes,
    set_interactive_controls,
)
from .._registry import _get_plotters
from .._tensor_elements_data import (
    _extract_playback_step_records,
    _PlaybackStepRecord,
)
from .._tensor_elements_support import _extract_tensor_records, _TensorRecord
from ..config import (
    EngineName,
    FocusRadius,
    PlotConfig,
    TensorNetworkDiagnosticsConfig,
    TensorNetworkFocus,
    ViewName,
)
from ..exceptions import TensorDataError, TensorDataTypeError
from .controls import _InteractiveControlsLayout, _InteractiveControlsPanel
from .state import (
    InteractiveFeatureState,
    InteractiveViewCache,
    feature_availability_from_scene,
    feature_state_from_config,
    normalize_feature_state,
)
from .tensor_inspector import _LinkedTensorInspectorController
from .views import _InteractiveViewManager

RenderedAxes = Axes | Axes3D
_FocusMode = Literal["off", "neighborhood", "path"]


def _coerce_focus_radius(radius: int) -> FocusRadius:
    resolved_radius = int(radius)
    if resolved_radius == 1:
        return 1
    if resolved_radius == 2:
        return 2
    raise ValueError("Focus radius must be 1 or 2.")


def _release_canvas_mouse_grabber(
    figure: Figure | None,
    *,
    target_axes: object | None = None,
) -> None:
    if figure is None:
        return
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return
    mouse_grabber = getattr(canvas, "mouse_grabber", None)
    if mouse_grabber is None:
        return
    if target_axes is not None and mouse_grabber is not target_axes:
        return
    release_mouse = getattr(canvas, "release_mouse", None)
    if callable(release_mouse):
        with suppress(AttributeError, RuntimeError, TypeError, ValueError):
            release_mouse(mouse_grabber)


def _node_records_by_name(
    network: Any,
    *,
    engine: EngineName,
) -> dict[str, _TensorRecord]:
    try:
        _, records = _extract_tensor_records(network, engine=engine)
    except (TensorDataError, TensorDataTypeError) as exc:
        package_logger.debug("Tensor inspector node records unavailable: %s", exc)
        return {}
    mapping: dict[str, _TensorRecord] = {}
    duplicate_names: set[str] = set()
    for record in records:
        if record.name in mapping:
            duplicate_names.add(record.name)
            continue
        mapping[record.name] = record
    for name in duplicate_names:
        mapping.pop(name, None)
    return mapping


def _hit_visible_node_id(
    scene: _InteractiveSceneState,
    event: Any,
) -> int | None:
    if event.x is None or event.y is None:
        return None
    visible_node_ids = tuple(int(node_id) for node_id in scene.visible_node_ids)
    if not visible_node_ids:
        return None
    if scene.dimensions == 2 and scene.node_patch_coll is not None:
        node_patch_coll = scene.node_patch_coll
        if isinstance(node_patch_coll, (PatchCollection, PathCollection)):
            hit, props = node_patch_coll.contains(event)
            if not hit:
                return None
            indices = props.get("ind")
            if indices is None or len(indices) == 0:
                return None
            index = int(indices[0])
            if 0 <= index < len(visible_node_ids):
                return visible_node_ids[index]
            return None
        for index, collection in enumerate(node_patch_coll):
            hit, _props = collection.contains(event)
            if hit and 0 <= index < len(visible_node_ids):
                return visible_node_ids[index]
        return None
    if scene.dimensions != 3:
        return None

    best_node_id: int | None = None
    best_distance_sq = float("inf")
    projection = scene.ax.get_proj()
    for node_id in visible_node_ids:
        position = np.asarray(scene.positions[node_id], dtype=float).reshape(-1)
        if position.size < 3:
            padded = np.zeros(3, dtype=float)
            padded[: position.size] = position
            position = padded
        radius_px = (
            float(scene.tensor_disk_radius_px_3d)
            if scene.tensor_disk_radius_px_3d is not None
            else float(_tensor_disk_radius_px(scene.ax, position, scene.params, 3))
        )
        x_proj, y_proj, _z_proj = proj3d.proj_transform(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            projection,
        )
        point = np.asarray(scene.ax.transData.transform((x_proj, y_proj)), dtype=float).ravel()
        dx = float(point[0]) - float(event.x)
        dy = float(point[1]) - float(event.y)
        distance_sq = dx * dx + dy * dy
        if distance_sq <= radius_px * radius_px and distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_node_id = node_id
    return best_node_id


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
        self._initial_ax = initial_ax
        self._external_ax = initial_ax is not None
        self._plot_2d, self._plot_3d = _get_plotters(engine)
        self._playback_step_records = _extract_playback_step_records(network)
        self._node_records_by_name = _node_records_by_name(network, engine=engine)
        self.tensor_inspector_available: bool = bool(
            self._playback_step_records is not None or self._node_records_by_name
        )
        self._desired_state = feature_state_from_config(
            config,
            tensor_inspector_available=self.tensor_inspector_available,
        )
        self._active_state = self._desired_state
        self._focus: TensorNetworkFocus | None = config.focus
        self._focus_mode: _FocusMode = "off"
        self._focus_radius: FocusRadius = 1
        self._focus_pending_start: str | None = None
        self._focus_interaction_enabled: bool = False
        if config.focus is not None:
            self._focus_mode = cast(_FocusMode, config.focus.kind)
            self._focus_radius = _coerce_focus_radius(config.focus.radius)
            self._focus_interaction_enabled = True
        self._view_manager = _InteractiveViewManager(
            render_view=lambda view, ax: self._render_view(view, ax=ax),
            initial_ax=initial_ax,
            external_ax=self._external_ax,
        )
        self.figure: Figure | None = None
        self._controls_panel: _InteractiveControlsPanel | None = None
        self._tensor_inspector: _LinkedTensorInspectorController | None = None
        self._focus_status_text: Text | None = None
        self._figure_close_cid: int | None = None
        self._button_press_cid: int | None = None
        self._initialized: bool = False
        if self.tensor_inspector_available:
            self._tensor_inspector = _LinkedTensorInspectorController(
                step_records=cast(
                    tuple[_PlaybackStepRecord, ...] | None,
                    self._playback_step_records,
                ),
                node_records_by_name=self._node_records_by_name,
                placeholder_engine=engine,
                on_closed=self._on_tensor_inspector_closed,
            )

    @property
    def _view_caches(self) -> dict[ViewName, InteractiveViewCache]:
        return self._view_manager.view_caches

    @property
    def _view_toggle_ax(self) -> Axes | None:
        return None if self._controls_panel is None else self._controls_panel.view_toggle_ax

    @property
    def _view_toggle_button(self) -> Button | None:
        return None if self._controls_panel is None else self._controls_panel.view_toggle_button

    @property
    def _check_ax(self) -> Axes | None:
        return None if self._controls_panel is None else self._controls_panel.check_ax

    @property
    def _checkbuttons(self) -> CheckButtons | None:
        return None if self._controls_panel is None else self._controls_panel.checkbuttons

    @property
    def hover_on(self) -> bool:
        return bool(self._active_state.hover)

    @hover_on.setter
    def hover_on(self, enabled: bool) -> None:
        self._set_requested_state(hover=bool(enabled))

    @property
    def nodes_on(self) -> bool:
        return bool(self._active_state.nodes)

    @nodes_on.setter
    def nodes_on(self, enabled: bool) -> None:
        self._set_requested_state(nodes=bool(enabled))

    @property
    def tensor_labels_on(self) -> bool:
        return bool(self._active_state.tensor_labels)

    @tensor_labels_on.setter
    def tensor_labels_on(self, enabled: bool) -> None:
        self._set_requested_state(tensor_labels=bool(enabled))

    @property
    def edge_labels_on(self) -> bool:
        return bool(self._active_state.edge_labels)

    @edge_labels_on.setter
    def edge_labels_on(self, enabled: bool) -> None:
        self._set_requested_state(edge_labels=bool(enabled))

    @property
    def scheme_on(self) -> bool:
        return bool(self._active_state.scheme)

    @scheme_on.setter
    def scheme_on(self, enabled: bool) -> None:
        self._set_requested_state(scheme=bool(enabled))

    @property
    def cost_hover_on(self) -> bool:
        return bool(self._active_state.cost_hover)

    @cost_hover_on.setter
    def cost_hover_on(self, enabled: bool) -> None:
        self._set_requested_state(cost_hover=bool(enabled))

    @property
    def tensor_inspector_on(self) -> bool:
        return bool(self._active_state.tensor_inspector)

    @tensor_inspector_on.setter
    def tensor_inspector_on(self, enabled: bool) -> None:
        self._set_requested_state(tensor_inspector=bool(enabled))

    @property
    def diagnostics_on(self) -> bool:
        return bool(self._active_state.diagnostics)

    @diagnostics_on.setter
    def diagnostics_on(self, enabled: bool) -> None:
        self._set_requested_state(diagnostics=bool(enabled))

    @property
    def current_scene(self) -> _InteractiveSceneState:
        scene = self._view_manager.current_scene(self.current_view)
        assert scene is not None
        return scene

    @property
    def focus(self) -> TensorNetworkFocus | None:
        return self._focus

    @property
    def focus_mode(self) -> _FocusMode:
        return self._focus_mode

    @property
    def focus_radius(self) -> int:
        return int(self._focus_radius)

    def initialize(self) -> tuple[Figure, RenderedAxes]:
        figure, ax, scene = self._view_manager.build_initial_view(
            self.current_view,
            show_nodes=self._desired_state.nodes,
        )
        self.figure = figure
        if scene is None:
            return figure, ax
        self._build_controls(scene)
        self._apply_scene_state(scene)
        self._initialized = True
        set_interactive_controls(figure, self)
        set_active_axes(figure, ax)
        figure._tensor_network_viz_tensor_inspector = self._tensor_inspector  # type: ignore[attr-defined]
        self._figure_close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)
        self._button_press_cid = figure.canvas.mpl_connect(
            "button_press_event",
            self._on_button_press,
        )
        return figure, ax

    def _set_requested_state(self, **changes: bool) -> None:
        self._desired_state = replace(self._desired_state, **changes)
        self._active_state = replace(self._active_state, **changes)

    def _base_config(self) -> PlotConfig:
        diagnostics = self.config.diagnostics or TensorNetworkDiagnosticsConfig()
        return replace(
            self.config,
            show_nodes=self._desired_state.nodes,
            hover_labels=False,
            show_tensor_labels=False,
            show_index_labels=False,
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
            contraction_tensor_inspector=False,
            diagnostics=replace(diagnostics, show_overlay=self._desired_state.diagnostics),
            focus=self._focus,
        )

    def _scene_availability(self, scene: _InteractiveSceneState) -> Any:
        return feature_availability_from_scene(
            scene,
            tensor_inspector_available=self.tensor_inspector_available,
        )

    def _controls_layout(self, scene: _InteractiveSceneState) -> _InteractiveControlsLayout:
        availability = self._scene_availability(scene)
        return _InteractiveControlsLayout(
            include_view_selector=not self._external_ax,
            include_scheme_toggles=availability.scheme,
            include_tensor_inspector=availability.tensor_inspector,
            include_diagnostics=True,
            include_focus_controls=True,
        )

    def _render_view(
        self,
        view: ViewName,
        *,
        ax: RenderedAxes | None,
    ) -> tuple[Figure, RenderedAxes]:
        plotter = self._plot_2d if view == "2d" else self._plot_3d
        dimensions = 2 if view == "2d" else 3
        config = _merge_grid_positions_into_config(
            self._base_config(),
            self.network,
            dimensions=dimensions,
        )
        fig, rendered_ax = plotter(
            self.network,
            ax=ax,
            config=config,
            _build_contraction_controls=True,
            _contraction_controls_build_ui=False,
            _register_contraction_controls_on_figure=False,
        )
        return fig, rendered_ax

    def _scene_requires_node_mode_rerender(self, scene: _InteractiveSceneState) -> bool:
        return self._view_manager.scene_requires_node_mode_rerender(
            scene,
            show_nodes=self._desired_state.nodes,
        )

    def _rerender_cached_view(self, view: ViewName) -> _InteractiveSceneState:
        return self._view_manager.rerender_cached_view(view)

    def _build_view(
        self,
        view: ViewName,
        *,
        ax: RenderedAxes | None,
    ) -> tuple[Figure, RenderedAxes]:
        return self._view_manager.build_view(view, ax=ax)

    def _interactive_scheme_chrome_on(self) -> bool:
        controls = self.current_scene.contraction_controls
        return bool(controls is not None and self._active_state.scheme)

    def _apply_interactive_figure_layout(self) -> None:
        self._view_manager.sync_layout(
            figure=self.figure,
            scheme_chrome_on=self._interactive_scheme_chrome_on(),
        )

    def _sync_data_axes_vertical_layout(self) -> None:
        self._view_manager.sync_layout(
            figure=self.figure,
            scheme_chrome_on=self._interactive_scheme_chrome_on(),
        )

    def _build_controls(self, scene: _InteractiveSceneState) -> None:
        assert self.figure is not None
        self._controls_panel = _InteractiveControlsPanel(
            fig=self.figure,
            layout=self._controls_layout(scene),
            initial_view=self.current_view,
            initial_state=self._active_state,
            on_view_selected=self.set_view,
            on_state_changed=self._on_controls_state_changed,
            initial_focus_mode=self._focus_mode,
            initial_focus_radius=self._focus_radius,
            on_focus_mode_selected=self.set_focus_mode,
            on_focus_radius_selected=self.set_focus_radius,
            on_focus_cleared=self.clear_focus,
        )

    def _ensure_focus_status_text(self) -> Text:
        assert self.figure is not None
        if self._focus_status_text is None:
            self._focus_status_text = self.figure.text(
                0.5,
                0.985,
                "",
                ha="center",
                va="top",
                fontsize=9.0,
                color="#991B1B",
                visible=False,
                bbox={
                    "facecolor": "#FEF2F2",
                    "edgecolor": "#FCA5A5",
                    "boxstyle": "round,pad=0.25",
                    "alpha": 0.96,
                },
            )
        return self._focus_status_text

    def _focus_status_message(self, scene: _InteractiveSceneState) -> str | None:
        feedback = scene.focus_feedback
        if feedback is None or feedback.disconnected_endpoints is None:
            return None
        start_name, end_name = feedback.disconnected_endpoints
        return f"No path exists between {start_name} and {end_name}; showing both endpoints."

    def _sync_focus_status(self, scene: _InteractiveSceneState) -> None:
        if self.figure is None:
            return
        status_text = self._ensure_focus_status_text()
        message = self._focus_status_message(scene)
        if message is None:
            status_text.set_text("")
            status_text.set_visible(False)
            return
        status_text.set_text(message)
        status_text.set_visible(True)

    def _sync_checkbuttons(self) -> None:
        if self._controls_panel is None:
            return
        self._controls_panel.sync(
            state=self._active_state,
            view=self.current_view,
            focus_mode=self._focus_mode,
            focus_radius=self._focus_radius,
        )

    def _rerender_cached_views(self) -> _InteractiveSceneState:
        rerendered_current: _InteractiveSceneState | None = None
        for view_name, cache in self._view_caches.items():
            if cache.ax is None or cache.scene is None:
                continue
            scene = self._rerender_cached_view(view_name)
            if view_name == self.current_view:
                rerendered_current = scene
        if rerendered_current is None:
            rerendered_current = self.current_scene
        self._deactivate_non_current_views()
        return rerendered_current

    def _on_controls_state_changed(self, requested_state: InteractiveFeatureState) -> None:
        diagnostics_changed = requested_state.diagnostics != self._active_state.diagnostics
        self._desired_state = requested_state
        self._active_state = requested_state
        if diagnostics_changed:
            _release_canvas_mouse_grabber(self.figure, target_axes=self.current_scene.ax)
            scene = self._rerender_cached_views()
            self._apply_scene_state(scene)
            set_active_axes(scene.ax.figure, scene.ax)
            return
        self._apply_scene_state(self.current_scene)

    def _on_view_clicked(self, label: str | None) -> None:
        if label is None:
            return
        self.set_view(cast(ViewName, label))

    def _on_toggle_clicked(self, _label: str | None) -> None:
        if self._controls_panel is None:
            return
        self._on_controls_state_changed(self._controls_panel._last_state)

    def _deactivate_non_current_views(self) -> None:
        self._view_manager.deactivate_non_current_views(self.current_view)

    def _on_tensor_inspector_closed(self) -> None:
        self._active_state = replace(self._active_state, tensor_inspector=False)
        self._desired_state = replace(self._desired_state, tensor_inspector=False)
        self._sync_checkbuttons()
        request_canvas_redraw(self.figure)

    def _on_figure_closed(self, _event: Any) -> None:
        if self._figure_close_cid is not None and self.figure is not None:
            self.figure.canvas.mpl_disconnect(self._figure_close_cid)
            self._figure_close_cid = None
        if self._button_press_cid is not None and self.figure is not None:
            self.figure.canvas.mpl_disconnect(self._button_press_cid)
            self._button_press_cid = None
        if self._tensor_inspector is not None:
            self._tensor_inspector.close_from_owner()

    def _on_button_press(self, event: Any) -> None:
        if event.button is None or event.x is None or event.y is None:
            return
        current_ax = self.current_scene.ax
        in_current_axes = bool(
            event.inaxes is current_ax or current_ax.bbox.contains(float(event.x), float(event.y))
        )
        if not in_current_axes:
            if (
                event.inaxes is None
                and self._tensor_inspector is not None
                and self._tensor_inspector.is_enabled
            ):
                self._tensor_inspector.clear_selected_node()
            return
        node_id = _hit_visible_node_id(self.current_scene, event)
        if node_id is None:
            if self._tensor_inspector is not None and self._tensor_inspector.is_enabled:
                self._tensor_inspector.clear_selected_node()
            return
        node = self.current_scene.graph.nodes.get(node_id)
        if node is None or node.is_virtual or not node.name:
            return
        if self._focus_interaction_enabled:
            self.select_focus_node(node.name)
        if self._tensor_inspector is None or node.name not in self._node_records_by_name:
            return
        if not self._desired_state.tensor_inspector:
            return
        if self._tensor_inspector.select_node(node.name, reveal=True):
            self._active_state = replace(self._active_state, tensor_inspector=True)
            self._sync_checkbuttons()

    def _apply_scene_state(self, scene: _InteractiveSceneState) -> None:
        availability = self._scene_availability(scene)
        resolved = normalize_feature_state(self._desired_state, availability)
        if self._scene_requires_node_mode_rerender(scene):
            scene = self._rerender_cached_view(self.current_view)
            availability = self._scene_availability(scene)
            resolved = normalize_feature_state(self._desired_state, availability)
        _set_scene_node_mode(scene, mode=_node_mode_from_show_nodes(resolved.nodes))
        if resolved.tensor_labels:
            _ensure_tensor_label_artists(scene)
        for artist in scene.tensor_label_artists:
            _set_artist_visible(artist, resolved.tensor_labels)
        if resolved.edge_labels:
            _ensure_edge_label_artists(scene)
        for artist in scene.edge_label_artists:
            _set_artist_visible(artist, resolved.edge_labels)

        controls = scene.contraction_controls
        if controls is not None:
            controls.set_states(
                scheme_on=resolved.scheme,
                cost_hover_on=resolved.cost_hover,
            )
            if resolved.tensor_inspector:
                controls.ensure_viewer()
            resolved = replace(
                resolved,
                scheme=bool(controls.scheme_on),
                playback=bool(controls.scheme_on),
                cost_hover=bool(controls.cost_hover_on),
            )
            if self._tensor_inspector is not None:
                self._tensor_inspector.bind_viewer(controls._viewer)
        if self._tensor_inspector is not None:
            reveal_inspector = bool(
                self._initialized
                and resolved.tensor_inspector
                and not self._tensor_inspector.is_enabled
            )
            self._tensor_inspector.set_enabled(
                resolved.tensor_inspector,
                reveal=reveal_inspector,
            )
        self._sync_focus_status(scene)
        _bring_scene_label_artists_to_front(scene)
        _apply_scene_hover_state(scene, hover_on=resolved.hover)
        self._active_state = resolved
        self._sync_checkbuttons()
        if not self._external_ax:
            self._apply_interactive_figure_layout()
        request_canvas_redraw(scene.ax.figure)

    def set_view(self, view: ViewName) -> None:
        if view == self.current_view:
            return
        package_logger.debug(
            "Switching interactive tensor view from %s to %s.",
            self.current_view,
            view,
        )
        fig, ax, scene = self._view_manager.ensure_view(
            view,
            figure=self.figure,
            show_nodes=self._desired_state.nodes,
        )
        self.current_view = view
        self._deactivate_non_current_views()
        if scene is not None:
            self._apply_scene_state(scene)
        set_active_axes(fig, ax)

    def set_hover_enabled(self, enabled: bool) -> None:
        self.hover_on = enabled
        package_logger.debug("Interactive hover enabled=%s.", self.hover_on)
        self._apply_scene_state(self.current_scene)

    def set_nodes_enabled(self, enabled: bool) -> None:
        self.nodes_on = enabled
        package_logger.debug("Interactive node visibility enabled=%s.", self.nodes_on)
        self._apply_scene_state(self.current_scene)

    def set_tensor_labels_enabled(self, enabled: bool) -> None:
        self.tensor_labels_on = enabled
        package_logger.debug("Interactive tensor labels enabled=%s.", self.tensor_labels_on)
        self._apply_scene_state(self.current_scene)

    def set_edge_labels_enabled(self, enabled: bool) -> None:
        self.edge_labels_on = enabled
        package_logger.debug("Interactive edge labels enabled=%s.", self.edge_labels_on)
        self._apply_scene_state(self.current_scene)

    def set_focus_mode(self, mode: str) -> None:
        resolved_mode = cast(_FocusMode, str(mode))
        if resolved_mode not in {"off", "neighborhood", "path"}:
            raise ValueError(f"Unsupported focus mode {mode!r}.")
        rerender_required = False
        if self._focus_mode != resolved_mode:
            self._focus_mode = resolved_mode
            self._focus_pending_start = None
            if resolved_mode == "off":
                if self._focus is not None:
                    self._focus = None
                    rerender_required = True
            elif self._focus is not None and self._focus.kind != resolved_mode:
                self._focus = None
                rerender_required = True
        self._focus_interaction_enabled = resolved_mode != "off"
        self._sync_checkbuttons()
        if rerender_required:
            scene = self._rerender_cached_views()
            self._apply_scene_state(scene)
            set_active_axes(scene.ax.figure, scene.ax)

    def set_focus_radius(self, radius: int) -> None:
        resolved_radius = _coerce_focus_radius(radius)
        self._focus_interaction_enabled = self._focus_mode != "off"
        if self._focus_radius == resolved_radius:
            self._sync_checkbuttons()
            return
        self._focus_radius = resolved_radius
        if (
            self._focus is not None
            and self._focus.kind == "neighborhood"
            and self._focus.center is not None
        ):
            self._focus = TensorNetworkFocus(
                kind="neighborhood",
                center=self._focus.center,
                radius=resolved_radius,
            )
            scene = self._rerender_cached_views()
            self._apply_scene_state(scene)
            set_active_axes(scene.ax.figure, scene.ax)
            return
        self._sync_checkbuttons()

    def clear_focus(self) -> None:
        had_focus = self._focus is not None or self._focus_pending_start is not None
        self._focus = None
        self._focus_pending_start = None
        self._focus_interaction_enabled = self._focus_mode != "off"
        self._sync_checkbuttons()
        if not had_focus:
            return
        scene = self._rerender_cached_views()
        self._apply_scene_state(scene)
        set_active_axes(scene.ax.figure, scene.ax)

    def select_focus_node(self, node_name: str) -> bool:
        if not node_name or not self._focus_interaction_enabled or self._focus_mode == "off":
            return False
        if self._focus_mode == "neighborhood":
            self._focus_pending_start = None
            next_focus = TensorNetworkFocus(
                kind="neighborhood",
                center=node_name,
                radius=self._focus_radius,
            )
            if self._focus == next_focus:
                self._sync_checkbuttons()
                return True
            self._focus = next_focus
            scene = self._rerender_cached_views()
            self._apply_scene_state(scene)
            set_active_axes(scene.ax.figure, scene.ax)
            return True

        if self._focus_pending_start is None:
            rerender_required = self._focus is not None
            self._focus_pending_start = node_name
            self._focus = None
            self._sync_checkbuttons()
            if rerender_required:
                scene = self._rerender_cached_views()
                self._apply_scene_state(scene)
                set_active_axes(scene.ax.figure, scene.ax)
            return False

        next_focus = TensorNetworkFocus(
            kind="path",
            endpoints=(self._focus_pending_start, node_name),
            radius=self._focus_radius,
        )
        self._focus_pending_start = None
        if self._focus == next_focus:
            self._sync_checkbuttons()
            return True
        self._focus = next_focus
        scene = self._rerender_cached_views()
        self._apply_scene_state(scene)
        set_active_axes(scene.ax.figure, scene.ax)
        return True


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
