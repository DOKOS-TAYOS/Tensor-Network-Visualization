from __future__ import annotations

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
    _ensure_tensor_label_artists,
    _node_mode_from_show_nodes,
    _set_artist_visible,
    _set_scene_node_mode,
)
from .._logging import package_logger
from .._matplotlib_state import set_active_axes, set_interactive_controls
from .._registry import _get_plotters
from .._tensor_elements_data import (
    _extract_playback_step_records,
    _PlaybackStepRecord,
)
from ..config import EngineName, PlotConfig, ViewName
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
        self.tensor_inspector_available: bool = self._playback_step_records is not None
        self._desired_state = feature_state_from_config(
            config,
            tensor_inspector_available=self.tensor_inspector_available,
        )
        self._active_state = self._desired_state
        self._view_manager = _InteractiveViewManager(
            render_view=lambda view, ax: self._render_view(view, ax=ax),
            initial_ax=initial_ax,
            external_ax=self._external_ax,
        )
        self.figure: Figure | None = None
        self._controls_panel: _InteractiveControlsPanel | None = None
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
    def _view_caches(self) -> dict[ViewName, InteractiveViewCache]:
        return self._view_manager.view_caches

    @property
    def _radio_ax(self) -> Axes | None:
        return None if self._controls_panel is None else self._controls_panel.radio_ax

    @property
    def _radio(self) -> RadioButtons | None:
        return None if self._controls_panel is None else self._controls_panel.radio

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
    def current_scene(self) -> _InteractiveSceneState:
        scene = self._view_manager.current_scene(self.current_view)
        assert scene is not None
        return scene

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
        return figure, ax

    def _set_requested_state(self, **changes: bool) -> None:
        self._desired_state = replace(self._desired_state, **changes)
        self._active_state = replace(self._active_state, **changes)

    def _base_config(self) -> PlotConfig:
        return replace(
            self.config,
            show_nodes=self._desired_state.nodes,
            hover_labels=False,
            show_tensor_labels=False,
            show_index_labels=False,
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
            contraction_tensor_inspector=False,
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
        )

    def _sync_checkbuttons(self) -> None:
        if self._controls_panel is None:
            return
        self._controls_panel.sync(state=self._active_state, view=self.current_view)

    def _on_controls_state_changed(self, requested_state: InteractiveFeatureState) -> None:
        self._desired_state = requested_state
        self._active_state = requested_state
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
        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def _on_figure_closed(self, _event: Any) -> None:
        if self._figure_close_cid is not None and self.figure is not None:
            self.figure.canvas.mpl_disconnect(self._figure_close_cid)
            self._figure_close_cid = None
        if self._tensor_inspector is not None:
            self._tensor_inspector.close_from_owner()

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
        _apply_scene_hover_state(scene, hover_on=resolved.hover)
        self._active_state = resolved
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
