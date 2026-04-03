from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, RadioButtons
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core.draw.constants import (
    _EDGE_INDEX_LABEL_GID,
    _TENSOR_LABEL_GID,
    _ZORDER_LAYER_BASE,
    _ZORDER_LAYER_EDGE_INDEX,
    _ZORDER_LAYER_STRIDE,
    _ZORDER_LAYER_TENSOR_NAME,
)
from ._core.draw.contraction_edges import _draw_contraction_edge_labels
from ._core.draw.dangling_self_edges import (
    _draw_dangling_edge_labels,
    _draw_self_loop_edge_labels,
)
from ._core.draw.fonts_and_scale import _register_2d_zoom_font_scaling
from ._core.draw.hover import _RenderHoverState
from ._core.draw.labels_misc import (
    _contraction_hover_label_text,
    _dangling_hover_label_text,
    _self_loop_hover_label_text,
)
from ._core.draw.render_prep import (
    _apply_render_hover_state,
    _should_refine_tensor_labels,
)
from ._core.draw.scene_state import _InteractiveSceneState
from ._core.draw.tensors import _draw_labels, _refit_tensor_labels_to_disks
from ._registry import _get_plotters
from ._typing import root_figure
from ._ui_utils import _reserve_figure_bottom, _set_axes_visible
from .config import EngineName, PlotConfig, ViewName

RenderedAxes = Axes | Axes3D

_VIEW_SELECTOR_BOUNDS: tuple[float, float, float, float] = (0.02, 0.182, 0.09, 0.055)
_BASE_INTERACTIVE_CHECKBOX_BOUNDS: tuple[float, float, float, float] = (0.02, 0.028, 0.17, 0.09)
_SCHEME_INTERACTIVE_CHECKBOX_BOUNDS: tuple[float, float, float, float] = (0.02, 0.028, 0.17, 0.142)
_INTERACTIVE_CONTROLS_BOTTOM: float = 0.26
_BASE_TOGGLE_LABELS: tuple[str, str, str] = ("Hover", "Tensor labels", "Edge labels")
_SCHEME_TOGGLE_LABELS: tuple[str, str, str] = ("Scheme", "Playback", "Cost hover")
_INTERACTIVE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}
_INTERACTIVE_RADIO_PROPS: dict[str, float] = {"s": 38.0, "linewidth": 0.9}


@dataclass
class _ViewCache:
    ax: RenderedAxes | None = None
    scene: _InteractiveSceneState | None = None


def _set_artist_visible(artist: Artist, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        setter(bool(visible))


def _scene_from_axes(ax: RenderedAxes | None) -> _InteractiveSceneState | None:
    if ax is None:
        return None
    scene = getattr(ax, "_tensor_network_viz_scene", None)
    if isinstance(scene, _InteractiveSceneState):
        return scene
    return None


def _interactive_checkbox_bounds(
    *,
    include_scheme_toggles: bool,
) -> tuple[float, float, float, float]:
    if include_scheme_toggles:
        return _SCHEME_INTERACTIVE_CHECKBOX_BOUNDS
    return _BASE_INTERACTIVE_CHECKBOX_BOUNDS


def _last_edge_label_zorder(scene: _InteractiveSceneState) -> float | None:
    if scene.dimensions == 2 and scene.visible_node_ids:
        return float(
            _ZORDER_LAYER_BASE
            + (len(scene.visible_node_ids) - 1) * _ZORDER_LAYER_STRIDE
            + _ZORDER_LAYER_EDGE_INDEX
        )
    return None


def _tensor_label_zorders(scene: _InteractiveSceneState) -> dict[int, float] | None:
    if scene.dimensions != 2:
        return None
    return {
        node_id: float(
            _ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_TENSOR_NAME
        )
        for index, node_id in enumerate(scene.visible_node_ids)
    }


def _refresh_2d_zoom_scaling(scene: _InteractiveSceneState) -> None:
    if scene.dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, scene.ax))


def _ensure_tensor_label_artists(scene: _InteractiveSceneState) -> None:
    if scene.tensor_label_artists:
        return
    before = len(scene.ax.texts)
    _draw_labels(
        plotter=scene.plotter,
        ax=scene.ax,
        graph=scene.graph,
        positions=scene.positions,
        show_tensor_labels=True,
        config=scene.config,
        p=scene.params,
        dimensions=scene.dimensions,
        tensor_hover_by_node=None,
        visible_draw_order=list(scene.visible_node_ids),
        tensor_label_zorder_by_node=_tensor_label_zorders(scene),
        tensor_disk_radius_px_3d=scene.tensor_disk_radius_px_3d,
    )
    if _should_refine_tensor_labels(scene.config, visible_tensor_count=len(scene.visible_node_ids)):
        _refit_tensor_labels_to_disks(
            ax=scene.ax,
            p=scene.params,
            dimensions=scene.dimensions,
            tensor_disk_radius_px_3d=scene.tensor_disk_radius_px_3d,
        )
    scene.tensor_label_artists = [
        text
        for text in scene.ax.texts[before:]
        if getattr(text, "get_gid", lambda: None)() == _TENSOR_LABEL_GID
    ]
    for node_id, artist in zip(scene.visible_node_ids, scene.tensor_label_artists, strict=False):
        artist._tensor_network_viz_node_id = int(node_id)  # type: ignore[attr-defined]
    scene.tensor_hover_payload = None
    _refresh_2d_zoom_scaling(scene)


def _ensure_edge_label_artists(scene: _InteractiveSceneState) -> None:
    if scene.edge_label_artists:
        return
    before = len(scene.ax.texts)
    zorder_label = _last_edge_label_zorder(scene)
    for entry in scene.edge_geometry:
        edge = entry.edge
        if edge.kind == "contraction":
            left_id, right_id = edge.node_ids
            _draw_contraction_edge_labels(
                plotter=scene.plotter,
                curve=entry.polyline,
                edge=edge,
                graph=scene.graph,
                positions=scene.positions,
                left_id=left_id,
                right_id=right_id,
                show_index_labels=True,
                config=scene.config,
                p=scene.params,
                dimensions=scene.dimensions,
                ax=scene.ax,
                scale=scene.scale,
                zorder_label=zorder_label,
            )
        elif edge.kind == "dangling":
            _draw_dangling_edge_labels(
                plotter=scene.plotter,
                edge=edge,
                graph=scene.graph,
                start=entry.polyline[0],
                end=entry.polyline[-1],
                show_index_labels=True,
                config=scene.config,
                dimensions=scene.dimensions,
                p=scene.params,
                ax=scene.ax,
                scale=scene.scale,
                zorder_label=zorder_label,
            )
        elif edge.kind == "self":
            _draw_self_loop_edge_labels(
                plotter=scene.plotter,
                edge=edge,
                graph=scene.graph,
                curve=entry.polyline,
                positions=scene.positions,
                directions=scene.directions,
                show_index_labels=True,
                config=scene.config,
                dimensions=scene.dimensions,
                p=scene.params,
                ax=scene.ax,
                scale=scene.scale,
                zorder_label=zorder_label,
            )
    scene.edge_label_artists = [
        text
        for text in scene.ax.texts[before:]
        if getattr(text, "get_gid", lambda: None)() == _EDGE_INDEX_LABEL_GID
    ]
    _refresh_2d_zoom_scaling(scene)


def _build_tensor_hover_payload(scene: _InteractiveSceneState) -> dict[int, tuple[str, float]]:
    if scene.tensor_hover_payload is not None:
        return dict(scene.tensor_hover_payload)
    payload: dict[int, tuple[str, float]] = {}
    for artist in scene.tensor_label_artists:
        node_id = getattr(artist, "_tensor_network_viz_node_id", None)
        if node_id is None:
            continue
        payload[int(node_id)] = (str(artist.get_text()), float(artist.get_fontsize()))
    if payload:
        scene.tensor_hover_payload = dict(payload)
        return payload

    hover_data: dict[int, tuple[str, float]] = {}
    _draw_labels(
        plotter=scene.plotter,
        ax=scene.ax,
        graph=scene.graph,
        positions=scene.positions,
        show_tensor_labels=False,
        config=scene.config,
        p=scene.params,
        dimensions=scene.dimensions,
        tensor_hover_by_node=hover_data,
        visible_draw_order=list(scene.visible_node_ids),
        tensor_label_zorder_by_node=None,
        tensor_disk_radius_px_3d=scene.tensor_disk_radius_px_3d,
    )
    scene.tensor_hover_payload = dict(hover_data)
    return hover_data


def _build_edge_hover_payload(scene: _InteractiveSceneState) -> tuple[tuple[Any, str], ...]:
    if scene.edge_hover_payload is not None:
        return tuple(scene.edge_hover_payload)
    payload: list[tuple[Any, str]] = []
    for entry in scene.edge_geometry:
        edge = entry.edge
        text = ""
        if edge.kind == "contraction":
            text = _contraction_hover_label_text(edge, scene.graph)
        elif edge.kind == "dangling":
            text = _dangling_hover_label_text(edge)
        elif edge.kind == "self":
            text = _self_loop_hover_label_text(edge, scene.graph)
        if text:
            payload.append((entry.polyline, text))
    scene.edge_hover_payload = tuple(payload)
    return tuple(payload)


def _apply_scene_hover_state(
    scene: _InteractiveSceneState,
    *,
    hover_on: bool,
) -> None:
    tensor_hover = _build_tensor_hover_payload(scene) if hover_on else {}
    edge_hover = _build_edge_hover_payload(scene) if hover_on else ()
    state = _RenderHoverState(
        ax=scene.hover_state.ax,
        figure=scene.hover_state.figure,
        dimensions=scene.hover_state.dimensions,
        node_patch_coll=scene.node_patch_coll if tensor_hover else None,
        visible_node_ids=scene.hover_state.visible_node_ids,
        tensor_hover=dict(tensor_hover),
        edge_hover=tuple(edge_hover),
        line_width_px_hint=float(scene.hover_state.line_width_px_hint),
        positions=scene.hover_state.positions,
        params=scene.hover_state.params,
        tensor_disk_radius_px_3d=scene.hover_state.tensor_disk_radius_px_3d,
    )
    scene.hover_state = state
    scheme_patches_2d: tuple[tuple[Any, str], ...] = ()
    scheme_aabbs_3d: tuple[
        tuple[tuple[float, float, float, float, float, float], str, Any],
        ...,
    ] = ()
    controls = scene.contraction_controls
    if controls is not None and controls.scheme_on and controls.cost_hover_on:
        scheme_patches_2d = controls._scheme_entries_2d()
        scheme_aabbs_3d = controls._scheme_entries_3d()
    _apply_render_hover_state(
        scene.hover_state,
        scheme_patches_2d=scheme_patches_2d,
        scheme_aabbs_3d=scheme_aabbs_3d,
    )


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
        if not self._external_ax:
            _reserve_figure_bottom(figure, _INTERACTIVE_CONTROLS_BOTTOM)
        self._build_controls()
        self._apply_scene_state(self.current_scene)
        figure._tensor_network_viz_interactive_controls = self  # type: ignore[attr-defined]
        figure._tensor_network_viz_active_axes = ax  # type: ignore[attr-defined]
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
            scene.contraction_controls = getattr(
                rendered_ax,
                "_tensor_network_viz_contraction_controls",
                None,
            )
        return fig, rendered_ax

    def _build_controls(self) -> None:
        assert self.figure is not None
        labels = list(_BASE_TOGGLE_LABELS)
        has_scheme_toggles = self.current_scene.contraction_controls is not None
        if has_scheme_toggles:
            labels.extend(_SCHEME_TOGGLE_LABELS)
        if not self._external_ax:
            radio_ax = self.figure.add_axes(_VIEW_SELECTOR_BOUNDS)
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
        check_ax = self.figure.add_axes(
            _interactive_checkbox_bounds(include_scheme_toggles=has_scheme_toggles)
        )
        self._check_ax = check_ax
        statuses = [
            self.hover_on,
            self.tensor_labels_on,
            self.edge_labels_on,
        ]
        if has_scheme_toggles:
            statuses.extend([self.scheme_on, self.playback_on, self.cost_hover_on])
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

    def _apply_scene_state(self, scene: _InteractiveSceneState) -> None:
        if self.tensor_labels_on:
            _ensure_tensor_label_artists(scene)
        for artist in scene.tensor_label_artists:
            _set_artist_visible(artist, self.tensor_labels_on)
        if self.edge_labels_on:
            _ensure_edge_label_artists(scene)
        for artist in scene.edge_label_artists:
            _set_artist_visible(artist, self.edge_labels_on)
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
        _apply_scene_hover_state(scene, hover_on=self.hover_on)
        self._sync_checkbuttons()
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
        fig._tensor_network_viz_active_axes = ax  # type: ignore[attr-defined]

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
