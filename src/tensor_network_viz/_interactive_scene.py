from __future__ import annotations

from typing import Any, cast

from matplotlib.artist import Artist
from matplotlib.axes import Axes
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
from ._matplotlib_state import get_artist_node_id, get_scene, set_artist_node_id

RenderedAxes = Axes | Axes3D


def _set_artist_visible(artist: Artist, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        setter(bool(visible))


def _scene_from_axes(ax: RenderedAxes | None) -> _InteractiveSceneState | None:
    if ax is None:
        return None
    scene = get_scene(ax)
    if isinstance(scene, _InteractiveSceneState):
        return scene
    return None


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
        set_artist_node_id(artist, node_id)
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
        node_id = get_artist_node_id(artist)
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
    _apply_render_hover_state(
        scene.hover_state,
        scheme_patches_2d=(),
        scheme_aabbs_3d=(),
    )


__all__ = [
    "_apply_scene_hover_state",
    "_ensure_edge_label_artists",
    "_ensure_tensor_label_artists",
    "_scene_from_axes",
    "_set_artist_visible",
]
