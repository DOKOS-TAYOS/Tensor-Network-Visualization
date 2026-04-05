from __future__ import annotations

from typing import Any, cast

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core._label_format import format_tensor_node_label
from ._core.draw.constants import (
    _EDGE_INDEX_LABEL_GID,
    _LABEL_FONT_3D_SCALE,
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
from ._core.draw.label_descriptors import (
    _AnyLabelDescriptor,
    _DeferredBondLabelDescriptor,
    _DeferredSelfLoopLabelDescriptor,
    _TextLabelDescriptor,
)
from ._core.draw.labels_misc import (
    _contraction_hover_label_text,
    _dangling_hover_label_text,
    _edge_index_text_kwargs,
    _self_loop_hover_label_text,
)
from ._core.draw.render_prep import (
    _apply_render_hover_state,
    _should_refine_tensor_labels,
)
from ._core.draw.scene_state import _InteractiveSceneState
from ._core.draw.tensors import (
    _refit_tensor_labels_to_disks,
    _tensor_label_fontsize_to_fit,
)
from ._core.draw.viewport_geometry import (
    _bond_index_label_perp_offset,
    _contraction_edge_index_label_2d_placement,
    _contraction_edge_index_label_3d_placement,
    _edge_index_along_bond_text_kw,
    _edge_index_fontsize_for_bond,
)
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


def _resolve_bond_label_descriptor(
    scene: _InteractiveSceneState,
    descriptor: _DeferredBondLabelDescriptor,
) -> _TextLabelDescriptor:
    fontsize = _edge_index_fontsize_for_bond(
        descriptor.text,
        bond_start=descriptor.bond_start,
        bond_end=descriptor.bond_end,
        ax=scene.ax,
        dimensions=scene.dimensions,
        is_physical=descriptor.is_physical,
        peer_captions_for_width=descriptor.peer_captions_for_width,
    )
    text_kwargs = _edge_index_text_kwargs(
        scene.config,
        fontsize=fontsize,
        stub_kind=descriptor.stub_kind,
        bbox_pad=scene.params.index_bbox_pad,
        zorder=descriptor.zorder,
    )
    if scene.dimensions == 2:
        position, align_kwargs = _contraction_edge_index_label_2d_placement(
            Q=descriptor.point,
            t_geom_2d=np.asarray(descriptor.tangent_geom[:2], dtype=float),
            t_align_2d=np.asarray(descriptor.tangent_align[:2], dtype=float),
            text_ep=descriptor.text_endpoint,
            p=scene.params,
            ax=cast(Axes, scene.ax),
            scale=scene.scale,
            fontsize_pt=float(fontsize),
        )
    else:
        position, align_kwargs = _contraction_edge_index_label_3d_placement(
            Q=np.asarray(descriptor.point, dtype=float).reshape(3),
            t_geom_3d=np.asarray(descriptor.tangent_geom, dtype=float).reshape(3),
            t_align_3d=np.asarray(descriptor.tangent_align, dtype=float).reshape(3),
            text_ep=descriptor.text_endpoint,
            p=scene.params,
            ax=scene.ax,
            scale=scene.scale,
            fontsize_pt=float(fontsize),
        )
    return _TextLabelDescriptor(
        position=np.asarray(position, dtype=float).copy(),
        text=descriptor.text,
        kwargs={**text_kwargs, **align_kwargs},
    )


def _resolve_self_loop_label_descriptor(
    scene: _InteractiveSceneState,
    descriptor: _DeferredSelfLoopLabelDescriptor,
) -> _TextLabelDescriptor:
    fontsize = _edge_index_fontsize_for_bond(
        descriptor.text,
        bond_start=descriptor.bond_start,
        bond_end=descriptor.bond_end,
        ax=scene.ax,
        dimensions=scene.dimensions,
        is_physical=False,
        peer_captions_for_width=descriptor.peer_captions_for_width,
    )
    world_perp = descriptor.offset_direction if scene.dimensions == 3 else None
    offset = (
        np.asarray(descriptor.offset_direction, dtype=float)
        * float(descriptor.offset_scale)
        * _bond_index_label_perp_offset(
            descriptor.text,
            p=scene.params,
            scale=scene.scale,
            dimensions=scene.dimensions,
            ax=scene.ax,
            anchor=descriptor.point,
            world_perp_dir=world_perp,
        )
    )
    text_kwargs = {
        **_edge_index_text_kwargs(
            scene.config,
            fontsize=fontsize,
            bbox_pad=scene.params.index_bbox_pad,
            zorder=descriptor.zorder,
        ),
        **_edge_index_along_bond_text_kw(
            endpoint=descriptor.text_endpoint,
            tangent=descriptor.tangent,
            ax=scene.ax,
            dimensions=scene.dimensions,
        ),
    }
    return _TextLabelDescriptor(
        position=np.asarray(descriptor.point, dtype=float).copy() + offset,
        text=descriptor.text,
        kwargs=text_kwargs,
    )


def _materialize_label_descriptors(
    scene: _InteractiveSceneState,
    descriptors: tuple[_AnyLabelDescriptor, ...],
) -> tuple[_TextLabelDescriptor, ...]:
    materialized: list[_TextLabelDescriptor] = []
    for descriptor in descriptors:
        if isinstance(descriptor, _TextLabelDescriptor):
            materialized.append(descriptor)
            continue
        if isinstance(descriptor, _DeferredBondLabelDescriptor):
            materialized.append(_resolve_bond_label_descriptor(scene, descriptor))
            continue
        materialized.append(_resolve_self_loop_label_descriptor(scene, descriptor))
    return tuple(materialized)


def _plot_label_descriptors(
    scene: _InteractiveSceneState,
    descriptors: tuple[_TextLabelDescriptor, ...],
) -> list[Artist]:
    before = len(scene.ax.texts)
    for descriptor in descriptors:
        kwargs = dict(descriptor.kwargs)
        if (
            descriptor.node_id is not None
            and kwargs.get("gid") == _TENSOR_LABEL_GID
            and "fontsize" not in kwargs
        ):
            node_position = scene.positions[descriptor.node_id]
            if scene.tensor_disk_radius_px_3d is not None and scene.dimensions == 3:
                pixel_radius = float(scene.tensor_disk_radius_px_3d)
            else:
                from ._core.draw.disk_metrics import _tensor_disk_radius_px

                pixel_radius = _tensor_disk_radius_px(
                    scene.ax,
                    node_position,
                    scene.params,
                    scene.dimensions,
                )
            fontsize = _tensor_label_fontsize_to_fit(
                text=descriptor.text,
                cap_pt=scene.params.font_tensor_label_max,
                pixel_radius=pixel_radius,
                fig=scene.ax.figure,
            )
            if scene.dimensions == 3:
                cap_tensor = float(scene.params.font_tensor_label_max) * _LABEL_FONT_3D_SCALE
                kwargs["fontsize"] = min(
                    float(fontsize) * _LABEL_FONT_3D_SCALE,
                    cap_tensor,
                )
            else:
                kwargs["fontsize"] = float(fontsize)
        scene.plotter.plot_text(
            np.asarray(descriptor.position, dtype=float),
            descriptor.text,
            **kwargs,
        )
    return list(scene.ax.texts[before:])


def _build_tensor_label_descriptors(
    scene: _InteractiveSceneState,
) -> tuple[_TextLabelDescriptor, ...]:
    if scene.tensor_label_descriptors is not None:
        return cast(tuple[_TextLabelDescriptor, ...], tuple(scene.tensor_label_descriptors))
    zorder_by_node = _tensor_label_zorders(scene)
    descriptors = []
    for node_id in scene.visible_node_ids:
        node = scene.graph.nodes.get(node_id)
        if node is None or node.is_virtual:
            continue
        if zorder_by_node is None:
            zorder = float(_ZORDER_LAYER_TENSOR_NAME)
        else:
            zorder = float(zorder_by_node.get(node_id, _ZORDER_LAYER_TENSOR_NAME))
        descriptors.append(
            _TextLabelDescriptor(
                position=np.asarray(scene.positions[node_id], dtype=float).copy(),
                text=format_tensor_node_label(node.name),
                kwargs={
                    "color": scene.config.tensor_label_color,
                    "ha": "center",
                    "va": "center",
                    "zorder": zorder,
                    "gid": _TENSOR_LABEL_GID,
                },
                node_id=int(node_id),
            )
        )
    scene.tensor_label_descriptors = tuple(descriptors)
    return cast(tuple[_TextLabelDescriptor, ...], tuple(scene.tensor_label_descriptors))


def _build_edge_label_descriptors(
    scene: _InteractiveSceneState,
) -> tuple[_TextLabelDescriptor, ...]:
    if scene.edge_label_descriptors is not None:
        return cast(tuple[_TextLabelDescriptor, ...], tuple(scene.edge_label_descriptors))
    descriptors: list[_TextLabelDescriptor] = []
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
                label_sink=descriptors,
            )
            continue
        if edge.kind == "dangling":
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
                label_sink=descriptors,
            )
            continue
        if edge.kind == "self":
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
                label_sink=descriptors,
            )
    scene.edge_label_descriptors = tuple(descriptors)
    return cast(tuple[_TextLabelDescriptor, ...], tuple(scene.edge_label_descriptors))


def _ensure_scene_label_descriptors(scene: _InteractiveSceneState) -> None:
    _build_tensor_label_descriptors(scene)
    _build_edge_label_descriptors(scene)


def _ensure_tensor_label_artists(scene: _InteractiveSceneState) -> None:
    if scene.tensor_label_artists:
        return
    descriptors = tuple(scene.tensor_label_descriptors or ())
    if not descriptors:
        descriptors = _build_tensor_label_descriptors(scene)
    concrete_descriptors = _materialize_label_descriptors(scene, descriptors)
    scene.tensor_label_descriptors = concrete_descriptors
    scene.tensor_label_artists = _plot_label_descriptors(scene, concrete_descriptors)
    if _should_refine_tensor_labels(scene.config, visible_tensor_count=len(scene.visible_node_ids)):
        _refit_tensor_labels_to_disks(
            ax=scene.ax,
            p=scene.params,
            dimensions=scene.dimensions,
            tensor_disk_radius_px_3d=scene.tensor_disk_radius_px_3d,
        )
    scene.tensor_label_artists = [
        text
        for text in scene.tensor_label_artists
        if getattr(text, "get_gid", lambda: None)() == _TENSOR_LABEL_GID
    ]
    for descriptor, artist in zip(concrete_descriptors, scene.tensor_label_artists, strict=False):
        if descriptor.node_id is not None:
            set_artist_node_id(artist, descriptor.node_id)
    scene.tensor_hover_payload = None
    _refresh_2d_zoom_scaling(scene)


def _ensure_edge_label_artists(scene: _InteractiveSceneState) -> None:
    if scene.edge_label_artists:
        return
    descriptors = tuple(scene.edge_label_descriptors or ())
    if not descriptors:
        descriptors = _build_edge_label_descriptors(scene)
    concrete_descriptors = _materialize_label_descriptors(scene, descriptors)
    scene.edge_label_descriptors = concrete_descriptors
    scene.edge_label_artists = [
        text
        for text in _plot_label_descriptors(scene, concrete_descriptors)
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

    for descriptor in _build_tensor_label_descriptors(scene):
        if not isinstance(descriptor, _TextLabelDescriptor) or descriptor.node_id is None:
            continue
        fontsize_raw = descriptor.kwargs.get("fontsize")
        if fontsize_raw is None:
            fontsize_raw = (
                float(scene.params.font_tensor_label_max) * _LABEL_FONT_3D_SCALE
                if scene.dimensions == 3
                else float(scene.params.font_tensor_label_max)
            )
        payload[int(descriptor.node_id)] = (descriptor.text, float(fontsize_raw))
    scene.tensor_hover_payload = dict(payload)
    return payload


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
    "_build_edge_label_descriptors",
    "_build_tensor_label_descriptors",
    "_ensure_edge_label_artists",
    "_ensure_scene_label_descriptors",
    "_ensure_tensor_label_artists",
    "_scene_from_axes",
    "_set_artist_visible",
]
