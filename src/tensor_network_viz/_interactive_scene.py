from __future__ import annotations

from typing import Any, cast

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.text import Text
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core._label_format import format_tensor_node_label
from ._core.draw.constants import (
    _EDGE_INDEX_LABEL_GID,
    _LABEL_FONT_3D_SCALE,
    _TENSOR_LABEL_GID,
    _ZORDER_LAYER_BASE,
    _ZORDER_LAYER_DISK,
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
from ._core.draw.plotter import (
    NodeRenderMode,
    _node_edge_degrees,
    _NodeArtistBundle,
    _visible_degree_one_mask,
)
from ._core.draw.render_prep import (
    _apply_render_hover_state,
    _should_refine_tensor_labels,
)
from ._core.draw.scene_state import _InteractiveSceneState
from ._core.draw.tensors import (
    _refit_tensor_labels_to_disks,
    _tensor_label_fontsize_for_render,
)
from ._core.draw.viewport_geometry import (
    _bond_index_label_perp_offset,
    _contraction_edge_index_label_2d_placement,
    _contraction_edge_index_label_3d_placement,
    _edge_index_along_bond_text_kw,
    _edge_index_fontsize_for_bond,
)
from ._matplotlib_state import get_artist_node_id, get_scene, set_artist_node_id
from .config import TensorNetworkDiagnosticsConfig

RenderedAxes = Axes | Axes3D


def _diagnostic_hover_enabled(scene: _InteractiveSceneState) -> bool:
    diagnostics = scene.config.diagnostics or TensorNetworkDiagnosticsConfig()
    return bool(diagnostics.include_hover)


def _format_memory_estimate(estimated_nbytes: int | None) -> str | None:
    if estimated_nbytes is None:
        return None
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(estimated_nbytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _join_hover_text(base_text: str, extra_lines: list[str]) -> str:
    filtered_lines = [line for line in extra_lines if line]
    if not filtered_lines:
        return base_text
    if not base_text:
        return "\n".join(filtered_lines)
    return base_text + "\n" + "\n".join(filtered_lines)


def _node_hover_text(scene: _InteractiveSceneState, node_id: int, base_text: str) -> str:
    if not _diagnostic_hover_enabled(scene):
        return base_text
    node = scene.graph.nodes[node_id]
    extra_lines: list[str] = []
    if node.shape is not None:
        extra_lines.append(f"shape: {node.shape}")
    if node.element_count is not None:
        extra_lines.append(f"elements: {node.element_count}")
    memory_text = _format_memory_estimate(node.estimated_nbytes)
    if memory_text is not None:
        extra_lines.append(f"memory: {memory_text}")
    return _join_hover_text(base_text, extra_lines)


def _edge_hover_text(scene: _InteractiveSceneState, edge: object, base_text: str) -> str:
    if not _diagnostic_hover_enabled(scene):
        return base_text
    bond_dimension = getattr(edge, "bond_dimension", None)
    if bond_dimension is None:
        return base_text
    return _join_hover_text(base_text, [f"bond dimension: {bond_dimension}"])


def _set_artist_visible(artist: Artist, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        setter(bool(visible))


def _scene_label_artists(scene: _InteractiveSceneState) -> tuple[Artist, ...]:
    return tuple(
        artist
        for artist in (
            *scene.tensor_label_artists,
            *scene.edge_label_artists,
            *scene.diagnostic_artists,
        )
        if isinstance(artist, Artist) and getattr(artist, "axes", None) is scene.ax
    )


def _scene_non_label_max_zorder(
    scene: _InteractiveSceneState,
    label_artists: tuple[Artist, ...],
) -> float:
    label_ids = {id(artist) for artist in label_artists}
    max_zorder = 0.0
    for artist in scene.ax.get_children():
        if id(artist) in label_ids:
            continue
        max_zorder = max(max_zorder, float(artist.get_zorder()))
    return max_zorder


def _readd_scene_artist(scene: _InteractiveSceneState, artist: Artist) -> None:
    remover = getattr(artist, "remove", None)
    if not callable(remover):
        return
    try:
        remover()
    except (NotImplementedError, ValueError):
        return
    if isinstance(artist, Text):
        add_text = getattr(scene.ax, "_add_text", None)
        if callable(add_text):
            add_text(artist)
            return
    add_artist = getattr(scene.ax, "add_artist", None)
    if callable(add_artist):
        add_artist(artist)


def _bring_scene_label_artists_to_front(scene: _InteractiveSceneState) -> None:
    label_artists = _scene_label_artists(scene)
    if not label_artists:
        return
    top_zorder = _scene_non_label_max_zorder(scene, label_artists) + 1.0
    for index, artist in enumerate(label_artists):
        setter = getattr(artist, "set_zorder", None)
        if callable(setter):
            setter(float(top_zorder + index * 0.01))
        _readd_scene_artist(scene, artist)


def _node_mode_from_show_nodes(show_nodes: bool) -> NodeRenderMode:
    return "normal" if bool(show_nodes) else "compact"


def _set_node_bundle_visible(bundle: _NodeArtistBundle, visible: bool) -> None:
    for artist in bundle.artists:
        _set_artist_visible(cast(Artist, artist), visible)


def _build_scene_node_artists(
    scene: _InteractiveSceneState,
    *,
    mode: NodeRenderMode,
) -> _NodeArtistBundle:
    if not scene.visible_node_ids:
        return _NodeArtistBundle(mode=mode, artists=(), hover_target=None)

    node_degrees = _node_edge_degrees(scene.graph)
    if scene.dimensions == 2:
        clear_nodes = getattr(scene.plotter, "clear_node_disk_collections", None)
        if callable(clear_nodes):
            clear_nodes()
        draw_one = scene.plotter.draw_tensor_node
        for index, node_id in enumerate(scene.visible_node_ids):
            z_disk = float(_ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DISK)
            draw_one(
                np.asarray(scene.positions[node_id], dtype=float),
                config=scene.config,
                p=scene.params,
                degree_one=node_degrees.get(int(node_id), 0) == 1,
                mode=mode,
                zorder=z_disk,
            )
    else:
        coords = np.stack(
            [
                np.asarray(scene.positions[node_id], dtype=float)
                for node_id in scene.visible_node_ids
            ]
        )
        degree_one_mask = _visible_degree_one_mask(
            scene.graph,
            list(scene.visible_node_ids),
            node_degrees=node_degrees,
        )
        scene.plotter.draw_tensor_nodes(
            coords,
            config=scene.config,
            p=scene.params,
            degree_one_mask=degree_one_mask,
            mode=mode,
        )
    bundle = scene.plotter.get_node_artist_bundle()
    if bundle is None:
        return _NodeArtistBundle(mode=mode, artists=(), hover_target=None)
    return bundle


def _ensure_scene_node_artists(
    scene: _InteractiveSceneState,
    *,
    mode: NodeRenderMode,
) -> _NodeArtistBundle:
    bundle = scene.node_artist_bundles.get(mode)
    if bundle is not None:
        return bundle
    bundle = _build_scene_node_artists(scene, mode=mode)
    scene.node_artist_bundles[mode] = bundle
    return bundle


def _set_scene_node_mode(
    scene: _InteractiveSceneState,
    *,
    mode: NodeRenderMode,
) -> None:
    target_bundle = _ensure_scene_node_artists(scene, mode=mode)
    for bundle_mode, bundle in scene.node_artist_bundles.items():
        _set_node_bundle_visible(bundle, bundle_mode == mode)
    scene.active_node_mode = mode
    scene.node_patch_coll = target_bundle.hover_target


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
        preferred_fontsize_pt=scene.config.edge_label_fontsize,
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
        preferred_fontsize_pt=scene.config.edge_label_fontsize,
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
            fontsize = _tensor_label_fontsize_for_render(
                text=descriptor.text,
                config=scene.config,
                p=scene.params,
                pixel_radius=float(pixel_radius),
                fig=scene.ax.figure,
                dimensions=scene.dimensions,
            )
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
        node_position = scene.positions[node_id]
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
        label_text = format_tensor_node_label(node.name)
        fontsize = _tensor_label_fontsize_for_render(
            text=label_text,
            config=scene.config,
            p=scene.params,
            pixel_radius=float(pixel_radius),
            fig=scene.ax.figure,
            dimensions=scene.dimensions,
        )
        descriptors.append(
            _TextLabelDescriptor(
                position=np.asarray(scene.positions[node_id], dtype=float).copy(),
                text=label_text,
                kwargs={
                    "color": scene.config.tensor_label_color,
                    "ha": "center",
                    "va": "center",
                    "fontsize": float(fontsize),
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
            config=scene.config,
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
        payload[int(node_id)] = (
            _node_hover_text(scene, int(node_id), str(artist.get_text())),
            float(artist.get_fontsize()),
        )
    if payload:
        scene.tensor_hover_payload = dict(payload)
        return payload

    if scene.config.tensor_label_fontsize is not None:
        fontsize_hint = float(max(3.0, float(scene.config.tensor_label_fontsize)))
    else:
        fontsize_hint = float(scene.params.font_tensor_label_max)
        if scene.dimensions == 3:
            fontsize_hint *= float(_LABEL_FONT_3D_SCALE)

    for node_id in scene.visible_node_ids:
        node = scene.graph.nodes.get(int(node_id))
        if node is None or node.is_virtual:
            continue
        label_text = format_tensor_node_label(node.name)
        payload[int(node_id)] = (
            _node_hover_text(scene, int(node_id), label_text),
            fontsize_hint,
        )
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
            payload.append((entry.polyline, _edge_hover_text(scene, edge, text)))
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
    "_bring_scene_label_artists_to_front",
    "_build_edge_label_descriptors",
    "_build_scene_node_artists",
    "_build_tensor_label_descriptors",
    "_ensure_scene_node_artists",
    "_ensure_edge_label_artists",
    "_ensure_scene_label_descriptors",
    "_ensure_tensor_label_artists",
    "_node_mode_from_show_nodes",
    "_scene_from_axes",
    "_set_scene_node_mode",
    "_set_artist_visible",
]
