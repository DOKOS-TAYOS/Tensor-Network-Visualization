from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from ..contractions import _ContractionGroups, _group_contractions
from ..graph import _GraphData
from ..layout import AxisDirections, NodePositions
from .constants import *
from .edges import _draw_edges, _draw_edges_2d_layered
from .fonts_and_scale import (
    _draw_scale_params,
    _DrawScaleParams,
    _figure_relative_font_scale,
    _register_2d_zoom_font_scaling,
)
from .hover import (
    _apply_saved_hover_state,
    _disconnect_tensor_network_hover,
    _RenderHoverState,
)
from .labels_misc import _estimate_drawn_label_count
from .plotter import _make_plotter, _node_edge_degrees, _PlotAdapter
from .scene_state import _InteractiveSceneState, _RenderedEdgeGeometry
from .tensors import (
    _draw_labels,
    _draw_nodes,
    _refit_tensor_labels_to_disks,
    _visible_node_ids_in_graph_order,
)
from .viewport_geometry import (
    _apply_axis_limits_with_outset,
    _stack_viewport_coords,
    _view_outset_margin_data_units,
)


@dataclass(frozen=True)
class _GraphRenderState:
    contraction_groups: _ContractionGroups
    visible_order: tuple[int, ...]
    node_degrees: dict[int, int]


@dataclass(frozen=True)
class _RenderPrepContext:
    graph: _GraphData
    positions: NodePositions
    config: PlotConfig
    dimensions: Literal[2, 3]
    scale: float
    contraction_groups: _ContractionGroups
    graph_state: _GraphRenderState
    label_slots: int
    params: _DrawScaleParams
    plotter: _PlotAdapter
    hover_edge_targets: list[tuple[np.ndarray, str]] | None
    tensor_hover_by_node: dict[int, tuple[str, float]] | None
    edge_geometry_sink: list[_RenderedEdgeGeometry]
    viewport_coords: np.ndarray
    view_margin: float


_graph_render_state_by_id: dict[int, _GraphRenderState] = {}
_AUTO_FAST_VISIBLE_TENSOR_THRESHOLD: int = 40


def _graph_render_state(
    graph: _GraphData,
) -> _GraphRenderState:
    key = id(graph)
    cached = _graph_render_state_by_id.get(key)
    if cached is not None:
        return cached

    state = _GraphRenderState(
        contraction_groups=_group_contractions(graph),
        visible_order=tuple(_visible_node_ids_in_graph_order(graph)),
        node_degrees=_node_edge_degrees(graph),
    )
    _graph_render_state_by_id[key] = state

    def _evict() -> None:
        _graph_render_state_by_id.pop(key, None)

    weakref.finalize(graph, _evict)
    return state


def _should_refine_tensor_labels(
    config: PlotConfig,
    *,
    visible_tensor_count: int,
) -> bool:
    if config.tensor_label_refinement == "always":
        return True
    if config.tensor_label_refinement == "never":
        return False
    return visible_tensor_count < _AUTO_FAST_VISIBLE_TENSOR_THRESHOLD


def _prepare_render_context(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float,
    show_tensor_labels: bool,
    show_index_labels: bool,
    contraction_groups: _ContractionGroups | None = None,
    bond_curve_pad: float | None = None,
) -> _RenderPrepContext:
    _disconnect_tensor_network_hover(ax.figure)
    ax.cla()

    graph_state = _graph_render_state(graph)
    resolved_groups = contraction_groups or graph_state.contraction_groups
    label_slots = _estimate_drawn_label_count(
        graph,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
    )
    font_figure_scale = _figure_relative_font_scale(ax.figure, label_slots)
    params = _draw_scale_params(
        config,
        scale,
        fig=ax.figure,
        is_3d=dimensions == 3,
        font_figure_scale=font_figure_scale,
        label_slots=max(1, label_slots),
    )
    hover_edge_targets: list[tuple[np.ndarray, str]] | None = None
    tensor_hover_by_node: dict[int, tuple[str, float]] | None = None
    if config.hover_labels:
        hover_edge_targets = []
        tensor_hover_by_node = {}
    plotter = _make_plotter(
        ax,
        dimensions=dimensions,
        hover_edge_targets=hover_edge_targets,
    )
    pre_coords = _stack_viewport_coords(positions)
    view_margin = _view_outset_margin_data_units(
        graph,
        positions,
        params,
        scale,
        resolved_groups,
        bond_curve_pad=bond_curve_pad,
    )
    _apply_axis_limits_with_outset(ax, pre_coords, view_margin=view_margin, dimensions=dimensions)
    return _RenderPrepContext(
        graph=graph,
        positions=positions,
        config=config,
        dimensions=dimensions,
        scale=scale,
        contraction_groups=resolved_groups,
        graph_state=graph_state,
        label_slots=label_slots,
        params=params,
        plotter=plotter,
        hover_edge_targets=hover_edge_targets,
        tensor_hover_by_node=tensor_hover_by_node,
        edge_geometry_sink=[],
        viewport_coords=pre_coords,
        view_margin=view_margin,
    )


def _draw_edges_nodes_and_labels(
    *,
    ax: Any,
    context: _RenderPrepContext,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    tensor_disk_radius_px_3d: float | None,
) -> np.ndarray:
    visible_order = list(context.graph_state.visible_order)
    node_degrees = context.graph_state.node_degrees
    tensor_z_by_node: dict[int, float] | None = None
    use_2d_layers = context.dimensions == 2 and bool(visible_order)

    if use_2d_layers:
        clear_nodes = getattr(context.plotter, "clear_node_disk_collections", None)
        if callable(clear_nodes):
            clear_nodes()
        _draw_edges_2d_layered(
            plotter=context.plotter,
            graph=context.graph,
            positions=context.positions,
            directions=directions,
            visible_order=visible_order,
            contraction_groups=context.contraction_groups,
            show_index_labels=show_index_labels,
            config=context.config,
            scale=context.scale,
            p=context.params,
            ax=ax,
            edge_geometry_sink=context.edge_geometry_sink,
        )
        flush = getattr(context.plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
        draw_one = context.plotter.draw_tensor_node
        tensor_z_by_node = {
            node_id: float(
                _ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_TENSOR_NAME
            )
            for index, node_id in enumerate(visible_order)
        }
        for index, node_id in enumerate(visible_order):
            z_disk = float(_ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DISK)
            draw_one(
                np.asarray(context.positions[node_id], dtype=float),
                config=context.config,
                p=context.params,
                degree_one=node_degrees.get(int(node_id), 0) == 1,
                zorder=z_disk,
            )
        coords = np.stack(
            [np.asarray(context.positions[node_id], dtype=float) for node_id in visible_order]
        )
    else:
        _draw_edges(
            plotter=context.plotter,
            graph=context.graph,
            positions=context.positions,
            directions=directions,
            contraction_groups=context.contraction_groups,
            show_index_labels=show_index_labels,
            config=context.config,
            scale=context.scale,
            dimensions=context.dimensions,
            p=context.params,
            ax=ax,
            edge_geometry_sink=context.edge_geometry_sink,
        )
        flush = getattr(context.plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
        coords = _draw_nodes(
            plotter=context.plotter,
            graph=context.graph,
            positions=context.positions,
            config=context.config,
            p=context.params,
            visible_node_ids=visible_order,
            node_degrees=node_degrees,
        )

    context.plotter.style_axes(context.viewport_coords, view_margin=context.view_margin)
    _draw_labels(
        plotter=context.plotter,
        ax=ax,
        graph=context.graph,
        positions=context.positions,
        show_tensor_labels=show_tensor_labels,
        config=context.config,
        p=context.params,
        dimensions=context.dimensions,
        tensor_hover_by_node=context.tensor_hover_by_node,
        visible_draw_order=visible_order,
        tensor_label_zorder_by_node=tensor_z_by_node,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )
    if _should_refine_tensor_labels(
        context.config,
        visible_tensor_count=len(visible_order),
    ):
        _refit_tensor_labels_to_disks(
            ax=ax,
            p=context.params,
            dimensions=context.dimensions,
            tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
        )
    if context.dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, ax))
    return coords


def _register_render_hover(
    *,
    ax: Any,
    context: _RenderPrepContext,
    show_tensor_labels: bool,
    show_index_labels: bool,
    scheme_patches_2d: list[tuple[Any, str]],
    scheme_aabbs_3d: list[tuple[tuple[float, float, float, float, float, float], str, Any]],
    tensor_disk_radius_px_3d: float | None,
) -> _RenderHoverState:
    visible_ids = list(context.graph_state.visible_order)
    if context.dimensions == 2:
        node_colls = getattr(context.plotter, "_node_disk_collections", None)
        node_collection = (
            node_colls
            if isinstance(node_colls, list) and len(node_colls) > 0
            else getattr(context.plotter, "_node_disk_collection", None)
        )
        state = _RenderHoverState(
            ax=cast(Axes, ax),
            figure=ax.figure,
            dimensions=2,
            node_patch_coll=node_collection if context.tensor_hover_by_node is not None else None,
            visible_node_ids=tuple(visible_ids),
            tensor_hover=dict(context.tensor_hover_by_node or {}),
            edge_hover=tuple(context.hover_edge_targets or ()),
            line_width_px_hint=float(context.params.lw),
        )
        _apply_saved_hover_state(
            state,
            scheme_patches_2d=scheme_patches_2d,
            scheme_aabbs_3d=scheme_aabbs_3d,
        )
        return state

    state = _RenderHoverState(
        ax=ax,
        figure=ax.figure,
        dimensions=3,
        node_patch_coll=None,
        visible_node_ids=tuple(visible_ids),
        tensor_hover=dict(context.tensor_hover_by_node or {}),
        edge_hover=tuple(context.hover_edge_targets or ()),
        line_width_px_hint=float(context.params.lw),
        positions=context.positions,
        params=context.params,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )
    _apply_saved_hover_state(
        state,
        scheme_patches_2d=scheme_patches_2d,
        scheme_aabbs_3d=scheme_aabbs_3d,
    )
    return state


def _apply_render_hover_state(
    state: _RenderHoverState,
    *,
    scheme_patches_2d: Sequence[tuple[Any, str]] | None = None,
    scheme_aabbs_3d: Sequence[tuple[tuple[float, float, float, float, float, float], str, Any]]
    | None = None,
) -> None:
    _apply_saved_hover_state(
        state,
        scheme_patches_2d=scheme_patches_2d,
        scheme_aabbs_3d=scheme_aabbs_3d,
    )


def _node_patch_collection_from_plotter(context: _RenderPrepContext) -> Any:
    node_colls = getattr(context.plotter, "_node_disk_collections", None)
    if isinstance(node_colls, list) and len(node_colls) > 0:
        return node_colls
    return getattr(context.plotter, "_node_disk_collection", None)


def _build_interactive_scene_state(
    *,
    ax: Any,
    context: _RenderPrepContext,
    directions: AxisDirections,
    scale: float,
    hover_state: _RenderHoverState,
    tensor_disk_radius_px_3d: float | None,
) -> _InteractiveSceneState:
    return _InteractiveSceneState(
        ax=ax,
        graph=context.graph,
        positions=context.positions,
        directions=directions,
        config=context.config,
        dimensions=context.dimensions,
        scale=scale,
        params=context.params,
        contraction_groups=context.contraction_groups,
        plotter=context.plotter,
        visible_node_ids=tuple(context.graph_state.visible_order),
        node_patch_coll=_node_patch_collection_from_plotter(context),
        edge_geometry=tuple(context.edge_geometry_sink),
        hover_state=hover_state,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )


__all__ = [
    "_apply_render_hover_state",
    "_build_interactive_scene_state",
    "_graph_render_state",
    "_node_patch_collection_from_plotter",
    "_should_refine_tensor_labels",
    "_prepare_render_context",
    "_draw_edges_nodes_and_labels",
    "_register_render_hover",
    "_GraphRenderState",
    "_RenderHoverState",
    "_RenderPrepContext",
]
