from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from ..contractions import _group_contractions
from ..graph import (
    _GraphData,
)
from ..layout import (
    AxisDirections,
    NodePositions,
)
from .constants import *
from .edges import _draw_edges, _draw_edges_2d_layered
from .fonts_and_scale import (
    _draw_scale_params,
    _figure_relative_font_scale,
    _register_2d_zoom_font_scaling,
)
from .hover import (
    _disconnect_tensor_network_hover,
    _register_2d_hover_labels,
    _register_3d_hover_labels,
)
from .labels_misc import _estimate_drawn_label_count
from .plotter import _graph_edge_degree, _make_plotter
from .tensors import (
    _draw_labels,
    _draw_nodes,
    _refit_tensor_labels_to_disks,
    _visible_node_ids_in_graph_order,
)
from .viewport_geometry import (
    _apply_axis_limits_with_outset,
    _stack_visible_tensor_coords,
    _view_outset_margin_data_units,
)


def _draw_graph(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float = 1.0,
) -> None:
    _disconnect_tensor_network_hover(ax.figure)
    ax.cla()
    contraction_groups = _group_contractions(graph)
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
    hover_edge_list: list[tuple[np.ndarray, str]] | None = None
    tensor_hover_map: dict[int, tuple[str, float]] | None = None
    if config.hover_labels:
        if show_index_labels:
            hover_edge_list = []
        if show_tensor_labels:
            tensor_hover_map = {}
    plotter = _make_plotter(
        ax,
        dimensions=dimensions,
        hover_edge_targets=hover_edge_list,
    )
    pre_coords = _stack_visible_tensor_coords(graph, positions)
    view_margin = _view_outset_margin_data_units(
        graph, positions, params, scale, contraction_groups
    )
    _apply_axis_limits_with_outset(ax, pre_coords, view_margin=view_margin, dimensions=dimensions)

    visible_order = _visible_node_ids_in_graph_order(graph)
    tensor_z_by_node: dict[int, float] | None = None
    use_2d_layers = dimensions == 2 and bool(visible_order)
    if use_2d_layers:
        clear_nodes = getattr(plotter, "clear_node_disk_collections", None)
        if callable(clear_nodes):
            clear_nodes()
        _draw_edges_2d_layered(
            plotter=plotter,
            graph=graph,
            positions=positions,
            directions=directions,
            visible_order=visible_order,
            contraction_groups=contraction_groups,
            show_index_labels=show_index_labels,
            config=config,
            scale=scale,
            p=params,
            ax=ax,
        )
        flush = getattr(plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
        draw_one = getattr(plotter, "draw_tensor_node")
        tensor_z_by_node = {
            nid: float(
                _ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_TENSOR_NAME
            )
            for i, nid in enumerate(visible_order)
        }
        for i, nid in enumerate(visible_order):
            zdisk = float(_ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DISK)
            coord = np.asarray(positions[nid], dtype=float)
            deg_one = _graph_edge_degree(graph, nid) == 1
            draw_one(
                coord,
                config=config,
                p=params,
                degree_one=deg_one,
                zorder=zdisk,
            )
        coords = np.stack([np.asarray(positions[nid], dtype=float) for nid in visible_order])
    elif dimensions == 2:
        _draw_edges(
            plotter=plotter,
            graph=graph,
            positions=positions,
            directions=directions,
            contraction_groups=contraction_groups,
            show_index_labels=show_index_labels,
            config=config,
            scale=scale,
            dimensions=2,
            p=params,
            ax=ax,
        )
        flush = getattr(plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
        coords = _draw_nodes(
            plotter=plotter,
            graph=graph,
            positions=positions,
            config=config,
            p=params,
        )
    else:
        _draw_edges(
            plotter=plotter,
            graph=graph,
            positions=positions,
            directions=directions,
            contraction_groups=contraction_groups,
            show_index_labels=show_index_labels,
            config=config,
            scale=scale,
            dimensions=dimensions,
            p=params,
            ax=ax,
        )
        flush = getattr(plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
        coords = _draw_nodes(
            plotter=plotter,
            graph=graph,
            positions=positions,
            config=config,
            p=params,
        )
    plotter.style_axes(coords, view_margin=view_margin)
    _draw_labels(
        plotter=plotter,
        ax=ax,
        graph=graph,
        positions=positions,
        show_tensor_labels=show_tensor_labels,
        config=config,
        p=params,
        dimensions=dimensions,
        tensor_hover_by_node=tensor_hover_map,
        visible_draw_order=visible_order if use_2d_layers else None,
        tensor_label_zorder_by_node=tensor_z_by_node,
    )
    if config.refine_tensor_labels:
        _refit_tensor_labels_to_disks(ax=ax, p=params, dimensions=dimensions)
    if dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, ax))
    if config.hover_labels and (show_tensor_labels or show_index_labels):
        vis_ids = _visible_node_ids_in_graph_order(graph)
        if dimensions == 2:
            node_colls = getattr(plotter, "_node_disk_collections", None)
            node_coll = (
                node_colls
                if (isinstance(node_colls, list) and len(node_colls) > 0)
                else getattr(plotter, "_node_disk_collection", None)
            )
            _register_2d_hover_labels(
                cast(Axes, ax),
                node_patch_coll=node_coll if show_tensor_labels else None,
                visible_node_ids=vis_ids,
                tensor_hover=tensor_hover_map or {},
                edge_hover=list(hover_edge_list or ()),
                line_width_px_hint=float(params.lw),
            )
        else:
            _register_3d_hover_labels(
                ax,
                ax.figure,
                positions=positions,
                visible_node_ids=vis_ids,
                tensor_hover=tensor_hover_map or {},
                edge_hover=list(hover_edge_list or ()),
                line_width_px_hint=float(params.lw),
                p=params,
            )


__all__ = ["_draw_graph"]
