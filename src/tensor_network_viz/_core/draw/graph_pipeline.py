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
from .edges import _draw_edges
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
from .plotter import _make_plotter
from .tensors import _draw_labels, _draw_nodes, _refit_tensor_labels_to_disks
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
    if dimensions == 2:
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
    )
    if config.refine_tensor_labels:
        _refit_tensor_labels_to_disks(ax=ax, p=params, dimensions=dimensions)
    if dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, ax))
    if config.hover_labels and (show_tensor_labels or show_index_labels):
        vis_ids = [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]
        if dimensions == 2:
            node_coll = getattr(plotter, "_node_disk_collection", None)
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
