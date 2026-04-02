from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from ...config import PlotConfig
from ..contractions import _ContractionGroups
from ..graph import _EdgeData, _GraphData
from ..layout import AxisDirections, NodePositions
from .constants import *
from .contraction_edges import _draw_contraction_edge
from .dangling_self_edges import _draw_dangling_edge, _draw_self_loop_edge
from .fonts_and_scale import _DrawScaleParams
from .plotter import _PlotAdapter
from .scene_state import _RenderedEdgeGeometry


def _edge_stable_bond_sort_key(edge: _EdgeData) -> tuple[str, tuple[int, ...], int]:
    return (edge.kind, edge.node_ids, id(edge))


def _edge_stable_dangling_sort_key(edge: _EdgeData) -> tuple[tuple[int, ...], int]:
    return (edge.node_ids, id(edge))


def _draw_edges_2d_layered(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    visible_order: list[int],
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    p: _DrawScaleParams,
    ax: Any,
    edge_geometry_sink: list[_RenderedEdgeGeometry] | None = None,
) -> None:
    """Enqueue 2D edges with per-node z-order; caller must flush collections once."""
    by_node: dict[int, list[_EdgeData]] = defaultdict(list)
    for edge in graph.edges:
        for node_id in edge.node_ids:
            by_node[node_id].append(edge)
    drawn: set[int] = set()
    for index, node_id in enumerate(visible_order):
        z_bond = _ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_BOND
        z_dangling = _ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DANGLING
        z_label = _ZORDER_LAYER_BASE + index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_EDGE_INDEX
        incident = [edge for edge in by_node.get(node_id, ()) if id(edge) not in drawn]
        bonds = sorted(
            [edge for edge in incident if edge.kind != "dangling"],
            key=_edge_stable_bond_sort_key,
        )
        dangles = sorted(
            [edge for edge in incident if edge.kind == "dangling"],
            key=_edge_stable_dangling_sort_key,
        )
        for edge in bonds:
            if edge.kind == "self":
                _draw_self_loop_edge(
                    plotter=plotter,
                    edge=edge,
                    graph=graph,
                    positions=positions,
                    directions=directions,
                    show_index_labels=show_index_labels,
                    config=config,
                    dimensions=2,
                    p=p,
                    ax=ax,
                    scale=scale,
                    zorder_line=z_bond,
                    zorder_label=z_label,
                    edge_geometry_sink=edge_geometry_sink,
                )
            else:
                _draw_contraction_edge(
                    plotter=plotter,
                    edge=edge,
                    graph=graph,
                    positions=positions,
                    contraction_groups=contraction_groups,
                    show_index_labels=show_index_labels,
                    config=config,
                    scale=scale,
                    dimensions=2,
                    p=p,
                    ax=ax,
                    zorder_line=z_bond,
                    zorder_label=z_label,
                    edge_geometry_sink=edge_geometry_sink,
                )
            drawn.add(id(edge))
        for edge in dangles:
            _draw_dangling_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=2,
                p=p,
                ax=ax,
                scale=scale,
                zorder_line=z_dangling,
                zorder_label=z_label,
                edge_geometry_sink=edge_geometry_sink,
            )
            drawn.add(id(edge))

    if visible_order:
        last_index = len(visible_order) - 1
        z_orphan_line = (
            _ZORDER_LAYER_BASE + last_index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DANGLING
        )
        z_orphan_label = (
            _ZORDER_LAYER_BASE + last_index * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_EDGE_INDEX
        )
    else:
        z_orphan_line = _ZORDER_LAYER_BASE + _ZORDER_LAYER_DANGLING
        z_orphan_label = _ZORDER_LAYER_BASE + _ZORDER_LAYER_EDGE_INDEX
    for edge in graph.edges:
        if edge.kind != "dangling" or id(edge) in drawn:
            continue
        _draw_dangling_edge(
            plotter=plotter,
            edge=edge,
            graph=graph,
            positions=positions,
            directions=directions,
            show_index_labels=show_index_labels,
            config=config,
            dimensions=2,
            p=p,
            ax=ax,
            scale=scale,
            zorder_line=z_orphan_line,
            zorder_label=z_orphan_label,
            edge_geometry_sink=edge_geometry_sink,
        )
        drawn.add(id(edge))


def _draw_edges(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    edge_geometry_sink: list[_RenderedEdgeGeometry] | None = None,
) -> None:
    for edge in graph.edges:
        if edge.kind == "dangling":
            _draw_dangling_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=dimensions,
                p=p,
                ax=ax,
                scale=scale,
                edge_geometry_sink=edge_geometry_sink,
            )
        elif edge.kind == "self":
            _draw_self_loop_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=dimensions,
                p=p,
                ax=ax,
                scale=scale,
                edge_geometry_sink=edge_geometry_sink,
            )
        else:
            _draw_contraction_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                contraction_groups=contraction_groups,
                show_index_labels=show_index_labels,
                config=config,
                scale=scale,
                dimensions=dimensions,
                p=p,
                ax=ax,
                edge_geometry_sink=edge_geometry_sink,
            )


__all__ = [
    "_draw_edges",
    "_draw_edges_2d_layered",
]
