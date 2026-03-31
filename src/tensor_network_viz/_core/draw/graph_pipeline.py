from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from ...contraction_viewer import attach_playback_to_tensor_network_figure
from ...einsum_module.contraction_cost import format_contraction_step_tooltip
from ..contractions import _ContractionGroups, _group_contractions
from ..graph import (
    _GraphData,
)
from ..layout import (
    AxisDirections,
    NodePositions,
)
from .constants import *
from .contraction_scheme import (
    _contraction_step_metrics_for_draw,
    _draw_contraction_scheme,
    _effective_contraction_steps,
)
from .disk_metrics import _tensor_disk_radius_px_3d_nominal
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
from .plotter import _make_plotter, _node_edge_degrees
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
    contraction_groups: _ContractionGroups | None = None,
    bond_curve_pad: float | None = None,
) -> None:
    _disconnect_tensor_network_hover(ax.figure)
    ax.cla()
    if contraction_groups is None:
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
        graph,
        positions,
        params,
        scale,
        contraction_groups,
        bond_curve_pad=bond_curve_pad,
    )
    _apply_axis_limits_with_outset(ax, pre_coords, view_margin=view_margin, dimensions=dimensions)

    if config.contraction_playback and not config.show_contraction_scheme:
        raise ValueError(
            "contraction_playback=True requires show_contraction_scheme=True so contraction "
            "steps can be drawn and stepped."
        )

    # Contraction highlights first (under bonds, nodes, and labels).
    per_step_artists: list[Any | None] | None = None
    scheme_steps_eff: tuple[frozenset[int], ...] | None = None
    scheme_aabb: (
        list[tuple[float, float, float, float, float, float] | None] | None
    ) = None
    if config.show_contraction_scheme:
        scheme_steps_eff = _effective_contraction_steps(graph, config)
        if scheme_steps_eff:
            per_step_artists, scheme_aabb = _draw_contraction_scheme(
                ax=ax,
                graph=graph,
                positions=positions,
                steps=scheme_steps_eff,
                config=config,
                dimensions=dimensions,
                scale=scale,
                p=params,
            )
            if config.contraction_playback:
                if not any(a is not None for a in per_step_artists):
                    raise ValueError(
                        "contraction_playback requires at least one drawable "
                        "contraction scheme step."
                    )
                attach_playback_to_tensor_network_figure(
                    artists_by_step=per_step_artists,
                    fig=ax.figure,
                    ax=ax,
                    config=config,
                )
        elif config.contraction_playback:
            raise ValueError(
                "contraction_playback requires a non-empty contraction step sequence on the graph."
            )

    visible_order = _visible_node_ids_in_graph_order(graph)
    node_degrees = _node_edge_degrees(graph)
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
        draw_one = plotter.draw_tensor_node
        tensor_z_by_node = {
            nid: float(_ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_TENSOR_NAME)
            for i, nid in enumerate(visible_order)
        }
        for i, nid in enumerate(visible_order):
            zdisk = float(_ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DISK)
            coord = np.asarray(positions[nid], dtype=float)
            deg_one = node_degrees.get(int(nid), 0) == 1
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
            visible_node_ids=visible_order,
            node_degrees=node_degrees,
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
            visible_node_ids=visible_order,
            node_degrees=node_degrees,
        )
    plotter.style_axes(coords, view_margin=view_margin)
    tensor_disk_radius_px_3d: float | None = None
    if dimensions == 3 and config.approximate_3d_tensor_disk_px:
        tensor_disk_radius_px_3d = _tensor_disk_radius_px_3d_nominal(ax, params)
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
        visible_draw_order=visible_order,
        tensor_label_zorder_by_node=tensor_z_by_node,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )
    if config.refine_tensor_labels:
        _refit_tensor_labels_to_disks(
            ax=ax,
            p=params,
            dimensions=dimensions,
            tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
        )
    if dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, ax))

    metrics_row = (
        _contraction_step_metrics_for_draw(graph, scheme_steps_eff)
        if scheme_steps_eff
        else None
    )
    scheme_patches_2d: list[tuple[Any, str]] = []
    scheme_aabbs_3d: list[
        tuple[tuple[float, float, float, float, float, float], str, Any]
    ] = []
    if (
        config.contraction_scheme_cost_hover
        and metrics_row is not None
        and per_step_artists is not None
        and scheme_aabb is not None
    ):
        for i, art in enumerate(per_step_artists):
            if i >= len(metrics_row):
                break
            m = metrics_row[i]
            if m is None or art is None:
                continue
            txt = format_contraction_step_tooltip(m)
            if dimensions == 2:
                scheme_patches_2d.append((art, txt))
            else:
                box = scheme_aabb[i] if i < len(scheme_aabb) else None
                if box is None:
                    continue
                scheme_aabbs_3d.append((box, txt, art))

    want_label_hover = config.hover_labels and (
        show_tensor_labels or show_index_labels
    )
    want_scheme_hover = bool(scheme_patches_2d) or bool(scheme_aabbs_3d)
    if want_label_hover or want_scheme_hover:
        vis_ids = visible_order
        if dimensions == 2:
            node_colls = getattr(plotter, "_node_disk_collections", None)
            node_coll = (
                node_colls
                if (isinstance(node_colls, list) and len(node_colls) > 0)
                else getattr(plotter, "_node_disk_collection", None)
            )
            if not want_label_hover:
                node_coll = None
            _register_2d_hover_labels(
                cast(Axes, ax),
                node_patch_coll=node_coll if show_tensor_labels else None,
                visible_node_ids=vis_ids,
                tensor_hover=tensor_hover_map or {},
                edge_hover=list(hover_edge_list or ()),
                line_width_px_hint=float(params.lw),
                scheme_hover_patches=scheme_patches_2d,
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
                tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
                scheme_hover_aabbs=scheme_aabbs_3d,
            )


__all__ = ["_draw_graph"]
