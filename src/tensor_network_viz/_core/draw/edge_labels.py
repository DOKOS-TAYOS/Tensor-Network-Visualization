from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from .._label_format import format_tensor_node_label
from ..graph import (
    _EdgeData,
    _EdgeEndpoint,
    _endpoint_index_caption,
    _GraphData,
    _require_contraction_endpoints,
)
from ..layout import NodePositions
from .fonts_and_scale import _DrawScaleParams
from .label_descriptors import _TextLabelDescriptor
from .labels_misc import _edge_index_text_kwargs
from .plotter import _PlotAdapter
from .viewport_geometry import (
    _blend_bond_tangent_with_chord_2d,
    _blend_bond_tangent_with_chord_3d,
    _contraction_edge_index_label_2d_placement,
    _contraction_edge_index_label_3d_placement,
    _edge_index_fontsize_for_bond,
    _edge_index_rim_arc_from_endpoint,
    _point_tangent_along_polyline_from_end,
    _point_tangent_along_polyline_from_start,
    _polyline_arc_length_total,
)


def _plot_contraction_index_captions(
    *,
    plotter: _PlotAdapter,
    curve: np.ndarray,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    left_id: int,
    right_id: int,
    config: PlotConfig,
    p: _DrawScaleParams,
    dimensions: Literal[2, 3],
    ax: Any,
    scale: float,
    zorder_label: float | None = None,
    label_sink: list[_TextLabelDescriptor] | None = None,
) -> None:
    ep_l, ep_r = _require_contraction_endpoints(edge)
    cap_l: str | None = _endpoint_index_caption(ep_l, edge, graph)
    cap_r: str | None = _endpoint_index_caption(ep_r, edge, graph)
    if not cap_l and not cap_r:
        return
    curve_matrix = np.asarray(curve, dtype=float)
    total_length = _polyline_arc_length_total(curve_matrix)
    half_length = 0.5 * total_length
    rim_arc = _edge_index_rim_arc_from_endpoint(
        r_global=float(p.r),
        half_polyline_length=half_length,
    )
    slot = 0.5 * (rim_arc + half_length)
    q_left, tangent_left = _point_tangent_along_polyline_from_start(curve_matrix, slot)
    q_right, tangent_right = _point_tangent_along_polyline_from_end(curve_matrix, slot)
    bond_start = np.asarray(positions[left_id], dtype=float)
    bond_end = np.asarray(positions[right_id], dtype=float)
    bond_start_2d = np.asarray(bond_start[:2], dtype=float)
    bond_end_2d = np.asarray(bond_end[:2], dtype=float)
    caption_pairs: tuple[
        tuple[_EdgeEndpoint, str | None, np.ndarray, np.ndarray, Literal["left", "right"]],
        tuple[_EdgeEndpoint, str | None, np.ndarray, np.ndarray, Literal["left", "right"]],
    ] = (
        (ep_l, cap_l, q_left, tangent_left, "left"),
        (ep_r, cap_r, q_right, tangent_right, "right"),
    )
    peer_caps: tuple[str, ...] = tuple(
        c for c in (cap_l, cap_r) if c and format_tensor_node_label(c).strip()
    )
    peer_for_width: tuple[str, ...] | None = peer_caps if len(peer_caps) > 1 else None
    for _, cap, point, tangent_forward, text_endpoint in caption_pairs:
        if not cap:
            continue
        fontsize = _edge_index_fontsize_for_bond(
            cap,
            bond_start=bond_start,
            bond_end=bond_end,
            ax=ax,
            dimensions=dimensions,
            is_physical=False,
            peer_captions_for_width=peer_for_width,
            preferred_fontsize_pt=config.edge_label_fontsize,
            fast_text_metrics=bool(p.fast_text_metrics),
        )
        text_kwargs = _edge_index_text_kwargs(
            config,
            fontsize=fontsize,
            stub_kind="bond",
            bbox_pad=p.index_bbox_pad,
            zorder=zorder_label,
        )
        if dimensions == 2:
            tangent_2d = np.asarray(tangent_forward[:2], dtype=float)
            tangent_blend = _blend_bond_tangent_with_chord_2d(
                tangent_2d,
                bond_start_2d,
                bond_end_2d,
            )
            position, align_kwargs = _contraction_edge_index_label_2d_placement(
                Q=point,
                t_geom_2d=tangent_2d,
                t_align_2d=tangent_blend,
                text_ep=text_endpoint,
                p=p,
                ax=cast(Axes, ax),
                scale=scale,
                fontsize_pt=float(fontsize),
            )
        else:
            tangent_blend_3d = _blend_bond_tangent_with_chord_3d(
                np.asarray(tangent_forward, dtype=float).reshape(3),
                bond_start,
                bond_end,
            )
            position, align_kwargs = _contraction_edge_index_label_3d_placement(
                Q=np.asarray(point, dtype=float),
                t_geom_3d=np.asarray(tangent_forward, dtype=float).reshape(3),
                t_align_3d=tangent_blend_3d,
                text_ep=text_endpoint,
                p=p,
                ax=ax,
                scale=scale,
                fontsize_pt=float(fontsize),
            )
        formatted = format_tensor_node_label(cap)
        kwargs = {**text_kwargs, **align_kwargs}
        if label_sink is not None:
            label_sink.append(
                _TextLabelDescriptor(
                    position=np.asarray(position, dtype=float).copy(),
                    text=formatted,
                    kwargs=dict(kwargs),
                )
            )
            continue
        plotter.plot_text(position, formatted, **kwargs)


__all__ = ["_plot_contraction_index_captions"]
