from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from .._label_format import format_tensor_node_label
from ..curves import _ellipse_points, _ellipse_points_3d, _require_self_endpoints
from ..graph import _EdgeData, _endpoint_index_caption, _GraphData
from ..layout import AxisDirections, NodePositions, _orthogonal_unit
from .constants import (
    _EDGE_INDEX_LABEL_ALONG_FRAC,
    _PHYS_DANGLING_2D_FRAC_FROM_TIP,
)
from .fonts_and_scale import _DrawScaleParams
from .label_descriptors import _TextLabelDescriptor
from .labels_misc import (
    _curve_index_outside_disk,
    _dangling_hover_label_text,
    _edge_index_text_kwargs,
    _node_label_clearance,
    _self_loop_hover_label_text,
)
from .plotter import _PlotAdapter
from .scene_state import _RenderedEdgeGeometry
from .vectors import _perpendicular_2d
from .viewport_geometry import (
    _blend_bond_tangent_with_chord_2d,
    _blend_bond_tangent_with_chord_3d,
    _bond_index_label_perp_offset,
    _contraction_edge_index_label_2d_placement,
    _contraction_edge_index_label_3d_placement,
    _edge_index_along_bond_text_kw,
    _edge_index_fontsize_for_bond,
    _point_tangent_along_polyline_from_end,
    _point_tangent_along_polyline_from_start,
    _polyline_arc_length_total,
)

_ZORDER_FLAT_DANGLE_SELF_LINE: float = 2.0


def _draw_dangling_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    zorder_line: float | None = None,
    zorder_label: float | None = None,
    edge_geometry_sink: list[_RenderedEdgeGeometry] | None = None,
) -> None:
    endpoint = edge.endpoints[0]
    direction = directions[(endpoint.node_id, endpoint.axis_index)]
    center = positions[endpoint.node_id]
    use_center_anchor = dimensions == 3 or graph.nodes[endpoint.node_id].is_virtual
    if dimensions == 2 and not config.show_nodes and not graph.nodes[endpoint.node_id].is_virtual:
        start = center
        end = center + direction * p.stub
    elif use_center_anchor:
        start = center
        end = center + direction * (p.r + p.stub)
    else:
        start = center + direction * p.r
        end = start + direction * p.stub
    zorder_stub = (
        float(_ZORDER_FLAT_DANGLE_SELF_LINE) if zorder_line is None else float(zorder_line)
    )
    plotter.plot_line(
        start,
        end,
        color=config.dangling_edge_color,
        linewidth=p.lw,
        zorder=zorder_stub,
    )
    if edge_geometry_sink is not None:
        stub_segment = np.stack(
            [np.asarray(start, dtype=float), np.asarray(end, dtype=float)],
            axis=0,
        )
        edge_geometry_sink.append(
            _RenderedEdgeGeometry(
                edge=edge,
                polyline=np.asarray(stub_segment[:, :dimensions], dtype=float, order="C"),
            )
        )
    _draw_dangling_edge_labels(
        plotter=plotter,
        edge=edge,
        graph=graph,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        show_index_labels=show_index_labels,
        config=config,
        dimensions=dimensions,
        p=p,
        ax=ax,
        scale=scale,
        zorder_label=zorder_label,
    )


def _draw_dangling_edge_labels(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    start: np.ndarray,
    end: np.ndarray,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    zorder_label: float | None = None,
    label_sink: list[_TextLabelDescriptor] | None = None,
) -> None:
    if not edge.label and not config.hover_labels:
        return
    hover_targets = getattr(plotter, "_hover_edge_targets", None)
    if config.hover_labels and hover_targets is not None:
        caption = _dangling_hover_label_text(edge)
        if caption:
            if dimensions == 2:
                hover_targets.append(
                    (
                        np.stack(
                            [
                                np.asarray(start[:2], dtype=float),
                                np.asarray(end[:2], dtype=float),
                            ],
                            axis=0,
                        ),
                        caption,
                    )
                )
            else:
                start_3d = np.zeros(3, dtype=float)
                end_3d = np.zeros(3, dtype=float)
                start_arr = np.asarray(start, dtype=float).reshape(-1)
                end_arr = np.asarray(end, dtype=float).reshape(-1)
                start_3d[: min(3, start_arr.size)] = start_arr[: min(3, start_arr.size)]
                end_3d[: min(3, end_arr.size)] = end_arr[: min(3, end_arr.size)]
                hover_targets.append((np.stack([start_3d, end_3d], axis=0), caption))
    if not (show_index_labels and edge.label):
        return

    raw_label = edge.label
    stub_segment = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)], axis=0)
    stub_length = _polyline_arc_length_total(stub_segment)
    distance_from_tip = float(_PHYS_DANGLING_2D_FRAC_FROM_TIP) * stub_length
    point, tangent = _point_tangent_along_polyline_from_end(stub_segment, distance_from_tip)

    fontsize = _edge_index_fontsize_for_bond(
        raw_label,
        bond_start=start,
        bond_end=end,
        ax=ax,
        dimensions=dimensions,
        is_physical=True,
    )
    text_kwargs = _edge_index_text_kwargs(
        config,
        fontsize=fontsize,
        stub_kind="dangling",
        bbox_pad=p.index_bbox_pad,
        zorder=zorder_label,
    )
    if dimensions == 2:
        start_2d = np.asarray(start[:2], dtype=float)
        end_2d = np.asarray(end[:2], dtype=float)
        tangent_2d = np.asarray(tangent[:2], dtype=float)
        tangent_blend = _blend_bond_tangent_with_chord_2d(tangent_2d, start_2d, end_2d)
        label_pos, align_kwargs = _contraction_edge_index_label_2d_placement(
            Q=point,
            t_geom_2d=tangent_2d,
            t_align_2d=tangent_blend,
            text_ep="left",
            p=p,
            ax=cast(Axes, ax),
            scale=scale,
            fontsize_pt=float(fontsize),
        )
    else:
        start_3d = np.asarray(start, dtype=float).reshape(3)
        end_3d = np.asarray(end, dtype=float).reshape(3)
        tangent_blend_3d = _blend_bond_tangent_with_chord_3d(
            np.asarray(tangent, dtype=float).reshape(3),
            start_3d,
            end_3d,
        )
        label_pos, align_kwargs = _contraction_edge_index_label_3d_placement(
            Q=point,
            t_geom_3d=np.asarray(tangent, dtype=float).reshape(3),
            t_align_3d=tangent_blend_3d,
            text_ep="left",
            p=p,
            ax=ax,
            scale=scale,
            fontsize_pt=float(fontsize),
        )
    formatted = format_tensor_node_label(raw_label)
    kwargs = {**text_kwargs, **align_kwargs}
    if label_sink is not None:
        label_sink.append(
            _TextLabelDescriptor(
                position=np.asarray(label_pos, dtype=float).copy(),
                text=formatted,
                kwargs=dict(kwargs),
            )
        )
        return
    plotter.plot_text(label_pos, formatted, **kwargs)


def _draw_self_loop_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    zorder_line: float | None = None,
    zorder_label: float | None = None,
    edge_geometry_sink: list[_RenderedEdgeGeometry] | None = None,
) -> None:
    endpoint_a, endpoint_b = _require_self_endpoints(edge)
    direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
    direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
    orientation = direction_a + direction_b
    if np.linalg.norm(orientation) < 1e-6:
        orientation = (
            np.array([1.0, 0.0, 0.0], dtype=float)
            if dimensions == 3
            else np.array([1.0, 0.0], dtype=float)
        )
    orientation = orientation / np.linalg.norm(orientation)
    center_pt = positions[endpoint_a.node_id] + orientation * (p.r + p.loop_r)
    if dimensions == 2:
        normal = _perpendicular_2d(orientation)
        curve = _ellipse_points(
            center_pt,
            orientation,
            normal,
            width=p.ellipse_w,
            height=p.ellipse_h,
        )
    else:
        normal = _orthogonal_unit(orientation)
        binormal = np.cross(orientation, normal)
        binormal = binormal / np.linalg.norm(binormal)
        curve = _ellipse_points_3d(
            center_pt,
            normal,
            binormal,
            width=p.ellipse_w,
            height=p.ellipse_h,
        )
    zorder_loop = (
        float(_ZORDER_FLAT_DANGLE_SELF_LINE) if zorder_line is None else float(zorder_line)
    )
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=zorder_loop)
    if edge_geometry_sink is not None:
        edge_geometry_sink.append(
            _RenderedEdgeGeometry(
                edge=edge,
                polyline=np.asarray(curve[:, :dimensions], dtype=float, order="C"),
            )
        )
    _draw_self_loop_edge_labels(
        plotter=plotter,
        edge=edge,
        graph=graph,
        curve=np.asarray(curve, dtype=float),
        positions=positions,
        directions=directions,
        show_index_labels=show_index_labels,
        config=config,
        dimensions=dimensions,
        p=p,
        ax=ax,
        scale=scale,
        zorder_label=zorder_label,
    )


def _draw_self_loop_edge_labels(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    curve: np.ndarray,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    zorder_label: float | None = None,
    label_sink: list[_TextLabelDescriptor] | None = None,
) -> None:
    hover_targets = getattr(plotter, "_hover_edge_targets", None)
    if config.hover_labels and hover_targets is not None:
        caption = _self_loop_hover_label_text(edge, graph)
        if caption:
            if dimensions == 2:
                hover_targets.append((np.asarray(curve[:, :2], dtype=float, order="C"), caption))
            else:
                hover_targets.append((np.asarray(curve[:, :3], dtype=float, order="C"), caption))
    if not show_index_labels:
        return

    endpoint_a, endpoint_b = _require_self_endpoints(edge)
    direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
    direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
    orientation = direction_a + direction_b
    if np.linalg.norm(orientation) < 1e-6:
        orientation = (
            np.array([1.0, 0.0, 0.0], dtype=float)
            if dimensions == 3
            else np.array([1.0, 0.0], dtype=float)
        )
    orientation = orientation / np.linalg.norm(orientation)
    if dimensions == 2:
        label_offset_dir = _perpendicular_2d(orientation) * p.ellipse_w
    else:
        normal = _orthogonal_unit(orientation)
        binormal = np.cross(orientation, normal)
        binormal = binormal / np.linalg.norm(binormal)
        label_offset_dir = binormal * p.ellipse_w
    caption_a = _endpoint_index_caption(endpoint_a, edge, graph)
    caption_b = _endpoint_index_caption(endpoint_b, edge, graph)
    count = int(curve.shape[0])
    node_center = positions[endpoint_a.node_id]
    clearance = _node_label_clearance(p)
    idx_a = _curve_index_outside_disk(curve, node_center, clearance, from_start=True)
    idx_b = _curve_index_outside_disk(curve, node_center, clearance, from_start=False)
    min_sep = max(2, count // 8)
    if abs(idx_a - idx_b) < min_sep:
        idx_b = min(count - 2, idx_a + min_sep)
    if idx_b <= idx_a:
        idx_a = max(1, idx_b - min_sep)
    direction_norm = float(np.linalg.norm(label_offset_dir))
    direction_unit = label_offset_dir / max(direction_norm, 1e-9)
    loop_span_a = np.asarray(curve[idx_a], dtype=float)
    loop_span_b = np.asarray(curve[idx_b], dtype=float)
    sub_loop = np.asarray(curve[idx_a : idx_b + 1], dtype=float)
    loop_length = _polyline_arc_length_total(sub_loop)
    along_loop = float(_EDGE_INDEX_LABEL_ALONG_FRAC) * loop_length
    q_a, tangent_a = _point_tangent_along_polyline_from_start(sub_loop, along_loop)
    q_b, tangent_b = _point_tangent_along_polyline_from_end(sub_loop, along_loop)
    peer_caps: tuple[str, ...] = tuple(
        c for c in (caption_a, caption_b) if c and format_tensor_node_label(c).strip()
    )
    peer_width: tuple[str, ...] | None = peer_caps if len(peer_caps) > 1 else None
    world_perp = direction_unit if dimensions == 3 else None
    if caption_a:
        offset_a = direction_unit * _bond_index_label_perp_offset(
            caption_a,
            p=p,
            scale=scale,
            dimensions=dimensions,
            ax=ax,
            anchor=q_a,
            world_perp_dir=world_perp,
        )
        fontsize_a = _edge_index_fontsize_for_bond(
            caption_a,
            bond_start=loop_span_a,
            bond_end=loop_span_b,
            ax=ax,
            dimensions=dimensions,
            is_physical=False,
            peer_captions_for_width=peer_width,
        )
        text_kwargs_a = {
            **_edge_index_text_kwargs(
                config,
                fontsize=fontsize_a,
                bbox_pad=p.index_bbox_pad,
                zorder=zorder_label,
            ),
            **_edge_index_along_bond_text_kw(
                endpoint="left",
                tangent=tangent_a,
                ax=ax,
                dimensions=dimensions,
            ),
        }
        position_a = np.asarray(q_a, dtype=float) + offset_a
        formatted_a = format_tensor_node_label(caption_a)
        if label_sink is not None:
            label_sink.append(
                _TextLabelDescriptor(
                    position=np.asarray(position_a, dtype=float).copy(),
                    text=formatted_a,
                    kwargs=dict(text_kwargs_a),
                )
            )
        else:
            plotter.plot_text(
                position_a,
                formatted_a,
                **text_kwargs_a,
            )
    if caption_b:
        offset_b = (
            -direction_unit
            * _bond_index_label_perp_offset(
                caption_b,
                p=p,
                scale=scale,
                dimensions=dimensions,
                ax=ax,
                anchor=q_b,
                world_perp_dir=world_perp,
            )
            * 0.88
        )
        fontsize_b = _edge_index_fontsize_for_bond(
            caption_b,
            bond_start=loop_span_a,
            bond_end=loop_span_b,
            ax=ax,
            dimensions=dimensions,
            is_physical=False,
            peer_captions_for_width=peer_width,
        )
        text_kwargs_b = {
            **_edge_index_text_kwargs(
                config,
                fontsize=fontsize_b,
                bbox_pad=p.index_bbox_pad,
                zorder=zorder_label,
            ),
            **_edge_index_along_bond_text_kw(
                endpoint="right",
                tangent=tangent_b,
                ax=ax,
                dimensions=dimensions,
            ),
        }
        formatted_b = format_tensor_node_label(caption_b)
        position_b = np.asarray(q_b, dtype=float) + offset_b
        if label_sink is not None:
            label_sink.append(
                _TextLabelDescriptor(
                    position=np.asarray(position_b, dtype=float).copy(),
                    text=formatted_b,
                    kwargs=dict(text_kwargs_b),
                )
            )
            return
        plotter.plot_text(
            position_b,
            formatted_b,
            **text_kwargs_b,
        )


__all__ = [
    "_draw_dangling_edge",
    "_draw_dangling_edge_labels",
    "_draw_self_loop_edge",
    "_draw_self_loop_edge_labels",
]
