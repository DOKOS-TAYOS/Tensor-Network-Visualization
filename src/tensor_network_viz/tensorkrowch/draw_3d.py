"""3D drawing for tensor networks."""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from ._draw_common import _draw_scale_params
from .curves import (
    _ellipse_points_3d,
    _group_contractions,
    _quadratic_curve,
    _require_self_endpoints,
)
from .graph import _GraphData
from .layout import AxisDirections, NodePositions, _orthogonal_unit


def _curved_edge_points_3d(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    scale: float = 1.0,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    perpendicular = np.cross(direction, reference)
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    offset = (offset_index - (edge_count - 1) / 2.0) * 0.18 * scale * distance
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


def _draw_3d(
    *,
    ax: Axes3D,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float = 1.0,
) -> None:
    ax.cla()
    pair_groups = _group_contractions(graph)
    p = _draw_scale_params(config, scale, is_3d=True)

    for edge in graph.edges:
        if edge.kind == "dangling":
            endpoint = edge.endpoints[0]
            direction = directions[(endpoint.node_id, endpoint.axis_index)]
            start = positions[endpoint.node_id] + direction * p.r
            end = start + direction * p.stub
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=config.dangling_edge_color,
                linewidth=p.lw,
                zorder=2,
            )
            if show_index_labels and edge.label:
                label_pos = end + direction * p.label_offset
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    edge.label,
                    color=config.label_color,
                    fontsize=p.font_dangling,
                    zorder=5,
                    ha="center",
                    va="bottom",
                )
        elif edge.kind == "self":
            endpoint_a, endpoint_b = _require_self_endpoints(edge)
            direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
            direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
            orientation = direction_a + direction_b
            if np.linalg.norm(orientation) < 1e-6:
                orientation = np.array([1.0, 0.0, 0.0], dtype=float)
            orientation = orientation / np.linalg.norm(orientation)
            normal = _orthogonal_unit(orientation)
            binormal = np.cross(orientation, normal)
            binormal = binormal / np.linalg.norm(binormal)
            center_pt = (
                positions[endpoint_a.node_id]
                + orientation * (p.r + p.loop_r)
            )
            curve = _ellipse_points_3d(
                center_pt, normal, binormal, width=p.ellipse_w, height=p.ellipse_h
            )
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color=config.bond_edge_color,
                linewidth=p.lw,
                zorder=2,
            )
            if show_index_labels and edge.label:
                label_pos = center_pt + binormal * p.ellipse_w
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    edge.label,
                    color=config.label_color,
                    fontsize=p.font_dangling,
                    zorder=5,
                    ha="center",
                    va="bottom",
                )
        else:
            key = tuple(sorted(edge.node_ids))
            group = pair_groups[key]
            offset_index = group.index(edge)
            curve = _curved_edge_points_3d(
                start=positions[edge.node_ids[0]],
                end=positions[edge.node_ids[1]],
                offset_index=offset_index,
                edge_count=len(group),
                scale=scale,
            )
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color=config.bond_edge_color,
                linewidth=p.lw,
                zorder=1,
            )
            if show_index_labels and edge.label:
                midpoint = curve[len(curve) // 2]
                delta = positions[edge.node_ids[1]] - positions[edge.node_ids[0]]
                dist = max(float(np.linalg.norm(delta)), 1e-6)
                direction = delta / dist
                perpendicular = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=float))
                if np.linalg.norm(perpendicular) < 1e-6:
                    perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                if perpendicular[2] < 0:
                    perpendicular = -perpendicular
                label_pos = midpoint + perpendicular * p.label_offset
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    edge.label,
                    color=config.label_color,
                    fontsize=p.font_bond,
                    zorder=5,
                    ha="center",
                    va="bottom",
                )

    coords = np.stack(list(positions.values()))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=p.scatter_s,
        c=config.node_color,
        edgecolors=config.node_edge_color,
        linewidths=p.lw,
        depthshade=False,
    )

    if show_tensor_labels:
        for node_id, node in graph.nodes.items():
            x, y, z = positions[node_id]
            ax.text(
                x, y, z,
                node.name,
                color=config.tensor_label_color,
                fontsize=p.font_node,
                ha="center",
                va="center",
                zorder=5,
            )

    _style_3d_axes(ax, coords)


def _style_3d_axes(ax: Axes3D, coords: np.ndarray) -> None:
    span = np.ptp(coords, axis=0)
    span = np.maximum(span, 1.0)
    center = coords.mean(axis=0)
    ax.set_xlim(center[0] - span[0] * 0.9, center[0] + span[0] * 0.9)
    ax.set_ylim(center[1] - span[1] * 0.9, center[1] + span[1] * 0.9)
    ax.set_zlim(center[2] - span[2] * 0.9, center[2] + span[2] * 0.9)
    ax.set_box_aspect(span)
    ax.set_axis_off()
