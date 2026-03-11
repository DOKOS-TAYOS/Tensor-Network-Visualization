"""Shared scale and style parameters for 2D and 3D drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import PlotConfig
from .curves import (
    _ellipse_points,
    _ellipse_points_3d,
    _quadratic_curve,
    _require_self_endpoints,
)
from .graph import _EdgeData, _GraphData
from .layout import AxisDirections, NodePositions, _orthogonal_unit

_CURVE_OFFSET_FACTOR: float = 0.18


class _PlotAdapter(Protocol):
    """Protocol for dimension-aware plotting (2D vs 3D)."""

    def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None: ...
    def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None: ...
    def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None: ...
    def scatter(self, coords: np.ndarray, **kwargs: Any) -> None: ...
    def style_axes(self, coords: np.ndarray) -> None: ...


def _make_2d_plotter(ax: Any) -> _PlotAdapter:
    """Create a 2D plot adapter."""

    class _2DPlotter:
        def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
            ax.plot([start[0], end[0]], [start[1], end[1]], **kwargs)

        def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
            ax.plot(curve[:, 0], curve[:, 1], **kwargs)

        def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
            ax.text(pos[0], pos[1], text, **kwargs)

        def scatter(self, coords: np.ndarray, **kwargs: Any) -> None:
            ax.scatter(coords[:, 0], coords[:, 1], **kwargs)

        def style_axes(self, coords: np.ndarray) -> None:
            span = np.ptp(coords, axis=0)
            span = np.maximum(span, 1.0)
            center = coords.mean(axis=0)
            ax.set_xlim(center[0] - span[0] * 0.9, center[0] + span[0] * 0.9)
            ax.set_ylim(center[1] - span[1] * 0.9, center[1] + span[1] * 0.9)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()

    return _2DPlotter()


def _make_3d_plotter(ax: Any) -> _PlotAdapter:
    """Create a 3D plot adapter."""

    class _3DPlotter:
        def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)

        def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], **kwargs)

        def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
            ax.text(pos[0], pos[1], pos[2], text, **kwargs)

        def scatter(self, coords: np.ndarray, **kwargs: Any) -> None:
            kwargs.setdefault("depthshade", False)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **kwargs)

        def style_axes(self, coords: np.ndarray) -> None:
            span = np.ptp(coords, axis=0)
            span = np.maximum(span, 1.0)
            center = coords.mean(axis=0)
            ax.set_xlim(center[0] - span[0] * 0.9, center[0] + span[0] * 0.9)
            ax.set_ylim(center[1] - span[1] * 0.9, center[1] + span[1] * 0.9)
            ax.set_zlim(center[2] - span[2] * 0.9, center[2] + span[2] * 0.9)
            ax.set_box_aspect(span)
            ax.set_axis_off()

    return _3DPlotter()


def _perpendicular_2d(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]], dtype=float)


def _perpendicular_3d(direction: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    perp = np.cross(direction, reference)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
    return perp / np.linalg.norm(perp)


def _curved_edge_points(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    dimensions: int,
    scale: float = 1.0,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    perpendicular = (
        _perpendicular_3d(direction) if dimensions == 3 else _perpendicular_2d(direction)
    )
    offset = (offset_index - (edge_count - 1) / 2.0) * _CURVE_OFFSET_FACTOR * scale * distance
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


def _draw_edges(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    pair_groups: dict[tuple[int, int], list[_EdgeData]],
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    dimensions: int,
    p: _DrawScaleParams,
) -> None:
    for edge in graph.edges:
        if edge.kind == "dangling":
            endpoint = edge.endpoints[0]
            direction = directions[(endpoint.node_id, endpoint.axis_index)]
            start = positions[endpoint.node_id] + direction * p.r
            end = start + direction * p.stub
            plotter.plot_line(
                start, end, color=config.dangling_edge_color, linewidth=p.lw, zorder=2
            )
            if show_index_labels and edge.label:
                label_pos = end + direction * p.label_offset
                plotter.plot_text(
                    label_pos,
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
                orientation = (
                    np.array([1.0, 0.0, 0.0], dtype=float)
                    if dimensions == 3
                    else np.array([1.0, 0.0], dtype=float)
                )
            orientation = orientation / np.linalg.norm(orientation)
            if dimensions == 2:
                normal = _perpendicular_2d(orientation)
                center_pt = positions[endpoint_a.node_id] + orientation * (p.r + p.loop_r)
                curve = _ellipse_points(
                    center_pt, orientation, normal,
                    width=p.ellipse_w, height=p.ellipse_h,
                )
                label_offset_dir = normal * p.ellipse_w
            else:
                normal = _orthogonal_unit(orientation)
                binormal = np.cross(orientation, normal)
                binormal = binormal / np.linalg.norm(binormal)
                center_pt = positions[endpoint_a.node_id] + orientation * (p.r + p.loop_r)
                curve = _ellipse_points_3d(
                    center_pt, normal, binormal,
                    width=p.ellipse_w, height=p.ellipse_h,
                )
                label_offset_dir = binormal * p.ellipse_w
            plotter.plot_curve(
                curve, color=config.bond_edge_color, linewidth=p.lw, zorder=2
            )
            if show_index_labels and edge.label:
                label_pos = center_pt + label_offset_dir
                plotter.plot_text(
                    label_pos,
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
            curve = _curved_edge_points(
                start=positions[edge.node_ids[0]],
                end=positions[edge.node_ids[1]],
                offset_index=offset_index,
                edge_count=len(group),
                dimensions=dimensions,
                scale=scale,
            )
            plotter.plot_curve(
                curve, color=config.bond_edge_color, linewidth=p.lw, zorder=1
            )
            if show_index_labels and edge.label:
                midpoint = curve[len(curve) // 2]
                delta = positions[edge.node_ids[1]] - positions[edge.node_ids[0]]
                dist = max(float(np.linalg.norm(delta)), 1e-6)
                direction = delta / dist
                perpendicular = (
                    _perpendicular_3d(direction)
                    if dimensions == 3
                    else _perpendicular_2d(direction)
                )
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                if (dimensions == 2 and perpendicular[1] < 0) or (
                    dimensions == 3 and perpendicular[2] < 0
                ):
                    perpendicular = -perpendicular
                label_pos = midpoint + perpendicular * p.label_offset
                plotter.plot_text(
                    label_pos,
                    edge.label,
                    color=config.label_color,
                    fontsize=p.font_bond,
                    zorder=5,
                    ha="center",
                    va="bottom",
                )


def _draw_nodes(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    config: PlotConfig,
    p: _DrawScaleParams,
) -> np.ndarray:
    visible_node_ids = [
        node_id for node_id, node in graph.nodes.items() if not node.is_virtual
    ]
    if visible_node_ids:
        coords = np.stack([positions[node_id] for node_id in visible_node_ids])
        plotter.scatter(
            coords,
            s=p.scatter_s,
            c=config.node_color,
            edgecolors=config.node_edge_color,
            linewidths=p.lw,
            zorder=3,
        )
    else:
        coords = np.stack(list(positions.values()))
    return coords


def _draw_labels(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    p: _DrawScaleParams,
) -> None:
    if show_tensor_labels:
        for node_id, node in graph.nodes.items():
            if node.is_virtual:
                continue
            pos = positions[node_id]
            plotter.plot_text(
                pos,
                node.name,
                color=config.tensor_label_color,
                ha="center",
                va="center",
                fontsize=p.font_node,
                zorder=4,
            )
    if show_index_labels:
        for node_id, node in graph.nodes.items():
            if not node.is_virtual or not node.label:
                continue
            pos = positions[node_id]
            plotter.plot_text(
                pos,
                node.label,
                color=config.label_color,
                fontsize=p.font_bond,
                zorder=5,
                ha="center",
                va="bottom",
            )


@dataclass(frozen=True)
class _DrawScaleParams:
    """Resolved scale-dependent parameters for drawing."""

    r: float
    stub: float
    loop_r: float
    lw: float
    font_dangling: int
    font_bond: int
    font_node: int
    label_offset: float
    ellipse_w: float
    ellipse_h: float
    scatter_s: float


def _draw_scale_params(config: PlotConfig, scale: float, *, is_3d: bool) -> _DrawScaleParams:
    """Compute scale-dependent drawing parameters from config."""
    r = (
        config.node_radius if config.node_radius is not None else PlotConfig.DEFAULT_NODE_RADIUS
    ) * scale
    stub = (
        config.stub_length if config.stub_length is not None else PlotConfig.DEFAULT_STUB_LENGTH
    ) * scale
    loop_r = (
        config.self_loop_radius
        if config.self_loop_radius is not None
        else PlotConfig.DEFAULT_SELF_LOOP_RADIUS
    ) * scale
    lw_default = PlotConfig.DEFAULT_LINE_WIDTH_3D if is_3d else PlotConfig.DEFAULT_LINE_WIDTH_2D
    lw_attr = config.line_width_3d if is_3d else config.line_width_2d
    lw = (lw_attr if lw_attr is not None else lw_default) * scale
    scatter_s = (120 if is_3d else 900) * (scale**2)

    return _DrawScaleParams(
        r=r,
        stub=stub,
        loop_r=loop_r,
        lw=lw,
        font_dangling=max(7, round(9 * scale)),
        font_bond=max(5, round(5 * scale)),
        font_node=max(8, round(10 * scale)),
        label_offset=0.08 * scale,
        ellipse_w=0.16 * scale,
        ellipse_h=0.12 * scale,
        scatter_s=scatter_s,
    )
