"""3D free-axis direction assignment."""

from __future__ import annotations

import math

import numpy as np

from ...config import PlotConfig
from ..graph import _GraphData
from ..layout_structure import _component_orthogonal_basis, _LayoutComponent
from .direction_common import (
    _direction_from_axis_name,
    _direction_has_space,
    _is_dangling_leg_axis,
    _orthogonal_unit,
    _used_axis_directions,
)
from .geometry import _dangling_stub_segment_3d, _segment_point_min_distance_sq_3d
from .types import AxisDirections, NodePositions


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    layout_components: tuple[_LayoutComponent, ...],
) -> None:
    assigned_segments: list[tuple[np.ndarray, np.ndarray]] = []
    component_by_node = {
        node_id: component for component in layout_components for node_id in component.node_ids
    }

    for node_id, node in graph.nodes.items():
        component = component_by_node[node_id]
        axis_count = max(node.degree, 1)
        axis, lateral, normal = _component_orthogonal_basis(component, positions)
        candidate_directions = [normal, -normal, lateral, -lateral, axis, -axis]

        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue

            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=3)
            strict_phys = _is_dangling_leg_axis(graph, node_id, axis_index)
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            origin = positions[node_id]
            if (
                named_direction is not None
                and _direction_has_space(named_direction, used_dirs)
                and not _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=named_direction,
                    assigned_segments=assigned_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                )
            ):
                directions[axis_key] = named_direction
                assigned_segments.append((origin.copy(), named_direction.copy()))
                continue

            for direction in candidate_directions:
                if not _direction_has_space(direction, used_dirs):
                    continue
                if _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=direction,
                    assigned_segments=assigned_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                ):
                    continue
                directions[axis_key] = direction
                assigned_segments.append((origin.copy(), direction.copy()))
                break
            else:
                if strict_phys:
                    dirs_try: list[np.ndarray] = []
                    if named_direction is not None:
                        dirs_try.append(np.asarray(named_direction, dtype=float))
                    dirs_try.extend(list(candidate_directions))
                    r_disk = (
                        float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
                    )
                    origin_3d = np.asarray(origin, dtype=float).reshape(-1)[:3]
                    other_ids = [other_id for other_id in positions if other_id != node_id]
                    best_direction: np.ndarray | None = None
                    best_margin = -1e300
                    for raw_direction in dirs_try:
                        direction_vec = np.asarray(raw_direction, dtype=float).reshape(-1)[:3]
                        norm = float(np.linalg.norm(direction_vec))
                        if norm < 1e-9:
                            continue
                        direction_unit = direction_vec / norm
                        start, end = _dangling_stub_segment_3d(
                            origin_3d,
                            direction_unit,
                            draw_scale=draw_scale,
                        )
                        if other_ids:
                            margin = min(
                                math.sqrt(
                                    _segment_point_min_distance_sq_3d(
                                        start,
                                        end,
                                        np.asarray(positions[other_id], dtype=float).reshape(-1)[
                                            :3
                                        ],
                                    )
                                )
                                - r_disk
                                for other_id in other_ids
                            )
                        else:
                            margin = 0.0
                        if margin > best_margin:
                            best_margin = margin
                            best_direction = direction_unit
                    if best_direction is not None:
                        directions[axis_key] = best_direction
                        assigned_segments.append((origin.copy(), best_direction.copy()))
                        continue

                fallback = (
                    named_direction if named_direction is not None else _orthogonal_unit(axis)
                )
                if not _direction_has_space(fallback, used_dirs):
                    fallback = -fallback
                directions[axis_key] = fallback / np.linalg.norm(fallback)
                assigned_segments.append((origin.copy(), directions[axis_key].copy()))


def _direction_conflicts_3d(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_segments: list[tuple[np.ndarray, np.ndarray]],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> bool:
    tip = origin + direction * 0.45

    if strict_physical_node_clearance:
        direction_vec = np.asarray(direction, dtype=float).reshape(-1)[:3]
        norm = float(np.linalg.norm(direction_vec))
        direction_unit = (
            direction_vec / norm if norm >= 1e-9 else np.array([0.0, 0.0, 1.0], dtype=float)
        )
        scale = max(float(draw_scale), 1e-6)
        start, end = _dangling_stub_segment_3d(origin, direction_unit, draw_scale=draw_scale)
        r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * scale * 1.08
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            other = np.asarray(other_position, dtype=float).reshape(-1)[:3]
            if math.sqrt(_segment_point_min_distance_sq_3d(start, end, other)) < r_disk:
                return True
    else:
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            if np.linalg.norm(tip - other_position) < 0.26:
                return True

    for other_origin, other_direction in assigned_segments:
        other_tip = other_origin + other_direction * 0.45
        if np.linalg.norm(tip - other_tip) < 0.26:
            return True
        if (
            np.linalg.norm(origin - other_origin) < 0.12
            and float(np.dot(direction, other_direction)) > 0.92
        ):
            return True

    return False


__all__ = [
    "_compute_free_directions_3d",
    "_direction_conflicts_3d",
]
