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
    _normalize_direction,
    _orthogonal_unit,
    _preferred_component_directions_3d,
    _used_axis_directions,
)
from .geometry import (
    _dangling_stub_segment_3d,
    _segment_point_min_distance_sq_3d,
    _segment_segment_min_distance_sq_3d,
)
from .parameters import _STUB_BOND_CLEAR_3D
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
    bond_segments_by_node = _non_incident_bond_segments_by_node(graph, positions)

    for node_id, node in graph.nodes.items():
        component = component_by_node[node_id]
        axis_count = max(node.degree, 1)
        axis, lateral, normal = _component_orthogonal_basis(component, positions)
        candidate_directions = _preferred_component_directions_3d(
            component,
            positions,
            node_id=node_id,
            origin=positions[node_id],
            axis=axis,
            lateral=lateral,
            normal=normal,
        )

        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue

            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=3)
            strict_phys = _is_dangling_leg_axis(graph, node_id, axis_index)
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            origin = positions[node_id]
            node_bond_segments = bond_segments_by_node.get(node_id, ())
            if (
                named_direction is not None
                and _direction_has_space(named_direction, used_dirs)
                and not _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=named_direction,
                    assigned_segments=assigned_segments,
                    bond_segments=node_bond_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                )
            ):
                normalized_named = _normalize_direction(named_direction, dimensions=3)
                directions[axis_key] = normalized_named
                assigned_segments.append((origin.copy(), normalized_named.copy()))
                continue

            for direction in candidate_directions:
                if not _direction_has_space(direction, used_dirs):
                    continue
                if _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=direction,
                    assigned_segments=assigned_segments,
                    bond_segments=node_bond_segments,
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
                        dirs_try.append(_normalize_direction(named_direction, dimensions=3))
                    dirs_try.extend(list(candidate_directions))
                    best_direction: np.ndarray | None = None
                    best_margin = -1e300
                    seen: set[tuple[float, float, float]] = set()
                    for raw_direction in dirs_try:
                        direction_unit = _normalize_direction(raw_direction, dimensions=3)
                        key = tuple(np.round(direction_unit, decimals=6))
                        if key in seen:
                            continue
                        seen.add(key)
                        margin = _direction_margin_3d(
                            node_id=node_id,
                            origin=origin,
                            direction=direction_unit,
                            assigned_segments=assigned_segments,
                            bond_segments=node_bond_segments,
                            positions=positions,
                            draw_scale=draw_scale,
                            strict_physical_node_clearance=strict_phys,
                        )
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
    bond_segments: tuple[tuple[np.ndarray, np.ndarray], ...],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> bool:
    return _direction_margin_3d(
        node_id=node_id,
        origin=origin,
        direction=direction,
        assigned_segments=assigned_segments,
        bond_segments=bond_segments,
        positions=positions,
        draw_scale=draw_scale,
        strict_physical_node_clearance=strict_physical_node_clearance,
    ) < 0.0


def _direction_margin_3d(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_segments: list[tuple[np.ndarray, np.ndarray]],
    bond_segments: tuple[tuple[np.ndarray, np.ndarray], ...],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> float:
    direction_unit = _normalize_direction(direction, dimensions=3)
    origin_3d = np.asarray(origin, dtype=float).reshape(-1)[:3]
    tip = origin_3d + direction_unit * 0.45
    margin = float("inf")

    if strict_physical_node_clearance:
        scale = max(float(draw_scale), 1e-6)
        start, end = _dangling_stub_segment_3d(origin_3d, direction_unit, draw_scale=draw_scale)
        r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * scale * 1.08
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            other = np.asarray(other_position, dtype=float).reshape(-1)[:3]
            margin = min(
                margin,
                math.sqrt(_segment_point_min_distance_sq_3d(start, end, other)) - r_disk,
            )
        for bond_start, bond_end in bond_segments:
            margin = min(
                margin,
                math.sqrt(_segment_segment_min_distance_sq_3d(start, end, bond_start, bond_end))
                - _STUB_BOND_CLEAR_3D,
            )
    else:
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            margin = min(margin, float(np.linalg.norm(tip - other_position)) - 0.26)

    for other_origin, other_direction in assigned_segments:
        other_origin_3d = np.asarray(other_origin, dtype=float).reshape(-1)[:3]
        other_direction_3d = _normalize_direction(other_direction, dimensions=3)
        other_tip = other_origin_3d + other_direction_3d * 0.45
        margin = min(margin, float(np.linalg.norm(tip - other_tip)) - 0.26)
        if (
            float(np.linalg.norm(origin_3d - other_origin_3d)) < 0.12
            and float(np.dot(direction_unit, other_direction_3d)) > 0.92
        ):
            margin = min(margin, -1.0)

    return margin


def _non_incident_bond_segments_by_node(
    graph: _GraphData,
    positions: NodePositions,
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    segments_by_node: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        node_id: [] for node_id in positions
    }
    for edge in graph.edges:
        if edge.kind != "contraction" or len(edge.node_ids) != 2:
            continue
        left_id, right_id = (int(edge.node_ids[0]), int(edge.node_ids[1]))
        start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:3]
        end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:3]
        for node_id in segments_by_node:
            if node_id in {left_id, right_id}:
                continue
            segments_by_node[node_id].append((start, end))
    return {node_id: tuple(segments) for node_id, segments in segments_by_node.items()}


__all__ = [
    "_compute_free_directions_3d",
    "_direction_conflicts_3d",
]
