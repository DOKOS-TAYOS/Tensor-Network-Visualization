"""3D free-axis direction assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...config import PlotConfig
from ..graph import _GraphData
from ..layout_structure import _component_orthogonal_basis, _LayoutComponent
from .direction_common import (
    _direction_from_axis_name,
    _direction_has_space,
    _forced_dangling_direction_from_axis_name,
    _is_dangling_leg_axis,
    _normalize_direction,
    _orthogonal_unit,
    _preferred_component_directions_3d,
    _used_axis_directions,
)
from .geometry import (
    _dangling_stub_segment_3d,
    _segment_point_min_distance_sq_3d,
    _segment_point_min_distance_sq_3d_many,
    _segment_segment_min_distance_sq_3d,
)
from .parameters import _STUB_BOND_CLEAR_3D
from .types import AxisDirections, NodePositions


@dataclass(frozen=True)
class _DirectionConflictData3D:
    positions_by_node: dict[int, np.ndarray]
    other_positions_by_node: dict[int, np.ndarray]
    bond_starts_by_node: dict[int, np.ndarray]
    bond_ends_by_node: dict[int, np.ndarray]
    bond_bbox_min_by_node: dict[int, np.ndarray]
    bond_bbox_max_by_node: dict[int, np.ndarray]


@dataclass
class _AssignedDirectionState3D:
    origins: np.ndarray
    directions: np.ndarray
    tips: np.ndarray
    count: int = 0

    @classmethod
    def empty(cls, capacity: int) -> _AssignedDirectionState3D:
        return cls(
            origins=np.zeros((max(int(capacity), 1), 3), dtype=float),
            directions=np.zeros((max(int(capacity), 1), 3), dtype=float),
            tips=np.zeros((max(int(capacity), 1), 3), dtype=float),
            count=0,
        )

    def append(
        self,
        *,
        origin: np.ndarray,
        direction: np.ndarray,
        tip: np.ndarray,
    ) -> None:
        index = int(self.count)
        self.origins[index] = origin
        self.directions[index] = direction
        self.tips[index] = tip
        self.count += 1

    def origins_view(self) -> np.ndarray:
        return self.origins[: self.count]

    def directions_view(self) -> np.ndarray:
        return self.directions[: self.count]

    def tips_view(self) -> np.ndarray:
        return self.tips[: self.count]


def _build_direction_conflict_data_3d(
    graph: _GraphData,
    positions: NodePositions,
) -> _DirectionConflictData3D:
    positions_by_node = {
        int(node_id): np.asarray(position, dtype=float).reshape(-1)[:3].copy()
        for node_id, position in positions.items()
    }
    node_ids = tuple(sorted(positions_by_node))
    all_positions = np.stack([positions_by_node[node_id] for node_id in node_ids], axis=0)

    other_positions_by_node: dict[int, np.ndarray] = {}
    for index, node_id in enumerate(node_ids):
        if all_positions.shape[0] <= 1:
            other_positions_by_node[node_id] = np.zeros((0, 3), dtype=float)
            continue
        other_positions_by_node[node_id] = np.concatenate(
            [all_positions[:index], all_positions[index + 1 :]],
            axis=0,
        )

    bond_left_ids: list[int] = []
    bond_right_ids: list[int] = []
    bond_starts_raw: list[np.ndarray] = []
    bond_ends_raw: list[np.ndarray] = []
    for edge in graph.edges:
        if edge.kind != "contraction" or len(edge.node_ids) != 2:
            continue
        left_id, right_id = (int(edge.node_ids[0]), int(edge.node_ids[1]))
        bond_left_ids.append(left_id)
        bond_right_ids.append(right_id)
        bond_starts_raw.append(positions_by_node[left_id])
        bond_ends_raw.append(positions_by_node[right_id])

    if bond_starts_raw:
        bond_starts = np.stack(bond_starts_raw, axis=0)
        bond_ends = np.stack(bond_ends_raw, axis=0)
        bond_bbox_min = np.minimum(bond_starts, bond_ends)
        bond_bbox_max = np.maximum(bond_starts, bond_ends)
        left_ids_arr = np.asarray(bond_left_ids, dtype=int)
        right_ids_arr = np.asarray(bond_right_ids, dtype=int)
    else:
        bond_starts = np.zeros((0, 3), dtype=float)
        bond_ends = np.zeros((0, 3), dtype=float)
        bond_bbox_min = np.zeros((0, 3), dtype=float)
        bond_bbox_max = np.zeros((0, 3), dtype=float)
        left_ids_arr = np.zeros((0,), dtype=int)
        right_ids_arr = np.zeros((0,), dtype=int)

    bond_starts_by_node: dict[int, np.ndarray] = {}
    bond_ends_by_node: dict[int, np.ndarray] = {}
    bond_bbox_min_by_node: dict[int, np.ndarray] = {}
    bond_bbox_max_by_node: dict[int, np.ndarray] = {}
    for node_id in node_ids:
        if bond_starts.shape[0] == 0:
            bond_starts_by_node[node_id] = np.zeros((0, 3), dtype=float)
            bond_ends_by_node[node_id] = np.zeros((0, 3), dtype=float)
            bond_bbox_min_by_node[node_id] = np.zeros((0, 3), dtype=float)
            bond_bbox_max_by_node[node_id] = np.zeros((0, 3), dtype=float)
            continue
        mask = (left_ids_arr != node_id) & (right_ids_arr != node_id)
        bond_starts_by_node[node_id] = bond_starts[mask]
        bond_ends_by_node[node_id] = bond_ends[mask]
        bond_bbox_min_by_node[node_id] = bond_bbox_min[mask]
        bond_bbox_max_by_node[node_id] = bond_bbox_max[mask]

    return _DirectionConflictData3D(
        positions_by_node=positions_by_node,
        other_positions_by_node=other_positions_by_node,
        bond_starts_by_node=bond_starts_by_node,
        bond_ends_by_node=bond_ends_by_node,
        bond_bbox_min_by_node=bond_bbox_min_by_node,
        bond_bbox_max_by_node=bond_bbox_max_by_node,
    )


def _bbox_distance_sq_batch_3d(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    if bbox_min.size == 0:
        return np.zeros((0,), dtype=float)
    segment_min = np.minimum(segment_start, segment_end)
    segment_max = np.maximum(segment_start, segment_end)
    gap = np.maximum(
        0.0,
        np.maximum(bbox_min - segment_max, segment_min - bbox_max),
    )
    return np.einsum("ij,ij->i", gap, gap)


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    layout_components: tuple[_LayoutComponent, ...],
) -> None:
    conflict_data = _build_direction_conflict_data_3d(graph, positions)
    component_by_node = {
        node_id: component for component in layout_components for node_id in component.node_ids
    }
    assigned_state = _AssignedDirectionState3D.empty(
        sum(max(node.degree, 1) for node in graph.nodes.values())
    )

    for node_id, node in graph.nodes.items():
        component = component_by_node[node_id]
        axis_count = max(node.degree, 1)
        unassigned_axis_indices = [
            axis_index
            for axis_index in range(axis_count)
            if (node_id, axis_index) not in directions
        ]
        forced_directions = {
            axis_index: _normalize_direction(forced_direction, dimensions=3)
            for axis_index in unassigned_axis_indices
            for forced_direction in [
                _forced_dangling_direction_from_axis_name(
                    graph,
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_name=(
                        node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
                    ),
                    dimensions=3,
                )
            ]
            if forced_direction is not None
        }
        origin = conflict_data.positions_by_node[node_id]
        if forced_directions and len(forced_directions) == len(unassigned_axis_indices):
            for axis_index in unassigned_axis_indices:
                forced_direction = forced_directions[axis_index]
                directions[(node_id, axis_index)] = forced_direction
                assigned_state.append(
                    origin=origin,
                    direction=forced_direction,
                    tip=origin + forced_direction * 0.45,
                )
            continue
        axis, lateral, normal = _component_orthogonal_basis(component, positions)
        candidate_directions = tuple(
            _normalize_direction(direction, dimensions=3)
            for direction in _preferred_component_directions_3d(
                component,
                positions,
                node_id=node_id,
                origin=positions[node_id],
                axis=axis,
                lateral=lateral,
                normal=normal,
            )
        )

        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue

            forced_direction = forced_directions.get(axis_index)
            if forced_direction is not None:
                directions[axis_key] = forced_direction
                assigned_state.append(
                    origin=origin,
                    direction=forced_direction,
                    tip=origin + forced_direction * 0.45,
                )
                continue

            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            strict_phys = _is_dangling_leg_axis(graph, node_id, axis_index)
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            named_direction = _direction_from_axis_name(axis_name, dimensions=3)
            if named_direction is not None:
                named_direction = _normalize_direction(named_direction, dimensions=3)
            if (
                named_direction is not None
                and _direction_has_space(named_direction, used_dirs)
                and not _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=named_direction,
                    assigned_segments=[],
                    bond_segments=(),
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                    conflict_data=conflict_data,
                    assigned_state=assigned_state,
                )
            ):
                directions[axis_key] = named_direction
                assigned_state.append(
                    origin=origin,
                    direction=named_direction,
                    tip=origin + named_direction * 0.45,
                )
                continue

            for direction in candidate_directions:
                if not _direction_has_space(direction, used_dirs):
                    continue
                if _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=direction,
                    assigned_segments=[],
                    bond_segments=(),
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                    conflict_data=conflict_data,
                    assigned_state=assigned_state,
                ):
                    continue
                directions[axis_key] = direction
                assigned_state.append(
                    origin=origin,
                    direction=direction,
                    tip=origin + direction * 0.45,
                )
                break
            else:
                if strict_phys:
                    dirs_try: list[np.ndarray] = []
                    if named_direction is not None:
                        dirs_try.append(named_direction)
                    dirs_try.extend(list(candidate_directions))
                    best_direction: np.ndarray | None = None
                    best_margin = -1e300
                    seen: set[tuple[float, float, float]] = set()
                    for raw_direction in dirs_try:
                        direction_unit = raw_direction
                        key = tuple(np.round(direction_unit, decimals=6))
                        if key in seen:
                            continue
                        seen.add(key)
                        margin = _direction_margin_3d(
                            node_id=node_id,
                            origin=origin,
                            direction=direction_unit,
                            assigned_segments=[],
                            bond_segments=(),
                            positions=positions,
                            draw_scale=draw_scale,
                            strict_physical_node_clearance=strict_phys,
                            conflict_data=conflict_data,
                            assigned_state=assigned_state,
                        )
                        if margin > best_margin:
                            best_margin = margin
                            best_direction = direction_unit
                    if best_direction is not None:
                        directions[axis_key] = best_direction
                        assigned_state.append(
                            origin=origin,
                            direction=best_direction,
                            tip=origin + best_direction * 0.45,
                        )
                        continue

                fallback = (
                    named_direction if named_direction is not None else _orthogonal_unit(axis)
                )
                if not _direction_has_space(fallback, used_dirs):
                    fallback = -fallback
                fallback = _normalize_direction(fallback, dimensions=3)
                directions[axis_key] = fallback
                assigned_state.append(
                    origin=origin,
                    direction=fallback,
                    tip=origin + fallback * 0.45,
                )


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
    conflict_data: _DirectionConflictData3D | None = None,
    assigned_state: _AssignedDirectionState3D | None = None,
) -> bool:
    return (
        _direction_margin_3d(
            node_id=node_id,
            origin=origin,
            direction=direction,
            assigned_segments=assigned_segments,
            bond_segments=bond_segments,
            positions=positions,
            draw_scale=draw_scale,
            strict_physical_node_clearance=strict_physical_node_clearance,
            conflict_data=conflict_data,
            assigned_state=assigned_state,
        )
        < 0.0
    )


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
    conflict_data: _DirectionConflictData3D | None = None,
    assigned_state: _AssignedDirectionState3D | None = None,
) -> float:
    direction_unit = _normalize_direction(direction, dimensions=3)
    origin_3d = np.asarray(origin, dtype=float).reshape(-1)[:3]
    tip = origin_3d + direction_unit * 0.45
    margin = float("inf")

    if conflict_data is not None and assigned_state is not None:
        if strict_physical_node_clearance:
            scale = max(float(draw_scale), 1e-6)
            start, end = _dangling_stub_segment_3d(origin_3d, direction_unit, draw_scale=draw_scale)
            r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * scale * 1.08

            other_positions = conflict_data.other_positions_by_node.get(node_id)
            if other_positions is not None and other_positions.size:
                point_dist_sq = _segment_point_min_distance_sq_3d_many(start, end, other_positions)
                margin = min(margin, math.sqrt(float(np.min(point_dist_sq))) - r_disk)

            bond_starts = conflict_data.bond_starts_by_node.get(node_id)
            bond_ends = conflict_data.bond_ends_by_node.get(node_id)
            bond_bbox_min = conflict_data.bond_bbox_min_by_node.get(node_id)
            bond_bbox_max = conflict_data.bond_bbox_max_by_node.get(node_id)
            if (
                bond_starts is not None
                and bond_ends is not None
                and bond_bbox_min is not None
                and bond_bbox_max is not None
                and bond_starts.size
            ):
                bbox_dist_sq = _bbox_distance_sq_batch_3d(
                    start,
                    end,
                    bbox_min=bond_bbox_min,
                    bbox_max=bond_bbox_max,
                )
                candidate_mask = bbox_dist_sq <= (_STUB_BOND_CLEAR_3D**2)
                if np.any(candidate_mask):
                    for bond_start, bond_end in zip(
                        bond_starts[candidate_mask],
                        bond_ends[candidate_mask],
                        strict=False,
                    ):
                        margin = min(
                            margin,
                            math.sqrt(
                                _segment_segment_min_distance_sq_3d(
                                    start, end, bond_start, bond_end
                                )
                            )
                            - _STUB_BOND_CLEAR_3D,
                        )
        else:
            other_positions = conflict_data.other_positions_by_node.get(node_id)
            if other_positions is not None and other_positions.size:
                deltas = other_positions - tip
                distances_sq = np.einsum("ij,ij->i", deltas, deltas)
                margin = min(margin, math.sqrt(float(np.min(distances_sq))) - 0.26)

        if assigned_state.count:
            assigned_tips = assigned_state.tips_view()
            tip_deltas = assigned_tips - tip
            tip_distances_sq = np.einsum("ij,ij->i", tip_deltas, tip_deltas)
            margin = min(margin, math.sqrt(float(np.min(tip_distances_sq))) - 0.26)

            origin_deltas = assigned_state.origins_view() - origin_3d
            origin_distances_sq = np.einsum("ij,ij->i", origin_deltas, origin_deltas)
            direction_dots = assigned_state.directions_view() @ direction_unit
            if np.any((origin_distances_sq < (0.12**2)) & (direction_dots > 0.92)):
                margin = min(margin, -1.0)

        return margin

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
    "_build_direction_conflict_data_3d",
    "_compute_free_directions_3d",
    "_direction_conflicts_3d",
]
