"""2D free-axis direction assignment with reusable geometric precomputation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...config import PlotConfig
from ..contractions import _ContractionGroups, _group_contractions, _iter_contractions
from ..graph import _GraphData
from .direction_common import (
    _direction_from_axis_name,
    _direction_has_space,
    _is_dangling_leg_axis,
    _used_axis_directions,
)
from .geometry import (
    _BondSegment2D,
    _dangling_stub_segment_2d,
    _normalize_2d,
    _planar_contraction_bond_segment_records_2d,
    _segment_bboxes_overlap_2d,
    _segment_point_min_distance_sq_2d_many,
    _segments_cross_2d,
)
from .parameters import (
    _FREE_DIR_SAMPLES_2D,
    _STUB_ORIGIN_PAIR_CLEAR,
    _STUB_PARALLEL_DOT,
    _STUB_TIP_NODE_CLEAR,
    _STUB_TIP_TIP_CLEAR,
)
from .types import AxisDirections, NodePositions


@dataclass(frozen=True)
class _AssignedStub2D:
    origin: np.ndarray
    direction: np.ndarray
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class _FreeDirection2DContext:
    unit_circle: np.ndarray
    coords_xy: np.ndarray
    positions_xy: dict[int, np.ndarray]
    other_coords_by_node: dict[int, np.ndarray]
    away_scores_by_node: dict[int, np.ndarray]
    non_incident_segments_by_node: dict[int, tuple[_BondSegment2D, ...]]
    dangling_axes: frozenset[tuple[int, int]]


def _build_context(
    graph: _GraphData,
    positions: NodePositions,
    *,
    draw_scale: float,
    contraction_groups: _ContractionGroups | None,
) -> _FreeDirection2DContext:
    node_list = sorted(positions.keys())
    coords_xy = np.stack(
        [np.asarray(positions[node_id], dtype=float).reshape(-1)[:2] for node_id in node_list]
    )
    positions_xy = {
        node_id: np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
        for node_id in node_list
    }
    index_of = {node_id: index for index, node_id in enumerate(node_list)}

    angles = np.linspace(0.0, 2.0 * math.pi, _FREE_DIR_SAMPLES_2D, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))

    neighbor_midpoints: dict[int, list[np.ndarray]] = {node_id: [] for node_id in node_list}
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        midpoint = (positions_xy[left_id] + positions_xy[right_id]) / 2.0
        neighbor_midpoints[left_id].append(midpoint)
        neighbor_midpoints[right_id].append(midpoint)

    other_coords_by_node: dict[int, np.ndarray] = {}
    away_scores_by_node: dict[int, np.ndarray] = {}
    for node_id in node_list:
        self_index = index_of[node_id]
        if len(node_list) == 1:
            other_coords = np.zeros((0, 2), dtype=float)
        elif self_index == 0:
            other_coords = coords_xy[1:]
        elif self_index == len(node_list) - 1:
            other_coords = coords_xy[:-1]
        else:
            other_coords = np.concatenate(
                (coords_xy[:self_index], coords_xy[self_index + 1 :]),
                axis=0,
            )
        other_coords_by_node[node_id] = other_coords

        obstacle_parts: list[np.ndarray] = []
        if other_coords.size:
            obstacle_parts.append(other_coords)
        if neighbor_midpoints[node_id]:
            obstacle_parts.append(np.asarray(neighbor_midpoints[node_id], dtype=float))
        if obstacle_parts:
            obstacles = np.concatenate(obstacle_parts, axis=0)
        else:
            origin = positions_xy[node_id]
            obstacles = np.asarray([[origin[0] + 1.0, origin[1]]], dtype=float)

        vecs_to_obstacles = obstacles - positions_xy[node_id]
        dists = np.linalg.norm(vecs_to_obstacles, axis=1, keepdims=True)
        dirs_to_obstacles = vecs_to_obstacles / np.maximum(dists, 1e-6)
        toward = dirs_to_obstacles @ unit_circle.T
        away_scores_by_node[node_id] = -np.min(toward, axis=0)

    groups = contraction_groups if contraction_groups is not None else _group_contractions(graph)
    non_incident_segments_by_node = {node_id: [] for node_id in node_list}
    for segment in _planar_contraction_bond_segment_records_2d(
        graph,
        positions,
        scale=draw_scale,
        contraction_groups=groups,
    ):
        left_id, right_id = segment.node_ids
        for node_id in node_list:
            if node_id in (left_id, right_id):
                continue
            non_incident_segments_by_node[node_id].append(segment)

    dangling_axes = frozenset(
        (edge.endpoints[0].node_id, edge.endpoints[0].axis_index)
        for edge in graph.edges
        if edge.kind == "dangling"
    )
    return _FreeDirection2DContext(
        unit_circle=unit_circle,
        coords_xy=coords_xy,
        positions_xy=positions_xy,
        other_coords_by_node=other_coords_by_node,
        away_scores_by_node=away_scores_by_node,
        non_incident_segments_by_node={
            node_id: tuple(segments) for node_id, segments in non_incident_segments_by_node.items()
        },
        dangling_axes=dangling_axes,
    )


def _direction_conflicts_2d(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_stub_segments: list[tuple[np.ndarray, np.ndarray]],
    bond_segments: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> bool:
    d = _normalize_2d(direction)
    o2 = np.asarray(origin, dtype=float).reshape(-1)[:2]
    p0, p1 = _dangling_stub_segment_2d(origin, d, draw_scale=draw_scale)
    s = max(float(draw_scale), 1e-6)

    for left_id, right_id, bond_start, bond_end in bond_segments:
        if left_id == node_id or right_id == node_id:
            continue
        if _segments_cross_2d(p0, p1, bond_start, bond_end):
            return True

    other_coords = np.stack(
        [
            np.asarray(other_position, dtype=float).reshape(-1)[:2]
            for other_id, other_position in positions.items()
            if other_id != node_id
        ]
    )
    if other_coords.size:
        if strict_physical_node_clearance:
            r_disk_sq = (float(PlotConfig.DEFAULT_NODE_RADIUS) * s * 1.08) ** 2
            if (
                float(np.min(_segment_point_min_distance_sq_2d_many(p0, p1, other_coords)))
                < r_disk_sq
            ):
                return True
        else:
            deltas = other_coords - p1
            if bool(np.any(np.einsum("ij,ij->i", deltas, deltas) < (_STUB_TIP_NODE_CLEAR**2))):
                return True

    for q0, q1 in assigned_stub_segments:
        if _segments_cross_2d(p0, p1, q0, q1):
            return True
        if float(np.linalg.norm(p1 - q1)) < _STUB_TIP_TIP_CLEAR:
            return True
        d_other = _normalize_2d(q1 - q0)
        if (
            float(np.linalg.norm(o2 - q0)) < _STUB_ORIGIN_PAIR_CLEAR
            and float(np.dot(d, d_other)) > _STUB_PARALLEL_DOT
        ):
            return True
    return False


def _candidate_conflicts(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_stubs: list[_AssignedStub2D],
    other_coords: np.ndarray,
    non_incident_segments: tuple[_BondSegment2D, ...],
    draw_scale: float,
    strict_physical_node_clearance: bool,
) -> bool:
    normalized_direction = _normalize_2d(direction)
    p0, p1 = _dangling_stub_segment_2d(origin, normalized_direction, draw_scale=draw_scale)

    for segment in non_incident_segments:
        if not _segment_bboxes_overlap_2d(p0, p1, segment.bbox):
            continue
        if _segments_cross_2d(p0, p1, segment.start, segment.end):
            return True

    if other_coords.size:
        if strict_physical_node_clearance:
            r_disk_sq = (
                float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
            ) ** 2
            if (
                float(np.min(_segment_point_min_distance_sq_2d_many(p0, p1, other_coords)))
                < r_disk_sq
            ):
                return True
        else:
            deltas = other_coords - p1
            if bool(np.any(np.einsum("ij,ij->i", deltas, deltas) < (_STUB_TIP_NODE_CLEAR**2))):
                return True

    for assigned in assigned_stubs:
        if _segments_cross_2d(p0, p1, assigned.start, assigned.end):
            return True
        if float(np.linalg.norm(p1 - assigned.end)) < _STUB_TIP_TIP_CLEAR:
            return True
        if (
            float(np.linalg.norm(origin - assigned.origin)) < _STUB_ORIGIN_PAIR_CLEAR
            and float(np.dot(normalized_direction, assigned.direction)) > _STUB_PARALLEL_DOT
        ):
            return True
    return False


def _compute_free_directions_2d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
) -> None:
    context = _build_context(
        graph,
        positions,
        draw_scale=draw_scale,
        contraction_groups=contraction_groups,
    )
    assigned_stubs: list[_AssignedStub2D] = []

    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        origin = context.positions_xy[node_id]
        axis_count = max(node.degree, 1)
        away_scores = context.away_scores_by_node[node_id]
        other_coords = context.other_coords_by_node[node_id]
        non_incident_segments = context.non_incident_segments_by_node.get(node_id, ())

        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue

            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=2)
            strict_phys = axis_key in context.dangling_axes

            if used_dirs:
                used_stack = np.stack(
                    [used_direction[:2].astype(float) for used_direction in used_dirs]
                )
                overlap = context.unit_circle @ used_stack.T
                separation = np.maximum(0.0, overlap).sum(axis=1) * 2.0
            else:
                separation = np.zeros(context.unit_circle.shape[0], dtype=float)
            scores = away_scores - separation
            order = np.argsort(-scores)

            picked: np.ndarray | None = None
            if named_direction is not None and _direction_has_space(named_direction, used_dirs):
                named_2d = _normalize_2d(named_direction[:2])
                if not _candidate_conflicts(
                    node_id=node_id,
                    origin=origin,
                    direction=named_2d,
                    assigned_stubs=assigned_stubs,
                    other_coords=other_coords,
                    non_incident_segments=non_incident_segments,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                ):
                    picked = named_2d

            if picked is None:
                for index in order:
                    candidate = context.unit_circle[int(index)]
                    if not _direction_has_space(candidate, used_dirs):
                        continue
                    if _candidate_conflicts(
                        node_id=node_id,
                        origin=origin,
                        direction=candidate,
                        assigned_stubs=assigned_stubs,
                        other_coords=other_coords,
                        non_incident_segments=non_incident_segments,
                        draw_scale=draw_scale,
                        strict_physical_node_clearance=strict_phys,
                    ):
                        continue
                    picked = candidate.copy()
                    break

            if picked is None and not strict_phys:
                for index in order:
                    candidate = context.unit_circle[int(index)]
                    if not _direction_has_space(candidate, used_dirs):
                        continue
                    picked = candidate.copy()
                    break

            if picked is None and strict_phys:
                r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
                best_index: int | None = None
                best_margin = -1e300
                for index in range(int(context.unit_circle.shape[0])):
                    candidate = context.unit_circle[index]
                    p0, p1 = _dangling_stub_segment_2d(origin, candidate, draw_scale=draw_scale)
                    if other_coords.size:
                        margin = float(
                            math.sqrt(
                                float(
                                    np.min(
                                        _segment_point_min_distance_sq_2d_many(
                                            p0,
                                            p1,
                                            other_coords,
                                        )
                                    )
                                )
                            )
                            - r_disk
                        )
                    else:
                        margin = 0.0
                    if margin > best_margin:
                        best_margin = margin
                        best_index = index
                if best_index is not None:
                    picked = context.unit_circle[int(best_index)].copy()

            if picked is None:
                picked = context.unit_circle[int(np.argmax(scores))].copy()
            if picked is None:
                raise RuntimeError("Failed to assign a free 2D axis direction.")

            normalized_pick = _normalize_2d(picked)
            directions[axis_key] = normalized_pick
            start, end = _dangling_stub_segment_2d(origin, normalized_pick, draw_scale=draw_scale)
            assigned_stubs.append(
                _AssignedStub2D(
                    origin=origin.copy(),
                    direction=normalized_pick.copy(),
                    start=start,
                    end=end,
                )
            )


__all__ = [
    "_compute_free_directions_2d",
    "_direction_conflicts_2d",
    "_direction_has_space",
    "_is_dangling_leg_axis",
    "_used_axis_directions",
]
