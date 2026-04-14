"""2D free-axis direction assignment with local rule-based heuristics."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...config import PlotConfig
from ..contractions import _ContractionGroups, _iter_contractions
from ..graph import _GraphData
from ..layout_structure import _LayoutComponent
from .direction_common import (
    _CARDINAL_DIRECTIONS_2D,
    _DIAGONAL_DIRECTIONS_2D,
    _SEMIDIAGONAL_DIRECTIONS_2D,
    _behavior_direction_order_2d,
    _component_centroid,
    _direction_from_axis_name,
    _is_dangling_leg_axis,
    _sorted_direction_candidates,
    _used_axis_directions,
)
from .geometry import (
    _dangling_stub_segment_2d,
    _normalize_2d,
    _segment_point_min_distance_sq_2d_many,
    _segments_cross_2d,
)
from .parameters import (
    _STUB_ORIGIN_PAIR_CLEAR,
    _STUB_PARALLEL_DOT,
    _STUB_TIP_NODE_CLEAR,
    _STUB_TIP_TIP_CLEAR,
)
from .types import AxisDirections, NodePositions

_ANGLE_THRESHOLD_DEGREES_2D: float = 5.0

_Behavior2D = Literal["north", "south", "east", "west"]


@dataclass(frozen=True)
class _ContractionSegment2D:
    edge_index: int
    left_id: int
    right_id: int
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class _FreeDirectionContext2D:
    positions_xy: dict[int, np.ndarray]
    neighbor_ids_by_node: dict[int, tuple[int, ...]]
    second_neighbor_ids_by_node: dict[int, tuple[int, ...]]
    contraction_segments_by_node: dict[int, tuple[_ContractionSegment2D, ...]]
    shared_virtual_hub_ids_by_node: dict[int, tuple[int, ...]]
    shared_virtual_hub_center_by_node: dict[int, np.ndarray]
    dangling_axes: frozenset[tuple[int, int]]
    component_by_node: dict[int, _LayoutComponent]


@dataclass(frozen=True)
class _NodeDirectionPlan2D:
    node_id: int
    behavior: _Behavior2D | None


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
            radius_sq = (float(PlotConfig.DEFAULT_NODE_RADIUS) * s) ** 2
            if (
                float(np.min(_segment_point_min_distance_sq_2d_many(p0, p1, other_coords)))
                < radius_sq
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


def _direction_angle_conflicts_2d(
    direction: np.ndarray,
    other_direction: np.ndarray,
    *,
    threshold_degrees: float = _ANGLE_THRESHOLD_DEGREES_2D,
    treat_opposite_as_conflict: bool = True,
) -> bool:
    dot = float(
        np.dot(
            _normalize_2d(direction),
            _normalize_2d(other_direction),
        )
    )
    threshold_dot = math.cos(math.radians(float(threshold_degrees)))
    if dot >= threshold_dot:
        return True
    return bool(treat_opposite_as_conflict and dot <= -threshold_dot)


def _pick_candidate_direction_2d(
    *,
    candidates: tuple[np.ndarray, ...],
    is_valid: Callable[[np.ndarray], bool],
) -> np.ndarray:
    last_tried: np.ndarray | None = None
    for candidate in candidates:
        normalized = _normalize_2d(candidate)
        last_tried = normalized.copy()
        if is_valid(normalized):
            return normalized
    if last_tried is None:
        raise RuntimeError(
            "No candidate directions were generated for the 2D free-axis assignment."
        )
    return last_tried


def _build_context(
    graph: _GraphData,
    positions: NodePositions,
    *,
    layout_components: tuple[_LayoutComponent, ...],
) -> _FreeDirectionContext2D:
    positions_xy = {
        node_id: np.asarray(position, dtype=float).reshape(-1)[:2]
        for node_id, position in positions.items()
    }
    neighbor_ids_by_node: dict[int, set[int]] = {
        int(node_id): set()
        for component in layout_components
        for node_id in component.geometry_node_ids
    }
    for component in layout_components:
        for left_id, right_id in component.geometry_contraction_graph.edges():
            neighbor_ids_by_node[int(left_id)].add(int(right_id))
            neighbor_ids_by_node[int(right_id)].add(int(left_id))

    second_neighbor_ids_by_node: dict[int, tuple[int, ...]] = {}
    for node_id in neighbor_ids_by_node:
        second_neighbors: set[int] = set()
        for neighbor_id in neighbor_ids_by_node[node_id]:
            second_neighbors.update(neighbor_ids_by_node[neighbor_id])
        second_neighbors.discard(node_id)
        second_neighbors.difference_update(neighbor_ids_by_node[node_id])
        second_neighbor_ids_by_node[node_id] = tuple(sorted(second_neighbors))

    contraction_segments_by_node_lists: dict[int, list[_ContractionSegment2D]] = {
        node_id: [] for node_id in graph.nodes
    }
    for edge_index, record in enumerate(_iter_contractions(graph)):
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        segment = _ContractionSegment2D(
            edge_index=edge_index,
            left_id=left_id,
            right_id=right_id,
            start=positions_xy[left_id],
            end=positions_xy[right_id],
        )
        contraction_segments_by_node_lists[left_id].append(segment)
        contraction_segments_by_node_lists[right_id].append(segment)

    dangling_axes = frozenset(
        (edge.endpoints[0].node_id, edge.endpoints[0].axis_index)
        for edge in graph.edges
        if edge.kind == "dangling"
    )
    component_by_node = {
        node_id: component
        for component in layout_components
        for node_id in component.geometry_node_ids
    }
    shared_virtual_hub_ids_by_node: dict[int, tuple[int, ...]] = {}
    shared_virtual_hub_center_by_node: dict[int, np.ndarray] = {}
    for component in layout_components:
        shared_virtual_groups: defaultdict[frozenset[int], list[int]] = defaultdict(list)
        for node_id in component.geometry_virtual_node_ids:
            visible_neighbor_ids = frozenset(
                neighbor_id
                for neighbor_id in component.geometry_contraction_graph.neighbors(node_id)
                if not graph.nodes[neighbor_id].is_virtual
            )
            if visible_neighbor_ids:
                shared_virtual_groups[visible_neighbor_ids].append(node_id)
        for node_ids in shared_virtual_groups.values():
            if len(node_ids) < 2:
                continue
            ordered_node_ids = tuple(sorted(node_ids))
            center = np.mean(
                np.stack([positions_xy[node_id] for node_id in ordered_node_ids]),
                axis=0,
            )
            for node_id in ordered_node_ids:
                shared_virtual_hub_ids_by_node[node_id] = ordered_node_ids
                shared_virtual_hub_center_by_node[node_id] = center.copy()
    return _FreeDirectionContext2D(
        positions_xy=positions_xy,
        neighbor_ids_by_node={
            node_id: tuple(sorted(neighbor_ids))
            for node_id, neighbor_ids in neighbor_ids_by_node.items()
        },
        second_neighbor_ids_by_node=second_neighbor_ids_by_node,
        contraction_segments_by_node={
            node_id: tuple(segments)
            for node_id, segments in contraction_segments_by_node_lists.items()
        },
        shared_virtual_hub_ids_by_node=shared_virtual_hub_ids_by_node,
        shared_virtual_hub_center_by_node=shared_virtual_hub_center_by_node,
        dangling_axes=dangling_axes,
        component_by_node=component_by_node,
    )


def _random_direction_bucket_2d(
    *,
    blocked: tuple[np.ndarray, ...],
    count: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    generator = rng if rng is not None else np.random.default_rng()
    blocked_norm = [_normalize_2d(direction) for direction in blocked]
    randoms: list[np.ndarray] = []
    attempts = 0
    while len(randoms) < int(count) and attempts < 4096:
        attempts += 1
        angle = float(generator.uniform(0.0, 2.0 * math.pi))
        candidate = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        if any(float(np.dot(candidate, other)) > 0.995 for other in [*blocked_norm, *randoms]):
            continue
        randoms.append(candidate)
    if len(randoms) < int(count):
        raise RuntimeError("Unable to generate the random fallback directions for the 2D layout.")
    return tuple(randoms)


def _sort_bucket_by_reference_2d(
    reference: np.ndarray,
    candidates: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    return _sorted_direction_candidates(reference, candidates, dimensions=2)


def _reference_for_behavior_2d(behavior: _Behavior2D) -> np.ndarray:
    return _behavior_direction_order_2d(behavior)[0]


def _behavior_candidates_2d(
    *,
    behavior: _Behavior2D,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    del rng
    return _behavior_direction_order_2d(behavior)


def _outward_candidates_2d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    del rng
    origin = np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
    centroid = _component_centroid(component, positions, dimensions=2)
    reference = origin - centroid
    if float(np.linalg.norm(reference)) < 1e-9:
        reference = np.array([0.0, 1.0], dtype=float)
    return (
        *_sort_bucket_by_reference_2d(reference, _CARDINAL_DIRECTIONS_2D),
        *_sort_bucket_by_reference_2d(reference, _DIAGONAL_DIRECTIONS_2D),
        *_sort_bucket_by_reference_2d(reference, _SEMIDIAGONAL_DIRECTIONS_2D),
    )


def _node_candidate_directions_2d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    behavior: _Behavior2D | None,
    outward_reference: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    if outward_reference is not None:
        del rng
        return (
            *_sort_bucket_by_reference_2d(outward_reference, _CARDINAL_DIRECTIONS_2D),
            *_sort_bucket_by_reference_2d(outward_reference, _DIAGONAL_DIRECTIONS_2D),
            *_sort_bucket_by_reference_2d(outward_reference, _SEMIDIAGONAL_DIRECTIONS_2D),
        )
    if behavior is not None:
        return _behavior_candidates_2d(behavior=behavior, rng=rng)
    return _outward_candidates_2d(component, positions, node_id=node_id, rng=rng)


def _random_fallback_candidates_2d(
    *,
    deterministic: tuple[np.ndarray, ...],
    reference: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    return _sort_bucket_by_reference_2d(
        reference,
        _random_direction_bucket_2d(blocked=deterministic, rng=rng),
    )


def _first_valid_candidate_2d(
    *,
    candidates: tuple[np.ndarray, ...],
    is_valid: Callable[[np.ndarray], bool],
) -> np.ndarray | None:
    for candidate in candidates:
        normalized = _normalize_2d(candidate)
        if is_valid(normalized):
            return normalized
    return None


def _shared_virtual_hub_outward_reference_2d(
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
) -> np.ndarray | None:
    center = context.shared_virtual_hub_center_by_node.get(node_id)
    if center is None:
        return None
    reference = context.positions_xy[node_id] - center
    if float(np.linalg.norm(reference)) < 1e-9:
        return None
    return reference


def _component_node_plans_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> tuple[_NodeDirectionPlan2D, ...]:
    if component.structure_kind == "chain":
        ordered_node_ids = [
            node_id for node_id in component.chain_order if node_id in component.geometry_node_ids
        ]
        remainder = sorted(
            node_id for node_id in component.geometry_node_ids if node_id not in ordered_node_ids
        )
        return tuple(
            _NodeDirectionPlan2D(node_id=node_id, behavior="north")
            for node_id in [*ordered_node_ids, *remainder]
        )
    if component.structure_kind == "grid" and component.grid_mapping is not None:
        return _grid_node_plans_2d(component, positions)
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        return _grid3d_node_plans_2d(component, positions)
    if component.structure_kind == "tree":
        return _tree_node_plans_2d(component)
    return _irregular_node_plans_2d(component, positions)


def _grid_node_plans_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> tuple[_NodeDirectionPlan2D, ...]:
    remaining = {
        node_id: np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
        for node_id in component.geometry_node_ids
        if node_id in positions
    }
    plans: list[_NodeDirectionPlan2D] = []
    while remaining:
        x_levels = {round(float(coord[0]), 6) for coord in remaining.values()}
        y_levels = {round(float(coord[1]), 6) for coord in remaining.values()}
        min_x = min(x_levels)
        max_x = max(x_levels)
        min_y = min(y_levels)
        max_y = max(y_levels)

        if len(y_levels) == 1:
            node_ids = sorted(
                remaining,
                key=lambda node_id: float(remaining[node_id][0]),
            )
            plans.extend(
                _NodeDirectionPlan2D(node_id=node_id, behavior="north") for node_id in node_ids
            )
            break

        if len(x_levels) == 1:
            node_ids = sorted(
                remaining,
                key=lambda node_id: float(remaining[node_id][1]),
            )
            plans.extend(
                _NodeDirectionPlan2D(node_id=node_id, behavior="east") for node_id in node_ids
            )
            break

        top_nodes = sorted(
            (node_id for node_id, coord in remaining.items() if round(float(coord[1]), 6) == max_y),
            key=lambda node_id: float(remaining[node_id][0]),
        )
        bottom_nodes = sorted(
            (node_id for node_id, coord in remaining.items() if round(float(coord[1]), 6) == min_y),
            key=lambda node_id: float(remaining[node_id][0]),
        )
        right_nodes = sorted(
            (
                node_id
                for node_id, coord in remaining.items()
                if round(float(coord[0]), 6) == max_x and min_y < round(float(coord[1]), 6) < max_y
            ),
            key=lambda node_id: (-float(remaining[node_id][1]), float(remaining[node_id][0])),
        )
        left_nodes = sorted(
            (
                node_id
                for node_id, coord in remaining.items()
                if round(float(coord[0]), 6) == min_x and min_y < round(float(coord[1]), 6) < max_y
            ),
            key=lambda node_id: (-float(remaining[node_id][1]), float(remaining[node_id][0])),
        )

        for node_id in top_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="north"))
            remaining.pop(node_id, None)
        for node_id in bottom_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="south"))
            remaining.pop(node_id, None)
        for node_id in right_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="east"))
            remaining.pop(node_id, None)
        for node_id in left_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="west"))
            remaining.pop(node_id, None)
    return tuple(plans)


def _grid3d_node_plans_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> tuple[_NodeDirectionPlan2D, ...]:
    remaining = {
        node_id: component.grid3d_mapping[node_id]
        for node_id in component.geometry_node_ids
        if component.grid3d_mapping is not None and node_id in component.grid3d_mapping
    }
    plans: list[_NodeDirectionPlan2D] = []
    while remaining:
        i_vals = {i for i, _, _ in remaining.values()}
        k_vals = {k for _, _, k in remaining.values()}
        pos_xy = {
            node_id: np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
            for node_id in remaining
            if node_id in positions
        }
        x_levels = {round(float(coord[0]), 6) for coord in pos_xy.values()}
        y_levels = {round(float(coord[1]), 6) for coord in pos_xy.values()}
        min_i = min(i_vals)
        max_i = max(i_vals)
        min_k = min(k_vals)
        max_k = max(k_vals)
        min_x = min(x_levels)
        max_x = max(x_levels)
        min_y = min(y_levels)
        max_y = max(y_levels)

        if len(y_levels) == 1 and len(x_levels) >= 1:
            if len(x_levels) == 1:
                front_nodes = sorted(
                    (node_id for node_id, (_i, _j, k) in remaining.items() if k == min_k),
                    key=lambda node_id: remaining[node_id][2],
                )
                back_nodes = sorted(
                    (
                        node_id
                        for node_id, (_i, _j, k) in remaining.items()
                        if k == max_k and node_id not in front_nodes
                    ),
                    key=lambda node_id: remaining[node_id][2],
                )
                for node_id in front_nodes:
                    plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="west"))
                    remaining.pop(node_id, None)
                for node_id in back_nodes:
                    plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="east"))
                    remaining.pop(node_id, None)
                continue

            row_nodes = sorted(
                remaining,
                key=lambda node_id: (float(pos_xy[node_id][0]), remaining[node_id][2]),
            )
            plans.extend(
                _NodeDirectionPlan2D(node_id=node_id, behavior="north") for node_id in row_nodes
            )
            break

        if len(x_levels) == 1 and len(k_vals) == 1:
            col_nodes = sorted(
                remaining,
                key=lambda node_id: (float(pos_xy[node_id][1]), remaining[node_id][2]),
            )
            plans.extend(
                _NodeDirectionPlan2D(node_id=node_id, behavior="east") for node_id in col_nodes
            )
            break

        top_nodes = sorted(
            (node_id for node_id, coord in pos_xy.items() if round(float(coord[1]), 6) == max_y),
            key=lambda node_id: (float(pos_xy[node_id][0]), remaining[node_id][2]),
        )
        bottom_nodes = sorted(
            (node_id for node_id, coord in pos_xy.items() if round(float(coord[1]), 6) == min_y),
            key=lambda node_id: (float(pos_xy[node_id][0]), remaining[node_id][2]),
        )
        right_nodes = sorted(
            (
                node_id
                for node_id, coord in pos_xy.items()
                if round(float(coord[0]), 6) == max_x and min_y < round(float(coord[1]), 6) < max_y
            ),
            key=lambda node_id: (-float(pos_xy[node_id][1]), remaining[node_id][2]),
        )
        left_nodes = sorted(
            (
                node_id
                for node_id, coord in pos_xy.items()
                if round(float(coord[0]), 6) == min_x and min_y < round(float(coord[1]), 6) < max_y
            ),
            key=lambda node_id: (-float(pos_xy[node_id][1]), remaining[node_id][2]),
        )
        front_nodes = sorted(
            (
                node_id
                for node_id, (i, _j, k) in remaining.items()
                if (
                    k == min_k
                    and min_i < i < max_i
                    and min_y < round(float(pos_xy[node_id][1]), 6) < max_y
                    and min_x < round(float(pos_xy[node_id][0]), 6) < max_x
                )
            ),
            key=lambda node_id: (-float(pos_xy[node_id][1]), float(pos_xy[node_id][0])),
        )
        back_nodes = sorted(
            (
                node_id
                for node_id, (i, _j, k) in remaining.items()
                if (
                    k == max_k
                    and min_i < i < max_i
                    and min_y < round(float(pos_xy[node_id][1]), 6) < max_y
                    and min_x < round(float(pos_xy[node_id][0]), 6) < max_x
                )
            ),
            key=lambda node_id: (-float(pos_xy[node_id][1]), float(pos_xy[node_id][0])),
        )

        for node_id in top_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="north"))
            remaining.pop(node_id, None)
        for node_id in bottom_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="south"))
            remaining.pop(node_id, None)
        for node_id in right_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="east"))
            remaining.pop(node_id, None)
        for node_id in left_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="west"))
            remaining.pop(node_id, None)
        for node_id in front_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="west"))
            remaining.pop(node_id, None)
        for node_id in back_nodes:
            plans.append(_NodeDirectionPlan2D(node_id=node_id, behavior="east"))
            remaining.pop(node_id, None)
    return tuple(plans)


def _tree_node_plans_2d(component: _LayoutComponent) -> tuple[_NodeDirectionPlan2D, ...]:
    root_id = (
        component.tree_root if component.tree_root is not None else min(component.geometry_node_ids)
    )
    seen = {root_id}
    queue: deque[int] = deque([root_id])
    ordered: list[int] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for neighbor_id in sorted(component.geometry_contraction_graph.neighbors(node_id)):
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            queue.append(neighbor_id)
    ordered.extend(
        sorted(node_id for node_id in component.geometry_node_ids if node_id not in seen)
    )
    return tuple(_NodeDirectionPlan2D(node_id=node_id, behavior="south") for node_id in ordered)


def _irregular_node_plans_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> tuple[_NodeDirectionPlan2D, ...]:
    centroid = _component_centroid(component, positions, dimensions=2)

    def radial_distance(node_id: int) -> float:
        point = np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
        return float(np.linalg.norm(point - centroid))

    ordered = sorted(
        (node_id for node_id in component.geometry_node_ids if node_id in positions),
        key=lambda node_id: (
            -radial_distance(node_id),
            node_id,
        ),
    )
    return tuple(_NodeDirectionPlan2D(node_id=node_id, behavior=None) for node_id in ordered)


def _neighbor_hops_for_component_2d(component: _LayoutComponent) -> int:
    if component.structure_kind in {"chain", "grid"}:
        return 0
    return 2


def _candidate_conflicts_with_node_axes_2d(
    graph: _GraphData,
    directions: AxisDirections,
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
    axis_index: int,
    candidate: np.ndarray,
) -> bool:
    grouped_node_ids = context.shared_virtual_hub_ids_by_node.get(node_id, (node_id,))
    for grouped_node_id in grouped_node_ids:
        axis_count = max(graph.nodes[grouped_node_id].degree, 1)
        for other_axis_index in range(axis_count):
            if grouped_node_id == node_id and other_axis_index == axis_index:
                continue
            other_direction = directions.get((grouped_node_id, other_axis_index))
            if other_direction is None:
                continue
            if _direction_angle_conflicts_2d(
                candidate,
                other_direction,
                treat_opposite_as_conflict=False,
            ):
                return True
    return False


def _candidate_conflicts_with_same_node_axes_2d(
    graph: _GraphData,
    directions: AxisDirections,
    *,
    node_id: int,
    axis_index: int,
    candidate: np.ndarray,
    pending_directions: AxisDirections | None = None,
) -> bool:
    axis_count = max(graph.nodes[node_id].degree, 1)
    for other_axis_index in range(axis_count):
        if other_axis_index == axis_index:
            continue
        other_axis_key = (node_id, other_axis_index)
        other_direction = (
            pending_directions.get(other_axis_key)
            if pending_directions is not None and other_axis_key in pending_directions
            else directions.get(other_axis_key)
        )
        if other_direction is None:
            continue
        if _direction_angle_conflicts_2d(
            candidate,
            other_direction,
            treat_opposite_as_conflict=False,
        ):
            return True
    return False


def _is_chain_fast_path_component_2d(component: _LayoutComponent) -> bool:
    if component.structure_kind != "chain" or component.geometry_virtual_node_ids:
        return False
    return all(
        component.geometry_contraction_graph.degree(node_id) <= 2
        for node_id in component.geometry_node_ids
    )


def _try_compute_chain_free_directions_2d(
    graph: _GraphData,
    directions: AxisDirections,
    *,
    layout_components: tuple[_LayoutComponent, ...],
) -> bool:
    if not layout_components:
        return False
    if not all(_is_chain_fast_path_component_2d(component) for component in layout_components):
        return False

    resolved: AxisDirections = {}
    candidate_order = _behavior_direction_order_2d("north")
    for component in layout_components:
        ordered_node_ids = [
            node_id for node_id in component.chain_order if node_id in component.geometry_node_ids
        ]
        ordered_node_ids.extend(
            sorted(
                node_id
                for node_id in component.geometry_node_ids
                if node_id not in ordered_node_ids
            )
        )
        for node_id in ordered_node_ids:
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            axis_count = max(node.degree, 1)
            for axis_index in range(axis_count):
                axis_key = (node_id, axis_index)
                if axis_key in directions or axis_key in resolved:
                    continue
                named_direction = _direction_from_axis_name(
                    node.axes_names[axis_index] if axis_index < len(node.axes_names) else None,
                    dimensions=2,
                )
                if named_direction is not None:
                    candidate = _normalize_2d(named_direction)
                    if not _candidate_conflicts_with_same_node_axes_2d(
                        graph,
                        directions,
                        node_id=node_id,
                        axis_index=axis_index,
                        candidate=candidate,
                        pending_directions=resolved,
                    ):
                        resolved[axis_key] = candidate
                        continue

                picked = _first_valid_candidate_2d(
                    candidates=candidate_order,
                    is_valid=lambda candidate, node_id=node_id, axis_index=axis_index: (
                        not (
                            _candidate_conflicts_with_same_node_axes_2d(
                                graph,
                                directions,
                                node_id=node_id,
                                axis_index=axis_index,
                                candidate=candidate,
                                pending_directions=resolved,
                            )
                        )
                    ),
                )
                if picked is None:
                    return False
                resolved[axis_key] = picked

    directions.update(resolved)
    return True


def _candidate_crosses_neighbor_bonds_2d(
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
    neighbor_ids: set[int],
    candidate: np.ndarray,
    draw_scale: float,
) -> bool:
    start, end = _dangling_stub_segment_2d(
        context.positions_xy[node_id],
        candidate,
        draw_scale=draw_scale,
    )
    seen_edge_indices: set[int] = set()
    for neighbor_id in sorted(neighbor_ids):
        for segment in context.contraction_segments_by_node.get(neighbor_id, ()):
            if segment.edge_index in seen_edge_indices:
                continue
            seen_edge_indices.add(segment.edge_index)
            if node_id in {segment.left_id, segment.right_id}:
                continue
            if _segments_cross_2d(start, end, segment.start, segment.end):
                return True
    return False


def _candidate_tip_enters_neighbor_space_2d(
    component: _LayoutComponent,
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
    candidate: np.ndarray,
    prohibited_neighbor_ids: set[int],
    processed_node_ids: set[int],
    draw_scale: float,
) -> bool:
    if component.structure_kind == "grid3d":
        blocked_neighbor_ids = {
            other_id for other_id in prohibited_neighbor_ids if other_id not in processed_node_ids
        }
    else:
        blocked_neighbor_ids = set(prohibited_neighbor_ids)

    if not blocked_neighbor_ids:
        return False
    origin = context.positions_xy[node_id]
    _start, tip = _dangling_stub_segment_2d(origin, candidate, draw_scale=draw_scale)
    radius = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6)
    prohibited_coords = np.stack(
        [context.positions_xy[other_id] for other_id in sorted(blocked_neighbor_ids)]
    )
    deltas = prohibited_coords - tip
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    return bool(np.any(distances_sq < (radius**2)))


def _candidate_crosses_neighbor_stubs_2d(
    graph: _GraphData,
    directions: AxisDirections,
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
    neighbor_ids: set[int],
    candidate: np.ndarray,
    draw_scale: float,
) -> bool:
    start, end = _dangling_stub_segment_2d(
        context.positions_xy[node_id],
        candidate,
        draw_scale=draw_scale,
    )
    for neighbor_id in sorted(neighbor_ids):
        axis_count = max(graph.nodes[neighbor_id].degree, 1)
        for axis_index in range(axis_count):
            axis_key = (neighbor_id, axis_index)
            if axis_key not in context.dangling_axes:
                continue
            other_direction = directions.get(axis_key)
            if other_direction is None:
                continue
            other_start, other_end = _dangling_stub_segment_2d(
                context.positions_xy[neighbor_id],
                other_direction,
                draw_scale=draw_scale,
            )
            if _segments_cross_2d(start, end, other_start, other_end):
                return True
    return False


def _candidate_is_valid_2d(
    graph: _GraphData,
    directions: AxisDirections,
    context: _FreeDirectionContext2D,
    *,
    node_id: int,
    axis_index: int,
    candidate: np.ndarray,
    neighbor_hops: int,
    processed_node_ids: set[int],
    draw_scale: float,
) -> bool:
    if _candidate_conflicts_with_node_axes_2d(
        graph,
        directions,
        context,
        node_id=node_id,
        axis_index=axis_index,
        candidate=candidate,
    ):
        return False

    if neighbor_hops <= 0:
        return True

    component = context.component_by_node[node_id]
    first_neighbors = set(context.neighbor_ids_by_node.get(node_id, ()))
    prohibited_neighbors = set(first_neighbors)
    if neighbor_hops > 1:
        prohibited_neighbors.update(context.second_neighbor_ids_by_node.get(node_id, ()))

    if _candidate_tip_enters_neighbor_space_2d(
        component,
        context,
        node_id=node_id,
        candidate=candidate,
        prohibited_neighbor_ids=prohibited_neighbors,
        processed_node_ids=processed_node_ids,
        draw_scale=draw_scale,
    ):
        return False

    if _candidate_crosses_neighbor_bonds_2d(
        context,
        node_id=node_id,
        neighbor_ids=prohibited_neighbors,
        candidate=candidate,
        draw_scale=draw_scale,
    ):
        return False

    return not _candidate_crosses_neighbor_stubs_2d(
        graph,
        directions,
        context,
        node_id=node_id,
        neighbor_ids=prohibited_neighbors,
        candidate=candidate,
        draw_scale=draw_scale,
    )


def _compute_free_directions_2d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
    layout_components: tuple[_LayoutComponent, ...],
) -> None:
    del contraction_groups
    if _try_compute_chain_free_directions_2d(
        graph,
        directions,
        layout_components=layout_components,
    ):
        return

    context = _build_context(graph, positions, layout_components=layout_components)
    rng = np.random.default_rng()
    processed_node_ids: set[int] = set()

    for component in layout_components:
        node_plans = _component_node_plans_2d(component, positions)
        neighbor_hops = _neighbor_hops_for_component_2d(component)
        for node_plan in node_plans:
            node_id = node_plan.node_id
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            axis_count = max(node.degree, 1)
            unassigned_axis_indices = [
                axis_index
                for axis_index in range(axis_count)
                if (node_id, axis_index) not in directions
            ]
            named_directions = {
                axis_index: _normalize_2d(named_direction)
                for axis_index in unassigned_axis_indices
                for named_direction in [
                    _direction_from_axis_name(
                        node.axes_names[axis_index] if axis_index < len(node.axes_names) else None,
                        dimensions=2,
                    )
                ]
                if named_direction is not None
            }
            outward_reference = _shared_virtual_hub_outward_reference_2d(
                context,
                node_id=node_id,
            )
            deterministic_candidates = _node_candidate_directions_2d(
                component,
                positions,
                node_id=node_id,
                behavior=node_plan.behavior,
                outward_reference=outward_reference,
                rng=rng,
            )
            fallback_reference = (
                outward_reference
                if outward_reference is not None
                else _reference_for_behavior_2d(node_plan.behavior)
                if node_plan.behavior is not None
                else (
                    context.positions_xy[node_id]
                    - _component_centroid(component, positions, dimensions=2)
                )
            )
            if float(np.linalg.norm(fallback_reference)) < 1e-9:
                fallback_reference = np.array([0.0, 1.0], dtype=float)
            random_candidates: tuple[np.ndarray, ...] | None = None

            for axis_index in range(axis_count):
                axis_key = (node_id, axis_index)
                if axis_key in directions:
                    continue

                def is_valid_candidate(
                    candidate: np.ndarray,
                    *,
                    node_id: int = node_id,
                    axis_index: int = axis_index,
                    neighbor_hops: int = neighbor_hops,
                ) -> bool:
                    return _candidate_is_valid_2d(
                        graph,
                        directions,
                        context,
                        node_id=node_id,
                        axis_index=axis_index,
                        candidate=candidate,
                        neighbor_hops=neighbor_hops,
                        processed_node_ids=processed_node_ids,
                        draw_scale=draw_scale,
                    )

                named_direction = named_directions.get(axis_index)
                if named_direction is not None and is_valid_candidate(named_direction):
                    directions[axis_key] = named_direction
                    continue

                picked = _first_valid_candidate_2d(
                    candidates=deterministic_candidates,
                    is_valid=is_valid_candidate,
                )
                if picked is None:
                    if random_candidates is None:
                        random_candidates = _random_fallback_candidates_2d(
                            deterministic=deterministic_candidates,
                            reference=np.asarray(fallback_reference, dtype=float).reshape(-1)[:2],
                            rng=rng,
                        )
                    picked = _pick_candidate_direction_2d(
                        candidates=random_candidates,
                        is_valid=is_valid_candidate,
                    )
                directions[axis_key] = _normalize_2d(picked)

            processed_node_ids.add(node_id)


__all__ = [
    "_build_context",
    "_compute_free_directions_2d",
    "_direction_angle_conflicts_2d",
    "_direction_conflicts_2d",
    "_is_dangling_leg_axis",
    "_pick_candidate_direction_2d",
    "_used_axis_directions",
]
