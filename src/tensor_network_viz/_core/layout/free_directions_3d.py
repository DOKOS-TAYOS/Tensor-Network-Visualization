"""3D free-axis direction assignment."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...config import PlotConfig
from ..graph import _GraphData
from ..layout_structure import _component_orthogonal_basis, _LayoutComponent
from .direction_common import (
    _component_centroid,
    _dedupe_candidate_directions,
    _direction_from_axis_name,
    _forced_dangling_direction_from_axis_name,
    _normalize_direction,
    _sorted_direction_candidates,
)
from .geometry import (
    _dangling_stub_segment_3d,
    _segment_point_min_distance_sq_3d,
    _segment_point_min_distance_sq_3d_many,
    _segment_segment_min_distance_sq_3d,
)
from .parameters import _STUB_BOND_CLEAR_3D
from .types import AxisDirections, NodePositions

_ANGLE_THRESHOLD_DEGREES_3D: float = 10.0
_RANDOM_FALLBACK_COUNT_3D: int = 16

_FRONT_LOCAL_3D: tuple[float, float, float] = (0.0, 0.0, 1.0)
_BACK_LOCAL_3D: tuple[float, float, float] = (0.0, 0.0, -1.0)
_UP_LOCAL_3D: tuple[float, float, float] = (0.0, 1.0, 0.0)
_DOWN_LOCAL_3D: tuple[float, float, float] = (0.0, -1.0, 0.0)
_RIGHT_LOCAL_3D: tuple[float, float, float] = (1.0, 0.0, 0.0)
_LEFT_LOCAL_3D: tuple[float, float, float] = (-1.0, 0.0, 0.0)

_AXIS_LOCAL_ORDER_3D: tuple[tuple[float, float, float], ...] = (
    _FRONT_LOCAL_3D,
    _BACK_LOCAL_3D,
    _UP_LOCAL_3D,
    _DOWN_LOCAL_3D,
    _RIGHT_LOCAL_3D,
    _LEFT_LOCAL_3D,
)
_DIAGONAL_LOCAL_ORDER_3D: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, -1.0),
)
_CUBE_LOCAL_ORDER_3D: tuple[tuple[float, float, float], ...] = (
    *_AXIS_LOCAL_ORDER_3D,
    *_DIAGONAL_LOCAL_ORDER_3D,
)


@dataclass(frozen=True)
class _LocalFrame3D:
    front: np.ndarray
    right: np.ndarray
    up: np.ndarray


@dataclass(frozen=True)
class _LayerInfo3D:
    layer_index_by_node: dict[int, int]
    layer_count: int


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


def _direction_angle_conflicts_3d(
    direction: np.ndarray,
    other_direction: np.ndarray,
    *,
    threshold_degrees: float = _ANGLE_THRESHOLD_DEGREES_3D,
) -> bool:
    dot = float(
        np.dot(
            _normalize_direction(direction, dimensions=3),
            _normalize_direction(other_direction, dimensions=3),
        )
    )
    threshold_dot = math.cos(math.radians(float(threshold_degrees)))
    return dot >= threshold_dot


def _pick_candidate_direction_3d(
    *,
    candidates: tuple[np.ndarray, ...],
    is_valid: Callable[[np.ndarray], bool],
) -> np.ndarray:
    last_tried: np.ndarray | None = None
    for candidate in candidates:
        normalized = _normalize_direction(candidate, dimensions=3)
        last_tried = normalized.copy()
        if is_valid(normalized):
            return normalized
    if last_tried is None:
        raise RuntimeError(
            "No candidate directions were generated for the 3D free-axis assignment."
        )
    return last_tried


def _direction_key_3d(direction: np.ndarray) -> tuple[float, float, float]:
    normalized = _normalize_direction(direction, dimensions=3)
    return tuple(float(value) for value in np.round(normalized, decimals=6))


def _orthogonal_hint_3d(vector: np.ndarray) -> np.ndarray:
    unit = _normalize_direction(vector, dimensions=3)
    basis = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    return min(basis, key=lambda candidate: abs(float(np.dot(unit, candidate)))).copy()


def _orthonormal_frame_3d(
    *,
    front: np.ndarray,
    right_hint: np.ndarray,
    up_hint: np.ndarray | None = None,
) -> _LocalFrame3D:
    front_unit = _normalize_direction(front, dimensions=3)
    right_raw = np.asarray(right_hint, dtype=float).reshape(-1)[:3]
    right_raw = right_raw - front_unit * float(np.dot(right_raw, front_unit))
    if float(np.linalg.norm(right_raw)) < 1e-9:
        right_raw = _orthogonal_hint_3d(front_unit)
        right_raw = right_raw - front_unit * float(np.dot(right_raw, front_unit))
    right_unit = _normalize_direction(right_raw, dimensions=3)
    up_unit = np.cross(front_unit, right_unit)
    up_unit = _normalize_direction(up_unit, dimensions=3)
    if up_hint is not None and float(np.dot(up_unit, up_hint)) < 0.0:
        right_unit = -right_unit
        up_unit = -up_unit
    return _LocalFrame3D(front=front_unit, right=right_unit, up=up_unit)


def _local_to_world_direction_3d(
    frame: _LocalFrame3D,
    local_direction: tuple[float, float, float],
) -> np.ndarray:
    x_coord, y_coord, z_coord = local_direction
    return _normalize_direction(
        frame.right * x_coord + frame.up * y_coord + frame.front * z_coord,
        dimensions=3,
    )


def _local_surface_reference_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    frame: _LocalFrame3D,
) -> np.ndarray:
    origin = np.asarray(positions[node_id], dtype=float).reshape(-1)[:3]
    centroid = _component_centroid(component, positions, dimensions=3)
    outward = origin - centroid
    return np.array(
        [
            float(np.dot(outward, frame.right)),
            float(np.dot(outward, frame.up)),
        ],
        dtype=float,
    )


def _sort_local_surface_directions_3d(
    local_directions: tuple[tuple[float, float, float], ...],
    reference_xy: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    reference = np.asarray(reference_xy, dtype=float).reshape(-1)[:2]
    if float(np.linalg.norm(reference)) < 1e-9:
        return local_directions
    reference = reference / np.linalg.norm(reference)
    scored: list[tuple[float, int, tuple[float, float, float]]] = []
    for index, local_direction in enumerate(local_directions):
        surface = np.array([local_direction[0], local_direction[1]], dtype=float)
        surface_norm = float(np.linalg.norm(surface))
        score = 0.0 if surface_norm < 1e-9 else -float(np.dot(surface / surface_norm, reference))
        scored.append((score, index, local_direction))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(local_direction for _, _, local_direction in scored)


def _surface_local_order_3d(reference_xy: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    surface_cardinals = _sort_local_surface_directions_3d(
        (_UP_LOCAL_3D, _DOWN_LOCAL_3D, _RIGHT_LOCAL_3D, _LEFT_LOCAL_3D),
        reference_xy,
    )
    surface_diagonals = _sort_local_surface_directions_3d(
        (
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
        ),
        reference_xy,
    )
    normal_diagonals = tuple(
        local_direction
        for local_direction in _DIAGONAL_LOCAL_ORDER_3D
        if abs(local_direction[2]) > 0.0
    )
    normal_diagonals = tuple(
        sorted(
            normal_diagonals,
            key=lambda direction: (
                -abs(direction[2]),
                0 if direction[2] > 0.0 else 1,
                _sort_index_for_surface_direction_3d(direction, reference_xy),
            ),
        )
    )
    return (
        _FRONT_LOCAL_3D,
        _BACK_LOCAL_3D,
        *surface_cardinals,
        *normal_diagonals,
        *surface_diagonals,
    )


def _sort_index_for_surface_direction_3d(
    local_direction: tuple[float, float, float],
    reference_xy: np.ndarray,
) -> float:
    reference = np.asarray(reference_xy, dtype=float).reshape(-1)[:2]
    surface = np.array([local_direction[0], local_direction[1]], dtype=float)
    if float(np.linalg.norm(reference)) < 1e-9 or float(np.linalg.norm(surface)) < 1e-9:
        return 0.0
    return -float(np.dot(surface / np.linalg.norm(surface), reference / np.linalg.norm(reference)))


def _stepped_surface_local_order_3d(
    *,
    layer_index: int,
    layer_count: int,
    reference_xy: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    surface_directions = _sort_local_surface_directions_3d(
        (
            _UP_LOCAL_3D,
            _DOWN_LOCAL_3D,
            _RIGHT_LOCAL_3D,
            _LEFT_LOCAL_3D,
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
        ),
        reference_xy,
    )
    front_diagonals = _sort_local_surface_directions_3d(
        tuple(direction for direction in _DIAGONAL_LOCAL_ORDER_3D if direction[2] > 0.0),
        reference_xy,
    )
    back_diagonals = _sort_local_surface_directions_3d(
        tuple(direction for direction in _DIAGONAL_LOCAL_ORDER_3D if direction[2] < 0.0),
        reference_xy,
    )
    if layer_index == 0:
        return (
            _FRONT_LOCAL_3D,
            *front_diagonals,
            _BACK_LOCAL_3D,
            *back_diagonals,
            *surface_directions,
        )
    if layer_index == layer_count - 1:
        return (
            _BACK_LOCAL_3D,
            *back_diagonals,
            _FRONT_LOCAL_3D,
            *front_diagonals,
            *surface_directions,
        )
    closer_to_first = layer_index <= (layer_count - 1) / 2.0
    first_side = _FRONT_LOCAL_3D if closer_to_first else _BACK_LOCAL_3D
    first_diagonals = front_diagonals if closer_to_first else back_diagonals
    second_side = _BACK_LOCAL_3D if closer_to_first else _FRONT_LOCAL_3D
    second_diagonals = back_diagonals if closer_to_first else front_diagonals
    return (
        *surface_directions,
        first_side,
        *first_diagonals,
        second_side,
        *second_diagonals,
    )


def _dedupe_local_order_3d(
    local_order: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    seen: set[tuple[float, float, float]] = set()
    ordered: list[tuple[float, float, float]] = []
    for local_direction in local_order:
        if local_direction in seen:
            continue
        seen.add(local_direction)
        ordered.append(local_direction)
    for local_direction in _CUBE_LOCAL_ORDER_3D:
        if local_direction in seen:
            continue
        seen.add(local_direction)
        ordered.append(local_direction)
    return tuple(ordered)


def _world_candidates_from_local_order_3d(
    frame: _LocalFrame3D,
    local_order: tuple[tuple[float, float, float], ...],
    *,
    skip_local_directions: frozenset[tuple[float, float, float]] = frozenset(),
) -> tuple[np.ndarray, ...]:
    candidates = [
        _local_to_world_direction_3d(frame, local_direction)
        for local_direction in _dedupe_local_order_3d(local_order)
        if local_direction not in skip_local_directions
    ]
    return _dedupe_candidate_directions(candidates, dimensions=3)


def _component_layer_info_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    normal: np.ndarray,
) -> _LayerInfo3D:
    node_values = {
        node_id: float(np.dot(np.asarray(positions[node_id], dtype=float).reshape(-1)[:3], normal))
        for node_id in component.node_ids
        if node_id in positions
    }
    if not node_values:
        return _LayerInfo3D(layer_index_by_node={}, layer_count=1)
    rounded_values = sorted({round(value, 6) for value in node_values.values()})
    layer_index_by_node = {
        node_id: min(
            range(len(rounded_values)),
            key=lambda index: abs(round(value, 6) - rounded_values[index]),
        )
        for node_id, value in node_values.items()
    }
    return _LayerInfo3D(
        layer_index_by_node=layer_index_by_node,
        layer_count=max(len(rounded_values), 1),
    )


def _chain_candidates_3d(
    component: _LayoutComponent,
    frame: _LocalFrame3D,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    chain_order = [chain_id for chain_id in component.chain_order if chain_id in component.node_ids]
    is_interior = node_id in chain_order[1:-1]
    skip = frozenset({_RIGHT_LOCAL_3D, _LEFT_LOCAL_3D}) if is_interior else frozenset()
    return _world_candidates_from_local_order_3d(
        frame,
        _CUBE_LOCAL_ORDER_3D,
        skip_local_directions=skip,
    )


def _grid3d_frame_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
) -> tuple[_LocalFrame3D, int]:
    if component.grid3d_mapping is None or node_id not in component.grid3d_mapping:
        axis, lateral, normal = _component_orthogonal_basis(component, positions)
        return _orthonormal_frame_3d(front=normal, right_hint=axis, up_hint=lateral), 0

    coords_by_node = component.grid3d_mapping
    i_vals = [coords[0] for coords in coords_by_node.values()]
    j_vals = [coords[1] for coords in coords_by_node.values()]
    k_vals = [coords[2] for coords in coords_by_node.values()]
    mins = (min(i_vals), min(j_vals), min(k_vals))
    maxs = (max(i_vals), max(j_vals), max(k_vals))
    coords = coords_by_node[node_id]
    signs = tuple(
        1.0 if coord == max_value else -1.0 if coord == min_value else 0.0
        for coord, min_value, max_value in zip(coords, mins, maxs, strict=True)
    )
    boundary_count = sum(1 for sign in signs if sign != 0.0)
    front = np.array(signs, dtype=float)
    if float(np.linalg.norm(front)) < 1e-9:
        centroid = _component_centroid(component, positions, dimensions=3)
        front = np.asarray(positions[node_id], dtype=float).reshape(-1)[:3] - centroid
    if float(np.linalg.norm(front)) < 1e-9:
        front = np.array([0.0, 0.0, 1.0], dtype=float)

    axis_basis = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    right_hint = next(
        (axis_basis[index] for index, sign in enumerate(signs) if sign == 0.0),
        _orthogonal_hint_3d(front),
    )
    up_hint = next(
        (
            axis_basis[index]
            for index, sign in enumerate(signs)
            if sign == 0.0 and not np.allclose(axis_basis[index], right_hint)
        ),
        None,
    )
    return (
        _orthonormal_frame_3d(
            front=front,
            right_hint=right_hint,
            up_hint=up_hint,
        ),
        boundary_count,
    )


def _grid3d_candidates_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    frame, boundary_count = _grid3d_frame_3d(component, positions, node_id=node_id)
    if boundary_count == 1:
        reference_xy = _local_surface_reference_3d(
            component,
            positions,
            node_id=node_id,
            frame=frame,
        )
        local_order = _surface_local_order_3d(reference_xy)
    elif boundary_count == 2:
        local_order = _CUBE_LOCAL_ORDER_3D
    elif boundary_count >= 3:
        local_order = (
            _FRONT_LOCAL_3D,
            _BACK_LOCAL_3D,
            _LEFT_LOCAL_3D,
            _RIGHT_LOCAL_3D,
            _UP_LOCAL_3D,
            _DOWN_LOCAL_3D,
            *_DIAGONAL_LOCAL_ORDER_3D,
        )
    else:
        local_order = _CUBE_LOCAL_ORDER_3D
    return _world_candidates_from_local_order_3d(frame, local_order)


def _surface_or_stepped_candidates_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    axis: np.ndarray,
    lateral: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, ...]:
    layer_info = _component_layer_info_3d(component, positions, normal=normal)
    front = -normal if layer_info.layer_count > 1 else normal
    frame = _orthonormal_frame_3d(front=front, right_hint=axis, up_hint=lateral)
    reference_xy = _local_surface_reference_3d(component, positions, node_id=node_id, frame=frame)
    if layer_info.layer_count > 1:
        local_order = _stepped_surface_local_order_3d(
            layer_index=layer_info.layer_index_by_node.get(node_id, 0),
            layer_count=layer_info.layer_count,
            reference_xy=reference_xy,
        )
    else:
        local_order = _surface_local_order_3d(reference_xy)
    return _world_candidates_from_local_order_3d(frame, local_order)


def _node_candidate_directions_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    axis, lateral, normal = _component_orthogonal_basis(component, positions)
    if component.structure_kind == "chain":
        frame = _orthonormal_frame_3d(front=normal, right_hint=axis, up_hint=lateral)
        return _chain_candidates_3d(component, frame, node_id=node_id)
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        return _grid3d_candidates_3d(component, positions, node_id=node_id)
    return _surface_or_stepped_candidates_3d(
        component,
        positions,
        node_id=node_id,
        axis=axis,
        lateral=lateral,
        normal=normal,
    )


def _random_direction_bucket_3d(
    *,
    blocked: tuple[np.ndarray, ...],
    count: int = _RANDOM_FALLBACK_COUNT_3D,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    generator = rng if rng is not None else np.random.default_rng()
    blocked_norm = [_normalize_direction(direction, dimensions=3) for direction in blocked]
    randoms: list[np.ndarray] = []
    attempts = 0
    while len(randoms) < int(count) and attempts < 8192:
        attempts += 1
        raw = generator.normal(size=3)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-12:
            continue
        candidate = raw / norm
        if any(float(np.dot(candidate, other)) > 0.995 for other in [*blocked_norm, *randoms]):
            continue
        randoms.append(candidate)
    if len(randoms) < int(count):
        raise RuntimeError("Unable to generate the random fallback directions for the 3D layout.")
    return tuple(randoms)


def _random_fallback_candidates_3d(
    *,
    deterministic: tuple[np.ndarray, ...],
    reference: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    return _sorted_direction_candidates(
        reference,
        _random_direction_bucket_3d(blocked=deterministic, rng=rng),
        dimensions=3,
    )


def _candidate_conflicts_with_node_axes_3d(
    directions: AxisDirections,
    *,
    node_id: int,
    axis_index: int,
    axis_count: int,
    candidate: np.ndarray,
) -> bool:
    for other_axis_index in range(axis_count):
        if other_axis_index == axis_index:
            continue
        other_direction = directions.get((node_id, other_axis_index))
        if other_direction is None:
            continue
        if _direction_angle_conflicts_3d(candidate, other_direction):
            return True
    return False


def _first_valid_candidate_3d(
    candidates: tuple[np.ndarray, ...],
    directions: AxisDirections,
    tried_direction_keys: set[tuple[float, float, float]],
    *,
    node_id: int,
    axis_index: int,
    axis_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    last_tried: np.ndarray | None = None
    for raw_candidate in candidates:
        candidate = _normalize_direction(raw_candidate, dimensions=3)
        candidate_key = _direction_key_3d(candidate)
        if candidate_key in tried_direction_keys:
            continue
        tried_direction_keys.add(candidate_key)
        last_tried = candidate
        if not _candidate_conflicts_with_node_axes_3d(
            directions,
            node_id=node_id,
            axis_index=axis_index,
            axis_count=axis_count,
            candidate=candidate,
        ):
            return candidate, last_tried
    return None, last_tried


def _component_node_order_3d(component: _LayoutComponent) -> tuple[int, ...]:
    if component.structure_kind == "chain" and component.chain_order:
        ordered = [node_id for node_id in component.chain_order if node_id in component.node_ids]
        ordered.extend(sorted(node_id for node_id in component.node_ids if node_id not in ordered))
        return tuple(ordered)
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        coords_by_node = component.grid3d_mapping
        i_vals = [coords[0] for coords in coords_by_node.values()]
        j_vals = [coords[1] for coords in coords_by_node.values()]
        k_vals = [coords[2] for coords in coords_by_node.values()]
        mins = (min(i_vals), min(j_vals), min(k_vals))
        maxs = (max(i_vals), max(j_vals), max(k_vals))

        def grid_priority(node_id: int) -> tuple[int, int, tuple[int, int, int], int]:
            coords = coords_by_node[node_id]
            shell = min(
                min(coord - min_value, max_value - coord)
                for coord, min_value, max_value in zip(coords, mins, maxs, strict=True)
            )
            boundary_count = sum(
                1
                for coord, min_value, max_value in zip(coords, mins, maxs, strict=True)
                if coord in {min_value, max_value}
            )
            return (shell, -boundary_count, coords, node_id)

        return tuple(
            sorted(
                (node_id for node_id in component.node_ids if node_id in coords_by_node),
                key=grid_priority,
            )
        )
    return tuple(sorted(component.node_ids))


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    layout_components: tuple[_LayoutComponent, ...],
) -> None:
    del draw_scale
    rng = np.random.default_rng()

    for component in layout_components:
        for node_id in _component_node_order_3d(component):
            if node_id not in graph.nodes or node_id not in positions:
                continue
            node = graph.nodes[node_id]
            axis_count = max(node.degree, 1)
            unassigned_axis_indices = [
                axis_index
                for axis_index in range(axis_count)
                if (node_id, axis_index) not in directions
            ]
            if not unassigned_axis_indices:
                continue

            forced_directions = {
                axis_index: _normalize_direction(forced_direction, dimensions=3)
                for axis_index in unassigned_axis_indices
                for forced_direction in [
                    _forced_dangling_direction_from_axis_name(
                        graph,
                        node_id=node_id,
                        axis_index=axis_index,
                        axis_name=(
                            node.axes_names[axis_index]
                            if axis_index < len(node.axes_names)
                            else None
                        ),
                        dimensions=3,
                    )
                ]
                if forced_direction is not None
            }
            if forced_directions and len(forced_directions) == len(unassigned_axis_indices):
                for axis_index in unassigned_axis_indices:
                    directions[(node_id, axis_index)] = forced_directions[axis_index]
                continue

            deterministic_candidates = _node_candidate_directions_3d(
                component,
                positions,
                node_id=node_id,
            )
            fallback_reference = deterministic_candidates[0]
            random_candidates: tuple[np.ndarray, ...] | None = None
            tried_direction_keys: set[tuple[float, float, float]] = set()
            last_tried: np.ndarray | None = None

            for axis_index in range(axis_count):
                axis_key = (node_id, axis_index)
                if axis_key in directions:
                    continue

                forced_direction = forced_directions.get(axis_index)
                if forced_direction is not None:
                    directions[axis_key] = forced_direction
                    continue

                axis_name = (
                    node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
                )
                named_direction = _direction_from_axis_name(axis_name, dimensions=3)
                if named_direction is not None:
                    named_direction = _normalize_direction(named_direction, dimensions=3)
                    named_key = _direction_key_3d(named_direction)
                    if named_key not in tried_direction_keys:
                        tried_direction_keys.add(named_key)
                        last_tried = named_direction
                        if not _candidate_conflicts_with_node_axes_3d(
                            directions,
                            node_id=node_id,
                            axis_index=axis_index,
                            axis_count=axis_count,
                            candidate=named_direction,
                        ):
                            directions[axis_key] = named_direction
                            continue

                picked, deterministic_last_tried = _first_valid_candidate_3d(
                    deterministic_candidates,
                    directions,
                    tried_direction_keys,
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_count=axis_count,
                )
                if deterministic_last_tried is not None:
                    last_tried = deterministic_last_tried
                if picked is None:
                    if random_candidates is None:
                        random_candidates = _random_fallback_candidates_3d(
                            deterministic=deterministic_candidates,
                            reference=fallback_reference,
                            rng=rng,
                        )
                    picked, random_last_tried = _first_valid_candidate_3d(
                        random_candidates,
                        directions,
                        tried_direction_keys,
                        node_id=node_id,
                        axis_index=axis_index,
                        axis_count=axis_count,
                    )
                    if random_last_tried is not None:
                        last_tried = random_last_tried
                if picked is None:
                    if last_tried is None:
                        raise RuntimeError(
                            "No candidate directions were generated for the 3D free-axis "
                            "assignment."
                        )
                    picked = last_tried
                directions[axis_key] = _normalize_direction(picked, dimensions=3)


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
    "_direction_angle_conflicts_3d",
    "_direction_conflicts_3d",
    "_pick_candidate_direction_3d",
    "_random_direction_bucket_3d",
]
