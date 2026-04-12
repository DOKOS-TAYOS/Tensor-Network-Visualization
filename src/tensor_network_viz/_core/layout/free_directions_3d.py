"""3D free-axis direction assignment."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..graph import _GraphData
from ..layout_structure import _component_orthogonal_basis, _LayoutComponent
from .direction_common import (
    _component_centroid,
    _direction_from_axis_name,
    _normalize_direction,
    _sorted_direction_candidates,
)
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
    rounded = np.round(normalized, decimals=6)
    return (float(rounded[0]), float(rounded[1]), float(rounded[2]))


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


def _iter_deduped_local_order_3d(
    local_order: tuple[tuple[float, float, float], ...],
) -> Iterator[tuple[float, float, float]]:
    seen: set[tuple[float, float, float]] = set()
    for local_direction in local_order:
        if local_direction in seen:
            continue
        seen.add(local_direction)
        yield local_direction
    for local_direction in _CUBE_LOCAL_ORDER_3D:
        if local_direction in seen:
            continue
        seen.add(local_direction)
        yield local_direction


def _world_candidates_from_local_order_3d(
    frame: _LocalFrame3D,
    local_order: tuple[tuple[float, float, float], ...],
    *,
    skip_local_directions: frozenset[tuple[float, float, float]] = frozenset(),
) -> Iterator[np.ndarray]:
    seen: set[tuple[float, float, float]] = set()
    for local_direction in _iter_deduped_local_order_3d(local_order):
        if local_direction in skip_local_directions:
            continue
        candidate = _local_to_world_direction_3d(frame, local_direction)
        candidate_key = _direction_key_3d(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        yield candidate


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
) -> Iterator[np.ndarray]:
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
) -> Iterator[np.ndarray]:
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
) -> Iterator[np.ndarray]:
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
) -> Iterator[np.ndarray]:
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
    candidates: Iterable[np.ndarray],
    directions: AxisDirections,
    tried_direction_keys: set[tuple[float, float, float]],
    *,
    node_id: int,
    axis_index: int,
    axis_count: int,
    tried_directions: list[np.ndarray] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    last_tried: np.ndarray | None = None
    for raw_candidate in candidates:
        candidate = _normalize_direction(raw_candidate, dimensions=3)
        candidate_key = _direction_key_3d(candidate)
        if candidate_key in tried_direction_keys:
            continue
        tried_direction_keys.add(candidate_key)
        last_tried = candidate
        if tried_directions is not None:
            tried_directions.append(candidate)
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

            named_directions = {
                axis_index: _normalize_direction(named_direction, dimensions=3)
                for axis_index in unassigned_axis_indices
                for named_direction in [
                    _direction_from_axis_name(
                        node.axes_names[axis_index] if axis_index < len(node.axes_names) else None,
                        dimensions=3,
                    )
                ]
                if named_direction is not None
            }

            deterministic_candidates = _node_candidate_directions_3d(
                component,
                positions,
                node_id=node_id,
            )
            random_candidates: tuple[np.ndarray, ...] | None = None
            tried_direction_keys: set[tuple[float, float, float]] = set()
            tried_directions: list[np.ndarray] = []
            last_tried: np.ndarray | None = None

            for axis_index in range(axis_count):
                axis_key = (node_id, axis_index)
                if axis_key in directions:
                    continue

                named_direction = named_directions.get(axis_index)
                if named_direction is not None:
                    named_key = _direction_key_3d(named_direction)
                    if named_key not in tried_direction_keys:
                        tried_direction_keys.add(named_key)
                        last_tried = named_direction
                        tried_directions.append(named_direction)
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
                    tried_directions=tried_directions,
                )
                if deterministic_last_tried is not None:
                    last_tried = deterministic_last_tried
                if picked is None:
                    if random_candidates is None:
                        random_candidates = _random_fallback_candidates_3d(
                            deterministic=tuple(tried_directions),
                            reference=(
                                last_tried
                                if last_tried is not None
                                else np.array([0.0, 0.0, 1.0], dtype=float)
                            ),
                            rng=rng,
                        )
                    picked, random_last_tried = _first_valid_candidate_3d(
                        random_candidates,
                        directions,
                        tried_direction_keys,
                        node_id=node_id,
                        axis_index=axis_index,
                        axis_count=axis_count,
                        tried_directions=tried_directions,
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
    conflict_data: object | None = None,
    assigned_state: object | None = None,
) -> bool:
    del (
        node_id,
        origin,
        bond_segments,
        positions,
        draw_scale,
        strict_physical_node_clearance,
        conflict_data,
    )
    direction_unit = _normalize_direction(direction, dimensions=3)

    for _other_origin, other_direction in assigned_segments:
        if _direction_angle_conflicts_3d(direction_unit, other_direction):
            return True

    if assigned_state is None or not hasattr(assigned_state, "directions_view"):
        return False
    assigned_count = int(getattr(assigned_state, "count", 0))
    if assigned_count <= 0:
        return False
    other_directions = assigned_state.directions_view()
    if other_directions.size == 0:
        return False
    threshold_dot = math.cos(math.radians(_ANGLE_THRESHOLD_DEGREES_3D))
    dots = other_directions @ direction_unit
    return bool(np.any(dots >= threshold_dot))


__all__ = [
    "_compute_free_directions_3d",
    "_direction_angle_conflicts_3d",
    "_direction_conflicts_3d",
    "_pick_candidate_direction_3d",
    "_random_direction_bucket_3d",
]
