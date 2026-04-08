"""Shared helpers for axis-direction assignment."""

from __future__ import annotations

import math

import numpy as np

from ..axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from ..graph import _GraphData
from ..layout_structure import _LayoutComponent
from .parameters import _FREE_DIR_OVERLAP_THRESHOLD
from .types import AxisDirections, NodePositions, Vector

_CARDINAL_DIRECTIONS_2D: tuple[np.ndarray, ...] = (
    np.array([0.0, 1.0], dtype=float),
    np.array([1.0, 0.0], dtype=float),
    np.array([0.0, -1.0], dtype=float),
    np.array([-1.0, 0.0], dtype=float),
)
_DIAGONAL_DIRECTIONS_2D: tuple[np.ndarray, ...] = (
    np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0),
    np.array([-1.0, 1.0], dtype=float) / np.sqrt(2.0),
    np.array([-1.0, -1.0], dtype=float) / np.sqrt(2.0),
    np.array([1.0, -1.0], dtype=float) / np.sqrt(2.0),
)
_SEMIDIAGONAL_DIRECTIONS_2D: tuple[np.ndarray, ...] = tuple(
    np.array(
        [math.cos(math.radians(angle_degrees)), math.sin(math.radians(angle_degrees))],
        dtype=float,
    )
    for angle_degrees in (67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 22.5)
)
_NORTH_BEHAVIOR_ORDER_2D: tuple[np.ndarray, ...] = (
    np.array([0.0, 1.0], dtype=float),
    np.array([0.0, -1.0], dtype=float),
    np.array([1.0, 0.0], dtype=float),
    np.array([-1.0, 0.0], dtype=float),
    *_DIAGONAL_DIRECTIONS_2D,
    *_SEMIDIAGONAL_DIRECTIONS_2D,
)


def _rotate_direction_2d(
    direction: np.ndarray,
    *,
    quarter_turns_clockwise: int,
) -> np.ndarray:
    x_coord, y_coord = np.asarray(direction, dtype=float).reshape(-1)[:2]
    turns = int(quarter_turns_clockwise) % 4
    if turns == 0:
        return np.array([x_coord, y_coord], dtype=float)
    if turns == 1:
        return np.array([y_coord, -x_coord], dtype=float)
    if turns == 2:
        return np.array([-x_coord, -y_coord], dtype=float)
    return np.array([-y_coord, x_coord], dtype=float)


def _behavior_direction_order_2d(behavior: str) -> tuple[np.ndarray, ...]:
    turns_by_behavior = {
        "north": 0,
        "east": 1,
        "south": 2,
        "west": 3,
    }
    turns = turns_by_behavior.get(behavior.lower().strip(), 0)
    return tuple(
        _normalize_direction(
            _rotate_direction_2d(direction, quarter_turns_clockwise=turns),
            dimensions=2,
        )
        for direction in _NORTH_BEHAVIOR_ORDER_2D
    )


def _is_dangling_leg_axis(graph: _GraphData, node_id: int, axis_index: int) -> bool:
    """True when this node axis is the tensor endpoint of a dangling edge."""
    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        endpoint = edge.endpoints[0]
        if endpoint.node_id == int(node_id) and endpoint.axis_index == int(axis_index):
            return True
    return False


def _direction_from_axis_name(
    axis_name: str | None,
    *,
    dimensions: int,
) -> np.ndarray | None:
    """Return direction vector for an axis name in 2D or 3D."""
    if not axis_name:
        return None
    key = axis_name.lower().strip()
    if dimensions == 2 and key in _AXIS_DIR_2D:
        return np.array(_AXIS_DIR_2D[key], dtype=float)
    if dimensions == 3 and key in _AXIS_DIR_3D:
        return np.array(_AXIS_DIR_3D[key], dtype=float)
    return None


def _used_axis_directions(
    directions: AxisDirections,
    *,
    node_id: int,
    axis_count: int,
) -> list[np.ndarray]:
    return [
        directions[(node_id, axis_index)]
        for axis_index in range(axis_count)
        if (node_id, axis_index) in directions
    ]


def _direction_has_space(
    direction: np.ndarray,
    used_dirs: list[np.ndarray],
    *,
    overlap_threshold: float = _FREE_DIR_OVERLAP_THRESHOLD,
    cumulative: bool = True,
) -> bool:
    dimensions = int(len(np.asarray(direction).reshape(-1)))
    direction_unit = _normalize_direction(direction, dimensions=dimensions)
    if cumulative:
        overlap = sum(
            max(0.0, float(np.dot(direction_unit, used_direction))) for used_direction in used_dirs
        )
        return overlap < overlap_threshold
    return all(
        float(
            np.dot(
                direction_unit,
                _normalize_direction(used_direction, dimensions=dimensions),
            )
        )
        < overlap_threshold
        for used_direction in used_dirs
    )


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)


def _normalize_direction(vector: np.ndarray, *, dimensions: int) -> np.ndarray:
    direction = np.asarray(vector, dtype=float).reshape(-1)[:dimensions]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        if dimensions == 2:
            return np.array([1.0, 0.0], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return direction / norm


def _dedupe_candidate_directions(
    candidates: list[np.ndarray],
    *,
    dimensions: int,
) -> tuple[np.ndarray, ...]:
    unique: list[np.ndarray] = []
    for candidate in candidates:
        normalized = _normalize_direction(candidate, dimensions=dimensions)
        if any(float(np.dot(normalized, other)) > 0.995 for other in unique):
            continue
        unique.append(normalized)
    return tuple(unique)


def _sorted_direction_candidates(
    reference: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    *,
    dimensions: int,
) -> tuple[np.ndarray, ...]:
    reference_vec = np.asarray(reference, dtype=float).reshape(-1)[:dimensions]
    if float(np.linalg.norm(reference_vec)) < 1e-9:
        return _dedupe_candidate_directions(list(candidates), dimensions=dimensions)

    reference_unit = _normalize_direction(reference_vec, dimensions=dimensions)
    scored: list[tuple[float, int, np.ndarray]] = []
    for index, candidate in enumerate(candidates):
        normalized = _normalize_direction(candidate, dimensions=dimensions)
        scored.append((-float(np.dot(normalized, reference_unit)), index, normalized))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(candidate for _, _, candidate in scored)


def _component_centroid(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    dimensions: int,
) -> np.ndarray:
    source_ids = component.anchor_node_ids or component.visible_node_ids or component.node_ids
    anchor_ids = tuple(node_id for node_id in source_ids if node_id in positions)
    if not anchor_ids:
        anchor_ids = tuple(sorted(positions))
    coords = np.stack(
        [
            np.asarray(positions[node_id], dtype=float).reshape(-1)[:dimensions]
            for node_id in anchor_ids
        ]
    )
    return coords.mean(axis=0)


def _component_main_axis_2d_like(
    component: _LayoutComponent,
    positions: NodePositions,
) -> np.ndarray:
    source_ids = component.anchor_node_ids or component.visible_node_ids or component.node_ids
    anchor_ids = tuple(node_id for node_id in source_ids if node_id in positions)
    if not anchor_ids:
        anchor_ids = tuple(sorted(positions))
    chain_node_ids = [node_id for node_id in component.chain_order if node_id in positions]
    if component.structure_kind == "chain" and len(chain_node_ids) >= 2:
        start = np.asarray(positions[chain_node_ids[0]], dtype=float).reshape(-1)[:2]
        end = np.asarray(positions[chain_node_ids[-1]], dtype=float).reshape(-1)[:2]
        return _normalize_direction(end - start, dimensions=2)
    coords = np.stack(
        [np.asarray(positions[node_id], dtype=float).reshape(-1)[:2] for node_id in anchor_ids]
    )
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return _normalize_direction(vh[0], dimensions=2)


def _grid_boundary_directions_2d(
    component: _LayoutComponent,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    if component.grid_mapping is None or node_id not in component.grid_mapping:
        return ()

    cols = [col for col, _ in component.grid_mapping.values()]
    rows = [row for _, row in component.grid_mapping.values()]
    col, row = component.grid_mapping[node_id]
    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)
    directions: list[np.ndarray] = []
    if row == max_row:
        directions.append(_CARDINAL_DIRECTIONS_2D[0])
    elif row == min_row:
        directions.append(_CARDINAL_DIRECTIONS_2D[2])
    if col == max_col:
        directions.append(_CARDINAL_DIRECTIONS_2D[1])
    elif col == min_col:
        directions.append(_CARDINAL_DIRECTIONS_2D[3])
    return tuple(directions)


def _grid3d_projected_basis_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if component.grid3d_mapping is None:
        return (
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0),
        )

    node_id_by_coords = {coords: node_id for node_id, coords in component.grid3d_mapping.items()}
    steps = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    fallback = (
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 1.0], dtype=float),
    )
    samples: list[list[np.ndarray]] = [[], [], []]

    for node_id, (i, j, k) in component.grid3d_mapping.items():
        origin = np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
        for axis_index, (di, dj, dk) in enumerate(steps):
            other_id = node_id_by_coords.get((i + di, j + dj, k + dk))
            if other_id is None:
                continue
            delta = np.asarray(positions[other_id], dtype=float).reshape(-1)[:2] - origin
            if float(np.linalg.norm(delta)) < 1e-9:
                continue
            samples[axis_index].append(delta)

    basis: list[np.ndarray] = []
    for axis_index, deltas in enumerate(samples):
        if deltas:
            basis.append(
                _normalize_direction(
                    np.sum(np.stack(deltas, axis=0), axis=0),
                    dimensions=2,
                )
            )
        else:
            basis.append(_normalize_direction(fallback[axis_index], dimensions=2))
    return basis[0], basis[1], basis[2]


def _grid3d_face_directions_2d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    if component.grid3d_mapping is None or node_id not in component.grid3d_mapping:
        return ()

    x_dir, y_dir, z_dir = _grid3d_projected_basis_2d(component, positions)
    i_vals = [i for i, _, _ in component.grid3d_mapping.values()]
    j_vals = [j for _, j, _ in component.grid3d_mapping.values()]
    k_vals = [k for _, _, k in component.grid3d_mapping.values()]
    i, j, k = component.grid3d_mapping[node_id]
    directions: list[np.ndarray] = []
    if i == max(i_vals):
        directions.append(x_dir)
    elif i == min(i_vals):
        directions.append(-x_dir)
    if j == max(j_vals):
        directions.append(y_dir)
    elif j == min(j_vals):
        directions.append(-y_dir)
    if k == max(k_vals):
        directions.append(z_dir)
    elif k == min(k_vals):
        directions.append(-z_dir)
    return tuple(directions)


def _grid3d_face_directions_3d(
    component: _LayoutComponent,
    *,
    node_id: int,
) -> tuple[np.ndarray, ...]:
    if component.grid3d_mapping is None or node_id not in component.grid3d_mapping:
        return ()

    i_vals = [i for i, _, _ in component.grid3d_mapping.values()]
    j_vals = [j for _, j, _ in component.grid3d_mapping.values()]
    k_vals = [k for _, _, k in component.grid3d_mapping.values()]
    i, j, k = component.grid3d_mapping[node_id]
    directions: list[np.ndarray] = []
    if i == max(i_vals):
        directions.append(np.array([1.0, 0.0, 0.0], dtype=float))
    elif i == min(i_vals):
        directions.append(np.array([-1.0, 0.0, 0.0], dtype=float))
    if j == max(j_vals):
        directions.append(np.array([0.0, 1.0, 0.0], dtype=float))
    elif j == min(j_vals):
        directions.append(np.array([0.0, -1.0, 0.0], dtype=float))
    if k == max(k_vals):
        directions.append(np.array([0.0, 0.0, 1.0], dtype=float))
    elif k == min(k_vals):
        directions.append(np.array([0.0, 0.0, -1.0], dtype=float))
    return tuple(directions)


def _outward_regularized_directions_2d(
    reference: np.ndarray,
    *,
    prefer_diagonals: bool,
) -> tuple[np.ndarray, ...]:
    cardinals = _sorted_direction_candidates(
        reference,
        _CARDINAL_DIRECTIONS_2D,
        dimensions=2,
    )
    diagonals = _sorted_direction_candidates(
        reference,
        _DIAGONAL_DIRECTIONS_2D,
        dimensions=2,
    )
    if prefer_diagonals:
        return _dedupe_candidate_directions([*diagonals, *cardinals], dimensions=2)
    return _dedupe_candidate_directions([*cardinals, *diagonals], dimensions=2)


def _preferred_component_directions_2d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    origin: np.ndarray,
    prefer_outward_chain_axis: bool = True,
) -> tuple[np.ndarray, ...]:
    axis = _component_main_axis_2d_like(component, positions)
    perpendicular = np.array([-axis[1], axis[0]], dtype=float)
    perpendicular = _normalize_direction(perpendicular, dimensions=2)
    centroid = _component_centroid(component, positions, dimensions=2)
    outward = np.asarray(origin, dtype=float).reshape(-1)[:2] - centroid
    outward_norm = float(np.linalg.norm(outward))

    if component.structure_kind == "chain":
        if (
            prefer_outward_chain_axis
            and component.chain_order
            and node_id in {component.chain_order[0], component.chain_order[-1]}
            and outward_norm > 1e-6
        ):
            outward_dir = outward / outward_norm
            axis_out = axis if float(np.dot(outward_dir, axis)) >= 0.0 else -axis
            return _dedupe_candidate_directions(
                [perpendicular, -perpendicular, axis_out, -axis_out],
                dimensions=2,
            )
        return _dedupe_candidate_directions(
            [perpendicular, -perpendicular, axis, -axis],
            dimensions=2,
        )

    if component.structure_kind == "grid" and component.grid_mapping is not None:
        boundary = _grid_boundary_directions_2d(component, node_id=node_id)
        reference = outward if outward_norm > 1e-6 else axis + perpendicular
        preferred = _outward_regularized_directions_2d(
            reference,
            prefer_diagonals=not boundary,
        )
        return _dedupe_candidate_directions(
            [
                *boundary,
                *preferred,
                axis,
                -axis,
                perpendicular,
                -perpendicular,
            ],
            dimensions=2,
        )

    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        face_directions = _grid3d_face_directions_2d(component, positions, node_id=node_id)
        x_dir, y_dir, z_dir = _grid3d_projected_basis_2d(component, positions)
        reference = outward if outward_norm > 1e-6 else (x_dir + y_dir + z_dir)
        sorted_faces = _sorted_direction_candidates(reference, face_directions, dimensions=2)
        preferred = _outward_regularized_directions_2d(reference, prefer_diagonals=False)
        return _dedupe_candidate_directions(
            [
                *sorted_faces,
                *preferred,
                axis,
                -axis,
                perpendicular,
                -perpendicular,
            ],
            dimensions=2,
        )

    reference = outward if outward_norm > 1e-6 else (axis + perpendicular)
    return _dedupe_candidate_directions(
        [
            *_outward_regularized_directions_2d(reference, prefer_diagonals=False),
            axis,
            -axis,
            perpendicular,
            -perpendicular,
        ],
        dimensions=2,
    )


def _preferred_component_directions_3d(
    component: _LayoutComponent,
    positions: NodePositions,
    *,
    node_id: int,
    origin: np.ndarray,
    axis: np.ndarray,
    lateral: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if component.structure_kind != "grid3d":
        return _dedupe_candidate_directions(
            [normal, -normal, lateral, -lateral, axis, -axis],
            dimensions=3,
        )

    centroid = _component_centroid(component, positions, dimensions=3)
    outward = np.asarray(origin, dtype=float).reshape(-1)[:3] - centroid
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-6:
        base_candidates = _dedupe_candidate_directions(
            [normal, -normal, lateral, -lateral, axis, -axis],
            dimensions=3,
        )
        face_directions = _grid3d_face_directions_3d(component, node_id=node_id)
        if len(face_directions) == 1:
            return _dedupe_candidate_directions(
                [*face_directions, *base_candidates],
                dimensions=3,
            )
        return base_candidates

    outward_dir = outward / outward_norm
    candidates: list[np.ndarray] = [outward_dir]
    for base in (normal, lateral, axis):
        projected = np.asarray(base, dtype=float).reshape(-1)[:3]
        projected = projected - outward_dir * float(np.dot(projected, outward_dir))
        if float(np.linalg.norm(projected)) > 1e-6:
            projected = projected / np.linalg.norm(projected)
            candidates.extend([projected, -projected])
    candidates.extend([-outward_dir, normal, -normal, lateral, -lateral, axis, -axis])
    face_directions = _grid3d_face_directions_3d(component, node_id=node_id)
    if len(face_directions) == 1:
        candidates = [*face_directions, *candidates]
    elif len(face_directions) > 1:
        sorted_faces = _sorted_direction_candidates(outward_dir, face_directions, dimensions=3)
        candidates.extend(sorted_faces)
    return _dedupe_candidate_directions(candidates, dimensions=3)


__all__ = [
    "_behavior_direction_order_2d",
    "_dedupe_candidate_directions",
    "_direction_from_axis_name",
    "_direction_has_space",
    "_is_dangling_leg_axis",
    "_normalize_direction",
    "_orthogonal_unit",
    "_preferred_component_directions_2d",
    "_preferred_component_directions_3d",
    "_used_axis_directions",
]
