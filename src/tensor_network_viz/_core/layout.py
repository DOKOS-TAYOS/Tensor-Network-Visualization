"""Layout computation for tensor network graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from .axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from .contractions import _contraction_weights, _iter_contractions
from .graph import _GraphData
from .layout_structure import (
    _analyze_layout_components,
    _component_orthogonal_basis,
    _LayoutComponent,
    _leaf_nodes,
    _specialized_anchor_positions,
)

Vector: TypeAlias = np.ndarray
NodePositions: TypeAlias = dict[int, Vector]
AxisDirections: TypeAlias = dict[tuple[int, int], Vector]

_LAYOUT_TARGET_NORM: float = 1.6
_FORCE_LAYOUT_K: float = 1.6
_FORCE_LAYOUT_COOLING_FACTOR: float = 0.985
_FREE_DIR_OVERLAP_THRESHOLD: float = 0.7
_FREE_DIR_SAMPLES_2D: int = 72
_COMPONENT_GAP: float = 1.4
_LAYER_SPACING: float = 0.55
_LAYER_SEQUENCE: tuple[int, ...] = (0, 1, -1, 2, -2, 3, -3)


def _normalize_positions(
    positions: NodePositions,
    node_ids: list[int],
    target_norm: float = _LAYOUT_TARGET_NORM,
) -> NodePositions:
    """Center and scale positions to target norm. Returns new dict."""
    arr = np.array([positions[node_id] for node_id in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(arr, axis=1).max()
    if max_norm > 1e-6:
        arr /= max_norm / target_norm
    return {node_id: arr[index].copy() for index, node_id in enumerate(node_ids)}


def _direction_from_axis_name(
    axis_name: str | None,
    dimensions: int,
) -> np.ndarray | None:
    """Return direction vector for axis name in 2D or 3D, or None if unknown."""
    if not axis_name:
        return None
    key = axis_name.lower().strip()
    if dimensions == 2 and key in _AXIS_DIR_2D:
        return np.array(_AXIS_DIR_2D[key], dtype=float)
    if dimensions == 3 and key in _AXIS_DIR_3D:
        return np.array(_AXIS_DIR_3D[key], dtype=float)
    return None


def _compute_layout(
    graph: _GraphData,
    dimensions: int,
    seed: int,
    *,
    iterations: int = 220,
) -> NodePositions:
    node_ids = list(graph.nodes)
    if len(node_ids) == 1:
        return {node_ids[0]: np.zeros(dimensions, dtype=float)}

    components = _analyze_layout_components(graph)
    component_positions: list[NodePositions] = []
    for index, component in enumerate(components):
        positions_2d = _compute_component_layout_2d(
            graph,
            component,
            seed=seed + index,
            iterations=iterations,
        )
        if dimensions == 2:
            component_positions.append(positions_2d)
            continue
        component_positions.append(_lift_component_layout_3d(graph, component, positions_2d))

    packed_positions = _pack_component_positions(component_positions, dimensions=dimensions)
    return _normalize_positions(packed_positions, node_ids)


def _compute_component_layout_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    seed: int,
    iterations: int,
) -> NodePositions:
    node_ids = list(component.node_ids)
    if len(node_ids) == 1:
        return {node_ids[0]: np.zeros(2, dtype=float)}

    fixed_positions = _specialized_anchor_positions(component)
    if fixed_positions:
        positions = _compute_force_layout(
            graph,
            node_ids=node_ids,
            dimensions=2,
            seed=seed,
            iterations=iterations,
            fixed_positions=fixed_positions,
        )
    else:
        positions = _compute_force_layout(
            graph,
            node_ids=node_ids,
            dimensions=2,
            seed=seed,
            iterations=iterations,
        )

    _snap_virtual_nodes_to_barycenters(component, positions)
    _place_trimmed_leaf_nodes_2d(component, positions)
    return _center_positions(positions, node_ids=node_ids)


def _lift_component_layout_3d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions_2d: NodePositions,
) -> NodePositions:
    positions = {
        node_id: np.array([coords[0], coords[1], 0.0], dtype=float)
        for node_id, coords in positions_2d.items()
    }
    _place_trimmed_leaf_nodes_3d(component, positions)
    _promote_3d_layers(graph, component, positions)
    return positions


def _center_positions(positions: NodePositions, *, node_ids: list[int]) -> NodePositions:
    arr = np.array([positions[node_id] for node_id in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    return {node_id: arr[index].copy() for index, node_id in enumerate(node_ids)}


def _pack_component_positions(
    component_positions: list[NodePositions],
    *,
    dimensions: int,
) -> NodePositions:
    if len(component_positions) == 1:
        return component_positions[0]

    packed: NodePositions = {}
    cursor_x = 0.0
    for positions in component_positions:
        node_ids = sorted(positions)
        coords = np.array([positions[node_id] for node_id in node_ids], dtype=float)
        min_x = float(coords[:, 0].min())
        max_x = float(coords[:, 0].max())
        shift = np.zeros(dimensions, dtype=float)
        shift[0] = cursor_x - min_x
        if dimensions > 1:
            shift[1:] = -coords[:, 1:].mean(axis=0)

        packed_coords = coords + shift
        for index, node_id in enumerate(node_ids):
            packed[node_id] = packed_coords[index].copy()

        cursor_x += max(max_x - min_x, 0.8) + _COMPONENT_GAP

    return packed


def _compute_force_layout(
    graph: _GraphData,
    *,
    node_ids: list[int],
    dimensions: int,
    seed: int,
    iterations: int,
    fixed_positions: NodePositions | None = None,
) -> NodePositions:
    positions = _initial_positions(node_ids, dimensions=dimensions, seed=seed)
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    pair_weights = _pair_weights_for_node_ids(graph, node_ids=node_ids)
    fixed_mask = np.zeros(len(node_ids), dtype=bool)
    fixed_array = np.zeros_like(positions)

    if fixed_positions:
        fixed_centroid = np.mean(np.stack(list(fixed_positions.values())), axis=0)
        for node_id, fixed_position in fixed_positions.items():
            if node_id not in index_by_node:
                continue
            index = index_by_node[node_id]
            fixed_mask[index] = True
            fixed_array[index] = fixed_position
            positions[index] = fixed_position
        for node_id in node_ids:
            index = index_by_node[node_id]
            if fixed_mask[index]:
                continue
            positions[index] = fixed_centroid + (positions[index] * 0.35)

    temperature = 0.12
    for _ in range(iterations):
        displacement = _repulsion_displacement(positions, k=_FORCE_LAYOUT_K)
        _apply_attraction_forces(
            displacement,
            positions,
            index_by_node=index_by_node,
            pair_weights=pair_weights,
            k=_FORCE_LAYOUT_K,
        )
        if fixed_positions:
            displacement[fixed_mask] = 0.0
        _apply_force_step(
            positions,
            displacement,
            temperature=temperature,
            fixed_mask=fixed_mask if fixed_positions else None,
            fixed_positions=fixed_array if fixed_positions else None,
        )
        temperature *= _FORCE_LAYOUT_COOLING_FACTOR

    return {node_id: positions[index].copy() for index, node_id in enumerate(node_ids)}


def _pair_weights_for_node_ids(
    graph: _GraphData,
    *,
    node_ids: list[int],
) -> dict[tuple[int, int], int]:
    node_id_set = set(node_ids)
    return {
        pair: weight
        for pair, weight in _contraction_weights(graph).items()
        if pair[0] in node_id_set and pair[1] in node_id_set
    }


def _repulsion_displacement(positions: np.ndarray, *, k: float) -> np.ndarray:
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, 1.0)
    directions = deltas / np.maximum(distances[..., None], 1e-6)
    repulsion = (k * k / np.maximum(distances, 1e-6) ** 2)[..., None] * directions
    return repulsion.sum(axis=1)


def _apply_attraction_forces(
    displacement: np.ndarray,
    positions: np.ndarray,
    *,
    index_by_node: dict[int, int],
    pair_weights: dict[tuple[int, int], int],
    k: float,
) -> None:
    for (left_id, right_id), weight in pair_weights.items():
        left_index = index_by_node[left_id]
        right_index = index_by_node[right_id]
        delta = positions[right_index] - positions[left_index]
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        direction = delta / distance
        attraction = weight * distance * distance / k * direction
        displacement[left_index] += attraction
        displacement[right_index] -= attraction


def _apply_force_step(
    positions: np.ndarray,
    displacement: np.ndarray,
    *,
    temperature: float,
    fixed_mask: np.ndarray | None = None,
    fixed_positions: np.ndarray | None = None,
) -> None:
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    step = displacement / np.maximum(norms, 1e-6) * temperature
    if fixed_mask is None or fixed_positions is None:
        positions += step
        positions -= positions.mean(axis=0, keepdims=True)
        max_norm = np.linalg.norm(positions, axis=1).max()
        if max_norm > _LAYOUT_TARGET_NORM:
            positions /= max_norm / _LAYOUT_TARGET_NORM
        return

    movable_mask = ~fixed_mask
    positions[movable_mask] += step[movable_mask]
    positions[fixed_mask] = fixed_positions[fixed_mask]


def _initial_positions(node_ids: list[int], dimensions: int, seed: int) -> Vector:
    count = len(node_ids)
    rng = np.random.default_rng(seed)

    if dimensions == 2:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        indices = np.arange(count, dtype=float)
        denom = max(count - 1, 1)
        y = 1.0 - (2.0 * indices) / denom
        radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        theta = golden_angle * indices
        positions = np.column_stack(
            (np.cos(theta) * radius, y, np.sin(theta) * radius),
        ).astype(float)

    positions += rng.normal(loc=0.0, scale=0.03, size=positions.shape)
    return positions


def _snap_virtual_nodes_to_barycenters(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    for node_id in component.virtual_node_ids:
        neighbors = sorted(component.contraction_graph.neighbors(node_id))
        if not neighbors:
            continue
        positions[node_id] = np.mean(
            np.stack([positions[neighbor_id] for neighbor_id in neighbors]),
            axis=0,
        )


def _place_trimmed_leaf_nodes_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    if not component.trimmed_leaf_parents:
        return

    axis = _component_main_axis_2d(component, positions)
    perpendicular = np.array([-axis[1], axis[0]], dtype=float)
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.array([0.0, 1.0], dtype=float)
    else:
        perpendicular /= np.linalg.norm(perpendicular)
    core_node_ids = [
        node_id
        for node_id in component.visible_node_ids
        if node_id not in {leaf_id for leaf_id, _ in component.trimmed_leaf_parents}
    ]
    if not core_node_ids:
        return
    centroid = np.mean(
        np.stack([positions[node_id] for node_id in core_node_ids]),
        axis=0,
    )

    assigned_targets: list[np.ndarray] = []
    for leaf_id, parent_id in component.trimmed_leaf_parents:
        positions[leaf_id] = _best_attachment_position_2d(
            component=component,
            origin=positions[parent_id],
            parent_id=parent_id,
            leaf_id=leaf_id,
            candidates=(perpendicular, -perpendicular),
            axis=axis,
            centroid=centroid,
            assigned_targets=assigned_targets,
            positions=positions,
        )
        assigned_targets.append(positions[leaf_id].copy())


def _place_trimmed_leaf_nodes_3d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    if not component.trimmed_leaf_parents:
        return

    _, lateral, normal = _component_orthogonal_basis(component, positions)
    candidates = (normal, -normal, lateral, -lateral)
    assigned_targets: list[np.ndarray] = []
    for leaf_id, parent_id in component.trimmed_leaf_parents:
        positions[leaf_id] = _best_attachment_position_3d(
            origin=positions[parent_id],
            candidates=candidates,
            assigned_targets=assigned_targets,
        )
        assigned_targets.append(positions[leaf_id].copy())


def _component_main_axis_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> np.ndarray:
    anchor_node_ids = component.anchor_node_ids or component.visible_node_ids
    if component.structure_kind == "chain" and len(component.chain_order) >= 2:
        start = positions[component.chain_order[0]]
        end = positions[component.chain_order[-1]]
        axis = end - start
    elif len(anchor_node_ids) >= 2:
        coords = np.stack([positions[node_id] for node_id in anchor_node_ids])
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    else:
        axis = np.array([1.0, 0.0], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return axis / norm


def _best_attachment_position_2d(
    *,
    component: _LayoutComponent,
    origin: np.ndarray,
    parent_id: int,
    leaf_id: int,
    candidates: tuple[np.ndarray, ...],
    axis: np.ndarray,
    centroid: np.ndarray,
    assigned_targets: list[np.ndarray],
    positions: NodePositions,
) -> np.ndarray:
    leaf_node_ids = {node_id for node_id, _ in component.trimmed_leaf_parents}
    direction_options = (
        candidates if component.structure_kind == "chain" else (*candidates, axis, -axis)
    )
    used_dirs = []
    for neighbor_id in component.contraction_graph.neighbors(parent_id):
        if neighbor_id == leaf_id or neighbor_id in leaf_node_ids:
            continue
        delta = positions[neighbor_id][:2] - origin[:2]
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            used_dirs.append(delta / norm)

    outward = origin[:2] - centroid[:2]
    outward_norm = np.linalg.norm(outward)
    outward_dir = outward / outward_norm if outward_norm > 1e-6 else np.zeros(2, dtype=float)

    existing_segments = []
    for left_id, right_id in component.contraction_graph.edges():
        if leaf_id in (left_id, right_id):
            continue
        if left_id in leaf_node_ids or right_id in leaf_node_ids:
            continue
        if parent_id in (left_id, right_id):
            continue
        existing_segments.append((positions[left_id][:2], positions[right_id][:2]))

    distance = 1.0
    best_score = -np.inf
    best_candidate = origin[:2] + candidates[0] * distance
    for direction in direction_options:
        candidate = origin + direction * distance
        score = 0.8 * float(np.dot(direction[:2], outward_dir))
        score -= 2.5 * sum(
            max(0.0, float(np.dot(direction[:2], used_dir))) for used_dir in used_dirs
        )
        score -= 2.0 * sum(
            1.0 for target in assigned_targets if np.linalg.norm(candidate[:2] - target[:2]) < 0.3
        )
        score -= 4.0 * sum(
            1.0
            for node_id, position in positions.items()
            if node_id not in {leaf_id, parent_id}
            and np.linalg.norm(candidate[:2] - position[:2]) < 0.25
        )
        score -= 4.0 * sum(
            1.0
            for start, end in existing_segments
            if _segment_hits_existing_geometry_2d(origin[:2], candidate[:2], start, end)
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate[:2].copy()
    return best_candidate


def _best_attachment_position_3d(
    *,
    origin: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    assigned_targets: list[np.ndarray],
) -> np.ndarray:
    distance = 1.0
    for direction in candidates:
        candidate = origin + direction * distance
        if all(np.linalg.norm(candidate - target) >= 0.3 for target in assigned_targets):
            return candidate
    return origin + candidates[0] * distance


def _segment_hits_existing_geometry_2d(
    start: np.ndarray,
    end: np.ndarray,
    other_start: np.ndarray,
    other_end: np.ndarray,
) -> bool:
    if _segments_cross_2d(start, end, other_start, other_end):
        return True
    midpoint = (start + end) / 2.0
    return _point_segment_distance_2d(midpoint, other_start, other_end) < 0.15


def _point_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _promote_3d_layers(
    graph: _GraphData,
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    layer_indices = dict.fromkeys(component.node_ids, 0)

    for node_id in component.virtual_node_ids:
        if _node_overlaps_component(node_id, component, positions):
            layer_indices[node_id] = _next_layer(used_layers=layer_indices.values())

    edges = [
        tuple(sorted(record.node_ids))
        for record in _iter_contractions(graph)
        if all(node_id in component.contraction_graph for node_id in record.node_ids)
    ]

    for index, left_edge in enumerate(edges):
        for right_edge in edges[index + 1 :]:
            if set(left_edge) & set(right_edge):
                continue
            if not _segments_cross_2d(
                positions[left_edge[0]][:2],
                positions[left_edge[1]][:2],
                positions[right_edge[0]][:2],
                positions[right_edge[1]][:2],
            ):
                continue
            promoted = _choose_promotable_node(
                graph,
                component,
                node_ids=left_edge + right_edge,
            )
            if promoted is None or layer_indices[promoted] != 0:
                continue
            layer_indices[promoted] = _next_layer(used_layers=layer_indices.values())

    for node_id, layer_index in layer_indices.items():
        positions[node_id][2] += layer_index * _LAYER_SPACING


def _next_layer(*, used_layers: Iterable[int | float]) -> int:
    used = {int(layer) for layer in used_layers}
    for candidate in _LAYER_SEQUENCE[1:]:
        if candidate not in used:
            return candidate
    return max(used or {0}, key=abs) + 1


def _node_overlaps_component(
    node_id: int,
    component: _LayoutComponent,
    positions: NodePositions,
) -> bool:
    point = positions[node_id][:2]
    for other_id in component.node_ids:
        if other_id == node_id:
            continue
        if np.linalg.norm(point - positions[other_id][:2]) < 0.18:
            return True
    return False


def _choose_promotable_node(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    node_ids: tuple[int, ...],
) -> int | None:
    leaf_node_ids = set(_leaf_nodes(component))
    anchor_node_ids = set(component.anchor_node_ids)

    def priority(node_id: int) -> tuple[int, int]:
        if graph.nodes[node_id].is_virtual:
            return (0, node_id)
        if node_id not in anchor_node_ids:
            return (1, node_id)
        if node_id in leaf_node_ids:
            return (2, node_id)
        return (3, node_id)

    ordered = sorted(dict.fromkeys(node_ids), key=priority)
    if not ordered:
        return None
    return ordered[0]


def _segments_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _compute_axis_directions(
    graph: _GraphData,
    positions: NodePositions,
    dimensions: int,
) -> AxisDirections:
    directions: AxisDirections = {}
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        left_endpoint, right_endpoint = record.endpoints
        left_position = positions[left_id]
        right_position = positions[right_id]
        delta = right_position - left_position
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        toward_right = delta / distance
        directions[(left_id, left_endpoint.axis_index)] = toward_right
        directions[(right_id, right_endpoint.axis_index)] = -toward_right

    if dimensions == 2:
        _compute_free_directions_2d(graph, positions, directions)
    else:
        _compute_free_directions_3d(graph, positions, directions)

    return directions


def _compute_free_directions_2d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
) -> None:
    node_ids = list(positions)
    angles = np.linspace(0.0, 2.0 * math.pi, _FREE_DIR_SAMPLES_2D, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    other_node_positions = {
        node_id: np.array(
            [positions[other_id] for other_id in node_ids if other_id != node_id],
            dtype=float,
        )
        for node_id in node_ids
    }
    neighbor_midpoints: dict[int, list[np.ndarray]] = {node_id: [] for node_id in node_ids}
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        midpoint = (positions[left_id] + positions[right_id]) / 2.0
        neighbor_midpoints[left_id].append(midpoint)
        neighbor_midpoints[right_id].append(midpoint)

    for node_id, node in graph.nodes.items():
        origin = positions[node_id]
        obstacle_parts: list[np.ndarray] = []
        if other_node_positions[node_id].size:
            obstacle_parts.append(other_node_positions[node_id])
        if neighbor_midpoints[node_id]:
            obstacle_parts.append(np.array(neighbor_midpoints[node_id], dtype=float))
        obstacles = (
            np.concatenate(obstacle_parts, axis=0)
            if obstacle_parts
            else np.array([[origin[0] + 1.0, origin[1]]], dtype=float)
        )

        vecs_to_obstacles = obstacles - origin
        dists = np.linalg.norm(vecs_to_obstacles, axis=1, keepdims=True)
        dirs_to_obstacles = vecs_to_obstacles / np.maximum(dists, 1e-6)

        axis_count = max(node.degree, 1)
        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=2)
            if named_direction is not None and _direction_has_space(named_direction, used_dirs):
                directions[axis_key] = named_direction
                continue

            toward = dirs_to_obstacles @ unit_circle.T
            away_scores = -np.min(toward, axis=0)
            if used_dirs:
                used_stack = np.stack(
                    [used_direction[:2].astype(float) for used_direction in used_dirs],
                )
                overlap = unit_circle @ used_stack.T
                separation = np.maximum(0.0, overlap).sum(axis=1) * 2.0
            else:
                separation = np.zeros(unit_circle.shape[0], dtype=float)
            scores = away_scores - separation
            best_direction = unit_circle[int(np.argmax(scores))].copy()
            directions[axis_key] = best_direction


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
) -> None:
    components = _analyze_layout_components(graph)
    assigned_segments: list[tuple[np.ndarray, np.ndarray]] = []
    component_by_node = {
        node_id: component for component in components for node_id in component.node_ids
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
                ):
                    continue
                directions[axis_key] = direction
                assigned_segments.append((origin.copy(), direction.copy()))
                break
            else:
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
) -> bool:
    tip = origin + direction * 0.45
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


def _direction_has_space(direction: np.ndarray, used_dirs: list[np.ndarray]) -> bool:
    overlap = sum(
        max(0.0, float(np.dot(direction, used_direction))) for used_direction in used_dirs
    )
    return overlap < _FREE_DIR_OVERLAP_THRESHOLD


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)
