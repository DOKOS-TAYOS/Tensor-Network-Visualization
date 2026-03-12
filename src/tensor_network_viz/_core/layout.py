"""Layout computation for tensor network graphs."""

from __future__ import annotations

import math
from collections import Counter
from typing import TypeAlias

import networkx as nx
import numpy as np

from .axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from .contractions import _contraction_weights, _iter_contractions
from .graph import _GraphData

Vector: TypeAlias = np.ndarray
NodePositions: TypeAlias = dict[int, Vector]
AxisDirections: TypeAlias = dict[tuple[int, int], Vector]

_LAYOUT_TARGET_NORM: float = 1.6
_GRID_LAYOUT_MAX_NODES: int = 50
_FORCE_LAYOUT_K: float = 1.6
_FORCE_LAYOUT_COOLING_FACTOR: float = 0.985
_FREE_DIR_OVERLAP_THRESHOLD: float = 0.7
_FREE_DIR_SAMPLES_2D: int = 72


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

    automatic_positions = _select_automatic_layout(graph, dimensions=dimensions)
    if automatic_positions is not None:
        return automatic_positions

    return _compute_force_layout(
        graph,
        node_ids=node_ids,
        dimensions=dimensions,
        seed=seed,
        iterations=iterations,
    )


def _select_automatic_layout(graph: _GraphData, *, dimensions: int) -> NodePositions | None:
    if dimensions != 2:
        return None

    grid_positions = _try_grid_layout_2d(graph)
    if grid_positions is not None:
        return grid_positions
    return _try_planar_layout_2d(graph)


def _compute_force_layout(
    graph: _GraphData,
    *,
    node_ids: list[int],
    dimensions: int,
    seed: int,
    iterations: int,
) -> NodePositions:
    positions = _initial_positions(node_ids, dimensions=dimensions, seed=seed)
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    pair_weights = _contraction_weights(graph)

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
        _apply_force_step(positions, displacement, temperature=temperature)
        temperature *= _FORCE_LAYOUT_COOLING_FACTOR

    return {node_id: positions[index].copy() for index, node_id in enumerate(node_ids)}


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
) -> None:
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    positions += displacement / np.maximum(norms, 1e-6) * temperature
    positions -= positions.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(positions, axis=1).max()
    if max_norm > _LAYOUT_TARGET_NORM:
        positions /= max_norm / _LAYOUT_TARGET_NORM


def _build_nx_graph_from_graph_data(graph: _GraphData) -> nx.Graph:
    """Build a NetworkX graph from _GraphData (contraction edges only)."""
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.nodes)
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        if left_id != right_id:
            nx_graph.add_edge(left_id, right_id)
    return nx_graph


def _try_grid_layout_2d(graph: _GraphData) -> NodePositions | None:
    """Attempt regular grid layout when the graph is a 2D grid. Returns None otherwise."""
    node_ids = list(graph.nodes)
    if len(node_ids) <= 1 or len(node_ids) > _GRID_LAYOUT_MAX_NODES:
        return None

    nx_graph = _build_nx_graph_from_graph_data(graph)
    n_nodes = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()
    if not nx.is_connected(nx_graph):
        return None

    degree_histogram = Counter(degree for _, degree in nx_graph.degree())
    if any(degree > 4 for degree in degree_histogram):
        return None

    for rows in range(1, n_nodes + 1):
        if n_nodes % rows != 0:
            continue
        cols = n_nodes // rows
        expected_edges = 2 * rows * cols - rows - cols
        if n_edges != expected_edges:
            continue
        if degree_histogram != _grid_degree_histogram(rows, cols):
            continue
        grid_graph = nx.grid_2d_graph(rows, cols)
        mapping = nx.vf2pp_isomorphism(nx_graph, grid_graph)
        if mapping is None:
            continue
        positions = {
            node_id: np.array([mapping[node_id][1], mapping[node_id][0]], dtype=float)
            for node_id in node_ids
        }
        return _normalize_positions(positions, node_ids)
    return None


def _try_planar_layout_2d(graph: _GraphData) -> NodePositions | None:
    """Attempt planar layout for 2D. Returns None if graph is not planar."""
    node_ids = list(graph.nodes)
    if len(node_ids) <= 1:
        return None

    nx_graph = _build_nx_graph_from_graph_data(graph)
    try:
        planar_positions = nx.planar_layout(nx_graph)
    except nx.NetworkXException:
        return None

    positions = {node_id: np.array(planar_positions[node_id], dtype=float) for node_id in node_ids}
    return _normalize_positions(positions, node_ids)


def _grid_degree_histogram(rows: int, cols: int) -> Counter[int]:
    if rows == 1:
        if cols == 2:
            return Counter({1: 2})
        return Counter({1: 2, 2: cols - 2})
    if cols == 1:
        if rows == 2:
            return Counter({1: 2})
        return Counter({1: 2, 2: rows - 2})

    histogram = Counter({2: 4})
    if rows > 2:
        histogram[3] += 2 * (rows - 2)
    if cols > 2:
        histogram[3] += 2 * (cols - 2)
    if rows > 2 and cols > 2:
        histogram[4] = (rows - 2) * (cols - 2)
    return histogram


def _initial_positions(node_ids: list[int], dimensions: int, seed: int) -> Vector:
    count = len(node_ids)
    rng = np.random.default_rng(seed)

    if dimensions == 2:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        positions = np.zeros((count, 3), dtype=float)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(count):
            y = 1.0 - (2.0 * index) / max(count - 1, 1)
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * index
            positions[index] = np.array(
                [math.cos(theta) * radius, y, math.sin(theta) * radius],
                dtype=float,
            )

    positions += rng.normal(loc=0.0, scale=0.03, size=positions.shape)
    return positions


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

            best_score = -np.inf
            best_direction = np.array([1.0, 0.0], dtype=float)
            for direction in unit_circle:
                toward_obstacles = np.dot(dirs_to_obstacles, direction)
                away_score = -float(np.min(toward_obstacles))
                separation_penalty = sum(
                    max(0.0, float(np.dot(direction, used_direction[:2]))) * 2.0
                    for used_direction in used_dirs
                )
                score = away_score - separation_penalty
                if score > best_score:
                    best_score = score
                    best_direction = direction.astype(float, copy=True)
            directions[axis_key] = best_direction


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
) -> None:
    center = np.mean(np.stack(list(positions.values())), axis=0)
    for node_id, node in graph.nodes.items():
        origin = positions[node_id]
        radial = origin - center
        if np.linalg.norm(radial) < 1e-6:
            radial = np.array([1.0, 0.0, 0.0], dtype=float)
        radial = radial / np.linalg.norm(radial)
        basis_a = _orthogonal_unit(radial)
        basis_b = np.cross(radial, basis_a)
        basis_b = basis_b / np.linalg.norm(basis_b)
        axis_count = max(node.degree, 1)
        free_indices = [
            axis_index
            for axis_index in range(axis_count)
            if (node_id, axis_index) not in directions
        ]

        for offset, axis_index in enumerate(free_indices):
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=3)
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            if named_direction is not None and _direction_has_space(named_direction, used_dirs):
                directions[(node_id, axis_index)] = named_direction
                continue
            angle = 2.0 * math.pi * offset / max(len(free_indices), 1)
            direction = radial + 0.55 * (
                math.cos(angle) * basis_a + math.sin(angle) * basis_b
            )
            directions[(node_id, axis_index)] = direction / np.linalg.norm(direction)


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
        max(0.0, float(np.dot(direction, used_direction)))
        for used_direction in used_dirs
    )
    return overlap < _FREE_DIR_OVERLAP_THRESHOLD


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)
