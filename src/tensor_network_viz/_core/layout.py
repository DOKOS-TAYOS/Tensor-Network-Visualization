"""Layout computation for tensor network graphs."""

from __future__ import annotations

import math
from collections import Counter
from typing import TypeAlias

import networkx as nx
import numpy as np

from .axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
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
    arr = np.array([positions[nid] for nid in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(arr, axis=1).max()
    if max_norm > 1e-6:
        arr /= max_norm / target_norm
    return {nid: arr[i].copy() for i, nid in enumerate(node_ids)}


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
        origin = np.zeros(dimensions, dtype=float)
        return {node_ids[0]: origin}

    if dimensions == 2:
        grid_pos = _try_grid_layout_2d(graph)
        if grid_pos is not None:
            return grid_pos
        planar_pos = _try_planar_layout_2d(graph)
        if planar_pos is not None:
            return planar_pos

    positions = _initial_positions(node_ids, dimensions=dimensions, seed=seed)
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}

    pair_weights: dict[tuple[int, int], int] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        left, right = edge.node_ids
        key = (left, right) if left < right else (right, left)
        pair_weights[key] = pair_weights.get(key, 0) + 1

    k = _FORCE_LAYOUT_K
    temperature = 0.12
    for _ in range(iterations):
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, 1.0)
        directions = deltas / np.maximum(distances[..., None], 1e-6)
        repulsion = (k * k / np.maximum(distances, 1e-6) ** 2)[..., None] * directions
        displacement = repulsion.sum(axis=1)

        for (left_id, right_id), weight in pair_weights.items():
            left_index = index_by_node[left_id]
            right_index = index_by_node[right_id]
            delta = positions[right_index] - positions[left_index]
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            direction = delta / distance
            attraction = weight * distance * distance / k * direction
            displacement[left_index] += attraction
            displacement[right_index] -= attraction

        norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        positions += displacement / np.maximum(norms, 1e-6) * temperature
        positions -= positions.mean(axis=0, keepdims=True)
        max_norm = np.linalg.norm(positions, axis=1).max()
        if max_norm > _LAYOUT_TARGET_NORM:
            positions /= max_norm / _LAYOUT_TARGET_NORM
        temperature *= _FORCE_LAYOUT_COOLING_FACTOR

    return {node_id: positions[index] for index, node_id in enumerate(node_ids)}


def _build_nx_graph_from_graph_data(graph: _GraphData) -> nx.Graph:
    """Build a NetworkX graph from _GraphData (contraction edges only)."""
    g = nx.Graph()
    g.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        left, right = edge.node_ids
        if left != right:
            g.add_edge(left, right)
    return g


def _try_grid_layout_2d(graph: _GraphData) -> NodePositions | None:
    """Attempt regular grid layout when the graph is a 2D grid. Returns None otherwise."""
    node_ids = list(graph.nodes)
    if len(node_ids) <= 1 or len(node_ids) > _GRID_LAYOUT_MAX_NODES:
        return None

    g = _build_nx_graph_from_graph_data(graph)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if not nx.is_connected(g):
        return None

    degree_histogram = Counter(degree for _, degree in g.degree())
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
        grid_g = nx.grid_2d_graph(rows, cols)
        mapping = nx.vf2pp_isomorphism(g, grid_g)
        if mapping is not None:
            positions = {
                nid: np.array([mapping[nid][1], mapping[nid][0]], dtype=float)
                for nid in node_ids
            }
            return _normalize_positions(positions, node_ids)
    return None


def _try_planar_layout_2d(graph: _GraphData) -> NodePositions | None:
    """Attempt planar layout for 2D. Returns None if graph is not planar."""
    node_ids = list(graph.nodes)
    if len(node_ids) <= 1:
        return None

    g = _build_nx_graph_from_graph_data(graph)
    try:
        pos = nx.planar_layout(g)
    except nx.NetworkXException:
        return None

    positions = {nid: np.array(pos[nid], dtype=float) for nid in node_ids}
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
    center = np.mean(np.stack(list(positions.values())), axis=0)

    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        a_id, b_id = edge.node_ids
        a_ep = next(ep for ep in edge.endpoints if ep.node_id == a_id)
        b_ep = next(ep for ep in edge.endpoints if ep.node_id == b_id)
        pa = positions[a_id]
        pb = positions[b_id]
        delta = pb - pa
        dist = max(float(np.linalg.norm(delta)), 1e-6)
        toward_b = delta / dist
        toward_a = -toward_b
        directions[(a_id, a_ep.axis_index)] = toward_b
        directions[(b_id, b_ep.axis_index)] = toward_a

    if dimensions == 2:
        _compute_free_directions_2d(graph, positions, directions)
    else:
        _compute_free_directions_3d(graph, positions, center, directions)

    return directions


def _compute_free_directions_2d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
) -> None:
    node_ids = list(positions)
    samples = _FREE_DIR_SAMPLES_2D
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    other_node_positions = {
        node_id: np.array(
            [positions[other_id] for other_id in node_ids if other_id != node_id],
            dtype=float,
        )
        for node_id in node_ids
    }
    neighbor_midpoints: dict[int, list[np.ndarray]] = {node_id: [] for node_id in node_ids}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        left_id, right_id = edge.node_ids
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
        obstacles_arr = (
            np.concatenate(obstacle_parts, axis=0)
            if obstacle_parts
            else np.array([[origin[0] + 1.0, origin[1]]], dtype=float)
        )

        vecs_to_obstacles = obstacles_arr - origin
        dists = np.linalg.norm(vecs_to_obstacles, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-6)
        dirs_to_obstacles = vecs_to_obstacles / dists

        axis_count = max(node.degree, 1)
        for axis_index in range(axis_count):
            if (node_id, axis_index) in directions:
                continue
            used_dirs = [
                directions[(node_id, j)]
                for j in range(axis_count)
                if (node_id, j) in directions
            ]
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_d = _direction_from_axis_name(axis_name, dimensions=2)
            if named_d is not None:
                overlap = sum(max(0.0, float(np.dot(named_d, u[:2]))) for u in used_dirs)
                if overlap < _FREE_DIR_OVERLAP_THRESHOLD:
                    directions[(node_id, axis_index)] = named_d
                    continue
            best_score = -np.inf
            best_d = np.array([1.0, 0.0], dtype=float)
            for d in unit_circle:
                d = d.astype(float)
                toward_obstacles = np.dot(dirs_to_obstacles, d)
                away_score = -float(np.min(toward_obstacles))
                sep_score = 0.0
                for u in used_dirs:
                    sim = float(np.dot(d, u[:2]))
                    sep_score += max(0.0, sim) * 2.0
                score = away_score - sep_score
                if score > best_score:
                    best_score = score
                    best_d = d.copy()
            directions[(node_id, axis_index)] = best_d


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    center: np.ndarray,
    directions: AxisDirections,
) -> None:
    for node_id, node in graph.nodes.items():
        origin = positions[node_id]
        radial = origin - center
        if np.linalg.norm(radial) < 1e-6:
            radial = np.array([1.0, 0.0, 0.0], dtype=float)
        radial = radial / np.linalg.norm(radial)
        basis_a = _orthogonal_unit(radial)
        basis_b = np.cross(radial, basis_a)
        basis_b = basis_b / np.linalg.norm(basis_b)
        free_indices = [
            j for j in range(max(node.degree, 1))
            if (node_id, j) not in directions
        ]
        for idx, axis_index in enumerate(free_indices):
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_d = _direction_from_axis_name(axis_name, dimensions=3)
            if named_d is not None:
                used_dirs = [
                    directions[(node_id, j)]
                    for j in range(max(node.degree, 1))
                    if (node_id, j) in directions
                ]
                overlap = sum(max(0.0, float(np.dot(named_d, u))) for u in used_dirs)
                if overlap < _FREE_DIR_OVERLAP_THRESHOLD:
                    directions[(node_id, axis_index)] = named_d
                    continue
            angle = 2.0 * math.pi * idx / max(len(free_indices), 1)
            direction = radial + 0.55 * (
                math.cos(angle) * basis_a + math.sin(angle) * basis_b
            )
            directions[(node_id, axis_index)] = direction / np.linalg.norm(direction)


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)
