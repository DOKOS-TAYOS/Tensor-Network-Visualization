"""Generic-layout graph coarsening helpers."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import networkx as nx
import numpy as np

from ..graph import _GraphData
from ..layout_structure import (
    _classify_anchor_graph,
    _layout_chain,
    _layout_circular,
    _layout_grid,
    _layout_grid3d_projection_2d,
    _layout_planar,
    _layout_tree,
    _layout_tube_projection_2d,
    _LayoutComponent,
)
from .force_directed import _compute_weighted_force_layout
from .types import NodePositions

_GENERIC_FORCE_MIN_ITERATIONS: int = 30
_GENERIC_FORCE_MAX_ITERATIONS: int = 60
_GENERIC_TREE_SPACING: float = 1.0
_GENERIC_PATH_MIN_ARC_HEIGHT: float = 0.35
_COARSENED_INITIAL_STRUCTURE_KINDS: frozenset[str] = frozenset(("generic", "planar"))


@dataclass(frozen=True)
class _CoarseningGraph:
    node_ids: tuple[int, ...]
    neighbors_by_node: dict[int, tuple[int, ...]]
    edge_weights: dict[tuple[int, int], float]


@dataclass(frozen=True)
class _PeeledDegreeOneTrees:
    core_node_ids: tuple[int, ...]
    parent_by_removed_node: dict[int, int]


@dataclass(frozen=True)
class _CompressedPath:
    endpoint_ids: tuple[int, int]
    internal_node_ids: tuple[int, ...]
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class _DegreeTwoCompression:
    skeleton_node_ids: tuple[int, ...]
    paths: tuple[_CompressedPath, ...]
    skeleton_graph: nx.Graph


def _distinct_neighbor_counts(
    nx_graph: nx.Graph,
    node_ids: tuple[int, ...],
) -> dict[int, int]:
    coarsening_graph = _coarsening_graph_from_nx(nx_graph, node_ids)
    return {
        node_id: len(coarsening_graph.neighbors_by_node[node_id])
        for node_id in coarsening_graph.node_ids
    }


def _coarsening_graph_from_nx(
    nx_graph: nx.Graph,
    node_ids: tuple[int, ...],
) -> _CoarseningGraph:
    sorted_node_ids = tuple(sorted(int(node_id) for node_id in node_ids))
    node_id_set = set(sorted_node_ids)
    neighbors_by_node: dict[int, tuple[int, ...]] = {}
    edge_weights: dict[tuple[int, int], float] = {}
    for node_id in sorted_node_ids:
        neighbors: list[int] = []
        for neighbor_id_raw, data in nx_graph[node_id].items():
            neighbor_id = int(neighbor_id_raw)
            if neighbor_id not in node_id_set:
                continue
            neighbors.append(neighbor_id)
            if node_id < neighbor_id:
                edge_weights[(node_id, neighbor_id)] = float(data.get("weight", 1.0))
        neighbors.sort()
        neighbors_by_node[node_id] = tuple(neighbors)
    return _CoarseningGraph(
        node_ids=sorted_node_ids,
        neighbors_by_node=neighbors_by_node,
        edge_weights=edge_weights,
    )


def _has_degree_one_neighbor(
    coarsening_graph: _CoarseningGraph,
) -> bool:
    return any(
        len(coarsening_graph.neighbors_by_node[node_id]) <= 1
        for node_id in coarsening_graph.node_ids
    )


def _nx_graph_from_coarsening(
    coarsening_graph: _CoarseningGraph,
    node_ids: tuple[int, ...],
) -> nx.Graph:
    node_id_set = set(node_ids)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(node_ids)
    for (left_id, right_id), weight in coarsening_graph.edge_weights.items():
        if left_id in node_id_set and right_id in node_id_set:
            nx_graph.add_edge(left_id, right_id, weight=weight)
    return nx_graph


def _peel_degree_one_trees(
    nx_graph: nx.Graph,
    node_ids: tuple[int, ...],
) -> _PeeledDegreeOneTrees:
    coarsening_graph = _coarsening_graph_from_nx(nx_graph, node_ids)
    return _peel_degree_one_trees_from_coarsening(coarsening_graph)


def _peel_degree_one_trees_from_coarsening(
    coarsening_graph: _CoarseningGraph,
) -> _PeeledDegreeOneTrees:
    """Remove degree-one trees recursively, using distinct contraction neighbors."""
    remaining = set(coarsening_graph.node_ids)
    degrees = {
        node_id: len(coarsening_graph.neighbors_by_node[node_id])
        for node_id in coarsening_graph.node_ids
    }
    parent_by_removed_node: dict[int, int] = {}
    queue = [node_id for node_id, degree in degrees.items() if degree <= 1]
    heapq.heapify(queue)

    while queue and len(remaining) > 1:
        node_id = heapq.heappop(queue)
        if node_id not in remaining:
            continue
        if degrees[node_id] > 1:
            continue
        parent_ids = [
            neighbor_id
            for neighbor_id in coarsening_graph.neighbors_by_node[node_id]
            if neighbor_id in remaining
        ]
        if len(parent_ids) > 1:
            degrees[node_id] = len(parent_ids)
            continue
        if parent_ids:
            parent_id = parent_ids[0]
            parent_by_removed_node[node_id] = parent_id
        else:
            parent_id = None

        remaining.remove(node_id)
        degrees[node_id] = 0
        for neighbor_id in coarsening_graph.neighbors_by_node[node_id]:
            if neighbor_id not in remaining:
                continue
            degrees[neighbor_id] -= 1
            if degrees[neighbor_id] <= 1:
                heapq.heappush(queue, neighbor_id)

    return _PeeledDegreeOneTrees(
        core_node_ids=tuple(sorted(remaining)),
        parent_by_removed_node=parent_by_removed_node,
    )


def _edge_key(left_id: int, right_id: int) -> tuple[int, int]:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _compress_degree_two_paths(
    nx_graph: nx.Graph,
    node_ids: tuple[int, ...],
) -> _DegreeTwoCompression:
    coarsening_graph = _coarsening_graph_from_nx(nx_graph, node_ids)
    return _compress_degree_two_paths_from_coarsening(coarsening_graph)


def _compress_degree_two_paths_from_coarsening(
    coarsening_graph: _CoarseningGraph,
) -> _DegreeTwoCompression:
    degrees = {
        node_id: len(coarsening_graph.neighbors_by_node[node_id])
        for node_id in coarsening_graph.node_ids
    }
    branch_node_ids = {node_id for node_id, degree in degrees.items() if degree != 2}
    if not branch_node_ids:
        return _DegreeTwoCompression(
            skeleton_node_ids=coarsening_graph.node_ids,
            paths=(),
            skeleton_graph=_nx_graph_from_coarsening(coarsening_graph, coarsening_graph.node_ids),
        )

    visited_edges: set[tuple[int, int]] = set()
    paths: list[_CompressedPath] = []
    compressed_internal_nodes: set[int] = set()

    for start_id in sorted(branch_node_ids):
        for neighbor_id in coarsening_graph.neighbors_by_node[start_id]:
            first_edge = _edge_key(start_id, neighbor_id)
            if first_edge in visited_edges:
                continue
            if neighbor_id in branch_node_ids:
                visited_edges.add(first_edge)
                continue

            path = [start_id, neighbor_id]
            previous_id = start_id
            current_id = neighbor_id
            seen_path_nodes = {start_id}
            while True:
                visited_edges.add(_edge_key(previous_id, current_id))
                if current_id in seen_path_nodes:
                    break
                seen_path_nodes.add(current_id)
                if current_id in branch_node_ids:
                    break
                candidates = [
                    node_id
                    for node_id in coarsening_graph.neighbors_by_node[current_id]
                    if node_id != previous_id
                ]
                if len(candidates) != 1:
                    break
                next_id = candidates[0]
                path.append(next_id)
                previous_id, current_id = current_id, next_id

            if path[-1] in branch_node_ids and path[-1] != start_id and len(path) > 2:
                internal_node_ids = tuple(path[1:-1])
                paths.append(
                    _CompressedPath(
                        endpoint_ids=(path[0], path[-1]),
                        internal_node_ids=internal_node_ids,
                        node_ids=tuple(path),
                    )
                )
                compressed_internal_nodes.update(internal_node_ids)
                continue

    skeleton_node_ids = tuple(
        sorted(
            int(node_id)
            for node_id in coarsening_graph.node_ids
            if int(node_id) not in compressed_internal_nodes
        )
    )
    skeleton_node_set = set(skeleton_node_ids)
    skeleton_graph = nx.Graph()
    skeleton_graph.add_nodes_from(skeleton_node_ids)

    for (left_id, right_id), weight in coarsening_graph.edge_weights.items():
        if left_id not in skeleton_node_set or right_id not in skeleton_node_set:
            continue
        _add_weighted_skeleton_edge(
            skeleton_graph,
            left_id,
            right_id,
            weight=weight,
        )

    for path in paths:
        left_id, right_id = path.endpoint_ids
        _add_weighted_skeleton_edge(skeleton_graph, left_id, right_id, weight=1.0)

    return _DegreeTwoCompression(
        skeleton_node_ids=skeleton_node_ids,
        paths=tuple(sorted(paths, key=lambda path: path.node_ids)),
        skeleton_graph=skeleton_graph,
    )


def _add_weighted_skeleton_edge(
    skeleton_graph: nx.Graph,
    left_id: int,
    right_id: int,
    *,
    weight: float,
) -> None:
    if skeleton_graph.has_edge(left_id, right_id):
        skeleton_graph[left_id][right_id]["weight"] = (
            float(skeleton_graph[left_id][right_id].get("weight", 1.0)) + weight
        )
        return
    skeleton_graph.add_edge(left_id, right_id, weight=weight)


def _classified_positions_2d(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> NodePositions | None:
    if nx_graph.number_of_nodes() == 0:
        return {}
    if nx_graph.number_of_nodes() == 1:
        node_id = int(next(iter(nx_graph.nodes)))
        return {node_id: np.zeros(2, dtype=float)}

    structure_kind, chain_order, grid_mapping, grid3d_mapping, tree_root = _classify_anchor_graph(
        nx_graph,
        graph,
    )
    if structure_kind == "generic":
        return None
    if structure_kind == "chain":
        return _layout_chain(chain_order)
    if structure_kind == "circular":
        return _layout_circular(chain_order)
    if structure_kind == "grid" and grid_mapping is not None:
        return _layout_grid(grid_mapping)
    if structure_kind == "tube" and grid_mapping is not None:
        return _layout_tube_projection_2d(grid_mapping)
    if structure_kind == "grid3d" and grid3d_mapping is not None:
        return _layout_grid3d_projection_2d(grid3d_mapping)
    if structure_kind == "tree" and tree_root is not None:
        return _layout_tree(nx_graph, tree_root)
    if structure_kind == "planar":
        return _layout_planar(nx_graph)
    return None


def _generic_force_iterations(iterations: int, node_count: int) -> int:
    auto_iterations = max(
        _GENERIC_FORCE_MIN_ITERATIONS,
        min(_GENERIC_FORCE_MAX_ITERATIONS, 8 * max(node_count, 1)),
    )
    return min(int(iterations), int(auto_iterations))


def _pair_weights_from_graph(nx_graph: nx.Graph) -> dict[tuple[int, int], float]:
    return {
        _edge_key(int(left_id), int(right_id)): float(data.get("weight", 1.0))
        for left_id, right_id, data in nx_graph.edges(data=True)
    }


def _pair_weights_from_coarsening(
    coarsening_graph: _CoarseningGraph,
) -> dict[tuple[int, int], float]:
    return dict(coarsening_graph.edge_weights)


def _connection_weight(
    coarsening_graph: _CoarseningGraph,
    left_id: int,
    right_id: int,
) -> float:
    return float(coarsening_graph.edge_weights.get(_edge_key(left_id, right_id), 0.0))


def _total_weight_to_node_ids(
    coarsening_graph: _CoarseningGraph,
    node_id: int,
    node_ids: set[int],
) -> float:
    return float(
        sum(
            _connection_weight(coarsening_graph, node_id, neighbor_id)
            for neighbor_id in coarsening_graph.neighbors_by_node[node_id]
            if neighbor_id in node_ids
        )
    )


def _ordered_endpoint_candidates(
    coarsening_graph: _CoarseningGraph,
    *,
    endpoint_id: int,
    placed_node_ids: set[int],
    degree_by_node_id: dict[int, int],
) -> list[int]:
    candidates = [
        neighbor_id
        for neighbor_id in coarsening_graph.neighbors_by_node[endpoint_id]
        if neighbor_id not in placed_node_ids
    ]
    return sorted(
        candidates,
        key=lambda node_id: (
            -_connection_weight(coarsening_graph, endpoint_id, node_id),
            -_total_weight_to_node_ids(coarsening_graph, node_id, placed_node_ids),
            degree_by_node_id[node_id],
            node_id,
        ),
    )


def _ordered_recovery_candidates(
    coarsening_graph: _CoarseningGraph,
    *,
    remaining_node_ids: set[int],
    placed_node_ids: set[int],
    degree_by_node_id: dict[int, int],
) -> list[int]:
    return sorted(
        remaining_node_ids,
        key=lambda node_id: (
            -_total_weight_to_node_ids(coarsening_graph, node_id, placed_node_ids),
            degree_by_node_id[node_id],
            node_id,
        ),
    )


def _ordered_initial_circle_node_ids(coarsening_graph: _CoarseningGraph) -> list[int]:
    node_ids = list(coarsening_graph.node_ids)
    if len(node_ids) <= 2:
        return node_ids

    degree_by_node_id = {
        node_id: len(coarsening_graph.neighbors_by_node[node_id])
        for node_id in coarsening_graph.node_ids
    }
    start_id = min(node_ids, key=lambda node_id: (degree_by_node_id[node_id], node_id))
    ordered_node_ids: deque[int] = deque((start_id,))
    placed_node_ids = {start_id}
    remaining_node_ids = set(node_ids)
    remaining_node_ids.remove(start_id)

    initial_candidates = _ordered_endpoint_candidates(
        coarsening_graph,
        endpoint_id=start_id,
        placed_node_ids=placed_node_ids,
        degree_by_node_id=degree_by_node_id,
    )
    if initial_candidates:
        right_id = initial_candidates[0]
        ordered_node_ids.append(right_id)
        placed_node_ids.add(right_id)
        remaining_node_ids.remove(right_id)
    if len(initial_candidates) > 1:
        left_id = initial_candidates[1]
        ordered_node_ids.appendleft(left_id)
        placed_node_ids.add(left_id)
        remaining_node_ids.remove(left_id)

    next_side = "right"
    while remaining_node_ids:
        chosen_node_id: int | None = None
        chosen_side: str | None = None

        for side in (next_side, "left" if next_side == "right" else "right"):
            endpoint_id = ordered_node_ids[-1] if side == "right" else ordered_node_ids[0]
            candidates = _ordered_endpoint_candidates(
                coarsening_graph,
                endpoint_id=endpoint_id,
                placed_node_ids=placed_node_ids,
                degree_by_node_id=degree_by_node_id,
            )
            if candidates:
                chosen_node_id = candidates[0]
                chosen_side = side
                break

        used_recovery = False
        if chosen_node_id is None or chosen_side is None:
            used_recovery = True
            chosen_side = next_side
            chosen_node_id = _ordered_recovery_candidates(
                coarsening_graph,
                remaining_node_ids=remaining_node_ids,
                placed_node_ids=placed_node_ids,
                degree_by_node_id=degree_by_node_id,
            )[0]

        if chosen_side == "right":
            ordered_node_ids.append(chosen_node_id)
        else:
            ordered_node_ids.appendleft(chosen_node_id)
        placed_node_ids.add(chosen_node_id)
        remaining_node_ids.remove(chosen_node_id)

        if used_recovery:
            next_side = chosen_side
        elif chosen_side == next_side:
            next_side = "left" if chosen_side == "right" else "right"
        else:
            next_side = chosen_side

    return list(ordered_node_ids)


def _normalize_direction(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-9:
        return fallback / fallback_norm
    return np.array([1.0, 0.0], dtype=float)


def _normalize_direction_2d(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=float).reshape(-1)
    x = float(flat[0])
    y = float(flat[1])
    norm = math.hypot(x, y)
    if norm > 1e-9:
        return np.array([x / norm, y / norm], dtype=float)
    fallback_flat = np.asarray(fallback, dtype=float).reshape(-1)
    fallback_x = float(fallback_flat[0])
    fallback_y = float(fallback_flat[1])
    fallback_norm = math.hypot(fallback_x, fallback_y)
    if fallback_norm > 1e-9:
        return np.array([fallback_x / fallback_norm, fallback_y / fallback_norm], dtype=float)
    return np.array([1.0, 0.0], dtype=float)


def _expand_compressed_paths_2d(
    compression: _DegreeTwoCompression,
    positions: NodePositions,
) -> None:
    if not compression.paths:
        return
    skeleton_coords = np.stack([positions[node_id] for node_id in compression.skeleton_node_ids])
    centroid = skeleton_coords.mean(axis=0)
    grouped_indices: dict[tuple[int, int], int] = defaultdict(int)
    for path in compression.paths:
        left_id, right_id = path.endpoint_ids
        start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:2]
        end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:2]
        delta = end - start
        perpendicular = np.array([-delta[1], delta[0]], dtype=float)
        midpoint = (start + end) / 2.0
        outward = _normalize_direction(midpoint - centroid[:2], perpendicular)
        path_key = _edge_key(left_id, right_id)
        parallel_index = grouped_indices[path_key]
        grouped_indices[path_key] += 1
        side_offset = (parallel_index - 0.5 * max(grouped_indices[path_key] - 1, 0)) * 0.25
        height = max(
            _GENERIC_PATH_MIN_ARC_HEIGHT,
            0.18 * float(len(path.internal_node_ids) + 1),
            0.16 * float(np.linalg.norm(delta)),
        )
        control = (
            midpoint + outward * height + _normalize_direction(perpendicular, outward) * side_offset
        )
        steps = len(path.internal_node_ids) + 1
        for index, node_id in enumerate(path.internal_node_ids, start=1):
            t = float(index) / float(steps)
            positions[node_id] = (
                ((1.0 - t) ** 2) * start + (2.0 * (1.0 - t) * t) * control + (t**2) * end
            )


def _stable_fallback_direction(node_id: int) -> np.ndarray:
    angle = 2.399963229728653 * float(abs(node_id) + 1)
    return np.array([float(np.cos(angle)), float(np.sin(angle))], dtype=float)


@lru_cache(maxsize=128)
def _child_offsets(child_count: int) -> tuple[float, ...]:
    return tuple(float(index) - 0.5 * float(child_count - 1) for index in range(child_count))


def _expand_peeled_trees_2d(
    peeled: _PeeledDegreeOneTrees,
    positions: NodePositions,
) -> None:
    if not peeled.parent_by_removed_node:
        return
    anchor_ids = [node_id for node_id in peeled.core_node_ids if node_id in positions]
    if anchor_ids:
        centroid = np.stack([positions[node_id] for node_id in anchor_ids]).mean(axis=0)
    else:
        centroid = np.zeros(2, dtype=float)
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for node_id, parent_id in peeled.parent_by_removed_node.items():
        children_by_parent[parent_id].append(node_id)
    for children in children_by_parent.values():
        children.sort()

    stack: list[tuple[int, np.ndarray]] = []
    for root_parent_id in sorted(children_by_parent, reverse=True):
        if root_parent_id not in peeled.core_node_ids or root_parent_id not in positions:
            continue
        root_direction = _normalize_direction_2d(
            np.asarray(positions[root_parent_id], dtype=float).reshape(-1)[:2] - centroid[:2],
            _stable_fallback_direction(root_parent_id),
        )
        stack.append((root_parent_id, root_direction))

    while stack:
        parent_id, direction = stack.pop()
        parent_position = np.asarray(positions[parent_id], dtype=float).reshape(-1)[:2]
        children = children_by_parent.get(parent_id, [])
        if not children:
            continue
        base_direction = _normalize_direction_2d(direction, _stable_fallback_direction(parent_id))
        perpendicular = np.array([-base_direction[1], base_direction[0]], dtype=float)
        offsets = _child_offsets(len(children))
        for child_id, offset in zip(children, offsets, strict=True):
            if len(children) == 1:
                child_direction = base_direction
            else:
                child_direction = _normalize_direction_2d(
                    base_direction + perpendicular * 0.55 * float(offset),
                    base_direction,
                )
            positions[child_id] = parent_position + child_direction * _GENERIC_TREE_SPACING
            stack.append((child_id, child_direction))


def _compute_coarsened_layout_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    seed: int,
    iterations: int,
) -> NodePositions | None:
    if component.structure_kind not in _COARSENED_INITIAL_STRUCTURE_KINDS:
        return None
    node_ids = tuple(sorted(int(node_id) for node_id in component.node_ids))
    if len(node_ids) <= 1:
        return {node_ids[0]: np.zeros(2, dtype=float)} if node_ids else {}

    component_coarsening_graph = _coarsening_graph_from_nx(component.contraction_graph, node_ids)
    if component.structure_kind == "planar" and not _has_degree_one_neighbor(
        component_coarsening_graph
    ):
        return None

    peeled = _peel_degree_one_trees_from_coarsening(component_coarsening_graph)
    if not peeled.core_node_ids:
        return None
    if component.structure_kind != "generic" and len(peeled.core_node_ids) == len(node_ids):
        return None

    core_graph = _nx_graph_from_coarsening(component_coarsening_graph, peeled.core_node_ids)
    positions = _classified_positions_2d(core_graph, graph)
    if positions is None:
        core_coarsening_graph = _coarsening_graph_from_nx(core_graph, peeled.core_node_ids)
        compression = _compress_degree_two_paths_from_coarsening(core_coarsening_graph)
        skeleton_graph = compression.skeleton_graph
        positions = _classified_positions_2d(skeleton_graph, graph)
        if positions is None:
            skeleton_node_ids = list(compression.skeleton_node_ids)
            if component.structure_kind == "generic":
                skeleton_node_ids = _ordered_initial_circle_node_ids(
                    _coarsening_graph_from_nx(skeleton_graph, compression.skeleton_node_ids)
                )
            positions = _compute_weighted_force_layout(
                node_ids=skeleton_node_ids,
                pair_weights=_pair_weights_from_graph(skeleton_graph),
                dimensions=2,
                seed=seed,
                iterations=_generic_force_iterations(iterations, len(skeleton_node_ids)),
            )
        else:
            positions = {
                node_id: np.asarray(position, dtype=float).copy()
                for node_id, position in positions.items()
            }
        _expand_compressed_paths_2d(compression, positions)
    else:
        positions = {
            node_id: np.asarray(position, dtype=float).copy()
            for node_id, position in positions.items()
        }

    _expand_peeled_trees_2d(peeled, positions)
    if set(positions) != set(node_ids):
        missing = sorted(set(node_ids) - set(positions))
        if missing:
            fallback_positions = _compute_weighted_force_layout(
                node_ids=list(node_ids),
                pair_weights=_pair_weights_from_coarsening(component_coarsening_graph),
                dimensions=2,
                seed=seed,
                iterations=_generic_force_iterations(iterations, len(node_ids)),
            )
            for node_id in missing:
                positions[node_id] = fallback_positions[node_id]

    return cast(NodePositions, positions)


def _compute_generic_coarsened_layout_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    seed: int,
    iterations: int,
) -> NodePositions | None:
    if component.structure_kind != "generic":
        return None
    return _compute_coarsened_layout_2d(
        graph,
        component,
        seed=seed,
        iterations=iterations,
    )


__all__ = [
    "_CompressedPath",
    "_DegreeTwoCompression",
    "_PeeledDegreeOneTrees",
    "_compress_degree_two_paths",
    "_compute_coarsened_layout_2d",
    "_compute_generic_coarsened_layout_2d",
    "_distinct_neighbor_counts",
    "_ordered_initial_circle_node_ids",
    "_peel_degree_one_trees",
]
