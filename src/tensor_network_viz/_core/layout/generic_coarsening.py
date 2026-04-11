"""Generic-layout graph coarsening helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
    node_id_set = set(node_ids)
    return {
        int(node_id): sum(
            1 for neighbor_id in nx_graph.neighbors(node_id) if neighbor_id in node_id_set
        )
        for node_id in node_ids
    }


def _peel_degree_one_trees(
    nx_graph: nx.Graph,
    node_ids: tuple[int, ...],
) -> _PeeledDegreeOneTrees:
    """Remove degree-one trees recursively, using distinct contraction neighbors."""
    remaining = {int(node_id) for node_id in node_ids}
    parent_by_removed_node: dict[int, int] = {}
    queue = sorted(
        node_id
        for node_id, degree in _distinct_neighbor_counts(nx_graph, node_ids).items()
        if degree <= 1
    )

    while queue and len(remaining) > 1:
        node_id = queue.pop(0)
        if node_id not in remaining:
            continue
        neighbors = sorted(
            int(neighbor_id)
            for neighbor_id in nx_graph.neighbors(node_id)
            if neighbor_id in remaining
        )
        if len(neighbors) > 1:
            continue
        if neighbors:
            parent_id = neighbors[0]
            parent_by_removed_node[node_id] = parent_id
        else:
            parent_id = None

        remaining.remove(node_id)
        if parent_id is not None and parent_id in remaining:
            degree = sum(
                1 for neighbor_id in nx_graph.neighbors(parent_id) if int(neighbor_id) in remaining
            )
            if degree <= 1:
                queue.append(parent_id)
                queue.sort()

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
    subgraph = nx_graph.subgraph(node_ids).copy()
    degrees = _distinct_neighbor_counts(subgraph, tuple(sorted(subgraph.nodes)))
    branch_node_ids = {node_id for node_id, degree in degrees.items() if degree != 2}
    if not branch_node_ids:
        return _DegreeTwoCompression(
            skeleton_node_ids=tuple(sorted(subgraph.nodes)),
            paths=(),
            skeleton_graph=subgraph,
        )

    visited_edges: set[tuple[int, int]] = set()
    paths: list[_CompressedPath] = []
    compressed_internal_nodes: set[int] = set()

    for start_id in sorted(branch_node_ids):
        for neighbor_id in sorted(int(node_id) for node_id in subgraph.neighbors(start_id)):
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
                candidates = sorted(
                    int(node_id)
                    for node_id in subgraph.neighbors(current_id)
                    if int(node_id) != previous_id
                )
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
            for node_id in subgraph.nodes
            if int(node_id) not in compressed_internal_nodes
        )
    )
    skeleton_node_set = set(skeleton_node_ids)
    skeleton_graph = nx.Graph()
    skeleton_graph.add_nodes_from(skeleton_node_ids)

    for left_id, right_id, data in subgraph.edges(data=True):
        left_int = int(left_id)
        right_int = int(right_id)
        if left_int not in skeleton_node_set or right_int not in skeleton_node_set:
            continue
        _add_weighted_skeleton_edge(
            skeleton_graph,
            left_int,
            right_int,
            weight=float(data.get("weight", 1.0)),
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


def _normalize_direction(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-9:
        return fallback / fallback_norm
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

    def place_children(parent_id: int, direction: np.ndarray) -> None:
        if parent_id not in positions:
            return
        parent_position = np.asarray(positions[parent_id], dtype=float).reshape(-1)[:2]
        children = children_by_parent.get(parent_id, [])
        if not children:
            return
        base_direction = _normalize_direction(direction, _stable_fallback_direction(parent_id))
        perpendicular = np.array([-base_direction[1], base_direction[0]], dtype=float)
        offsets = np.linspace(
            -0.5 * float(len(children) - 1),
            0.5 * float(len(children) - 1),
            len(children),
        )
        for child_id, offset in zip(children, offsets, strict=True):
            child_direction = _normalize_direction(
                base_direction + perpendicular * 0.55 * float(offset),
                base_direction,
            )
            positions[child_id] = parent_position + child_direction * _GENERIC_TREE_SPACING
            place_children(child_id, child_direction)

    for root_parent_id in sorted(children_by_parent):
        if root_parent_id not in positions:
            continue
        root_direction = _normalize_direction(
            np.asarray(positions[root_parent_id], dtype=float).reshape(-1)[:2] - centroid[:2],
            _stable_fallback_direction(root_parent_id),
        )
        place_children(root_parent_id, root_direction)


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

    component_graph = component.contraction_graph.subgraph(node_ids).copy()
    peeled = _peel_degree_one_trees(component_graph, node_ids)
    if not peeled.core_node_ids:
        return None
    if component.structure_kind != "generic" and len(peeled.core_node_ids) == len(node_ids):
        return None

    core_graph = component_graph.subgraph(peeled.core_node_ids).copy()
    positions = _classified_positions_2d(core_graph, graph)
    if positions is None:
        compression = _compress_degree_two_paths(core_graph, peeled.core_node_ids)
        skeleton_graph = compression.skeleton_graph
        positions = _classified_positions_2d(skeleton_graph, graph)
        if positions is None:
            skeleton_node_ids = list(compression.skeleton_node_ids)
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
                pair_weights=_pair_weights_from_graph(component_graph),
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
    "_peel_degree_one_trees",
]
