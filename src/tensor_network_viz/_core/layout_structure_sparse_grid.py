"""Sparse-grid reconstruction helpers for deterministic layout detection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from .layout_structure_coordinates import (
    _coordinate_axis_lengths,
    _expected_grid_edges_from_coords,
    _node_edge_set,
)


@dataclass(frozen=True)
class _ThetaClass2D:
    """Theta-class metadata used by sparse-grid reconstruction."""

    edge_ids: tuple[tuple[int, int], ...]
    negative_side: frozenset[int]
    positive_side: frozenset[int]
    axis: int


_EdgeId2D = tuple[int, int]


def _sorted_edge_id_2d(left_id: int, right_id: int) -> _EdgeId2D:
    """Return a stable 2D edge id with endpoints stored in ascending order."""
    left = int(left_id)
    right = int(right_id)
    return (left, right) if left <= right else (right, left)


def _theta_classes_2d(nx_graph: nx.Graph) -> tuple[_ThetaClass2D, ...] | None:
    """Group graph edges into theta classes for sparse-grid reconstruction."""
    root_id = min(int(node_id) for node_id in nx_graph.nodes)
    distances = {
        int(node_id): {int(other_id): distance for other_id, distance in lengths.items()}
        for node_id, lengths in nx.all_pairs_shortest_path_length(nx_graph)
    }
    edge_ids = tuple(
        _sorted_edge_id_2d(int(left_id), int(right_id)) for left_id, right_id in nx_graph.edges()
    )
    parent: dict[_EdgeId2D, _EdgeId2D] = {edge_id: edge_id for edge_id in edge_ids}

    def find(edge_id: _EdgeId2D) -> _EdgeId2D:
        while parent[edge_id] != edge_id:
            parent[edge_id] = parent[parent[edge_id]]
            edge_id = parent[edge_id]
        return edge_id

    def union(left_edge: _EdgeId2D, right_edge: _EdgeId2D) -> None:
        left_root = find(left_edge)
        right_root = find(right_edge)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, (left_a, right_a) in enumerate(edge_ids):
        for left_b, right_b in edge_ids[index + 1 :]:
            if (
                distances[left_a][left_b] + distances[right_a][right_b]
                != distances[left_a][right_b] + distances[right_a][left_b]
            ):
                union((left_a, right_a), (left_b, right_b))

    grouped_edges: dict[_EdgeId2D, list[_EdgeId2D]] = {}
    for edge_id in edge_ids:
        grouped_edges.setdefault(find(edge_id), []).append(edge_id)

    theta_classes: list[_ThetaClass2D] = []
    all_node_ids = frozenset(int(node_id) for node_id in nx_graph.nodes)
    for edge_group in grouped_edges.values():
        left_id, right_id = edge_group[0]
        negative_side = frozenset(
            int(node_id)
            for node_id in nx_graph.nodes
            if distances[int(node_id)][left_id] < distances[int(node_id)][right_id]
        )
        positive_side = frozenset(all_node_ids - negative_side)
        if root_id in positive_side:
            negative_side, positive_side = positive_side, negative_side
        theta_classes.append(
            _ThetaClass2D(
                edge_ids=tuple(sorted(edge_group)),
                negative_side=negative_side,
                positive_side=positive_side,
                axis=0,
            )
        )

    crossing_graph = _crossing_graph_2d(tuple(theta_classes))
    if not nx.is_bipartite(crossing_graph):
        return None
    axis_by_class = nx.bipartite.color(crossing_graph)
    return tuple(
        _ThetaClass2D(
            edge_ids=theta_class.edge_ids,
            negative_side=theta_class.negative_side,
            positive_side=theta_class.positive_side,
            axis=int(axis_by_class[class_index]),
        )
        for class_index, theta_class in enumerate(theta_classes)
    )


def _crossing_graph_2d(theta_classes: tuple[_ThetaClass2D, ...]) -> nx.Graph:
    """Build the theta-class crossing graph used to separate sparse-grid axes."""
    crossing_graph = nx.Graph()
    crossing_graph.add_nodes_from(range(len(theta_classes)))
    for left_index, right_index in combinations(range(len(theta_classes)), 2):
        left_class = theta_classes[left_index]
        right_class = theta_classes[right_index]
        if (
            left_class.negative_side & right_class.negative_side
            and left_class.negative_side & right_class.positive_side
            and left_class.positive_side & right_class.negative_side
            and left_class.positive_side & right_class.positive_side
        ):
            crossing_graph.add_edge(left_index, right_index)
    return crossing_graph


def _crossing_component_ids_2d(
    theta_classes: tuple[_ThetaClass2D, ...],
) -> dict[int, int] | None:
    """Return connected-component ids for the theta-class crossing graph."""
    crossing_graph = _crossing_graph_2d(theta_classes)
    if not nx.is_bipartite(crossing_graph):
        return None

    component_ids_by_class: dict[int, int] = {}
    for component_index, component_node_ids in enumerate(nx.connected_components(crossing_graph)):
        for class_index in component_node_ids:
            component_ids_by_class[int(class_index)] = component_index
    return component_ids_by_class


def _seed_sparse_grid_flip_mask_2d(
    nx_graph: nx.Graph,
    *,
    theta_classes: tuple[_ThetaClass2D, ...],
    component_ids_by_class: dict[int, int],
) -> int:
    """Pick a preferred crossing-component flip mask from ``planar_layout``."""
    planar_positions = nx.planar_layout(nx_graph)
    preferred_axis_by_class: dict[int, int] = {}
    for class_index, theta_class in enumerate(theta_classes):
        horizontal_score = sum(
            abs(float(planar_positions[left_id][0]) - float(planar_positions[right_id][0]))
            for left_id, right_id in theta_class.edge_ids
        )
        vertical_score = sum(
            abs(float(planar_positions[left_id][1]) - float(planar_positions[right_id][1]))
            for left_id, right_id in theta_class.edge_ids
        )
        preferred_axis_by_class[class_index] = 0 if horizontal_score >= vertical_score else 1

    component_flip_mask = 0
    for component_id in sorted(set(component_ids_by_class.values())):
        component_class_indices = [
            class_index
            for class_index, current_component_id in component_ids_by_class.items()
            if current_component_id == component_id
        ]
        no_flip_score = sum(
            int(theta_classes[class_index].axis == preferred_axis_by_class[class_index])
            for class_index in component_class_indices
        )
        flip_score = sum(
            int((theta_classes[class_index].axis ^ 1) == preferred_axis_by_class[class_index])
            for class_index in component_class_indices
        )
        if flip_score > no_flip_score:
            component_flip_mask |= 1 << component_id
    return component_flip_mask


def _grid_mapping_from_theta_classes(
    nx_graph: nx.Graph,
    *,
    theta_classes: tuple[_ThetaClass2D, ...],
    class_axes: dict[int, int],
) -> dict[int, tuple[int, int]] | None:
    """Reconstruct and validate sparse-grid coordinates from theta classes."""
    axis_counts = [0, 0]
    for class_index, theta_class in enumerate(theta_classes):
        axis_index = int(class_axes[class_index])
        axis_counts[axis_index] += len(theta_class.edge_ids)

    if axis_counts[0] == 0 or axis_counts[1] == 0:
        return None

    coords_by_node = {
        int(node_id): [
            sum(
                int(node_id) in theta_class.positive_side
                for class_index, theta_class in enumerate(theta_classes)
                if int(class_axes[class_index]) == 0
            ),
            sum(
                int(node_id) in theta_class.positive_side
                for class_index, theta_class in enumerate(theta_classes)
                if int(class_axes[class_index]) == 1
            ),
        ]
        for node_id in nx_graph.nodes
    }
    normalized_mapping = {
        node_id: (int(coords[0]), int(coords[1])) for node_id, coords in coords_by_node.items()
    }
    if len(set(normalized_mapping.values())) != len(normalized_mapping):
        return None
    if any(axis_length <= 1 for axis_length in _coordinate_axis_lengths(normalized_mapping)):
        return None
    if _node_edge_set(nx_graph) != _expected_grid_edges_from_coords(
        normalized_mapping, dimensions=2
    ):
        return None
    return normalized_mapping
