"""Structural analysis helpers for deterministic graph layouts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import networkx as nx
import numpy as np

from .contractions import _iter_contractions
from .graph import _GraphData

StructureKind = Literal["chain", "grid", "tree", "planar", "generic"]


@dataclass(frozen=True)
class _LayoutComponent:
    node_ids: tuple[int, ...]
    visible_node_ids: tuple[int, ...]
    virtual_node_ids: tuple[int, ...]
    trimmed_leaf_parents: tuple[tuple[int, int], ...]
    contraction_graph: nx.Graph
    visible_graph: nx.Graph
    proxy_visible_graph: nx.Graph
    anchor_node_ids: tuple[int, ...]
    anchor_graph: nx.Graph
    structure_kind: StructureKind
    chain_order: tuple[int, ...]
    grid_mapping: dict[int, tuple[int, int]] | None
    tree_root: int | None


def _analyze_layout_components(graph: _GraphData) -> tuple[_LayoutComponent, ...]:
    contraction_graph = _build_contraction_graph(graph)
    components: list[_LayoutComponent] = []
    for component_node_ids in _sorted_connected_components(contraction_graph):
        component_graph = contraction_graph.subgraph(component_node_ids).copy()
        visible_node_ids = tuple(
            sorted(node_id for node_id in component_node_ids if not graph.nodes[node_id].is_virtual)
        )
        virtual_node_ids = tuple(
            sorted(node_id for node_id in component_node_ids if graph.nodes[node_id].is_virtual)
        )
        visible_graph = _build_visible_graph(graph, component_node_ids=component_node_ids)
        trimmed_visible_graph, trimmed_leaf_parents = _trim_visible_leaf_nodes(
            graph,
            visible_graph,
        )
        proxy_visible_graph = _build_proxy_visible_graph(
            graph,
            component_node_ids=component_node_ids,
        )
        anchor_graph = _select_anchor_graph(
            component_graph=component_graph,
            visible_graph=(
                trimmed_visible_graph
                if trimmed_visible_graph.number_of_nodes() > 0
                else visible_graph
            ),
            proxy_visible_graph=proxy_visible_graph,
        )
        structure_kind, chain_order, grid_mapping, tree_root = _classify_anchor_graph(anchor_graph)
        components.append(
            _LayoutComponent(
                node_ids=tuple(sorted(component_node_ids)),
                visible_node_ids=visible_node_ids,
                virtual_node_ids=virtual_node_ids,
                trimmed_leaf_parents=trimmed_leaf_parents,
                contraction_graph=component_graph,
                visible_graph=visible_graph,
                proxy_visible_graph=proxy_visible_graph,
                anchor_node_ids=tuple(sorted(anchor_graph.nodes)),
                anchor_graph=anchor_graph,
                structure_kind=structure_kind,
                chain_order=chain_order,
                grid_mapping=grid_mapping,
                tree_root=tree_root,
            )
        )

    return tuple(components)


def _build_contraction_graph(graph: _GraphData) -> nx.Graph:
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.nodes)
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        if nx_graph.has_edge(left_id, right_id):
            nx_graph[left_id][right_id]["weight"] += 1
            continue
        nx_graph.add_edge(left_id, right_id, weight=1)
    return nx_graph


def _build_visible_graph(
    graph: _GraphData,
    *,
    component_node_ids: tuple[int, ...],
) -> nx.Graph:
    visible_node_ids = [
        node_id for node_id in component_node_ids if not graph.nodes[node_id].is_virtual
    ]
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(visible_node_ids)
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        if left_id not in nx_graph or right_id not in nx_graph:
            continue
        nx_graph.add_edge(left_id, right_id)
    return nx_graph


def _build_proxy_visible_graph(
    graph: _GraphData,
    *,
    component_node_ids: tuple[int, ...],
) -> nx.Graph:
    proxy_graph = _build_visible_graph(graph, component_node_ids=component_node_ids)
    contraction_graph = _build_contraction_graph(graph).subgraph(component_node_ids)
    for node_id in component_node_ids:
        if not graph.nodes[node_id].is_virtual:
            continue
        visible_neighbors = sorted(
            neighbor_id
            for neighbor_id in contraction_graph.neighbors(node_id)
            if not graph.nodes[neighbor_id].is_virtual
        )
        for left_id, right_id in combinations(visible_neighbors, 2):
            proxy_graph.add_edge(left_id, right_id)
    return proxy_graph


def _select_anchor_graph(
    *,
    component_graph: nx.Graph,
    visible_graph: nx.Graph,
    proxy_visible_graph: nx.Graph,
) -> nx.Graph:
    direct_component = _largest_nontrivial_component(visible_graph)
    if direct_component is not None:
        return visible_graph.subgraph(direct_component).copy()

    proxy_component = _largest_nontrivial_component(proxy_visible_graph)
    if proxy_component is not None:
        return proxy_visible_graph.subgraph(proxy_component).copy()

    if proxy_visible_graph.number_of_nodes() > 0:
        visible_nodes = tuple(sorted(proxy_visible_graph.nodes))
        return proxy_visible_graph.subgraph(visible_nodes[:1]).copy()

    component_nodes = tuple(sorted(component_graph.nodes))
    return component_graph.subgraph(component_nodes[:1]).copy()


def _largest_nontrivial_component(nx_graph: nx.Graph) -> tuple[int, ...] | None:
    if nx_graph.number_of_nodes() == 0:
        return None
    nontrivial = [
        tuple(sorted(component))
        for component in nx.connected_components(nx_graph)
        if len(component) > 1 or any(nx_graph.degree(node_id) > 0 for node_id in component)
    ]
    if not nontrivial:
        return None
    nontrivial.sort(key=lambda node_ids: (-len(node_ids), node_ids[0]))
    return nontrivial[0]


def _trim_visible_leaf_nodes(
    graph: _GraphData,
    visible_graph: nx.Graph,
) -> tuple[nx.Graph, tuple[tuple[int, int], ...]]:
    trimmed_graph = visible_graph.copy()
    leaf_parents = []
    for node_id, degree in visible_graph.degree():
        if degree != 1 or graph.nodes[node_id].degree != 1:
            continue
        parent_id = next(iter(visible_graph.neighbors(node_id)))
        leaf_parents.append((node_id, parent_id))

    if len(leaf_parents) >= visible_graph.number_of_nodes():
        return visible_graph.copy(), ()

    trimmed_graph.remove_nodes_from([node_id for node_id, _ in leaf_parents])
    leaf_parents.sort()
    return trimmed_graph, tuple(leaf_parents)


def _sorted_connected_components(nx_graph: nx.Graph) -> list[tuple[int, ...]]:
    components = [tuple(sorted(component)) for component in nx.connected_components(nx_graph)]
    components.sort(key=lambda node_ids: (node_ids[0], len(node_ids)))
    return components


def _classify_anchor_graph(
    anchor_graph: nx.Graph,
) -> tuple[StructureKind, tuple[int, ...], dict[int, tuple[int, int]] | None, int | None]:
    chain_order = _detect_chain(anchor_graph)
    if chain_order is not None:
        return "chain", chain_order, None, None

    grid_mapping = _detect_grid(anchor_graph)
    if grid_mapping is not None:
        return "grid", (), grid_mapping, None

    if anchor_graph.number_of_nodes() > 0 and nx.is_tree(anchor_graph):
        return "tree", (), None, _tree_root(anchor_graph)

    is_planar, _ = nx.check_planarity(anchor_graph)
    if anchor_graph.number_of_nodes() > 0 and is_planar:
        return "planar", (), None, None

    return "generic", (), None, None


def _specialized_anchor_positions(component: _LayoutComponent) -> dict[int, np.ndarray]:
    if not component.anchor_node_ids:
        return {}
    if component.structure_kind == "chain":
        return _layout_chain(component.chain_order)
    if component.structure_kind == "grid" and component.grid_mapping is not None:
        return _layout_grid(component.grid_mapping)
    if component.structure_kind == "tree" and component.tree_root is not None:
        return _layout_tree(component.anchor_graph, component.tree_root)
    if component.structure_kind == "planar":
        return _layout_planar(component.anchor_graph)
    return {}


def _detect_chain(nx_graph: nx.Graph) -> tuple[int, ...] | None:
    if nx_graph.number_of_nodes() < 2 or not nx.is_tree(nx_graph):
        return None
    degrees = dict(nx_graph.degree())
    if any(degree > 2 for degree in degrees.values()):
        return None

    endpoints = sorted(node_id for node_id, degree in degrees.items() if degree <= 1)
    if len(endpoints) != 2:
        return None

    order = [endpoints[0]]
    previous_id: int | None = None
    current_id = endpoints[0]
    while True:
        neighbors = sorted(
            neighbor_id
            for neighbor_id in nx_graph.neighbors(current_id)
            if neighbor_id != previous_id
        )
        if not neighbors:
            break
        next_id = neighbors[0]
        order.append(next_id)
        previous_id, current_id = current_id, next_id

    if len(order) != nx_graph.number_of_nodes():
        return None
    return tuple(order)


def _detect_grid(nx_graph: nx.Graph) -> dict[int, tuple[int, int]] | None:
    if nx_graph.number_of_nodes() <= 1:
        return None
    if not nx.is_connected(nx_graph):
        return None

    degree_histogram = {}
    for _, degree in nx_graph.degree():
        degree_histogram[degree] = degree_histogram.get(degree, 0) + 1
    if any(degree > 4 for degree in degree_histogram):
        return None

    n_nodes = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()
    for rows in range(1, n_nodes + 1):
        if n_nodes % rows != 0:
            continue
        cols = n_nodes // rows
        expected_edges = 2 * rows * cols - rows - cols
        if n_edges != expected_edges:
            continue
        if degree_histogram != dict(_grid_degree_histogram(rows, cols)):
            continue
        grid_graph = nx.grid_2d_graph(rows, cols)
        mapping = nx.vf2pp_isomorphism(nx_graph, grid_graph)
        if mapping is None:
            continue
        return {node_id: (grid_col, grid_row) for node_id, (grid_row, grid_col) in mapping.items()}
    return None


def _layout_chain(node_order: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {
        node_id: np.array([float(index), 0.0], dtype=float)
        for index, node_id in enumerate(node_order)
    }


def _layout_grid(grid_mapping: dict[int, tuple[int, int]]) -> dict[int, np.ndarray]:
    return {
        node_id: np.array([float(col), float(row)], dtype=float)
        for node_id, (col, row) in grid_mapping.items()
    }


def _layout_planar(nx_graph: nx.Graph) -> dict[int, np.ndarray]:
    positions = nx.planar_layout(nx_graph)
    return {node_id: np.array(positions[node_id], dtype=float) for node_id in nx_graph.nodes}


def _layout_tree(nx_graph: nx.Graph, root_id: int) -> dict[int, np.ndarray]:
    children_by_node: dict[int, list[int]] = {node_id: [] for node_id in nx_graph.nodes}
    parent_by_node: dict[int, int | None] = {root_id: None}
    queue = [root_id]
    for node_id in queue:
        children = sorted(
            neighbor_id
            for neighbor_id in nx_graph.neighbors(node_id)
            if neighbor_id != parent_by_node[node_id]
        )
        children_by_node[node_id] = children
        for child_id in children:
            parent_by_node[child_id] = node_id
            queue.append(child_id)

    positions: dict[int, np.ndarray] = {}

    def place(node_id: int, next_x: float, depth: int) -> float:
        children = children_by_node[node_id]
        if not children:
            positions[node_id] = np.array([next_x, -float(depth)], dtype=float)
            return next_x + 1.0

        child_centers: list[float] = []
        cursor = next_x
        for child_id in children:
            cursor = place(child_id, cursor, depth + 1)
            child_centers.append(float(positions[child_id][0]))
        positions[node_id] = np.array(
            [(child_centers[0] + child_centers[-1]) / 2.0, -float(depth)],
            dtype=float,
        )
        return cursor

    place(root_id, 0.0, 0)
    return positions


def _tree_root(nx_graph: nx.Graph) -> int:
    centers = sorted(nx.center(nx_graph))
    return centers[0]


def _grid_degree_histogram(rows: int, cols: int) -> dict[int, int]:
    if rows == 1:
        if cols == 2:
            return {1: 2}
        return {1: 2, 2: cols - 2}
    if cols == 1:
        if rows == 2:
            return {1: 2}
        return {1: 2, 2: rows - 2}

    histogram = {2: 4}
    if rows > 2:
        histogram[3] = histogram.get(3, 0) + 2 * (rows - 2)
    if cols > 2:
        histogram[3] = histogram.get(3, 0) + 2 * (cols - 2)
    if rows > 2 and cols > 2:
        histogram[4] = (rows - 2) * (cols - 2)
    return histogram


def _leaf_nodes(component: _LayoutComponent) -> tuple[int, ...]:
    return tuple(
        sorted(
            node_id
            for node_id in component.visible_graph.nodes
            if component.visible_graph.degree(node_id) <= 1
        )
    )


def _component_main_axis(
    component: _LayoutComponent,
    positions: dict[int, np.ndarray],
) -> np.ndarray:
    if component.structure_kind == "chain" and component.chain_order:
        start = positions[component.chain_order[0]]
        end = positions[component.chain_order[-1]]
        axis = end - start
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            return axis / norm
    if component.anchor_node_ids:
        coords = np.stack([positions[node_id] for node_id in component.anchor_node_ids])
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            return axis / norm
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _component_orthogonal_basis(
    component: _LayoutComponent,
    positions: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = _component_main_axis(component, positions)
    lateral = np.cross(normal, axis)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        lateral /= np.linalg.norm(lateral)
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    return axis, lateral, normal
