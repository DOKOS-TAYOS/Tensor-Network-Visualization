"""Structural analysis helpers for deterministic graph layouts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import networkx as nx
import numpy as np

from .contractions import _iter_contractions
from .graph import _GraphData

StructureKind = Literal[
    "chain",
    "grid",
    "grid3d",
    "tube",
    "circular",
    "tree",
    "planar",
    "generic",
]

# Oblique 2D projection (i,j,k) → layout xy. The depth axis uses an asymmetric skew so
# free stubs on projected cubic grids get a cleaner outward corridor between bonds.
_GRID3D_PROJECTION_X: float = -0.45
_GRID3D_PROJECTION_Y: float = -0.25


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
    grid3d_mapping: dict[int, tuple[int, int, int]] | None
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
        visible_graph = _visible_graph_for_component(
            graph,
            component_graph,
            component_node_ids=component_node_ids,
        )
        trimmed_visible_graph, trimmed_leaf_parents = _trim_visible_leaf_nodes(
            graph,
            visible_graph,
        )
        proxy_visible_graph = _proxy_visible_graph_for_component(
            graph,
            component_graph,
            component_node_ids=component_node_ids,
            visible_graph=visible_graph,
        )
        untrimmed_anchor_graph = _select_anchor_graph(
            component_graph=component_graph,
            visible_graph=visible_graph,
            proxy_visible_graph=proxy_visible_graph,
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
        untrimmed_classification = _classify_untrimmed_structured_anchor_graph(
            untrimmed_anchor_graph,
            graph,
        )
        if untrimmed_classification is not None:
            anchor_graph = untrimmed_anchor_graph
            trimmed_leaf_parents = ()
            structure_kind, chain_order, grid_mapping, grid3d_mapping, tree_root = (
                untrimmed_classification
            )
        else:
            structure_kind, chain_order, grid_mapping, grid3d_mapping, tree_root = (
                _classify_anchor_graph(anchor_graph, graph)
            )
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
                grid3d_mapping=grid3d_mapping,
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


def _visible_graph_for_component(
    graph: _GraphData,
    component_graph: nx.Graph,
    *,
    component_node_ids: tuple[int, ...],
) -> nx.Graph:
    """Visible-only induced subgraph of contractions, using only *component_graph* edges."""
    visible_node_ids = [
        node_id for node_id in component_node_ids if not graph.nodes[node_id].is_virtual
    ]
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(visible_node_ids)
    for left_id, right_id in component_graph.edges():
        if left_id in nx_graph and right_id in nx_graph:
            nx_graph.add_edge(left_id, right_id)
    return nx_graph


def _proxy_visible_graph_for_component(
    graph: _GraphData,
    component_graph: nx.Graph,
    *,
    component_node_ids: tuple[int, ...],
    visible_graph: nx.Graph,
) -> nx.Graph:
    """Augment *visible_graph* with proxy edges via virtual hubs.

    Does not rescan all contractions in the full graph.
    """
    proxy_graph = visible_graph.copy()
    for node_id in component_node_ids:
        if not graph.nodes[node_id].is_virtual:
            continue
        visible_neighbors = sorted(
            neighbor_id
            for neighbor_id in component_graph.neighbors(node_id)
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
    graph: _GraphData,
) -> tuple[
    StructureKind,
    tuple[int, ...],
    dict[int, tuple[int, int]] | None,
    dict[int, tuple[int, int, int]] | None,
    int | None,
]:
    chain_order = _detect_chain(anchor_graph)
    if chain_order is not None:
        return "chain", chain_order, None, None, None

    grid3d_mapping = _detect_coordinate_grid_3d(anchor_graph, graph)
    if grid3d_mapping is not None:
        return "grid3d", (), None, grid3d_mapping, None

    grid_mapping = _detect_coordinate_grid_2d(anchor_graph, graph)
    if grid_mapping is not None:
        return "grid", (), grid_mapping, None, None

    tube_mapping = _detect_coordinate_tube(anchor_graph, graph)
    if tube_mapping is not None:
        return "tube", (), tube_mapping, None, None

    if anchor_graph.number_of_nodes() > 0 and nx.is_tree(anchor_graph):
        return "tree", (), None, None, _tree_root(anchor_graph)

    grid_mapping = _detect_grid(anchor_graph)
    if grid_mapping is not None:
        return "grid", (), grid_mapping, None, None

    grid3d_mapping = _detect_grid_3d(anchor_graph)
    if grid3d_mapping is not None:
        return "grid3d", (), None, grid3d_mapping, None

    tube_mapping = _detect_topological_tube(anchor_graph)
    if tube_mapping is not None:
        return "tube", (), tube_mapping, None, None

    circular_order = _detect_circular_order(anchor_graph, graph)
    if circular_order is not None:
        return "circular", circular_order, None, None, None

    is_planar, _ = nx.check_planarity(anchor_graph)
    if anchor_graph.number_of_nodes() > 0 and is_planar:
        return "planar", (), None, None, None

    return "generic", (), None, None, None


def _classify_untrimmed_structured_anchor_graph(
    anchor_graph: nx.Graph,
    graph: _GraphData,
) -> (
    tuple[
        StructureKind,
        tuple[int, ...],
        dict[int, tuple[int, int]] | None,
        dict[int, tuple[int, int, int]] | None,
        int | None,
    ]
    | None
):
    if _detect_chain(anchor_graph) is not None:
        return None

    if anchor_graph.number_of_nodes() > 0 and nx.is_tree(anchor_graph):
        return None

    grid3d_mapping = _detect_coordinate_grid_3d(anchor_graph, graph)
    if grid3d_mapping is not None:
        return "grid3d", (), None, grid3d_mapping, None

    grid_mapping = _detect_coordinate_grid_2d(anchor_graph, graph)
    if grid_mapping is not None:
        return "grid", (), grid_mapping, None, None

    tube_mapping = _detect_coordinate_tube(anchor_graph, graph)
    if tube_mapping is not None:
        return "tube", (), tube_mapping, None, None

    circular_order = _detect_named_ring_order(anchor_graph, graph)
    if circular_order is not None:
        return "circular", circular_order, None, None, None

    return None


def _specialized_anchor_positions(component: _LayoutComponent) -> dict[int, np.ndarray]:
    if not component.anchor_node_ids:
        return {}
    if component.structure_kind == "chain":
        return _layout_chain(component.chain_order)
    if component.structure_kind == "circular":
        return _layout_circular(component.chain_order)
    if component.structure_kind == "grid" and component.grid_mapping is not None:
        return _layout_grid(component.grid_mapping)
    if component.structure_kind == "tube" and component.grid_mapping is not None:
        return _layout_tube_projection_2d(component.grid_mapping)
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        return _layout_grid3d_projection_2d(component.grid3d_mapping)
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


def _layout_circular(node_order: tuple[int, ...]) -> dict[int, np.ndarray]:
    if not node_order:
        return {}
    radius = max(float(len(node_order)) / (2.0 * math.pi), 1.0)
    return {
        node_id: np.array(
            [
                radius * math.cos(2.0 * math.pi * index / len(node_order)),
                radius * math.sin(2.0 * math.pi * index / len(node_order)),
            ],
            dtype=float,
        )
        for index, node_id in enumerate(node_order)
    }


def _tube_radius(periodic_count: int) -> float:
    return max(float(periodic_count) / (2.0 * math.pi), 1.0)


def _layout_tube_projection_2d(
    tube_mapping: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    if not tube_mapping:
        return {}
    periodic_count = max(theta for theta, _ in tube_mapping.values()) + 1
    radius = _tube_radius(periodic_count)
    axial_spacing = 2.25 * radius
    axial_skew = 0.28 * radius
    return {
        node_id: np.array(
            [
                float(axial_index) * axial_spacing
                + radius * math.cos(2.0 * math.pi * float(theta_index) / periodic_count),
                float(axial_index) * axial_skew
                + radius * math.sin(2.0 * math.pi * float(theta_index) / periodic_count),
            ],
            dtype=float,
        )
        for node_id, (theta_index, axial_index) in tube_mapping.items()
    }


def _layout_tube_3d(
    tube_mapping: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    if not tube_mapping:
        return {}
    periodic_count = max(theta for theta, _ in tube_mapping.values()) + 1
    radius = _tube_radius(periodic_count)
    return {
        node_id: np.array(
            [
                radius * math.cos(2.0 * math.pi * float(theta_index) / periodic_count),
                radius * math.sin(2.0 * math.pi * float(theta_index) / periodic_count),
                float(axial_index),
            ],
            dtype=float,
        )
        for node_id, (theta_index, axial_index) in tube_mapping.items()
    }


def _expected_edges_3d_grid(lx: int, ly: int, lz: int) -> int:
    return (lx - 1) * ly * lz + lx * (ly - 1) * lz + lx * ly * (lz - 1)


def _detect_grid_3d(nx_graph: nx.Graph) -> dict[int, tuple[int, int, int]] | None:
    """If *nx_graph* is a 3D rectangular grid (all factors >= 1), return node → (i,j,k)."""
    n = nx_graph.number_of_nodes()
    if n <= 1 or not nx.is_connected(nx_graph):
        return None
    m = nx_graph.number_of_edges()
    best_mn = -1
    best_shape: tuple[int, int, int] = (-1, -1, -1)
    best_mapping: dict[int, tuple[int, int, int]] | None = None

    for lx in range(1, n + 1):
        if n % lx != 0:
            continue
        rest = n // lx
        for ly in range(1, rest + 1):
            if rest % ly != 0:
                continue
            lz = rest // ly
            if _expected_edges_3d_grid(lx, ly, lz) != m:
                continue
            template = nx.grid_graph((lx, ly, lz))
            mapping = nx.vf2pp_isomorphism(nx_graph, template)
            if mapping is None:
                continue
            mn = min(lx, ly, lz)
            shape = (lx, ly, lz)
            if mn > best_mn or (mn == best_mn and shape > best_shape):
                best_mn = mn
                best_shape = shape
                best_mapping = {
                    int(node_id): (int(a), int(b), int(c)) for node_id, (a, b, c) in mapping.items()
                }

    return best_mapping


def _node_coordinate_from_metadata(
    graph: _GraphData,
    node_id: int,
    *,
    dimensions: int,
) -> tuple[str, tuple[int, ...]] | None:
    node = graph.nodes[int(node_id)]
    pattern = re.compile(
        r"^obs_(?P<prefix>.+)"
        + "".join(rf"_(?P<c{index}>-?\d+)" for index in range(dimensions))
        + r"$"
    )
    for axis_name in node.axes_names:
        match = pattern.match(axis_name)
        if match is None:
            continue
        return (
            match.group("prefix"),
            tuple(int(match.group(f"c{index}")) for index in range(dimensions)),
        )

    name_parts = node.name.rsplit("_", dimensions)
    if len(name_parts) == dimensions + 1 and all(
        re.fullmatch(r"-?\d+", part) for part in name_parts[1:]
    ):
        return name_parts[0], tuple(int(part) for part in name_parts[1:])

    return None


def _coordinate_map_from_node_metadata(
    nx_graph: nx.Graph,
    graph: _GraphData,
    *,
    dimensions: int,
) -> dict[int, tuple[int, ...]] | None:
    coords_by_node: dict[int, tuple[int, ...]] = {}
    prefix: str | None = None
    seen_coords: set[tuple[int, ...]] = set()
    for node_id in nx_graph.nodes:
        parsed = _node_coordinate_from_metadata(graph, int(node_id), dimensions=dimensions)
        if parsed is None:
            return None
        node_prefix, coords = parsed
        if prefix is None:
            prefix = node_prefix
        elif node_prefix != prefix:
            return None
        if coords in seen_coords:
            return None
        seen_coords.add(coords)
        coords_by_node[int(node_id)] = coords
    return coords_by_node


def _sorted_node_pair(left_id: int, right_id: int) -> tuple[int, int]:
    if left_id < right_id:
        return (left_id, right_id)
    return (right_id, left_id)


def _node_edge_set(nx_graph: nx.Graph) -> set[tuple[int, int]]:
    return {
        _sorted_node_pair(int(left_id), int(right_id)) for left_id, right_id in nx_graph.edges()
    }


def _expected_grid_edges_from_coords(
    coords_by_node: dict[int, tuple[int, ...]],
    *,
    dimensions: int,
) -> set[tuple[int, int]]:
    node_by_coord = {coords: node_id for node_id, coords in coords_by_node.items()}
    expected_edges: set[tuple[int, int]] = set()
    for node_id, coords in coords_by_node.items():
        for axis_index in range(dimensions):
            neighbor = list(coords)
            neighbor[axis_index] += 1
            neighbor_id = node_by_coord.get(tuple(neighbor))
            if neighbor_id is None:
                continue
            expected_edges.add(_sorted_node_pair(node_id, neighbor_id))
    return expected_edges


def _coordinate_axis_lengths(coords_by_node: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    if not coords_by_node:
        return ()
    dimensions = len(next(iter(coords_by_node.values())))
    return tuple(
        len({coords[axis_index] for coords in coords_by_node.values()})
        for axis_index in range(dimensions)
    )


def _detect_coordinate_grid_2d(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> dict[int, tuple[int, int]] | None:
    if nx_graph.number_of_nodes() <= 1 or not nx.is_connected(nx_graph):
        return None
    coords_by_node = _coordinate_map_from_node_metadata(nx_graph, graph, dimensions=2)
    if coords_by_node is None:
        return None
    if any(axis_length <= 1 for axis_length in _coordinate_axis_lengths(coords_by_node)):
        return None
    if _node_edge_set(nx_graph) != _expected_grid_edges_from_coords(
        coords_by_node,
        dimensions=2,
    ):
        return None
    return {node_id: (int(coords[1]), int(coords[0])) for node_id, coords in coords_by_node.items()}


def _detect_coordinate_grid_3d(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> dict[int, tuple[int, int, int]] | None:
    if nx_graph.number_of_nodes() <= 1 or not nx.is_connected(nx_graph):
        return None
    coords_by_node = _coordinate_map_from_node_metadata(nx_graph, graph, dimensions=3)
    if coords_by_node is None:
        return None
    if any(axis_length <= 1 for axis_length in _coordinate_axis_lengths(coords_by_node)):
        return None
    if _node_edge_set(nx_graph) != _expected_grid_edges_from_coords(
        coords_by_node,
        dimensions=3,
    ):
        return None
    return {
        node_id: (int(coords[0]), int(coords[1]), int(coords[2]))
        for node_id, coords in coords_by_node.items()
    }


def _detect_tube(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> dict[int, tuple[int, int]] | None:
    coordinate_mapping = _detect_coordinate_tube(nx_graph, graph)
    if coordinate_mapping is not None:
        return coordinate_mapping
    return _detect_topological_tube(nx_graph)


def _detect_coordinate_tube(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> dict[int, tuple[int, int]] | None:
    if nx_graph.number_of_nodes() <= 5 or not nx.is_connected(nx_graph):
        return None
    coords_by_node = _coordinate_map_from_node_metadata(nx_graph, graph, dimensions=2)
    if coords_by_node is None:
        return None
    node_by_coord = {coords: node_id for node_id, coords in coords_by_node.items()}
    actual_edges = _node_edge_set(nx_graph)

    for periodic_axis in (0, 1):
        axial_axis = 1 - periodic_axis
        periodic_values = sorted({coords[periodic_axis] for coords in coords_by_node.values()})
        axial_values = sorted({coords[axial_axis] for coords in coords_by_node.values()})
        if len(periodic_values) < 3 or len(axial_values) < 2:
            continue
        if len(node_by_coord) != len(periodic_values) * len(axial_values):
            continue

        periodic_index = {value: index for index, value in enumerate(periodic_values)}
        axial_index = {value: index for index, value in enumerate(axial_values)}
        expected_edges: set[tuple[int, int]] = set()
        for periodic_value in periodic_values:
            for axial_value in axial_values:
                coord = [0, 0]
                coord[periodic_axis] = periodic_value
                coord[axial_axis] = axial_value
                node_id = node_by_coord.get(tuple(coord))
                if node_id is None:
                    expected_edges.clear()
                    break

                next_periodic_value = periodic_values[
                    (periodic_index[periodic_value] + 1) % len(periodic_values)
                ]
                periodic_coord = list(coord)
                periodic_coord[periodic_axis] = next_periodic_value
                expected_edges.add(_sorted_node_pair(node_id, node_by_coord[tuple(periodic_coord)]))

                next_axial_idx = axial_index[axial_value] + 1
                if next_axial_idx < len(axial_values):
                    axial_coord = list(coord)
                    axial_coord[axial_axis] = axial_values[next_axial_idx]
                    expected_edges.add(
                        _sorted_node_pair(node_id, node_by_coord[tuple(axial_coord)])
                    )
            else:
                continue
            break

        if expected_edges and actual_edges == expected_edges:
            return {
                node_id: (
                    periodic_index[coords[periodic_axis]],
                    axial_index[coords[axial_axis]],
                )
                for node_id, coords in coords_by_node.items()
            }
    return None


def _detect_topological_tube(nx_graph: nx.Graph) -> dict[int, tuple[int, int]] | None:
    n_nodes = nx_graph.number_of_nodes()
    if n_nodes <= 5 or not nx.is_connected(nx_graph):
        return None
    n_edges = nx_graph.number_of_edges()
    for periodic in range(3, n_nodes + 1):
        if n_nodes % periodic != 0:
            continue
        length = n_nodes // periodic
        if length < 2 or n_edges != periodic * length + periodic * (length - 1):
            continue
        template = nx.cartesian_product(nx.cycle_graph(periodic), nx.path_graph(length))
        mapping = nx.vf2pp_isomorphism(nx_graph, template)
        if mapping is None:
            continue
        return {
            int(node_id): (int(coords[0]), int(coords[1])) for node_id, coords in mapping.items()
        }
    return None


def _detect_circular_order(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> tuple[int, ...] | None:
    named_order = _detect_named_ring_order(nx_graph, graph)
    if named_order is not None:
        return named_order
    return _detect_simple_cycle_order(nx_graph)


def _detect_named_ring_order(
    nx_graph: nx.Graph,
    graph: _GraphData,
) -> tuple[int, ...] | None:
    if nx_graph.number_of_nodes() < 3 or not nx.is_connected(nx_graph):
        return None
    entries: list[tuple[str, int, int]] = []
    for node_id in nx_graph.nodes:
        node_name = graph.nodes[int(node_id)].name
        match = re.fullmatch(r"(?P<prefix>[^_\d]+)(?P<index>\d+)", node_name)
        if match is None:
            return None
        entries.append((match.group("prefix"), int(match.group("index")), int(node_id)))

    prefixes = {prefix for prefix, _, _ in entries}
    if len(prefixes) != 1:
        return None
    entries.sort(key=lambda item: item[1])
    indices = [index for _, index, _ in entries]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        return None
    order = tuple(node_id for _, _, node_id in entries)
    if all(
        nx_graph.has_edge(order[index], order[(index + 1) % len(order)])
        for index in range(len(order))
    ):
        return order
    return None


def _detect_simple_cycle_order(nx_graph: nx.Graph) -> tuple[int, ...] | None:
    if nx_graph.number_of_nodes() < 3 or not nx.is_connected(nx_graph):
        return None
    if nx_graph.number_of_edges() != nx_graph.number_of_nodes():
        return None
    if any(degree != 2 for _, degree in nx_graph.degree()):
        return None

    start = min(int(node_id) for node_id in nx_graph.nodes)
    order = [start]
    previous_id: int | None = None
    current_id = start
    while len(order) < nx_graph.number_of_nodes():
        candidates = sorted(
            int(neighbor_id)
            for neighbor_id in nx_graph.neighbors(current_id)
            if int(neighbor_id) != previous_id
        )
        next_candidates = [node_id for node_id in candidates if node_id != start]
        if not next_candidates:
            return None
        next_id = next_candidates[0]
        if next_id in order:
            return None
        order.append(next_id)
        previous_id, current_id = current_id, next_id

    if not nx_graph.has_edge(order[-1], start):
        return None
    return tuple(order)


def _layout_grid3d_projection_2d(
    grid3d_mapping: dict[int, tuple[int, int, int]],
) -> dict[int, np.ndarray]:
    return {
        node_id: np.array(
            [
                float(i) + _GRID3D_PROJECTION_X * float(k),
                float(j) + _GRID3D_PROJECTION_Y * float(k),
            ],
            dtype=float,
        )
        for node_id, (i, j, k) in grid3d_mapping.items()
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
