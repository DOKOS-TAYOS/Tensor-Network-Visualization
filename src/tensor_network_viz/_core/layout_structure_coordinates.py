"""Coordinate-based structure detection helpers for deterministic layouts."""

from __future__ import annotations

import re

import networkx as nx

from .graph import _GraphData


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
