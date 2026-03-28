from __future__ import annotations

import math

import numpy as np

from tensor_network_viz._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from tensor_network_viz._core.layout import (
    _compute_axis_directions,
    _compute_layout,
    _dangling_stub_segment_2d,
    _planar_contraction_bond_segments_2d,
    _segments_cross_2d,
)
from tensor_network_viz._core.renderer import _resolve_draw_scale


def _build_chain_graph(
    *,
    length: int,
    dangling_axis_counts: dict[int, int] | None = None,
) -> _GraphData:
    dangling_axis_counts = dangling_axis_counts or {}
    nodes = {}
    edges = []

    for node_id in range(length):
        axes_names = []
        if node_id > 0:
            axes_names.append("left")
        if node_id < length - 1:
            axes_names.append("right")
        for dangling_index in range(dangling_axis_counts.get(node_id, 0)):
            axes_names.append(f"d{node_id}_{dangling_index}")
        nodes[node_id] = _make_node(f"N{node_id}", tuple(axes_names))

    for node_id in range(length - 1):
        left_axis = len(nodes[node_id].axes_names) - dangling_axis_counts.get(node_id, 0) - 1
        right_axis = 0
        edges.append(
            _make_contraction_edge(
                _EdgeEndpoint(node_id, left_axis, "right"),
                _EdgeEndpoint(node_id + 1, right_axis, "left"),
                name=f"e{node_id}",
                label=None,
            )
        )

    for node_id, count in dangling_axis_counts.items():
        start_axis = len(nodes[node_id].axes_names) - count
        for offset in range(count):
            axis_index = start_axis + offset
            axis_name = nodes[node_id].axes_names[axis_index]
            edges.append(
                _make_dangling_edge(
                    _EdgeEndpoint(node_id, axis_index, axis_name),
                    name=axis_name,
                    label=None,
                )
            )

    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_grid_graph(rows: int, cols: int) -> _GraphData:
    nodes = {}
    edge_specs: list[tuple[int, int, str, str]] = []

    def node_index(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            axes_names = []
            if col > 0:
                axes_names.append("left")
            if col < cols - 1:
                axes_names.append("right")
            if row > 0:
                axes_names.append("down")
            if row < rows - 1:
                axes_names.append("up")
            nodes[node_index(row, col)] = _make_node(f"N{row}_{col}", tuple(axes_names))

    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }

    for row in range(rows):
        for col in range(cols):
            node_id = node_index(row, col)
            if col < cols - 1:
                right_id = node_index(row, col + 1)
                edge_specs.append((node_id, right_id, "right", "left"))
            if row < rows - 1:
                up_id = node_index(row + 1, col)
                edge_specs.append((node_id, up_id, "up", "down"))

    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][left_name], left_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][right_name], right_name),
            name=f"{left_id}_{right_id}",
            label=None,
        )
        for left_id, right_id, left_name, right_name in edge_specs
    ]
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_planar_cycle_graph() -> _GraphData:
    nodes = {
        0: _make_node("A", ("right", "up")),
        1: _make_node("B", ("left", "up")),
        2: _make_node("C", ("down", "left")),
        3: _make_node("D", ("right", "down")),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "right"),
            _EdgeEndpoint(1, 0, "left"),
            name="ab",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "up"),
            _EdgeEndpoint(2, 0, "down"),
            name="bc",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 1, "left"),
            _EdgeEndpoint(3, 0, "right"),
            name="cd",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 1, "down"),
            _EdgeEndpoint(0, 1, "up"),
            name="da",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_star_with_free_axes() -> _GraphData:
    nodes = {
        0: _make_node("Center", ("left", "right", "up", "down", "f0", "f1", "f2", "f3")),
        1: _make_node("L", ("bond",)),
        2: _make_node("R", ("bond",)),
        3: _make_node("U", ("bond",)),
        4: _make_node("D", ("bond",)),
    }
    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "left"),
            _EdgeEndpoint(1, 0, "bond"),
            name="l",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "right"),
            _EdgeEndpoint(2, 0, "bond"),
            name="r",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 2, "up"),
            _EdgeEndpoint(3, 0, "bond"),
            name="u",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 3, "down"),
            _EdgeEndpoint(4, 0, "bond"),
            name="d",
            label=None,
        ),
    ]
    for axis_index in range(4, 8):
        axis_name = nodes[0].axes_names[axis_index]
        edges.append(
            _make_dangling_edge(
                _EdgeEndpoint(0, axis_index, axis_name),
                name=axis_name,
                label=None,
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_hypergraph_with_virtual_hub() -> _GraphData:
    nodes = {
        0: _make_node("A", ("h",)),
        1: _make_node("B", ("h",)),
        2: _make_node("C", ("h",)),
        -1: _make_node("", ("h0", "h1", "h2"), label="h", is_virtual=True),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "h"),
            _EdgeEndpoint(-1, 0, "h0"),
            name="h",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 0, "h"),
            _EdgeEndpoint(-1, 1, "h1"),
            name="h",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 0, "h"),
            _EdgeEndpoint(-1, 2, "h2"),
            name="h",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_einsum_like_mps_graph() -> _GraphData:
    nodes = {
        0: _make_node("A0", ("p", "a")),
        1: _make_node("x0", ("p",)),
        2: _make_node("A1", ("a", "p", "b")),
        3: _make_node("x1", ("p",)),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "p"),
            _EdgeEndpoint(1, 0, "p"),
            name="p0",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "a"),
            _EdgeEndpoint(2, 0, "a"),
            name="a",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 1, "p"),
            _EdgeEndpoint(3, 0, "p"),
            name="p1",
            label=None,
        ),
        _make_dangling_edge(
            _EdgeEndpoint(2, 2, "b"),
            name="b",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_complete_graph_5() -> _GraphData:
    """K5 complete graph: non-planar, classified as 'generic', uses force-directed fallback."""
    nodes = {}
    for i in range(5):
        axes = [f"e{i}_{j}" for j in range(5) if j != i]
        nodes[i] = _make_node(f"N{i}", tuple(axes))

    axis_by_pair: dict[tuple[int, int], int] = {}
    for i in range(5):
        idx = 0
        for j in range(5):
            if j != i:
                axis_by_pair[(i, j)] = idx
                idx += 1

    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            name = f"e{i}_{j}"
            edges.append(
                _make_contraction_edge(
                    _EdgeEndpoint(i, axis_by_pair[(i, j)], name),
                    _EdgeEndpoint(j, axis_by_pair[(j, i)], name),
                    name=name,
                    label=None,
                )
            )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_grid_with_leaf_nodes() -> _GraphData:
    nodes = {
        0: _make_node("P00", ("right", "up", "s")),
        1: _make_node("P01", ("left", "up", "t")),
        2: _make_node("P10", ("right", "down", "v")),
        3: _make_node("P11", ("left", "down", "w")),
        4: _make_node("x00", ("s",)),
        5: _make_node("x01", ("t",)),
        6: _make_node("x10", ("v",)),
        7: _make_node("x11", ("w",)),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "right"),
            _EdgeEndpoint(1, 0, "left"),
            name="h0",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 0, "right"),
            _EdgeEndpoint(3, 0, "left"),
            name="h1",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "up"),
            _EdgeEndpoint(2, 1, "down"),
            name="v0",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "up"),
            _EdgeEndpoint(3, 1, "down"),
            name="v1",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 2, "s"),
            _EdgeEndpoint(4, 0, "s"),
            name="s",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 2, "t"),
            _EdgeEndpoint(5, 0, "t"),
            name="t",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 2, "v"),
            _EdgeEndpoint(6, 0, "v"),
            name="v",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 2, "w"),
            _EdgeEndpoint(7, 0, "w"),
            name="w",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    return axis / np.linalg.norm(axis)


def test_compute_layout_line_chain_2d_is_colinear_and_evenly_spaced() -> None:
    graph = _build_chain_graph(length=4, dangling_axis_counts={0: 1, 3: 1})

    positions = _compute_layout(graph, dimensions=2, seed=0)
    coords = np.stack([positions[node_id] for node_id in range(4)])

    assert np.allclose(coords[:, 1], 0.0, atol=1e-6)
    assert np.all(np.diff(coords[:, 0]) > 0.0)
    assert np.allclose(np.diff(coords[:, 0]), np.diff(coords[:, 0])[0], atol=1e-6)


def test_compute_layout_line_chain_3d_keeps_backbone_straight_and_planar() -> None:
    graph = _build_chain_graph(length=4, dangling_axis_counts={0: 1, 3: 1})

    positions = _compute_layout(graph, dimensions=3, seed=0)
    coords = np.stack([positions[node_id] for node_id in range(4)])

    assert np.allclose(coords[:, 1], 0.0, atol=1e-6)
    assert np.allclose(coords[:, 2], 0.0, atol=1e-6)
    assert np.all(np.diff(coords[:, 0]) > 0.0)
    assert np.allclose(np.diff(coords[:, 0]), np.diff(coords[:, 0])[0], atol=1e-6)


def test_compute_layout_einsum_like_mps_2d_reattaches_degree_one_nodes_as_Ls() -> None:
    graph = _build_einsum_like_mps_graph()

    positions = _compute_layout(graph, dimensions=2, seed=0)

    assert np.allclose(positions[0][1], positions[2][1], atol=1e-6)
    assert positions[0][0] < positions[2][0]
    assert np.allclose(positions[1][0], positions[0][0], atol=1e-6)
    assert np.allclose(positions[3][0], positions[2][0], atol=1e-6)
    assert abs(positions[1][1] - positions[0][1]) > 0.2
    assert abs(positions[3][1] - positions[2][1]) > 0.2


def test_compute_layout_einsum_like_mps_3d_reattaches_degree_one_nodes_orthogonally() -> None:
    graph = _build_einsum_like_mps_graph()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    left_vector = positions[1] - positions[0]
    right_vector = positions[3] - positions[2]

    assert np.allclose(positions[0][1:], positions[2][1:], atol=1e-6)
    assert positions[0][0] < positions[2][0]
    assert abs(left_vector[0]) < 1e-6
    assert abs(right_vector[0]) < 1e-6
    assert np.linalg.norm(left_vector[1:]) > 0.2
    assert np.linalg.norm(right_vector[1:]) > 0.2


def test_compute_axis_directions_einsum_like_mps_3d_avoids_reusing_leaf_direction() -> None:
    graph = _build_einsum_like_mps_graph()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)

    assert np.allclose(directions[(2, 1)], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    assert np.allclose(directions[(2, 2)], np.array([0.0, 0.0, -1.0]), atol=1e-6)


def test_compute_layout_grid_with_leaf_nodes_2d_keeps_leaf_positions_off_core_nodes() -> None:
    graph = _build_grid_with_leaf_nodes()

    positions = _compute_layout(graph, dimensions=2, seed=0)
    core_positions = [positions[node_id] for node_id in range(4)]

    for leaf_id in range(4, 8):
        assert all(
            not np.allclose(positions[leaf_id], core_position, atol=1e-6)
            for core_position in core_positions
        )


def test_compute_layout_grid_2d_places_nodes_on_regular_lattice() -> None:
    graph = _build_grid_graph(2, 2)

    positions = _compute_layout(graph, dimensions=2, seed=0)
    coords = np.stack([positions[node_id] for node_id in sorted(graph.nodes)])
    x_values = np.unique(np.round(coords[:, 0], decimals=6))
    y_values = np.unique(np.round(coords[:, 1], decimals=6))

    assert len(x_values) == 2
    assert len(y_values) == 2
    assert math.isclose(
        abs(x_values[1] - x_values[0]),
        abs(y_values[1] - y_values[0]),
        rel_tol=1e-6,
    )


def test_compute_layout_planar_cycle_3d_stays_on_single_plane() -> None:
    graph = _build_planar_cycle_graph()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    coords = np.stack([positions[node_id] for node_id in sorted(graph.nodes)])

    assert np.allclose(coords[:, 2], 0.0, atol=1e-6)


def test_compute_axis_directions_chain_3d_assigns_orthogonal_open_end() -> None:
    graph = _build_chain_graph(length=3, dangling_axis_counts={0: 1, 2: 1})

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)

    assert np.allclose(directions[(0, 1)], np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_compute_axis_directions_competing_free_axes_prefer_unused_orthogonal_directions() -> None:
    graph = _build_star_with_free_axes()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)

    assert np.allclose(directions[(0, 4)], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(directions[(0, 5)], np.array([0.0, -1.0, 0.0]), atol=1e-6)
    for axis_index in range(6, 8):
        assert np.isclose(np.linalg.norm(directions[(0, axis_index)]), 1.0, atol=1e-6)


def test_compute_layout_virtual_hub_uses_incident_barycenter() -> None:
    graph = _build_hypergraph_with_virtual_hub()

    positions = _compute_layout(graph, dimensions=2, seed=0)
    visible_coords = np.stack([positions[node_id] for node_id in (0, 1, 2)])

    assert np.allclose(positions[-1], visible_coords.mean(axis=0), atol=1e-6)


def test_compute_layout_hypergraph_3d_keeps_visible_nodes_planar() -> None:
    graph = _build_hypergraph_with_virtual_hub()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    visible_coords = np.stack([positions[node_id] for node_id in (0, 1, 2)])
    axis = _principal_axis(visible_coords)
    residuals = visible_coords - visible_coords.mean(axis=0, keepdims=True)
    off_axis = residuals - np.outer(residuals @ axis, axis)

    assert np.max(np.linalg.norm(off_axis, axis=1)) > 0.2
    assert np.allclose(visible_coords[:, 2], 0.0, atol=1e-6)


def test_compute_layout_generic_graph_uses_force_fallback_produces_valid_positions() -> None:
    """Generic (non-planar) graphs fall back to force-directed layout; output must be valid."""
    graph = _build_complete_graph_5()

    positions = _compute_layout(graph, dimensions=2, seed=0, iterations=50)

    assert len(positions) == 5
    coords = np.stack([positions[i] for i in range(5)])
    assert not np.any(np.isnan(coords))
    assert not np.any(np.isinf(coords))
    assert np.all(np.isfinite(coords))


def test_weird_topology_2d_phys_stubs_do_not_cross() -> None:
    """Regression: example 'weird' graph must not draw crossing dangling (phys) segments in 2D."""
    nodes = {
        0: _make_node("center", ("north", "east", "south", "west", "phys")),
        1: _make_node("north", ("center", "east", "phys")),
        2: _make_node("east", ("center", "north", "south", "phys")),
        3: _make_node("south", ("center", "east", "west_a", "west_b", "phys")),
        4: _make_node("west", ("center", "south_a", "south_b", "phys")),
    }
    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "north"), _EdgeEndpoint(1, 0, "center"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "east"), _EdgeEndpoint(2, 0, "center"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 2, "south"), _EdgeEndpoint(3, 0, "center"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 3, "west"), _EdgeEndpoint(4, 0, "center"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "east"), _EdgeEndpoint(2, 1, "north"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 2, "south"), _EdgeEndpoint(3, 1, "east"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 2, "west_a"), _EdgeEndpoint(4, 1, "south_a"), name=None
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 3, "west_b"), _EdgeEndpoint(4, 2, "south_b"), name=None
        ),
        _make_dangling_edge(_EdgeEndpoint(0, 4, "phys"), name="phys"),
        _make_dangling_edge(_EdgeEndpoint(1, 2, "phys"), name="phys"),
        _make_dangling_edge(_EdgeEndpoint(2, 3, "phys"), name="phys"),
        _make_dangling_edge(_EdgeEndpoint(3, 4, "phys"), name="phys"),
        _make_dangling_edge(_EdgeEndpoint(4, 3, "phys"), name="phys"),
    ]
    graph = _GraphData(nodes=nodes, edges=tuple(edges))
    positions = _compute_layout(graph, dimensions=2, seed=0)
    ds = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=ds)

    stubs: list[tuple[np.ndarray, np.ndarray]] = []
    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        ep = edge.endpoints[0]
        direction = directions[(ep.node_id, ep.axis_index)]
        origin = positions[ep.node_id]
        stubs.append(_dangling_stub_segment_2d(origin, direction, draw_scale=ds))

    for i in range(len(stubs)):
        for j in range(i + 1, len(stubs)):
            p0, p1 = stubs[i]
            q0, q1 = stubs[j]
            assert not _segments_cross_2d(p0, p1, q0, q1), f"stubs {i} and {j} cross"

    bond_segments = _planar_contraction_bond_segments_2d(graph, positions, scale=ds)
    stub_idx = 0
    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        ep = edge.endpoints[0]
        p0, p1 = stubs[stub_idx]
        stub_idx += 1
        for a, b, ba, bb in bond_segments:
            if a == ep.node_id or b == ep.node_id:
                continue
            assert not _segments_cross_2d(p0, p1, ba, bb), (
                f"stub from node {ep.node_id} crosses bond {a}-{b}"
            )
