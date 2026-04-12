from __future__ import annotations

import math

import numpy as np
import pytest

from tensor_network_viz import PlotConfig
from tensor_network_viz._core._draw_common import _graph_edge_degree
from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
    _NodeData,
)
from tensor_network_viz._core.layout import (
    _analyze_layout_components_cached,
    _compute_axis_directions,
    _compute_component_layout_2d,
    _compute_layout,
    _dangling_stub_segment_2d,
    _is_dangling_leg_axis,
    _planar_contraction_bond_segments_2d,
    _segment_point_min_distance_sq_2d,
    _segments_cross_2d,
)
from tensor_network_viz._core.layout.direction_common import _behavior_direction_order_2d
from tensor_network_viz._core.layout.free_directions_2d import (
    _direction_angle_conflicts_2d,
    _pick_candidate_direction_2d,
)
from tensor_network_viz._core.layout.free_directions_3d import (
    _direction_angle_conflicts_3d,
    _direction_conflicts_3d,
    _pick_candidate_direction_3d,
)
from tensor_network_viz._core.layout.generic_coarsening import (
    _CoarseningGraph,
    _compress_degree_two_paths,
    _compute_coarsened_layout_2d,
    _distinct_neighbor_counts,
    _ordered_initial_circle_node_ids,
    _peel_degree_one_trees,
)
from tensor_network_viz._core.layout.parameters import _LAYER_SPACING
from tensor_network_viz._core.layout.positions import _promote_3d_layers
from tensor_network_viz._core.renderer import (
    _SHORTEST_EDGE_RADIUS_FRACTION,
    _resolve_draw_scale,
)


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


def _build_disconnected_singletons_graph(count: int) -> _GraphData:
    nodes = {node_id: _make_node(f"S{node_id}", ("phys",)) for node_id in range(count)}
    edges = tuple(
        _make_dangling_edge(
            _EdgeEndpoint(node_id, 0, "phys"),
            name=f"phys_{node_id}",
            label=None,
        )
        for node_id in range(count)
    )
    return _GraphData(nodes=nodes, edges=edges)


def test_graph_edge_degree_counts_all_edge_kinds() -> None:
    graph = _build_chain_graph(length=3)
    assert _graph_edge_degree(graph, 0) == 1
    assert _graph_edge_degree(graph, 1) == 2
    assert _graph_edge_degree(graph, 2) == 1

    isolated_dangling = _build_chain_graph(length=1, dangling_axis_counts={0: 1})
    assert _graph_edge_degree(isolated_dangling, 0) == 1

    leaf_with_phys = _build_chain_graph(length=2, dangling_axis_counts={1: 1})
    assert _graph_edge_degree(leaf_with_phys, 1) == 2


def test_dangling_leg_axis_matches_open_edges() -> None:
    g = _GraphData(
        nodes={
            0: _make_node("A", ("left", "p")),
            1: _make_node("B", ("right",)),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "left"),
                _EdgeEndpoint(1, 0, "right"),
                name=None,
            ),
            _make_dangling_edge(_EdgeEndpoint(0, 1, "p"), name="p"),
        ),
    )
    assert _is_dangling_leg_axis(g, 0, 1) is True
    assert _is_dangling_leg_axis(g, 0, 0) is False
    assert _is_dangling_leg_axis(g, 1, 0) is False


def test_behavior_direction_order_2d_rotates_north_template() -> None:
    north = _behavior_direction_order_2d("north")
    south = _behavior_direction_order_2d("south")
    east = _behavior_direction_order_2d("east")
    west = _behavior_direction_order_2d("west")

    expected_north = (
        np.array([0.0, 1.0], dtype=float),
        np.array([0.0, -1.0], dtype=float),
        np.array([1.0, 0.0], dtype=float),
        np.array([-1.0, 0.0], dtype=float),
        np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0),
        np.array([-1.0, 1.0], dtype=float) / np.sqrt(2.0),
        np.array([-1.0, -1.0], dtype=float) / np.sqrt(2.0),
        np.array([1.0, -1.0], dtype=float) / np.sqrt(2.0),
    )
    expected_north_semidiagonals = (
        np.array([math.cos(math.radians(67.5)), math.sin(math.radians(67.5))], dtype=float),
        np.array([math.cos(math.radians(112.5)), math.sin(math.radians(112.5))], dtype=float),
        np.array([math.cos(math.radians(157.5)), math.sin(math.radians(157.5))], dtype=float),
        np.array([math.cos(math.radians(202.5)), math.sin(math.radians(202.5))], dtype=float),
        np.array([math.cos(math.radians(247.5)), math.sin(math.radians(247.5))], dtype=float),
        np.array([math.cos(math.radians(292.5)), math.sin(math.radians(292.5))], dtype=float),
        np.array([math.cos(math.radians(337.5)), math.sin(math.radians(337.5))], dtype=float),
        np.array([math.cos(math.radians(22.5)), math.sin(math.radians(22.5))], dtype=float),
    )

    for idx, expected in enumerate(expected_north):
        assert np.allclose(north[idx], expected, atol=1e-9)
    for idx, expected in enumerate(expected_north_semidiagonals, start=8):
        assert np.allclose(north[idx], expected, atol=1e-9)

    assert np.allclose(south[0], np.array([0.0, -1.0], dtype=float), atol=1e-9)
    assert np.allclose(east[0], np.array([1.0, 0.0], dtype=float), atol=1e-9)
    assert np.allclose(west[0], np.array([-1.0, 0.0], dtype=float), atol=1e-9)


def test_direction_angle_conflicts_2d_respects_five_degree_threshold() -> None:
    assert _direction_angle_conflicts_2d(
        np.array([0.0, 1.0], dtype=float),
        np.array([math.sin(math.radians(4.0)), math.cos(math.radians(4.0))], dtype=float),
    )
    assert _direction_angle_conflicts_2d(
        np.array([0.0, 1.0], dtype=float),
        np.array([math.sin(math.radians(176.0)), math.cos(math.radians(176.0))], dtype=float),
    )
    assert not _direction_angle_conflicts_2d(
        np.array([0.0, 1.0], dtype=float),
        np.array([math.sin(math.radians(10.0)), math.cos(math.radians(10.0))], dtype=float),
    )


def test_pick_candidate_direction_2d_returns_last_tried_when_all_fail() -> None:
    candidates = (
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, -1.0], dtype=float),
    )
    picked = _pick_candidate_direction_2d(
        candidates=candidates,
        is_valid=lambda _candidate: False,
    )

    assert np.allclose(picked, candidates[-1], atol=1e-9)


def test_direction_angle_conflicts_3d_uses_ten_degree_threshold_without_opposites() -> None:
    assert _direction_angle_conflicts_3d(
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([math.sin(math.radians(9.0)), 0.0, math.cos(math.radians(9.0))], dtype=float),
    )
    assert not _direction_angle_conflicts_3d(
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([math.sin(math.radians(10.5)), 0.0, math.cos(math.radians(10.5))], dtype=float),
    )
    assert not _direction_angle_conflicts_3d(
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 0.0, -1.0], dtype=float),
    )


def test_direction_conflicts_3d_ignores_nonlocal_bond_geometry() -> None:
    conflicts = _direction_conflicts_3d(
        node_id=0,
        origin=np.array([0.0, 0.0, 0.0], dtype=float),
        direction=np.array([1.0, 0.0, 0.0], dtype=float),
        assigned_segments=[],
        bond_segments=(
            (
                np.array([0.5, -1.0, 0.0], dtype=float),
                np.array([0.5, 1.0, 0.0], dtype=float),
            ),
        ),
        positions={
            0: np.array([0.0, 0.0, 0.0], dtype=float),
            1: np.array([0.5, -1.0, 0.0], dtype=float),
            2: np.array([0.5, 1.0, 0.0], dtype=float),
        },
        draw_scale=1.0,
        strict_physical_node_clearance=True,
        conflict_data=object(),
    )

    assert not conflicts


def test_pick_candidate_direction_3d_returns_last_tried_when_all_fail() -> None:
    candidates = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    picked = _pick_candidate_direction_3d(
        candidates=candidates,
        is_valid=lambda _candidate: False,
    )

    assert np.allclose(picked, candidates[-1], atol=1e-9)


def test_compute_axis_directions_chain_2d_allows_opposite_free_axes_on_same_node() -> None:
    graph = _build_chain_graph(length=3, dangling_axis_counts={1: 2})
    positions = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0], dtype=float),
        2: np.array([2.0, 0.0], dtype=float),
    }

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)
    free_axis_indices = [
        axis_index
        for axis_index, axis_name in enumerate(graph.nodes[1].axes_names)
        if axis_name.startswith("d1_")
    ]
    picked = [
        np.asarray(directions[(1, axis_index)], dtype=float) for axis_index in free_axis_indices
    ]

    assert np.allclose(picked[0], np.array([0.0, 1.0], dtype=float), atol=1e-6)
    assert np.allclose(picked[1], np.array([0.0, -1.0], dtype=float), atol=1e-6)


def test_compute_axis_directions_2d_named_dangling_axes_fall_back_when_repeated() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("right", "right")),
        },
        edges=(
            _make_dangling_edge(_EdgeEndpoint(0, 0, "right"), name="right"),
            _make_dangling_edge(_EdgeEndpoint(0, 1, "right"), name="right"),
        ),
    )
    positions = {0: np.array([0.0, 0.0], dtype=float)}

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    assert np.allclose(directions[(0, 0)], np.array([1.0, 0.0], dtype=float), atol=1e-6)
    assert not np.allclose(directions[(0, 1)], np.array([1.0, 0.0], dtype=float), atol=1e-6)
    assert not _direction_angle_conflicts_2d(
        directions[(0, 1)],
        directions[(0, 0)],
        treat_opposite_as_conflict=False,
    )


def test_compute_axis_directions_2d_only_forces_named_dangling_axes() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("right", "phys")),
            1: _make_node("B", ("left",)),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "right"),
                _EdgeEndpoint(1, 0, "left"),
                name="bond",
            ),
            _make_dangling_edge(_EdgeEndpoint(0, 1, "phys"), name="phys"),
        ),
    )
    positions = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([0.0, 1.0], dtype=float),
    }

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    assert np.allclose(directions[(0, 0)], np.array([0.0, 1.0], dtype=float), atol=1e-6)


def test_compute_axis_directions_chain_2d_skips_random_bucket_when_cardinals_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_2d as free_directions_2d

    graph = _build_chain_graph(length=3, dangling_axis_counts={1: 1})
    positions = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0], dtype=float),
        2: np.array([2.0, 0.0], dtype=float),
    }
    calls: list[int] = []

    def counting_random_bucket(
        *,
        blocked: tuple[np.ndarray, ...],
        count: int = 8,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, ...]:
        calls.append(1)
        return (
            np.array([1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
        )

    monkeypatch.setattr(
        free_directions_2d,
        "_random_direction_bucket_2d",
        counting_random_bucket,
    )

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    assert np.allclose(directions[(1, 2)], np.array([0.0, 1.0], dtype=float), atol=1e-6)
    assert calls == []


def test_compute_axis_directions_large_chain_2d_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_2d as free_directions_2d

    def fail_build_context(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("_build_context should not run for pure 2D chains")

    def fail_random_bucket(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        del args, kwargs
        raise AssertionError("_random_direction_bucket_2d should not run for pure 2D chains")

    monkeypatch.setattr(free_directions_2d, "_build_context", fail_build_context)
    monkeypatch.setattr(free_directions_2d, "_random_direction_bucket_2d", fail_random_bucket)

    dangling_axis_counts = dict.fromkeys(range(80), 1)
    dangling_axis_counts[40] = 2
    graph = _build_chain_graph(length=80, dangling_axis_counts=dangling_axis_counts)
    positions = {node_id: np.array([float(node_id), 0.0], dtype=float) for node_id in graph.nodes}

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    assert np.allclose(directions[(40, 2)], np.array([0.0, 1.0], dtype=float), atol=1e-6)
    assert np.allclose(directions[(40, 3)], np.array([0.0, -1.0], dtype=float), atol=1e-6)


def test_physical_stub_2d_segment_clears_neighbor_node_disk() -> None:
    """Physical dangling legs use strict clearance: stub polyline must not pierce neighbor disks."""
    nodes = {
        0: _make_node("A", ("right", "phys")),
        1: _make_node("B", ("left",)),
    }
    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "right"),
            _EdgeEndpoint(1, 0, "left"),
            name="bond",
        ),
        _make_dangling_edge(_EdgeEndpoint(0, 1, "phys"), name="phys"),
    ]
    graph = _GraphData(nodes=nodes, edges=tuple(edges))
    positions = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([0.35, 0.0], dtype=float),
    }
    ds = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=ds)
    direction = directions[(0, 1)]
    p0, p1 = _dangling_stub_segment_2d(positions[0], direction, draw_scale=ds)
    r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(ds), 1e-6) * 1.08
    dist = math.sqrt(_segment_point_min_distance_sq_2d(p0, p1, positions[1]))
    assert dist >= r_disk - 1e-9


def test_resolve_draw_scale_from_shortest_contraction_edge() -> None:
    graph = _build_chain_graph(length=3)
    positions = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([0.4, 0.0], dtype=float),
        2: np.array([1.4, 0.0], dtype=float),
    }
    frac = float(_SHORTEST_EDGE_RADIUS_FRACTION)
    nr = float(PlotConfig.DEFAULT_NODE_RADIUS)
    s = _resolve_draw_scale(graph, positions)
    d_min = 0.4
    assert abs(s - frac * d_min / nr) < 1e-9
    assert abs(frac * d_min - nr * s) < 1e-9

    tight = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([0.25, 0.0], dtype=float),
        2: np.array([2.0, 0.0], dtype=float),
    }
    st = _resolve_draw_scale(graph, tight)
    assert abs(st - frac * 0.25 / nr) < 1e-9
    assert abs(frac * 0.25 - nr * st) < 1e-9


def test_resolve_draw_scale_heuristic_when_no_contraction_edges() -> None:
    graph = _build_chain_graph(length=1)
    positions = {0: np.array([0.0, 0.0], dtype=float)}
    s = _resolve_draw_scale(graph, positions)
    lo, hi = 0.35, 1.85
    assert lo <= s <= hi


@pytest.mark.parametrize(
    ("component_count", "expected_columns", "expected_rows"),
    [
        (3, 2, 2),
        (4, 2, 2),
        (5, 3, 2),
    ],
)
def test_compute_layout_disconnected_singletons_2d_uses_compact_grid_packing(
    component_count: int,
    expected_columns: int,
    expected_rows: int,
) -> None:
    graph = _build_disconnected_singletons_graph(component_count)

    positions = _compute_layout(graph, dimensions=2, seed=0, iterations=1)
    coords = np.stack([positions[node_id] for node_id in sorted(graph.nodes)])
    unique_x = {round(float(value), 6) for value in coords[:, 0]}
    unique_y = {round(float(value), 6) for value in coords[:, 1]}

    assert len(unique_x) == expected_columns
    assert len(unique_y) == expected_rows


@pytest.mark.parametrize(
    ("component_count", "expected_columns", "expected_rows"),
    [
        (4, 2, 2),
        (5, 3, 2),
    ],
)
def test_compute_layout_disconnected_singletons_3d_uses_compact_grid_packing(
    component_count: int,
    expected_columns: int,
    expected_rows: int,
) -> None:
    graph = _build_disconnected_singletons_graph(component_count)

    positions = _compute_layout(graph, dimensions=3, seed=0, iterations=1)
    coords = np.stack([positions[node_id] for node_id in sorted(graph.nodes)])
    unique_x = {round(float(value), 6) for value in coords[:, 0]}
    unique_y = {round(float(value), 6) for value in coords[:, 1]}
    unique_z = {round(float(value), 6) for value in coords[:, 2]}

    assert len(unique_x) == expected_columns
    assert len(unique_y) == expected_rows
    assert len(unique_z) == 1


def _build_3d_grid_graph(lx: int, ly: int, lz: int) -> _GraphData:
    """3D nearest-neighbor cubic lattice; same bond topology as cubic PEPS (no physical legs)."""

    def node_index(i: int, j: int, k: int) -> int:
        return i * ly * lz + j * lz + k

    nodes = {}
    edge_specs: list[tuple[int, int, str, str]] = []

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                axes_names: list[str] = []
                if i > 0:
                    axes_names.append("xm")
                if i < lx - 1:
                    axes_names.append("xp")
                if j > 0:
                    axes_names.append("ym")
                if j < ly - 1:
                    axes_names.append("yp")
                if k > 0:
                    axes_names.append("zm")
                if k < lz - 1:
                    axes_names.append("zp")
                nid = node_index(i, j, k)
                nodes[nid] = _make_node(f"P{i}_{j}_{k}", tuple(axes_names))

    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                nid = node_index(i, j, k)
                if i < lx - 1:
                    rid = node_index(i + 1, j, k)
                    edge_specs.append((nid, rid, "xp", "xm"))
                if j < ly - 1:
                    rid = node_index(i, j + 1, k)
                    edge_specs.append((nid, rid, "yp", "ym"))
                if k < lz - 1:
                    rid = node_index(i, j, k + 1)
                    edge_specs.append((nid, rid, "zp", "zm"))

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


def _build_3d_grid_graph_with_dangling_phys(lx: int, ly: int, lz: int) -> _GraphData:
    """3D nearest-neighbor cubic lattice with one dangling phys leg per tensor."""

    def node_index(i: int, j: int, k: int) -> int:
        return i * ly * lz + j * lz + k

    nodes = {}
    edge_specs: list[tuple[int, int, str, str]] = []

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                axes_names: list[str] = ["phys"]
                if i > 0:
                    axes_names.append("xm")
                if i < lx - 1:
                    axes_names.append("xp")
                if j > 0:
                    axes_names.append("ym")
                if j < ly - 1:
                    axes_names.append("yp")
                if k > 0:
                    axes_names.append("zm")
                if k < lz - 1:
                    axes_names.append("zp")
                nid = node_index(i, j, k)
                nodes[nid] = _make_node(f"P{i}_{j}_{k}", tuple(axes_names))

    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }

    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                nid = node_index(i, j, k)
                if i < lx - 1:
                    rid = node_index(i + 1, j, k)
                    edge_specs.append((nid, rid, "xp", "xm"))
                if j < ly - 1:
                    rid = node_index(i, j + 1, k)
                    edge_specs.append((nid, rid, "yp", "ym"))
                if k < lz - 1:
                    rid = node_index(i, j, k + 1)
                    edge_specs.append((nid, rid, "zp", "zm"))

    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][left_name], left_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][right_name], right_name),
            name=f"{left_id}_{right_id}",
            label=None,
        )
        for left_id, right_id, left_name, right_name in edge_specs
    ]
    for node_id, axis_map in axis_lookup.items():
        edges.append(
            _make_dangling_edge(
                _EdgeEndpoint(node_id, axis_map["phys"], "phys"),
                name=f"p{node_id}",
                label=None,
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_binary_tree_graph_with_leaf_phys() -> _GraphData:
    nodes = {
        0: _make_node("R", ("left", "right")),
        1: _make_node("L", ("parent", "left", "right")),
        2: _make_node("M", ("parent", "left", "right")),
        3: _make_node("LL", ("parent", "phys")),
        4: _make_node("LR", ("parent", "phys")),
        5: _make_node("ML", ("parent", "phys")),
        6: _make_node("MR", ("parent", "phys")),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "left"),
            _EdgeEndpoint(1, 0, "parent"),
            name="e01",
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "right"),
            _EdgeEndpoint(2, 0, "parent"),
            name="e02",
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "left"),
            _EdgeEndpoint(3, 0, "parent"),
            name="e13",
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 2, "right"),
            _EdgeEndpoint(4, 0, "parent"),
            name="e14",
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 1, "left"),
            _EdgeEndpoint(5, 0, "parent"),
            name="e25",
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 2, "right"),
            _EdgeEndpoint(6, 0, "parent"),
            name="e26",
        ),
        _make_dangling_edge(_EdgeEndpoint(3, 1, "phys"), name="phys_3", label=None),
        _make_dangling_edge(_EdgeEndpoint(4, 1, "phys"), name="phys_4", label=None),
        _make_dangling_edge(_EdgeEndpoint(5, 1, "phys"), name="phys_5", label=None),
        _make_dangling_edge(_EdgeEndpoint(6, 1, "phys"), name="phys_6", label=None),
    )
    return _GraphData(nodes=nodes, edges=edges)


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


def _build_grid_graph_with_dangling_phys(rows: int, cols: int) -> _GraphData:
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
            axes_names.append("phys")
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
    for node_id in sorted(nodes):
        edges.append(
            _make_dangling_edge(
                _EdgeEndpoint(node_id, axis_lookup[node_id]["phys"], "phys"),
                name=f"p{node_id}",
                label=None,
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_coord_graph_2d(
    active: set[tuple[int, int]],
    *,
    prefix: str = "SG",
) -> _GraphData:
    node_id_by_coord = {coord: index for index, coord in enumerate(sorted(active))}
    axes_by_node = {
        node_id: [f"obs_{prefix}_{row}_{col}"] for (row, col), node_id in node_id_by_coord.items()
    }
    edge_specs: list[tuple[int, int, str]] = []
    for row, col in sorted(active):
        node_id = node_id_by_coord[(row, col)]
        for dr, dc, label in ((1, 0, "down"), (0, 1, "right")):
            neighbor = (row + dr, col + dc)
            if neighbor not in node_id_by_coord:
                continue
            other_id = node_id_by_coord[neighbor]
            edge_name = f"grid_{prefix}_{row}_{col}_{label}"
            axes_by_node[node_id].append(edge_name)
            axes_by_node[other_id].append(edge_name)
            edge_specs.append((node_id, other_id, edge_name))

    nodes = {
        node_id: _make_node(
            f"{prefix}{row}_{col}",
            tuple(axes_by_node[node_id]),
        )
        for (row, col), node_id in node_id_by_coord.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = [
        _make_dangling_edge(
            _EdgeEndpoint(node_id, axis_lookup[node_id][f"obs_{prefix}_{row}_{col}"], "obs"),
            name=f"obs_{prefix}_{row}_{col}",
            label=None,
        )
        for (row, col), node_id in node_id_by_coord.items()
    ]
    edges.extend(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][edge_name], edge_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][edge_name], edge_name),
            name=edge_name,
            label=None,
        )
        for left_id, right_id, edge_name in edge_specs
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_coord_graph_3d(
    active: set[tuple[int, int, int]],
    *,
    prefix: str = "S3",
) -> _GraphData:
    node_id_by_coord = {coord: index for index, coord in enumerate(sorted(active))}
    axes_by_node = {
        node_id: [f"obs_{prefix}_{x}_{y}_{z}"] for (x, y, z), node_id in node_id_by_coord.items()
    }
    edge_specs: list[tuple[int, int, str]] = []
    for x, y, z in sorted(active):
        node_id = node_id_by_coord[(x, y, z)]
        for dx, dy, dz, label in (
            (1, 0, 0, "xp"),
            (0, 1, 0, "yp"),
            (0, 0, 1, "zp"),
        ):
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor not in node_id_by_coord:
                continue
            other_id = node_id_by_coord[neighbor]
            edge_name = f"grid_{prefix}_{x}_{y}_{z}_{label}"
            axes_by_node[node_id].append(edge_name)
            axes_by_node[other_id].append(edge_name)
            edge_specs.append((node_id, other_id, edge_name))

    nodes = {
        node_id: _make_node(
            f"{prefix}{x}_{y}_{z}",
            tuple(axes_by_node[node_id]),
        )
        for (x, y, z), node_id in node_id_by_coord.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = [
        _make_dangling_edge(
            _EdgeEndpoint(node_id, axis_lookup[node_id][f"obs_{prefix}_{x}_{y}_{z}"], "obs"),
            name=f"obs_{prefix}_{x}_{y}_{z}",
            label=None,
        )
        for (x, y, z), node_id in node_id_by_coord.items()
    ]
    edges.extend(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][edge_name], edge_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][edge_name], edge_name),
            name=edge_name,
            label=None,
        )
        for left_id, right_id, edge_name in edge_specs
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_named_ring_graph(
    length: int,
    *,
    chords: tuple[tuple[int, int], ...] = (),
) -> _GraphData:
    edge_pairs = [(node_id, (node_id + 1) % length, f"ring_{node_id}") for node_id in range(length)]
    edge_pairs.extend(
        (left_id, right_id, f"chord_{index}") for index, (left_id, right_id) in enumerate(chords)
    )
    axes_by_node = {node_id: [] for node_id in range(length)}
    for left_id, right_id, edge_name in edge_pairs:
        axes_by_node[left_id].append(edge_name)
        axes_by_node[right_id].append(edge_name)

    nodes = {
        node_id: _make_node(f"C{node_id:02d}", tuple(axes_by_node[node_id]))
        for node_id in range(length)
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = [
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][edge_name], edge_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][edge_name], edge_name),
            name=edge_name,
            label=None,
        )
        for left_id, right_id, edge_name in edge_pairs
    ]
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_tubular_grid_graph(periodic: int, length: int) -> _GraphData:
    node_id_by_coord = {
        (theta, z): theta * length + z for theta in range(periodic) for z in range(length)
    }
    axes_by_node = {
        node_id: [f"obs_TB_{theta}_{z}"] for (theta, z), node_id in node_id_by_coord.items()
    }
    edge_specs: list[tuple[int, int, str]] = []
    for theta in range(periodic):
        for z in range(length):
            node_id = node_id_by_coord[(theta, z)]
            wrapped = node_id_by_coord[((theta + 1) % periodic, z)]
            ring_edge = f"tube_TB_{theta}_{z}_ring"
            axes_by_node[node_id].append(ring_edge)
            axes_by_node[wrapped].append(ring_edge)
            edge_specs.append((node_id, wrapped, ring_edge))
            if z < length - 1:
                axial = node_id_by_coord[(theta, z + 1)]
                axial_edge = f"tube_TB_{theta}_{z}_axial"
                axes_by_node[node_id].append(axial_edge)
                axes_by_node[axial].append(axial_edge)
                edge_specs.append((node_id, axial, axial_edge))

    nodes = {
        node_id: _make_node(
            f"TB{theta}_{z}",
            tuple(axes_by_node[node_id]),
        )
        for (theta, z), node_id in node_id_by_coord.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = [
        _make_dangling_edge(
            _EdgeEndpoint(node_id, axis_lookup[node_id][f"obs_TB_{theta}_{z}"], "obs"),
            name=f"obs_TB_{theta}_{z}",
            label=None,
        )
        for (theta, z), node_id in node_id_by_coord.items()
    ]
    edges.extend(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][edge_name], edge_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][edge_name], edge_name),
            name=edge_name,
            label=None,
        )
        for left_id, right_id, edge_name in edge_specs
    )
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


def _build_planar_house_graph_with_phys() -> _GraphData:
    nodes = {
        0: _make_node("A", ("right", "up", "phys")),
        1: _make_node("B", ("left", "up", "phys")),
        2: _make_node("C", ("down", "right", "roof", "phys")),
        3: _make_node("D", ("down", "left", "roof", "phys")),
        4: _make_node("R", ("left_roof", "right_roof", "phys")),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "right"),
            _EdgeEndpoint(1, 0, "left"),
            name="ab",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "up"),
            _EdgeEndpoint(2, 0, "down"),
            name="ac",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "up"),
            _EdgeEndpoint(3, 0, "down"),
            name="bd",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 1, "right"),
            _EdgeEndpoint(3, 1, "left"),
            name="cd",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 2, "roof"),
            _EdgeEndpoint(4, 0, "left_roof"),
            name="ce",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 2, "roof"),
            _EdgeEndpoint(4, 1, "right_roof"),
            name="de",
            label=None,
        ),
        _make_dangling_edge(_EdgeEndpoint(0, 2, "phys"), name="p0", label=None),
        _make_dangling_edge(_EdgeEndpoint(1, 2, "phys"), name="p1", label=None),
        _make_dangling_edge(_EdgeEndpoint(2, 3, "phys"), name="p2", label=None),
        _make_dangling_edge(_EdgeEndpoint(3, 3, "phys"), name="p3", label=None),
        _make_dangling_edge(_EdgeEndpoint(4, 2, "phys"), name="p4", label=None),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_placement_triangle_with_chains_graph() -> _GraphData:
    nodes = {
        0: _make_node("Prior", ("sample_state", "latent_prev")),
        1: _make_node("Transition", ("latent_prev", "latent_now", "control_signal")),
        2: _make_node("Emission", ("latent_now", "sensor_reading", "readout_latent")),
        3: _make_node("Calibrator", ("sensor_reading", "calibration_feature")),
        4: _make_node("Readout", ("readout_latent", "calibration_feature", "class_score")),
        5: _make_node("Loss", ("class_score", "target_label")),
    }
    edges = (
        _make_dangling_edge(_EdgeEndpoint(0, 0, "sample_state"), name="sample_state"),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "latent_prev"),
            _EdgeEndpoint(1, 0, "latent_prev"),
            name="latent_prev",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 1, "latent_now"),
            _EdgeEndpoint(2, 0, "latent_now"),
            name="latent_now",
            label=None,
        ),
        _make_dangling_edge(_EdgeEndpoint(1, 2, "control_signal"), name="control_signal"),
        _make_contraction_edge(
            _EdgeEndpoint(2, 1, "sensor_reading"),
            _EdgeEndpoint(3, 0, "sensor_reading"),
            name="sensor_reading",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 2, "readout_latent"),
            _EdgeEndpoint(4, 0, "readout_latent"),
            name="readout_latent",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(3, 1, "calibration_feature"),
            _EdgeEndpoint(4, 1, "calibration_feature"),
            name="calibration_feature",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(4, 2, "class_score"),
            _EdgeEndpoint(5, 0, "class_score"),
            name="class_score",
            label=None,
        ),
        _make_dangling_edge(_EdgeEndpoint(5, 1, "target_label"), name="target_label"),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_planar_cycle_with_tails_graph(
    *,
    core_count: int,
    tail_length: int,
) -> _GraphData:
    axes_by_node: dict[int, list[str]] = {node_id: [] for node_id in range(core_count)}
    edge_specs: list[tuple[int, int, str]] = []
    next_node_id = core_count

    def add_edge(left_id: int, right_id: int, name: str) -> None:
        axes_by_node.setdefault(left_id, []).append(name)
        axes_by_node.setdefault(right_id, []).append(name)
        edge_specs.append((left_id, right_id, name))

    for node_id in range(core_count):
        add_edge(node_id, (node_id + 1) % core_count, f"cycle_{node_id}")

    for core_id in range(core_count):
        previous_id = core_id
        for step in range(tail_length):
            current_id = next_node_id
            next_node_id += 1
            add_edge(previous_id, current_id, f"tail_{core_id}_{step}")
            previous_id = current_id

    nodes = {
        node_id: _make_node(f"P{node_id}", tuple(axis_names))
        for node_id, axis_names in axes_by_node.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = tuple(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][name], name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][name], name),
            name=name,
            label=None,
        )
        for left_id, right_id, name in edge_specs
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


def _build_demo_hyper_graph() -> _GraphData:
    axes_by_node: dict[int, list[str]] = {node_id: ["phys"] for node_id in range(12)}
    hyper_bonds = (
        ((0, "alpha"), (1, "alpha"), (2, "alpha"), (3, "alpha")),
        ((3, "beta"), (4, "beta"), (5, "beta")),
        ((6, "gamma"), (7, "gamma"), (8, "gamma"), (9, "gamma"), (10, "gamma")),
    )
    pair_edges = (
        (0, 4),
        (1, 6),
        (2, 7),
        (5, 8),
        (8, 11),
        (11, 9),
        (9, 4),
        (10, 2),
        (11, 3),
    )

    for bond in hyper_bonds:
        for node_id, axis_name in bond:
            axes_by_node[node_id].append(axis_name)

    pair_bonds: list[tuple[tuple[int, str], tuple[int, str]]] = []
    for edge_index, (left_id, right_id) in enumerate(pair_edges):
        axis_name = f"ring_{edge_index}"
        axes_by_node[left_id].append(axis_name)
        axes_by_node[right_id].append(axis_name)
        pair_bonds.append(((left_id, axis_name), (right_id, axis_name)))

    nodes = {
        node_id: _make_node(f"H{node_id}", tuple(axis_names))
        for node_id, axis_names in axes_by_node.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }

    edges: list[_EdgeData] = []
    next_virtual_id = -1
    for bond in hyper_bonds:
        hub_id = next_virtual_id
        next_virtual_id -= 1
        hub_axes = tuple(f"{bond[0][1]}_{axis_index}" for axis_index in range(len(bond)))
        nodes[hub_id] = _make_node("", hub_axes, label=bond[0][1], is_virtual=True)
        for axis_slot, (node_id, axis_name) in enumerate(bond):
            edges.append(
                _make_contraction_edge(
                    _EdgeEndpoint(node_id, axis_lookup[node_id][axis_name], axis_name),
                    _EdgeEndpoint(hub_id, axis_slot, hub_axes[axis_slot]),
                    name=axis_name,
                    label=None,
                )
            )

    edges.extend(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][axis_name], axis_name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][axis_name], axis_name),
            name=axis_name,
            label=None,
        )
        for (left_id, axis_name), (right_id, _axis_name) in pair_bonds
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


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


def _build_parallel_pair_graph() -> _GraphData:
    nodes = {
        0: _make_node("A", ("ab0", "ab1")),
        1: _make_node("B", ("ab0", "ab1")),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "ab0"),
            _EdgeEndpoint(1, 0, "ab0"),
            name="ab0",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "ab1"),
            _EdgeEndpoint(1, 1, "ab1"),
            name="ab1",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_crossed_pair_graph() -> _GraphData:
    nodes = {
        0: _make_node("A", ("ab", "ac")),
        1: _make_node("D", ("ab",)),
        2: _make_node("B", ("cd", "ac")),
        3: _make_node("C", ("cd",)),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "ab"),
            _EdgeEndpoint(1, 0, "ab"),
            name="ab",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(2, 0, "cd"),
            _EdgeEndpoint(3, 0, "cd"),
            name="cd",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(0, 1, "ac"),
            _EdgeEndpoint(2, 1, "ac"),
            name="ac",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_virtual_overlap_graph() -> _GraphData:
    nodes = {
        0: _make_node("A", ("hub_a",)),
        1: _make_node("B", ("hub_b",)),
        -1: _make_node("Hub", ("hub_a", "hub_b"), is_virtual=True),
    }
    edges = (
        _make_contraction_edge(
            _EdgeEndpoint(0, 0, "hub_a"),
            _EdgeEndpoint(-1, 0, "hub_a"),
            name="hub_a",
            label=None,
        ),
        _make_contraction_edge(
            _EdgeEndpoint(1, 0, "hub_b"),
            _EdgeEndpoint(-1, 1, "hub_b"),
            name="hub_b",
            label=None,
        ),
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_cycle_with_recursive_tail_graph() -> _GraphData:
    axes_by_node = {node_id: [] for node_id in range(6)}
    edge_pairs = (
        (0, 1, "c01"),
        (1, 2, "c12"),
        (2, 3, "c23"),
        (3, 0, "c30"),
        (0, 4, "t04"),
        (4, 5, "t45"),
    )
    for left_id, right_id, name in edge_pairs:
        axes_by_node[left_id].append(name)
        axes_by_node[right_id].append(name)
    nodes = {
        node_id: _make_node(f"C{node_id}", tuple(axis_names))
        for node_id, axis_names in axes_by_node.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = tuple(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][name], name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][name], name),
            name=name,
            label=None,
        )
        for left_id, right_id, name in edge_pairs
    )
    return _GraphData(nodes=nodes, edges=edges)


def _build_subdivided_generic_graph(
    *,
    subdivisions: int,
    pendant_length: int,
) -> _GraphData:
    axes_by_node: dict[int, list[str]] = {node_id: [] for node_id in range(5)}
    edge_specs: list[tuple[int, int, str]] = []
    next_node_id = 5

    def add_edge(left_id: int, right_id: int, name: str) -> None:
        axes_by_node.setdefault(left_id, []).append(name)
        axes_by_node.setdefault(right_id, []).append(name)
        edge_specs.append((left_id, right_id, name))

    for left_id in range(5):
        for right_id in range(left_id + 1, 5):
            previous_id = left_id
            for step in range(subdivisions):
                current_id = next_node_id
                next_node_id += 1
                axes_by_node[current_id] = []
                add_edge(previous_id, current_id, f"k5_{left_id}_{right_id}_{step}")
                previous_id = current_id
            add_edge(previous_id, right_id, f"k5_{left_id}_{right_id}_end")

    for core_id in range(5):
        previous_id = core_id
        for step in range(pendant_length):
            current_id = next_node_id
            next_node_id += 1
            axes_by_node[current_id] = []
            add_edge(previous_id, current_id, f"tail_{core_id}_{step}")
            previous_id = current_id

    nodes = {
        node_id: _make_node(f"G_{node_id}", tuple(axis_names))
        for node_id, axis_names in axes_by_node.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = tuple(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][name], name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][name], name),
            name=name,
            label=None,
        )
        for left_id, right_id, name in edge_specs
    )
    return _GraphData(nodes=nodes, edges=edges)


def _with_one_dangling_phys_per_node(graph: _GraphData) -> _GraphData:
    nodes = {
        node_id: _make_node(
            node.name,
            (*node.axes_names, "phys"),
            label=node.label,
            is_virtual=node.is_virtual,
        )
        for node_id, node in graph.nodes.items()
    }
    edges = list(graph.edges)
    for node_id, node in sorted(nodes.items()):
        axis_index = len(node.axes_names) - 1
        edges.append(
            _make_dangling_edge(
                _EdgeEndpoint(node_id, axis_index, "phys"),
                name=f"p{node_id}",
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


def _build_peps_grid_with_leaf_vectors(rows: int, cols: int) -> tuple[_GraphData, dict[int, int]]:
    nodes: dict[int, _NodeData] = {}
    edges = []
    leaf_parent_by_id: dict[int, int] = {}

    def core_node_id(row: int, col: int) -> int:
        return 2 * (row * cols + col)

    def leaf_node_id(row: int, col: int) -> int:
        return core_node_id(row, col) + 1

    for row in range(rows):
        for col in range(cols):
            axes_names: list[str] = []
            if col > 0:
                axes_names.append("left")
            if col < cols - 1:
                axes_names.append("right")
            if row > 0:
                axes_names.append("down")
            if row < rows - 1:
                axes_names.append("up")
            phys_name = f"p_{row}_{col}"
            axes_names.append(phys_name)
            core_id = core_node_id(row, col)
            leaf_id = leaf_node_id(row, col)
            nodes[core_id] = _make_node(f"P{row}_{col}", tuple(axes_names))
            nodes[leaf_id] = _make_node(f"x{row}_{col}", (phys_name,))
            leaf_parent_by_id[leaf_id] = core_id

    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }

    for row in range(rows):
        for col in range(cols):
            core_id = core_node_id(row, col)
            if col < cols - 1:
                right_id = core_node_id(row, col + 1)
                edges.append(
                    _make_contraction_edge(
                        _EdgeEndpoint(core_id, axis_lookup[core_id]["right"], "right"),
                        _EdgeEndpoint(right_id, axis_lookup[right_id]["left"], "left"),
                        name=f"h_{row}_{col}",
                        label=None,
                    )
                )
            if row < rows - 1:
                up_id = core_node_id(row + 1, col)
                edges.append(
                    _make_contraction_edge(
                        _EdgeEndpoint(core_id, axis_lookup[core_id]["up"], "up"),
                        _EdgeEndpoint(up_id, axis_lookup[up_id]["down"], "down"),
                        name=f"v_{row}_{col}",
                        label=None,
                    )
                )
            phys_name = f"p_{row}_{col}"
            leaf_id = leaf_node_id(row, col)
            edges.append(
                _make_contraction_edge(
                    _EdgeEndpoint(core_id, axis_lookup[core_id][phys_name], phys_name),
                    _EdgeEndpoint(leaf_id, 0, phys_name),
                    name=phys_name,
                    label=None,
                )
            )
    return _GraphData(nodes=nodes, edges=tuple(edges)), leaf_parent_by_id


def _segment_segment_min_distance_sq_3d(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    """Squared minimum distance between two 3D segments."""

    delta_a = end_a - start_a
    delta_b = end_b - start_b
    offset = start_a - start_b
    aa = float(np.dot(delta_a, delta_a))
    ab = float(np.dot(delta_a, delta_b))
    bb = float(np.dot(delta_b, delta_b))
    ao = float(np.dot(delta_a, offset))
    bo = float(np.dot(delta_b, offset))
    denom = aa * bb - ab * ab
    small = 1e-12
    s_num = 0.0
    s_den = denom
    t_num = 0.0
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = bo
        t_den = bb
    else:
        s_num = ab * bo - bb * ao
        t_num = aa * bo - ab * ao
        if s_num < 0.0:
            s_num = 0.0
            t_num = bo
            t_den = bb
        elif s_num > s_den:
            s_num = s_den
            t_num = bo + ab
            t_den = bb

    if t_num < 0.0:
        t_num = 0.0
        if -ao < 0.0:
            s_num = 0.0
        elif -ao > aa:
            s_num = s_den
        else:
            s_num = -ao
            s_den = aa
    elif t_num > t_den:
        t_num = t_den
        if (-ao + ab) < 0.0:
            s_num = 0.0
        elif (-ao + ab) > aa:
            s_num = s_den
        else:
            s_num = -ao + ab
            s_den = aa

    s_param = 0.0 if abs(s_num) < small else s_num / s_den
    t_param = 0.0 if abs(t_num) < small else t_num / t_den
    delta = offset + s_param * delta_a - t_param * delta_b
    return float(np.dot(delta, delta))


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    return axis / np.linalg.norm(axis)


def _unit_2d(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(flat))
    assert norm > 1e-9
    return flat / norm


def _assert_outward_chain_2d(
    positions: dict[int, np.ndarray],
    *,
    anchor_id: int,
    chain_node_ids: tuple[int, ...],
    core_centroid: np.ndarray,
) -> None:
    previous = np.asarray(positions[anchor_id], dtype=float).reshape(-1)[:2]
    outward = _unit_2d(previous - core_centroid[:2])
    previous_direction = outward
    for node_id in chain_node_ids:
        current = np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
        step_direction = _unit_2d(current - previous)
        assert float(np.dot(step_direction, outward)) > 0.90
        assert float(np.dot(step_direction, previous_direction)) > 0.97
        previous = current
        previous_direction = step_direction


def test_compute_layout_line_chain_2d_is_colinear_and_evenly_spaced() -> None:
    graph = _build_chain_graph(length=4, dangling_axis_counts={0: 1, 3: 1})

    positions = _compute_layout(graph, dimensions=2, seed=0)
    coords = np.stack([positions[node_id] for node_id in range(4)])

    assert np.allclose(coords[:, 1], 0.0, atol=1e-6)
    assert np.all(np.diff(coords[:, 0]) > 0.0)
    assert np.allclose(np.diff(coords[:, 0]), np.diff(coords[:, 0])[0], atol=1e-6)


def test_compute_layout_chain_short_circuits_before_grid_isomorphism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout_structure as layout_structure

    def unexpected_vf2pp_isomorphism(*args: object, **kwargs: object) -> object:
        raise AssertionError("chain detection should not ask VF2 to test grid-like layouts")

    monkeypatch.setattr(
        layout_structure.nx,
        "vf2pp_isomorphism",
        unexpected_vf2pp_isomorphism,
    )

    graph = _build_chain_graph(length=40)

    _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]

    assert component.structure_kind == "chain"
    assert component.chain_order == tuple(range(1, 39))


def test_compute_layout_line_chain_3d_keeps_backbone_straight_and_planar() -> None:
    graph = _build_chain_graph(length=4, dangling_axis_counts={0: 1, 3: 1})

    positions = _compute_layout(graph, dimensions=3, seed=0)
    coords = np.stack([positions[node_id] for node_id in range(4)])

    assert np.allclose(coords[:, 1], 0.0, atol=1e-6)
    assert np.allclose(coords[:, 2], 0.0, atol=1e-6)
    assert np.all(np.diff(coords[:, 0]) > 0.0)
    assert np.allclose(np.diff(coords[:, 0]), np.diff(coords[:, 0])[0], atol=1e-6)


def test_compute_layout_large_chain_3d_skips_layer_promotion_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.positions as layout_positions

    def fail_node_overlap(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise AssertionError("_node_overlaps_component should not run for pure 3D chains")

    def fail_crossing_edges(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise AssertionError(
            "_component_has_crossing_contraction_edges_2d should not run for pure 3D chains"
        )

    monkeypatch.setattr(layout_positions, "_node_overlaps_component", fail_node_overlap)
    monkeypatch.setattr(
        layout_positions,
        "_component_has_crossing_contraction_edges_2d",
        fail_crossing_edges,
    )

    positions = _compute_layout(_build_chain_graph(length=400), dimensions=3, seed=0)

    assert len(positions) == 400
    assert all(np.all(np.isfinite(position)) for position in positions.values())


def test_compute_axis_directions_large_chain_3d_reuses_component_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_3d as free_directions_3d

    original = free_directions_3d._component_orthogonal_basis
    calls: list[int] = []

    def counting_component_orthogonal_basis(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        free_directions_3d,
        "_component_orthogonal_basis",
        counting_component_orthogonal_basis,
    )

    graph = _build_chain_graph(length=400, dangling_axis_counts=dict.fromkeys(range(400), 1))
    positions = {
        node_id: np.array([float(node_id), 0.0, 0.0], dtype=float) for node_id in graph.nodes
    }

    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)

    assert len(directions) == sum(max(node.degree, 1) for node in graph.nodes.values())
    assert len(calls) == 1


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


def _assert_peps_leaf_vectors_clear_2d(rows: int, cols: int) -> None:
    graph, leaf_parent_by_id = _build_peps_grid_with_leaf_vectors(rows, cols)
    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
    bond_segments = _planar_contraction_bond_segments_2d(graph, positions, scale=draw_scale)

    for leaf_id, parent_id in leaf_parent_by_id.items():
        parent = np.asarray(positions[parent_id], dtype=float).reshape(-1)[:2]
        leaf = np.asarray(positions[leaf_id], dtype=float).reshape(-1)[:2]
        for node_id, node_position in positions.items():
            if node_id in {leaf_id, parent_id}:
                continue
            other = np.asarray(node_position, dtype=float).reshape(-1)[:2]
            assert np.linalg.norm(leaf - other) >= r_disk - 1e-9
        for left_id, right_id, bond_start, bond_end in bond_segments:
            if left_id in {leaf_id, parent_id} or right_id in {leaf_id, parent_id}:
                continue
            assert not _segments_cross_2d(parent, leaf, bond_start, bond_end)


def test_compute_layout_peps_leaf_vectors_2d_clear_other_core_nodes_3x3() -> None:
    _assert_peps_leaf_vectors_clear_2d(3, 3)


def test_compute_layout_peps_leaf_vectors_2d_clear_other_core_nodes_4x4() -> None:
    _assert_peps_leaf_vectors_clear_2d(4, 4)


def test_compute_layout_peps_leaf_vectors_2d_regular_grid_boundary_faces_prefer_cardinals() -> None:
    graph, _leaf_parent_by_id = _build_peps_grid_with_leaf_vectors(3, 3)
    positions = _compute_layout(graph, dimensions=2, seed=0)

    expected_by_site = {
        (2, 0): np.array([0.0, 1.0], dtype=float),
        (2, 1): np.array([0.0, 1.0], dtype=float),
        (2, 2): np.array([0.0, 1.0], dtype=float),
        (0, 0): np.array([0.0, -1.0], dtype=float),
        (0, 1): np.array([0.0, -1.0], dtype=float),
        (0, 2): np.array([0.0, -1.0], dtype=float),
        (1, 0): np.array([-1.0, 0.0], dtype=float),
        (1, 2): np.array([1.0, 0.0], dtype=float),
    }

    for (row, col), expected in expected_by_site.items():
        parent_id = 2 * (row * 3 + col)
        leaf_id = parent_id + 1
        direction = np.asarray(
            positions[leaf_id] - positions[parent_id],
            dtype=float,
        ).reshape(-1)[:2]
        direction /= np.linalg.norm(direction)
        assert np.allclose(direction, expected, atol=1e-6), (row, col, direction)


def test_compute_axis_directions_grid_2d_phys_legs_follow_shell_behaviors() -> None:
    graph = _build_grid_graph_with_dangling_phys(3, 3)
    positions = {
        row * 3 + col: np.array([float(col), float(row)], dtype=float)
        for row in range(3)
        for col in range(3)
    }

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    expected_by_site = {
        (2, 0): np.array([0.0, 1.0], dtype=float),
        (2, 1): np.array([0.0, 1.0], dtype=float),
        (2, 2): np.array([0.0, 1.0], dtype=float),
        (0, 0): np.array([0.0, -1.0], dtype=float),
        (0, 1): np.array([0.0, -1.0], dtype=float),
        (0, 2): np.array([0.0, -1.0], dtype=float),
        (1, 0): np.array([-1.0, 0.0], dtype=float),
        (1, 1): np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0),
        (1, 2): np.array([1.0, 0.0], dtype=float),
    }

    for (row, col), expected in expected_by_site.items():
        node_id = row * 3 + col
        axis_index = graph.nodes[node_id].axes_names.index("phys")
        assert np.allclose(directions[(node_id, axis_index)], expected, atol=1e-6)


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


def test_compute_layout_2x2_grid_remains_grid_not_circular() -> None:
    graph = _build_grid_graph(2, 2)

    _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]

    assert component.structure_kind == "grid"
    assert component.grid_mapping is not None


def test_compute_layout_circular_ring_2d_places_nodes_on_circle() -> None:
    graph = _build_named_ring_graph(10)

    positions = _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]
    coords = np.stack([positions[node_id] for node_id in range(10)])
    radii = np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1)

    assert component.structure_kind == "circular"
    assert component.chain_order == tuple(range(10))
    assert float(np.ptp(radii)) < 1e-6
    assert float(radii.min()) > 0.1


def test_compute_layout_circular_chords_2d_uses_named_ring_order() -> None:
    graph = _build_named_ring_graph(12, chords=((0, 4), (3, 9), (6, 10)))

    positions = _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]
    coords = np.stack([positions[node_id] for node_id in range(12)])
    radii = np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1)

    assert component.structure_kind == "circular"
    assert component.chain_order == tuple(range(12))
    assert float(np.ptp(radii)) < 1e-6
    assert float(radii.min()) > 0.1


def test_compute_layout_tubular_grid_uses_circular_layers_in_2d_and_3d() -> None:
    graph = _build_tubular_grid_graph(periodic=6, length=4)

    positions_2d = _compute_layout(graph, dimensions=2, seed=0)
    positions_3d = _compute_layout(graph, dimensions=3, seed=0)
    component = _analyze_layout_components_cached(graph)[0]

    assert component.structure_kind == "tube"
    assert component.grid_mapping is not None
    for axial_index in range(4):
        layer_node_ids = [
            node_id
            for node_id, (_theta, z_index) in component.grid_mapping.items()
            if z_index == axial_index
        ]
        coords_2d = np.stack([positions_2d[node_id] for node_id in layer_node_ids])
        radii_2d = np.linalg.norm(
            coords_2d - coords_2d.mean(axis=0, keepdims=True),
            axis=1,
        )
        coords_3d = np.stack([positions_3d[node_id] for node_id in layer_node_ids])
        radii_3d = np.linalg.norm(
            coords_3d[:, :2] - coords_3d[:, :2].mean(axis=0, keepdims=True),
            axis=1,
        )
        assert float(np.ptp(radii_2d)) < 1e-6
        assert float(np.ptp(radii_3d)) < 1e-6

    z_values = [
        float(
            np.mean(
                [
                    positions_3d[node_id][2]
                    for node_id, (_theta, z_index) in component.grid_mapping.items()
                    if z_index == axial_index
                ]
            )
        )
        for axial_index in range(4)
    ]
    assert len({round(value, 6) for value in z_values}) == 4


def test_compute_layout_sparse_grid_2d_recovers_coordinate_holes() -> None:
    size = 6
    active = {(row, col) for row in range(size) for col in range(row, size)}
    graph = _build_coord_graph_2d(active, prefix="UT")

    _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]

    assert component.structure_kind == "grid"
    assert component.grid_mapping is not None
    assert set(component.grid_mapping.values()) == {(col, row) for row, col in active}


def test_compute_layout_sparse_grid_3d_recovers_coordinate_holes() -> None:
    active = {
        (x, y, z)
        for z in range(3)
        for x in range(5)
        for y in range(5)
        if abs(x - 2) + abs(y - 2) <= 2 - z
    }
    graph = _build_coord_graph_3d(active, prefix="PY")

    positions = _compute_layout(graph, dimensions=3, seed=0)
    component = _analyze_layout_components_cached(graph)[0]

    assert component.structure_kind == "grid3d"
    assert component.grid3d_mapping is not None
    assert set(component.grid3d_mapping.values()) == active
    assert len({round(float(position[2]), 6) for position in positions.values()}) == 3


def test_compute_layout_3d_grid_spans_three_axes() -> None:
    graph = _build_3d_grid_graph(2, 2, 2)

    positions = _compute_layout(graph, dimensions=3, seed=0)
    coords = np.stack([positions[node_id] for node_id in sorted(graph.nodes)])

    assert float(coords[:, 0].std()) > 1e-6
    assert float(coords[:, 1].std()) > 1e-6
    assert float(coords[:, 2].std()) > 1e-6


def test_compute_layout_3d_grid_uniform_nearest_neighbor_spacing() -> None:
    graph = _build_3d_grid_graph(2, 3, 2)

    positions = _compute_layout(graph, dimensions=3, seed=0)
    lengths: list[float] = []
    for edge in graph.edges:
        left_ep, right_ep = edge.endpoints
        delta = positions[left_ep.node_id] - positions[right_ep.node_id]
        lengths.append(float(np.linalg.norm(delta)))

    assert lengths
    mean_len = float(np.mean(lengths))
    assert all(math.isclose(L, mean_len, rel_tol=1e-5, abs_tol=1e-8) for L in lengths)


def test_compute_layout_2d_grid3d_depth_projects_with_negative_x_and_negative_y() -> None:
    graph = _build_3d_grid_graph(2, 2, 3)

    positions = _compute_layout(graph, dimensions=2, seed=0)
    component = _analyze_layout_components_cached(graph)[0]
    assert component.grid3d_mapping is not None

    node_id_by_coords = {coords: node_id for node_id, coords in component.grid3d_mapping.items()}
    p000 = positions[node_id_by_coords[(0, 0, 0)]]
    p001 = positions[node_id_by_coords[(0, 0, 1)]]
    p002 = positions[node_id_by_coords[(0, 0, 2)]]

    assert float(p001[0]) < float(p000[0])
    assert float(p001[1]) < float(p000[1])
    assert float(p002[0]) < float(p001[0])
    assert float(p002[1]) < float(p001[1])


def test_layered_visible_order_2d_draws_grid3d_from_far_face_to_near_face() -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    graph = _build_3d_grid_graph(2, 2, 3)

    layered_order = render_prep._layered_visible_order_2d(graph)
    component = _analyze_layout_components_cached(graph)[0]
    k_sequence = [component.grid3d_mapping[node_id][2] for node_id in layered_order]

    assert k_sequence == sorted(k_sequence)


def test_layered_visible_order_2d_draws_tube_by_axial_layers() -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    graph = _build_tubular_grid_graph(periodic=5, length=3)

    layered_order = render_prep._layered_visible_order_2d(graph)
    component = _analyze_layout_components_cached(graph)[0]
    assert component.grid_mapping is not None
    z_sequence = [component.grid_mapping[node_id][1] for node_id in layered_order]

    assert z_sequence == sorted(z_sequence)


def test_layered_tensor_label_zorders_2d_stay_above_all_node_disks() -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    graph = _build_3d_grid_graph(2, 2, 3)

    visible_order = list(render_prep._layered_visible_order_2d(graph))
    label_zorders = render_prep._layered_tensor_label_zorders_2d(visible_order)
    max_disk_zorder = max(
        render_prep._ZORDER_LAYER_BASE
        + index * render_prep._ZORDER_LAYER_STRIDE
        + render_prep._ZORDER_LAYER_DISK
        for index, _node_id in enumerate(visible_order)
    )

    assert label_zorders
    assert all(zorder > max_disk_zorder for zorder in label_zorders.values())


def test_register_render_hover_2d_uses_layered_visible_order_for_grid3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    import tensor_network_viz._core.draw.render_prep as render_prep

    graph = _build_3d_grid_graph(2, 2, 3)
    positions = _compute_layout(graph, dimensions=2, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=2)
    scale = _resolve_draw_scale(graph, positions)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    try:
        context = render_prep._prepare_render_context(
            ax=ax,
            graph=graph,
            positions=positions,
            config=PlotConfig(hover_labels=True),
            dimensions=2,
            scale=scale,
            show_tensor_labels=True,
            show_index_labels=False,
        )
        render_prep._draw_edges_nodes_and_labels(
            ax=ax,
            context=context,
            directions=directions,
            show_tensor_labels=True,
            show_index_labels=False,
            tensor_disk_radius_px_3d=None,
        )

        captured: dict[str, tuple[int, ...]] = {}

        def _capture_hover_state(
            state: render_prep._RenderHoverState,
            *,
            scheme_patches_2d: object | None = None,
            scheme_aabbs_3d: object | None = None,
        ) -> None:
            del scheme_patches_2d, scheme_aabbs_3d
            captured["visible_node_ids"] = state.visible_node_ids

        monkeypatch.setattr(render_prep, "_apply_saved_hover_state", _capture_hover_state)

        render_prep._register_render_hover(
            ax=ax,
            context=context,
            show_tensor_labels=True,
            show_index_labels=False,
            scheme_patches_2d=[],
            scheme_aabbs_3d=[],
            tensor_disk_radius_px_3d=None,
        )

        assert captured["visible_node_ids"] == render_prep._layered_visible_order_2d(graph)
    finally:
        fig.clear()


def test_build_interactive_scene_state_2d_uses_layered_visible_order_for_grid3d() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    import tensor_network_viz._core.draw.render_prep as render_prep

    graph = _build_3d_grid_graph(2, 2, 3)
    positions = _compute_layout(graph, dimensions=2, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=2)
    scale = _resolve_draw_scale(graph, positions)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    try:
        context = render_prep._prepare_render_context(
            ax=ax,
            graph=graph,
            positions=positions,
            config=PlotConfig(hover_labels=True),
            dimensions=2,
            scale=scale,
            show_tensor_labels=True,
            show_index_labels=False,
        )
        render_prep._draw_edges_nodes_and_labels(
            ax=ax,
            context=context,
            directions=directions,
            show_tensor_labels=True,
            show_index_labels=False,
            tensor_disk_radius_px_3d=None,
        )
        hover_state = render_prep._RenderHoverState(
            ax=ax,
            figure=ax.figure,
            dimensions=2,
            node_patch_coll=render_prep._node_patch_collection_from_plotter(context),
            visible_node_ids=(),
            tensor_hover=dict(context.tensor_hover_by_node or {}),
            edge_hover=tuple(context.hover_edge_targets or ()),
            line_width_px_hint=float(context.params.lw),
        )

        scene = render_prep._build_interactive_scene_state(
            ax=ax,
            context=context,
            directions=directions,
            scale=scale,
            hover_state=hover_state,
            tensor_disk_radius_px_3d=None,
        )

        assert scene.visible_node_ids == render_prep._layered_visible_order_2d(graph)
    finally:
        fig.clear()


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


def test_compute_axis_directions_chain_3d_uses_ordered_cube_candidates() -> None:
    graph = _build_chain_graph(length=3, dangling_axis_counts={1: 5})
    positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0, 0.0], dtype=float),
        2: np.array([2.0, 0.0, 0.0], dtype=float),
    }

    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)
    free_axis_indices = [
        axis_index
        for axis_index, axis_name in enumerate(graph.nodes[1].axes_names)
        if axis_name.startswith("d1_")
    ]
    picked = [
        np.asarray(directions[(1, axis_index)], dtype=float) for axis_index in free_axis_indices
    ]

    expected = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 0.0, -1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, -1.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 1.0], dtype=float) / np.sqrt(2.0),
    )
    for got, want in zip(picked, expected, strict=True):
        assert np.allclose(got, want, atol=1e-6)


def test_compute_axis_directions_chain_3d_skips_random_bucket_when_ordered_candidates_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_3d as free_directions_3d

    graph = _build_chain_graph(length=3, dangling_axis_counts={1: 1})
    positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0, 0.0], dtype=float),
        2: np.array([2.0, 0.0, 0.0], dtype=float),
    }
    calls: list[int] = []

    def counting_random_bucket(
        *,
        blocked: tuple[np.ndarray, ...],
        count: int = 16,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, ...]:
        del blocked, count, rng
        calls.append(1)
        return (
            np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0),
            np.array([-1.0, -1.0, -1.0], dtype=float) / np.sqrt(3.0),
        )

    monkeypatch.setattr(
        free_directions_3d,
        "_random_direction_bucket_3d",
        counting_random_bucket,
    )

    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)

    assert np.allclose(directions[(1, 2)], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    assert calls == []


def test_compute_axis_directions_chain_3d_generates_deterministic_candidates_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_3d as free_directions_3d

    graph = _build_chain_graph(length=3, dangling_axis_counts={1: 1})
    positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0, 0.0], dtype=float),
        2: np.array([2.0, 0.0, 0.0], dtype=float),
    }
    original = free_directions_3d._local_to_world_direction_3d
    calls: list[tuple[float, float, float]] = []

    def counting_local_to_world(
        frame: free_directions_3d._LocalFrame3D,
        local_direction: tuple[float, float, float],
    ) -> np.ndarray:
        calls.append(local_direction)
        return original(frame, local_direction)

    monkeypatch.setattr(
        free_directions_3d,
        "_local_to_world_direction_3d",
        counting_local_to_world,
    )

    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)

    assert np.allclose(directions[(1, 2)], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    assert calls == [(0.0, 0.0, 1.0)]


def test_compute_axis_directions_3d_supports_xp_alias_for_free_axes() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("up", "xp")),
            1: _make_node("B", ("down",)),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "up"),
                _EdgeEndpoint(1, 0, "down"),
                name="vertical",
                label=None,
            ),
        ),
    )
    positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([0.0, 1.0, 0.0], dtype=float),
    }

    directions = _compute_axis_directions(graph, positions, dimensions=3)

    assert np.allclose(directions[(0, 1)], np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_compute_axis_directions_3d_named_dangling_axes_fall_back_when_repeated() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("front", "front")),
        },
        edges=(
            _make_dangling_edge(_EdgeEndpoint(0, 0, "front"), name="front"),
            _make_dangling_edge(_EdgeEndpoint(0, 1, "front"), name="front"),
        ),
    )
    positions = {0: np.array([0.0, 0.0, 0.0], dtype=float)}

    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)

    assert np.allclose(directions[(0, 0)], np.array([0.0, 1.0, 0.0], dtype=float), atol=1e-6)
    assert not np.allclose(directions[(0, 1)], np.array([0.0, 1.0, 0.0], dtype=float), atol=1e-6)
    assert not _direction_angle_conflicts_3d(directions[(0, 1)], directions[(0, 0)])


def test_compute_axis_directions_competing_free_axes_prefer_unused_orthogonal_directions() -> None:
    graph = _build_star_with_free_axes()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)

    assert np.allclose(directions[(0, 4)], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(directions[(0, 5)], np.array([0.0, -1.0, 0.0]), atol=1e-6)
    for axis_index in range(6, 8):
        assert np.isclose(np.linalg.norm(directions[(0, axis_index)]), 1.0, atol=1e-6)


def test_compute_axis_directions_cubic_phys_stubs_3d_shell_roles_point_outward() -> None:
    graph = _build_3d_grid_graph_with_dangling_phys(3, 3, 3)

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)
    component = _analyze_layout_components_cached(graph)[0]
    node_id_by_coords = {coords: node_id for node_id, coords in component.grid3d_mapping.items()}

    expected_by_coords = {
        (2, 1, 1): np.array([1.0, 0.0, 0.0], dtype=float),
        (2, 2, 1): np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0),
        (2, 2, 2): np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0),
    }
    for coords, expected in expected_by_coords.items():
        node_id = node_id_by_coords[coords]
        axis_index = graph.nodes[node_id].axes_names.index("phys")
        direction = np.asarray(directions[(node_id, axis_index)], dtype=float).reshape(-1)[:3]
        direction /= np.linalg.norm(direction)
        assert np.allclose(direction, expected, atol=1e-6), (coords, direction)


def test_compute_axis_directions_cubic_phys_stubs_3d_face_centers_point_outward() -> None:
    graph = _build_3d_grid_graph_with_dangling_phys(3, 3, 3)

    positions = _compute_layout(graph, dimensions=3, seed=0)
    directions = _compute_axis_directions(graph, positions, dimensions=3)
    component = _analyze_layout_components_cached(graph)[0]
    node_id_by_coords = {coords: node_id for node_id, coords in component.grid3d_mapping.items()}
    centroid = np.mean(
        np.stack(
            [
                np.asarray(positions[node_id], dtype=float).reshape(-1)[:3]
                for node_id in component.anchor_node_ids
            ],
            axis=0,
        ),
        axis=0,
    )

    face_center_coords = (
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    for coords in face_center_coords:
        node_id = node_id_by_coords[coords]
        axis_index = graph.nodes[node_id].axes_names.index("phys")
        direction = np.asarray(directions[(node_id, axis_index)], dtype=float).reshape(-1)[:3]
        direction /= np.linalg.norm(direction)
        outward = np.asarray(positions[node_id], dtype=float).reshape(-1)[:3] - centroid
        outward /= np.linalg.norm(outward)
        assert float(np.dot(direction, outward)) > 0.95, (coords, direction, outward)


def test_compute_axis_directions_cubic_phys_stubs_2d_clear_nonincident_projected_bonds() -> None:
    graph = _build_3d_grid_graph_with_dangling_phys(3, 3, 2)

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)
    component = _analyze_layout_components_cached(graph)[0]
    radius = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6)

    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        endpoint = edge.endpoints[0]
        origin = np.asarray(positions[endpoint.node_id], dtype=float).reshape(-1)[:2]
        direction = np.asarray(
            directions[(endpoint.node_id, endpoint.axis_index)],
            dtype=float,
        ).reshape(-1)[:2]
        _start, end = _dangling_stub_segment_2d(origin, direction, draw_scale=draw_scale)
        first_neighbors = set(component.contraction_graph.neighbors(endpoint.node_id))
        second_neighbors = {
            second_id
            for neighbor_id in first_neighbors
            for second_id in component.contraction_graph.neighbors(neighbor_id)
            if second_id != endpoint.node_id
        }
        for neighbor_id in sorted(first_neighbors | second_neighbors):
            other = np.asarray(positions[neighbor_id], dtype=float).reshape(-1)[:2]
            assert np.linalg.norm(end - other) >= radius - 1e-9


def test_compute_axis_directions_grid3d_2d_top_and_bottom_shells_keep_free_cardinals() -> None:
    graph = _build_3d_grid_graph_with_dangling_phys(2, 3, 2)

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)

    y_by_node = {
        node_id: round(float(np.asarray(position, dtype=float).reshape(-1)[1]), 6)
        for node_id, position in positions.items()
    }
    top_y = max(y_by_node.values())
    bottom_y = min(y_by_node.values())
    top_node_ids = [node_id for node_id, y in y_by_node.items() if y == top_y]
    bottom_node_ids = [node_id for node_id, y in y_by_node.items() if y == bottom_y]

    assert top_node_ids
    assert bottom_node_ids

    for node_id in top_node_ids:
        axis_index = graph.nodes[node_id].axes_names.index("phys")
        assert np.allclose(
            directions[(node_id, axis_index)],
            np.array([0.0, 1.0], dtype=float),
            atol=1e-6,
        ), graph.nodes[node_id].name

    for node_id in bottom_node_ids:
        axis_index = graph.nodes[node_id].axes_names.index("phys")
        assert np.allclose(
            directions[(node_id, axis_index)],
            np.array([0.0, -1.0], dtype=float),
            atol=1e-6,
        ), graph.nodes[node_id].name


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


def test_compute_layout_demo_hypergraph_3d_lifts_virtual_hubs_off_plane() -> None:
    graph = _build_demo_hyper_graph()

    positions = _compute_layout(graph, dimensions=3, seed=0)
    component = _analyze_layout_components_cached(graph)[0]
    virtual_z_levels = {
        round(float(positions[node_id][2]), 6) for node_id in component.virtual_node_ids
    }

    assert len(virtual_z_levels) >= 2
    assert any(abs(float(positions[node_id][2])) > 1e-6 for node_id in component.virtual_node_ids)


def test_compute_layout_generic_graph_uses_force_fallback_produces_valid_positions() -> None:
    """Generic (non-planar) graphs fall back to force-directed layout; output must be valid."""
    graph = _build_complete_graph_5()

    positions = _compute_layout(graph, dimensions=2, seed=0, iterations=50)

    assert len(positions) == 5
    coords = np.stack([positions[i] for i in range(5)])
    assert not np.any(np.isnan(coords))
    assert not np.any(np.isinf(coords))
    assert np.all(np.isfinite(coords))


def _make_coarsening_graph(
    *,
    neighbors_by_node: dict[int, tuple[int, ...]],
    edge_weights: dict[tuple[int, int], float],
) -> _CoarseningGraph:
    normalized_edge_weights: dict[tuple[int, int], float] = {}
    for (left_id, right_id), weight in edge_weights.items():
        left_key = int(left_id)
        right_key = int(right_id)
        if right_key < left_key:
            left_key, right_key = right_key, left_key
        normalized_edge_weights[(left_key, right_key)] = float(weight)
    return _CoarseningGraph(
        node_ids=tuple(sorted(neighbors_by_node)),
        neighbors_by_node={
            int(node_id): tuple(int(neighbor_id) for neighbor_id in neighbors)
            for node_id, neighbors in sorted(neighbors_by_node.items())
        },
        edge_weights=normalized_edge_weights,
    )


def test_ordered_initial_circle_node_ids_prefers_heaviest_start_neighbor() -> None:
    coarsening_graph = _make_coarsening_graph(
        neighbors_by_node={
            0: (1, 2),
            1: (0, 3),
            2: (0, 3),
            3: (1, 2),
        },
        edge_weights={
            (0, 1): 2.0,
            (0, 2): 1.0,
            (1, 3): 1.0,
            (2, 3): 1.0,
        },
    )

    assert _ordered_initial_circle_node_ids(coarsening_graph) == [2, 0, 1, 3]


def test_ordered_initial_circle_node_ids_breaks_endpoint_ties_by_placed_connections() -> None:
    coarsening_graph = _make_coarsening_graph(
        neighbors_by_node={
            0: (1, 2),
            1: (0, 3, 4),
            2: (0, 3, 6),
            3: (1, 2),
            4: (1, 6),
            6: (2, 4),
        },
        edge_weights={
            (0, 1): 2.0,
            (0, 2): 1.0,
            (1, 3): 1.0,
            (1, 4): 1.0,
            (2, 3): 1.0,
            (2, 6): 1.0,
            (4, 6): 1.0,
        },
    )

    order = _ordered_initial_circle_node_ids(coarsening_graph)

    assert order[order.index(1) + 1] == 3


def test_ordered_initial_circle_node_ids_breaks_remaining_ties_by_lower_degree() -> None:
    coarsening_graph = _make_coarsening_graph(
        neighbors_by_node={
            0: (1, 2),
            1: (0, 3, 4),
            2: (0, 6),
            3: (1, 5),
            4: (1, 5, 6),
            5: (3, 4),
            6: (2, 4),
        },
        edge_weights={
            (0, 1): 2.0,
            (0, 2): 1.0,
            (1, 3): 1.0,
            (1, 4): 1.0,
            (2, 6): 1.0,
            (3, 5): 1.0,
            (4, 5): 1.0,
            (4, 6): 1.0,
        },
    )

    order = _ordered_initial_circle_node_ids(coarsening_graph)

    assert order[order.index(1) + 1] == 3


def test_ordered_initial_circle_node_ids_skips_blocked_side_and_continues_other_branch() -> None:
    coarsening_graph = _make_coarsening_graph(
        neighbors_by_node={
            0: (1, 2),
            1: (0, 2),
            2: (0, 1, 3, 4),
            3: (2, 4),
            4: (2, 3),
        },
        edge_weights={
            (0, 1): 2.0,
            (0, 2): 1.0,
            (1, 2): 1.0,
            (2, 3): 1.0,
            (2, 4): 1.0,
            (3, 4): 1.0,
        },
    )

    assert _ordered_initial_circle_node_ids(coarsening_graph) == [4, 3, 2, 0, 1]


def test_ordered_initial_circle_node_ids_uses_recovery_when_both_ends_block() -> None:
    coarsening_graph = _make_coarsening_graph(
        neighbors_by_node={
            0: (1, 2),
            1: (0, 2),
            2: (0, 1, 3, 4, 5, 6),
            3: (2, 4),
            4: (2, 3),
            5: (2, 6),
            6: (2, 5),
        },
        edge_weights={
            (0, 1): 2.0,
            (0, 2): 1.0,
            (1, 2): 1.0,
            (2, 3): 3.0,
            (2, 4): 2.0,
            (2, 5): 2.0,
            (2, 6): 1.0,
            (3, 4): 1.0,
            (5, 6): 1.0,
        },
    )

    assert _ordered_initial_circle_node_ids(coarsening_graph) == [4, 3, 2, 0, 1, 5, 6]


def test_generic_coarsening_counts_distinct_neighbors_for_parallel_edges() -> None:
    graph = _build_parallel_pair_graph()
    component = _analyze_layout_components_cached(graph)[0]

    counts = _distinct_neighbor_counts(component.contraction_graph, component.node_ids)

    assert counts == {0: 1, 1: 1}


def test_generic_coarsening_recursively_peels_degree_one_trees() -> None:
    graph = _build_cycle_with_recursive_tail_graph()
    component = _analyze_layout_components_cached(graph)[0]

    peeled = _peel_degree_one_trees(component.contraction_graph, component.node_ids)

    assert peeled.core_node_ids == (0, 1, 2, 3)
    assert peeled.parent_by_removed_node == {5: 4, 4: 0}


def test_coarsened_planar_without_degree_one_neighbors_skips_full_peel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.generic_coarsening as generic_coarsening

    graph = _build_planar_house_graph_with_phys()
    component = _analyze_layout_components_cached(graph)[0]

    def fail_peel(*args: object, **kwargs: object) -> object:
        raise AssertionError("planar without contraction leaves should not enter full peel")

    monkeypatch.setattr(
        generic_coarsening,
        "_peel_degree_one_trees_from_coarsening",
        fail_peel,
    )

    assert component.structure_kind == "planar"
    assert (
        _compute_coarsened_layout_2d(
            graph,
            component,
            seed=0,
            iterations=50,
        )
        is None
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_coarsened_decorated_planar_layout_places_chains_outward(
    dimensions: int,
) -> None:
    graph = _build_placement_triangle_with_chains_graph()
    component = _analyze_layout_components_cached(graph)[0]

    positions = _compute_layout(graph, dimensions=dimensions, seed=0, iterations=50)
    core_centroid = np.mean(
        np.stack(
            [np.asarray(positions[node_id], dtype=float).reshape(-1)[:2] for node_id in (2, 3, 4)]
        ),
        axis=0,
    )
    core_area = abs(
        float(
            np.linalg.det(
                np.stack(
                    [
                        positions[3][:2] - positions[2][:2],
                        positions[4][:2] - positions[2][:2],
                    ]
                )
            )
        )
    )

    assert component.structure_kind == "planar"
    assert core_area > 0.15
    _assert_outward_chain_2d(
        positions, anchor_id=2, chain_node_ids=(1, 0), core_centroid=core_centroid
    )
    _assert_outward_chain_2d(
        positions, anchor_id=4, chain_node_ids=(5,), core_centroid=core_centroid
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_coarsened_decorated_planar_long_tails_are_finite_and_spread(
    dimensions: int,
) -> None:
    graph = _build_planar_cycle_with_tails_graph(core_count=12, tail_length=8)

    first = _compute_layout(graph, dimensions=dimensions, seed=0, iterations=50)
    second = _compute_layout(graph, dimensions=dimensions, seed=0, iterations=50)
    coords = np.stack([first[node_id] for node_id in sorted(graph.nodes)])
    pairwise_distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(pairwise_distance, np.inf)

    assert np.all(np.isfinite(coords))
    assert float(pairwise_distance.min()) > 1e-4
    for node_id in sorted(graph.nodes):
        assert np.allclose(first[node_id], second[node_id], atol=1e-9)


def test_generic_coarsening_preserves_closed_degree_two_cycles() -> None:
    graph = _build_named_ring_graph(8)
    component = _analyze_layout_components_cached(graph)[0]

    compressed = _compress_degree_two_paths(component.contraction_graph, component.node_ids)

    assert compressed.skeleton_node_ids == tuple(range(8))
    assert compressed.paths == ()


def test_generic_coarsening_force_layout_uses_small_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.generic_coarsening as generic_coarsening

    graph = _build_subdivided_generic_graph(subdivisions=2, pendant_length=3)
    captured_node_ids: list[int] = []
    original = generic_coarsening._compute_weighted_force_layout

    def counting_force_layout(*args: object, **kwargs: object) -> dict[int, np.ndarray]:
        node_ids = kwargs["node_ids"]
        assert isinstance(node_ids, list)
        captured_node_ids[:] = [int(node_id) for node_id in node_ids]
        return original(*args, **kwargs)

    monkeypatch.setattr(
        generic_coarsening,
        "_compute_weighted_force_layout",
        counting_force_layout,
    )

    positions = _compute_layout(graph, dimensions=2, seed=0, iterations=50)

    assert set(positions) == set(graph.nodes)
    assert set(captured_node_ids) == set(range(5))
    assert len(captured_node_ids) < len(graph.nodes) / 4
    assert captured_node_ids == [4, 2, 0, 1, 3]


def test_generic_coarsening_layout_is_finite_spread_and_deterministic() -> None:
    graph = _build_subdivided_generic_graph(subdivisions=2, pendant_length=3)

    first_2d = _compute_layout(graph, dimensions=2, seed=0, iterations=50)
    second_2d = _compute_layout(graph, dimensions=2, seed=0, iterations=50)
    first_3d = _compute_layout(graph, dimensions=3, seed=0, iterations=50)

    coords_2d = np.stack([first_2d[node_id] for node_id in sorted(graph.nodes)])
    coords_3d = np.stack([first_3d[node_id] for node_id in sorted(graph.nodes)])
    pairwise_distance_2d = np.linalg.norm(
        coords_2d[:, None, :] - coords_2d[None, :, :],
        axis=2,
    )
    pairwise_distance_3d = np.linalg.norm(
        coords_3d[:, None, :] - coords_3d[None, :, :],
        axis=2,
    )
    np.fill_diagonal(pairwise_distance_2d, np.inf)
    np.fill_diagonal(pairwise_distance_3d, np.inf)

    assert np.all(np.isfinite(coords_2d))
    assert np.all(np.isfinite(coords_3d))
    for node_id in sorted(graph.nodes):
        assert np.allclose(first_2d[node_id], second_2d[node_id], atol=1e-9)
    assert float(pairwise_distance_2d.min()) > 1e-4
    assert float(pairwise_distance_3d.min()) > 1e-4


def test_promote_3d_layers_moves_overlapping_node_to_next_even_layer() -> None:
    graph = _build_chain_graph(length=2)
    component = _analyze_layout_components_cached(graph)[0]
    positions = {
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([0.0, 0.0, 0.0], dtype=float),
    }

    _promote_3d_layers(graph, component, positions)

    assert positions[0][2] == pytest.approx(0.0)
    assert positions[1][2] == pytest.approx(2.0 * _LAYER_SPACING)


def test_promote_3d_layers_uses_odd_layer_for_edge_crossing_without_node_overlap() -> None:
    graph = _build_crossed_pair_graph()
    component = _analyze_layout_components_cached(graph)[0]
    positions = {
        0: np.array([-1.0, -1.0, 0.0], dtype=float),
        1: np.array([1.0, 1.0, 0.0], dtype=float),
        2: np.array([-1.0, 1.0, 0.0], dtype=float),
        3: np.array([1.0, -1.0, 0.0], dtype=float),
    }

    _promote_3d_layers(graph, component, positions)

    assert positions[0][2] == pytest.approx(0.0)
    assert positions[2][2] == pytest.approx(0.0)
    assert positions[3][2] == pytest.approx(0.0)
    assert positions[1][2] == pytest.approx(_LAYER_SPACING)


def test_promote_3d_layers_includes_virtual_nodes_in_layer_assignment() -> None:
    graph = _build_virtual_overlap_graph()
    component = _analyze_layout_components_cached(graph)[0]
    positions = {
        -1: np.array([0.0, 0.0, 0.0], dtype=float),
        0: np.array([0.0, 0.0, 0.0], dtype=float),
        1: np.array([1.0, 0.0, 0.0], dtype=float),
    }

    _promote_3d_layers(graph, component, positions)

    assert positions[0][2] == pytest.approx(0.0)
    assert positions[1][2] == pytest.approx(0.0)
    assert positions[-1][2] == pytest.approx(2.0 * _LAYER_SPACING)


def test_tree_component_layout_2d_skips_force_layout_when_anchors_cover_all_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_binary_tree_graph_with_leaf_phys()
    component = _analyze_layout_components_cached(graph)[0]

    def fail_force_layout(*args: object, **kwargs: object) -> dict[int, np.ndarray]:
        del args, kwargs
        raise AssertionError("_compute_force_layout should not run for fully anchored trees")

    monkeypatch.setattr(
        "tensor_network_viz._core.layout.positions._compute_force_layout",
        fail_force_layout,
    )

    positions = _compute_component_layout_2d(graph, component, seed=0, iterations=1)

    assert set(positions) == set(graph.nodes)


def test_compute_axis_directions_tree_2d_leaf_phys_legs_prefer_south_when_corridor_is_free() -> (
    None
):
    graph = _build_binary_tree_graph_with_leaf_phys()

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)

    for leaf_id in (3, 4, 5, 6):
        axis_index = graph.nodes[leaf_id].axes_names.index("phys")
        assert np.allclose(
            directions[(leaf_id, axis_index)],
            np.array([0.0, -1.0], dtype=float),
            atol=1e-6,
        ), graph.nodes[leaf_id].name


def _neighbor_shell_ids(component: object, node_id: int) -> tuple[set[int], set[int]]:
    first_neighbors = set(component.contraction_graph.neighbors(node_id))
    second_neighbors = {
        second_id
        for neighbor_id in first_neighbors
        for second_id in component.contraction_graph.neighbors(neighbor_id)
        if second_id != node_id and second_id not in first_neighbors
    }
    return first_neighbors, second_neighbors


def _assert_local_neighbor_rules_hold_2d(
    graph: _GraphData,
    positions: dict[int, np.ndarray],
    directions: dict[tuple[int, int], np.ndarray],
    *,
    draw_scale: float,
) -> None:
    component = _analyze_layout_components_cached(graph)[0]
    radius = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6)

    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        endpoint = edge.endpoints[0]
        first_neighbors, second_neighbors = _neighbor_shell_ids(component, endpoint.node_id)
        origin = np.asarray(positions[endpoint.node_id], dtype=float).reshape(-1)[:2]
        direction = np.asarray(
            directions[(endpoint.node_id, endpoint.axis_index)],
            dtype=float,
        ).reshape(-1)[:2]
        _start, end = _dangling_stub_segment_2d(origin, direction, draw_scale=draw_scale)

        for neighbor_id in sorted(first_neighbors | second_neighbors):
            other = np.asarray(positions[neighbor_id], dtype=float).reshape(-1)[:2]
            assert np.linalg.norm(end - other) >= radius - 1e-9

        for other_edge in graph.edges:
            if other_edge.kind == "contraction":
                left_endpoint, right_endpoint = other_edge.endpoints
                left_id = left_endpoint.node_id
                right_id = right_endpoint.node_id
                if endpoint.node_id in {left_id, right_id}:
                    continue
                if left_id not in first_neighbors | second_neighbors and right_id not in (
                    first_neighbors | second_neighbors
                ):
                    continue
                bond_start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:2]
                bond_end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:2]
                assert not _segments_cross_2d(_start, end, bond_start, bond_end)
                continue

            other_endpoint = other_edge.endpoints[0]
            if other_endpoint.node_id not in first_neighbors | second_neighbors:
                continue
            other_direction = np.asarray(
                directions[(other_endpoint.node_id, other_endpoint.axis_index)],
                dtype=float,
            ).reshape(-1)[:2]
            other_start, other_end = _dangling_stub_segment_2d(
                positions[other_endpoint.node_id],
                other_direction,
                draw_scale=draw_scale,
            )
            assert not _segments_cross_2d(_start, end, other_start, other_end)


def test_compute_axis_directions_tree_2d_phys_legs_respect_local_neighbor_rules() -> None:
    graph = _build_binary_tree_graph_with_leaf_phys()

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)

    _assert_local_neighbor_rules_hold_2d(graph, positions, directions, draw_scale=draw_scale)


def test_compute_axis_directions_planar_2d_phys_legs_respect_local_neighbor_rules() -> None:
    graph = _build_planar_house_graph_with_phys()

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)

    _assert_local_neighbor_rules_hold_2d(graph, positions, directions, draw_scale=draw_scale)


def test_compute_axis_directions_generic_2d_phys_legs_respect_local_neighbor_rules() -> None:
    graph = _with_one_dangling_phys_per_node(_build_complete_graph_5())

    positions = _compute_layout(graph, dimensions=2, seed=0, iterations=50)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)

    _assert_local_neighbor_rules_hold_2d(graph, positions, directions, draw_scale=draw_scale)


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
