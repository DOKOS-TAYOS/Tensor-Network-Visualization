"""Precomputed node degrees vs naive per-node scans (see ``_node_edge_degrees``).

Timing (example dev machine, 2026-03-29): chain |V|≈8000 — batched mask build ~1.8 ms;
naive ``_graph_edge_degree`` per node in a list comprehension ~4.9 s (≈2800× slower).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tensor_network_viz._core.draw.plotter import (
    _graph_edge_degree,
    _node_edge_degrees,
    _visible_degree_one_mask,
)
from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)


def _chain_graph(n_nodes: int) -> _GraphData:
    nodes = {i: _make_node(f"T{i}", ("L", "R")) for i in range(n_nodes)}
    edges: list[_EdgeData] = []
    for i in range(n_nodes - 1):
        edges.append(
            _make_contraction_edge(
                _EdgeEndpoint(i, 1, f"b{i}"),
                _EdgeEndpoint(i + 1, 0, f"b{i}"),
                name=f"b{i}",
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def test_node_edge_degrees_matches_graph_edge_degree() -> None:
    g = _chain_graph(50)
    d = _node_edge_degrees(g)
    for nid in g.nodes:
        assert d.get(int(nid), 0) == _graph_edge_degree(g, nid), nid


def test_visible_degree_one_mask_with_shared_degrees() -> None:
    g = _chain_graph(20)
    visible = [i for i in g.nodes if not g.nodes[i].is_virtual]
    d = _node_edge_degrees(g)
    m1 = _visible_degree_one_mask(g, visible, node_degrees=d)
    m2 = _visible_degree_one_mask(g, visible)
    np.testing.assert_array_equal(m1, m2)


@pytest.mark.perf
def test_precomputed_degrees_much_faster_than_per_node_scan() -> None:
    n = 8000
    g = _chain_graph(n)
    visible = [i for i in g.nodes if not g.nodes[i].is_virtual]

    t0 = time.perf_counter()
    d = _node_edge_degrees(g)
    _ = _visible_degree_one_mask(g, visible, node_degrees=d)
    fast = time.perf_counter() - t0

    t1 = time.perf_counter()
    np.array([_graph_edge_degree(g, nid) == 1 for nid in visible], dtype=bool)
    slow = time.perf_counter() - t1

    assert slow > fast * 20.0, (
        f"expected batched degrees << naive loop (fast={fast:.5f}s slow={slow:.5f}s)"
    )
