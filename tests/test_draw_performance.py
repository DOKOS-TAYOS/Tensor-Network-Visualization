"""Lightweight timing smoke tests for dense 2D draws (Agg); not a strict benchmark gate."""

from __future__ import annotations

import io
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensor_network_viz import PlotConfig
from tensor_network_viz._core._draw_common import _draw_graph
from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)
from tensor_network_viz._core.layout import _compute_axis_directions
from tensor_network_viz._core.renderer import _resolve_draw_scale


def _chain_graph(n_nodes: int) -> _GraphData:
    """Linear chain with index labels on every bond (captions from axis names)."""
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


def test_dense_2d_draw_fast_flags_complete() -> None:
    """Fast path (skip refit + index separation) must finish without error."""
    n = 48
    graph = _chain_graph(n)
    positions = {i: np.array([float(i) * 0.5, 0.0], dtype=float) for i in range(n)}
    ds = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=ds)
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        _draw_graph(
            ax=ax,
            graph=graph,
            positions=positions,
            directions=directions,
            show_tensor_labels=True,
            show_index_labels=True,
            config=PlotConfig(
                positions=positions,
                refine_tensor_labels=False,
                separate_index_labels=False,
            ),
            dimensions=2,
            scale=ds,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 800
    finally:
        plt.close(fig)


def test_dense_2d_draw_full_quality_completes_reasonably() -> None:
    """Default label polish should still complete for moderate size (regression guard)."""
    n = 24
    graph = _chain_graph(n)
    positions = {i: np.array([float(i) * 0.5, 0.0], dtype=float) for i in range(n)}
    ds = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=ds)
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        t0 = time.perf_counter()
        _draw_graph(
            ax=ax,
            graph=graph,
            positions=positions,
            directions=directions,
            show_tensor_labels=True,
            show_index_labels=True,
            config=PlotConfig(positions=positions),
            dimensions=2,
            scale=ds,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 45.0, f"dense 2D draw took {elapsed:.1f}s"
    finally:
        plt.close(fig)
