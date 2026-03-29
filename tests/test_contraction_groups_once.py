"""One ``_group_contractions`` per plot orchestration (axis directions + draw).

Timing notes (duplicate grouping cost **before** deduplication, ``perf_counter``, example dev
machine 2026-03-29): chain graphs with **500 / 2000 / 5000** contraction edges → one
``_group_contractions`` ≈ **0.8 / 3.5 / 8.6 ms**; two calls in a row ≈ **1.5 / 6.4 / 16.4 ms**
— i.e. skipping the second call saves roughly one grouping time. Full ``_plot_graph`` remains
dominated by Matplotlib for typical sizes; see ``test_plot_graph_calls_group_contractions_once``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.config import PlotConfig


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


def test_plot_graph_calls_group_contractions_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensor_network_viz._core.renderer as renderer_mod

    calls: list[int] = []
    orig = renderer_mod._group_contractions

    def wrapped(graph: _GraphData):
        calls.append(1)
        return orig(graph)

    monkeypatch.setattr(renderer_mod, "_group_contractions", wrapped)

    graph = _chain_graph(24)
    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(figsize=(6, 4), refine_tensor_labels=False),
        renderer_name="test",
    )
    plt.close(fig)

    assert calls == [1], f"expected single grouping, got {len(calls)} calls"


def test_group_contractions_offsets_stable_across_calls() -> None:
    from tensor_network_viz._core.contractions import _group_contractions

    g = _chain_graph(2000)
    a = _group_contractions(g)
    b = _group_contractions(g)
    assert a.offsets == b.offsets
    assert a.groups.keys() == b.groups.keys()
