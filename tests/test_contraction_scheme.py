"""Contraction scheme metadata and drawing hooks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.patches import FancyBboxPatch

from tensor_network_viz import PlotConfig, pair_tensor
from tensor_network_viz._core.graph import (
    _GraphData,
    _make_node,
    _resolve_contraction_scheme_by_name,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.einsum_module.graph import _build_graph


def test_einsum_graph_records_contraction_steps_chain() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
    ]
    graph = _build_graph(trace)
    assert graph.contraction_steps is not None
    assert len(graph.contraction_steps) == 3
    assert graph.contraction_steps[0] == frozenset({0, 1})
    # Running union: second step includes A0/x0 lineage plus A1.
    assert graph.contraction_steps[1] == frozenset({0, 1, 2})
    assert graph.contraction_steps[2] == frozenset({0, 1, 2, 3})
    for step in graph.contraction_steps:
        assert len(step) >= 1


def test_einsum_peps_trace_includes_environment_tensors_in_scheme() -> None:
    """PEPS-style operands (P** / x**) use lineage union so x** stay highlighted with P**."""
    trace = [
        pair_tensor("P00", "x00", "r0", "sad,s->ad"),
        pair_tensor("r0", "P01", "r1", "ad,atbe->dtbe"),
    ]
    graph = _build_graph(trace)

    def _id_for(name: str) -> int:
        return next(nid for nid, n in graph.nodes.items() if n.name == name)

    id_p00, id_x00, id_p01 = _id_for("P00"), _id_for("x00"), _id_for("P01")
    assert graph.contraction_steps is not None
    assert graph.contraction_steps[0] == frozenset({id_p00, id_x00})
    assert graph.contraction_steps[1] == frozenset({id_p00, id_x00, id_p01})


def test_einsum_contraction_steps_parallel_branches_then_merge() -> None:
    trace = [
        pair_tensor("A", "B", "ab", "i,j->ij"),
        pair_tensor("C", "D", "cd", "k,l->kl"),
        pair_tensor("ab", "cd", "out", "ij,kl->ijkl"),
    ]
    graph = _build_graph(trace)
    assert graph.contraction_steps == (
        frozenset({0, 1}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 3}),
    )


def test_resolve_contraction_scheme_by_name_maps_steps() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("i",)),
            1: _make_node("B", ("i",)),
        },
        edges=(),
    )
    resolved = _resolve_contraction_scheme_by_name(graph, (("A", "B"),))
    assert resolved == (frozenset({0, 1}),)


def test_resolve_contraction_scheme_by_name_unknown_raises() -> None:
    graph = _GraphData(nodes={0: _make_node("A", ("i",))}, edges=())
    with pytest.raises(ValueError, match="unknown tensor name"):
        _resolve_contraction_scheme_by_name(graph, (("A", "missing"),))


def test_resolve_contraction_scheme_by_name_duplicate_name_raises() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("i",)),
            1: _make_node("A", ("j",)),
        },
        edges=(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_contraction_scheme_by_name(graph, (("A",),))


def test_plot_graph_draws_scheme_patches_2d() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(figsize=(4, 3), show_contraction_scheme=True),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme",
    )
    patches = [p for p in ax.patches if p.get_gid() == "tnv_contraction_scheme"]
    assert len(patches) == 2
    assert all(isinstance(p, FancyBboxPatch) for p in patches)
    labels = [t for t in ax.texts if t.get_gid() == "tnv_contraction_scheme_label"]
    assert labels == []
    assert fig is not None


def test_plot_graph_scheme_override_by_name() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=True,
            contraction_scheme_by_name=(("A0",),),
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_override",
    )
    patches = [p for p in ax.patches if p.get_gid() == "tnv_contraction_scheme"]
    assert len(patches) == 1
    assert isinstance(patches[0], FancyBboxPatch)
    labels = [t for t in ax.texts if t.get_gid() == "tnv_contraction_scheme_label"]
    assert labels == []
