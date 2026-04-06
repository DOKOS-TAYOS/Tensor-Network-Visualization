"""Contraction scheme metadata and drawing hooks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz import PlotConfig, pair_tensor
from tensor_network_viz._core.graph import (
    _GraphData,
    _make_node,
    _resolve_contraction_scheme_by_name,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.einsum_module.graph import _build_graph as _build_einsum_graph
from tensor_network_viz.tensorkrowch.graph import _build_graph as _build_tensorkrowch_graph


class _DummyEdge:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.node1: object | None = None
        self.node2: object | None = None


class _DummySuccessor:
    def __init__(self, *, node_ref: object | tuple[object, ...], child: object) -> None:
        self.node_ref = node_ref
        self.child = child


class _DummyTensorKrowchNode:
    def __init__(self, name: str, axes_names: tuple[str, ...]) -> None:
        self.name = name
        self.axes_names = list(axes_names)
        self.edges: list[_DummyEdge | None] = [None] * len(self.axes_names)
        self.successors: dict[str, dict[tuple[object, ...], _DummySuccessor]] = {}


class _DummyTensorKrowchNetwork:
    def __init__(
        self,
        *,
        nodes: dict[str, _DummyTensorKrowchNode],
        leaf_nodes: dict[str, _DummyTensorKrowchNode],
        resultant_nodes: dict[str, _DummyTensorKrowchNode] | None = None,
    ) -> None:
        self.nodes = nodes
        self.leaf_nodes = leaf_nodes
        if resultant_nodes is not None:
            self.resultant_nodes = resultant_nodes


def _connect(
    node1: _DummyTensorKrowchNode,
    axis1: int,
    node2: _DummyTensorKrowchNode | None = None,
    axis2: int | None = None,
    *,
    name: str | None = None,
) -> _DummyEdge:
    edge = _DummyEdge(name=name)
    edge.node1 = node1
    edge.node2 = node2
    node1.edges[axis1] = edge
    if node2 is not None and axis2 is not None:
        node2.edges[axis2] = edge
    return edge


def _attach_successor(
    source: _DummyTensorKrowchNode,
    *,
    op_name: str,
    parents: tuple[_DummyTensorKrowchNode, ...],
    child: _DummyTensorKrowchNode,
) -> None:
    source.successors.setdefault(op_name, {})[parents] = _DummySuccessor(
        node_ref=parents,
        child=child,
    )


def _node_id_by_name(graph: _GraphData, name: str) -> int:
    return next(node_id for node_id, node in graph.nodes.items() if node.name == name)


def test_einsum_graph_records_contraction_steps_chain() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
    ]
    graph = _build_einsum_graph(trace)
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
    graph = _build_einsum_graph(trace)

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
    graph = _build_einsum_graph(trace)
    assert graph.contraction_steps == (
        frozenset({0, 1}),
        frozenset({2, 3}),
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


def test_plot_graph_contraction_scheme_is_dynamic_only() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_einsum_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(figsize=(4, 3), show_contraction_scheme=True),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme",
    )
    patches = [p for p in ax.patches if p.get_gid() == "tnv_contraction_scheme"]
    assert patches == []
    labels = [t for t in ax.texts if t.get_gid() == "tnv_contraction_scheme_label"]
    assert labels == []
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None
    assert viewer.slider is not None
    assert viewer.slider.ax.get_visible()
    assert fig is not None


def test_plot_graph_scheme_override_by_name() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    graph = _build_einsum_graph(trace)
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
    assert patches == []
    labels = [t for t in ax.texts if t.get_gid() == "tnv_contraction_scheme_label"]
    assert labels == []
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None
    assert viewer.current_step == 1


def test_tensorkrowch_graph_records_single_auto_contraction_step() -> None:
    left = _DummyTensorKrowchNode("A", ("a", "b"))
    right = _DummyTensorKrowchNode("B", ("b", "c"))
    result = _DummyTensorKrowchNode("contract_edges", ("a", "c"))
    _connect(left, 0, result, 0, name="A[a] <-> None")
    _connect(left, 1, right, 0, name="A[b] <-> B[b]")
    _connect(right, 1, result, 1, name="B[c] <-> None")
    _attach_successor(left, op_name="contract_edges", parents=(left, right), child=result)
    network = _DummyTensorKrowchNetwork(
        nodes={"A": left, "B": right, "contract_edges": result},
        leaf_nodes={"A": left, "B": right},
        resultant_nodes={"contract_edges": result},
    )

    graph = _build_tensorkrowch_graph(network)

    left_id = _node_id_by_name(graph, "A")
    right_id = _node_id_by_name(graph, "B")
    assert graph.contraction_steps == (frozenset({left_id, right_id}),)
    assert graph.contraction_step_metrics is None


def test_tensorkrowch_graph_records_eventwise_auto_contraction_steps() -> None:
    left = _DummyTensorKrowchNode("A", ("a", "b"))
    middle = _DummyTensorKrowchNode("B", ("b", "c"))
    right = _DummyTensorKrowchNode("C", ("c", "d"))
    result0 = _DummyTensorKrowchNode("contract_edges_0", ("a", "c"))
    result1 = _DummyTensorKrowchNode("contract_edges_1", ("a", "d"))
    _connect(left, 0, result0, 0, name="A[a] <-> None")
    _connect(left, 1, middle, 0, name="A[b] <-> B[b]")
    _connect(middle, 1, result0, 1, name="B[c] <-> None")
    _connect(result0, 0, result1, 0, name="A[a] <-> None")
    _connect(result0, 1, right, 0, name="B[c] <-> C[c]")
    _connect(right, 1, result1, 1, name="C[d] <-> None")
    _attach_successor(left, op_name="contract_edges", parents=(left, middle), child=result0)
    _attach_successor(
        result0,
        op_name="contract_edges",
        parents=(result0, right),
        child=result1,
    )
    network = _DummyTensorKrowchNetwork(
        nodes={
            "A": left,
            "B": middle,
            "C": right,
            "contract_edges_0": result0,
            "contract_edges_1": result1,
        },
        leaf_nodes={"A": left, "B": middle, "C": right},
        resultant_nodes={"contract_edges_0": result0, "contract_edges_1": result1},
    )

    graph = _build_tensorkrowch_graph(network)

    left_id = _node_id_by_name(graph, "A")
    middle_id = _node_id_by_name(graph, "B")
    right_id = _node_id_by_name(graph, "C")
    assert graph.contraction_steps == (
        frozenset({left_id, middle_id}),
        frozenset({left_id, middle_id, right_id}),
    )
    assert graph.contraction_step_metrics is None


def test_tensorkrowch_graph_records_parallel_eventwise_steps() -> None:
    left = _DummyTensorKrowchNode("A", ("a", "b"))
    middle = _DummyTensorKrowchNode("B", ("b", "c"))
    right = _DummyTensorKrowchNode("C", ("d", "e"))
    far = _DummyTensorKrowchNode("D", ("e", "f"))
    result0 = _DummyTensorKrowchNode("contract_edges_0", ("a", "c"))
    result1 = _DummyTensorKrowchNode("contract_edges_1", ("d", "f"))
    result2 = _DummyTensorKrowchNode("contract_edges_2", ("a", "f"))

    _connect(left, 0, result0, 0, name="A[a] <-> None")
    _connect(left, 1, middle, 0, name="A[b] <-> B[b]")
    _connect(middle, 1, result0, 1, name="B[c] <-> None")

    _connect(right, 0, result1, 0, name="C[d] <-> None")
    _connect(right, 1, far, 0, name="C[e] <-> D[e]")
    _connect(far, 1, result1, 1, name="D[f] <-> None")

    _connect(result0, 0, result2, 0, name="A[a] <-> None")
    _connect(result1, 1, result2, 1, name="D[f] <-> None")

    _attach_successor(left, op_name="contract_edges", parents=(left, middle), child=result0)
    _attach_successor(right, op_name="contract_edges", parents=(right, far), child=result1)
    _attach_successor(
        result0,
        op_name="contract_edges",
        parents=(result0, result1),
        child=result2,
    )
    network = _DummyTensorKrowchNetwork(
        nodes={
            "A": left,
            "B": middle,
            "C": right,
            "D": far,
            "contract_edges_0": result0,
            "contract_edges_1": result1,
            "contract_edges_2": result2,
        },
        leaf_nodes={"A": left, "B": middle, "C": right, "D": far},
        resultant_nodes={
            "contract_edges_0": result0,
            "contract_edges_1": result1,
            "contract_edges_2": result2,
        },
    )

    graph = _build_tensorkrowch_graph(network)

    left_id = _node_id_by_name(graph, "A")
    middle_id = _node_id_by_name(graph, "B")
    right_id = _node_id_by_name(graph, "C")
    far_id = _node_id_by_name(graph, "D")
    assert graph.contraction_steps == (
        frozenset({left_id, middle_id}),
        frozenset({right_id, far_id}),
        frozenset({left_id, middle_id, right_id, far_id}),
    )


def test_tensorkrowch_graph_ignores_helper_nodes_when_recovering_contractions() -> None:
    left = _DummyTensorKrowchNode("A", ("a",))
    helper = _DummyTensorKrowchNode("stack_0", ("a",))
    right = _DummyTensorKrowchNode("B", ("a", "b"))
    result = _DummyTensorKrowchNode("contract_edges", ("b",))
    _connect(left, 0, helper, 0, name="lift")
    _connect(helper, 0, right, 0, name="stack_0[a] <-> B[a]")
    _connect(right, 1, result, 0, name="B[b] <-> None")
    _attach_successor(left, op_name="stack", parents=(left,), child=helper)
    _attach_successor(helper, op_name="contract_edges", parents=(helper, right), child=result)
    network = _DummyTensorKrowchNetwork(
        nodes={"A": left, "B": right, "stack_0": helper, "contract_edges": result},
        leaf_nodes={"A": left, "B": right},
        resultant_nodes={"contract_edges": result},
    )

    graph = _build_tensorkrowch_graph(network)

    left_id = _node_id_by_name(graph, "A")
    right_id = _node_id_by_name(graph, "B")
    assert graph.contraction_steps == (frozenset({left_id, right_id}),)
    assert graph.contraction_step_metrics is None


def test_tensorkrowch_graph_skips_auto_scheme_without_resultant_history() -> None:
    left = _DummyTensorKrowchNode("A", ("a",))
    right = _DummyTensorKrowchNode("B", ("a",))
    _connect(left, 0, right, 0, name="bond")
    network = _DummyTensorKrowchNetwork(
        nodes={"A": left, "B": right},
        leaf_nodes={"A": left, "B": right},
        resultant_nodes={},
    )

    graph = _build_tensorkrowch_graph(network)

    assert graph.contraction_steps is None
    assert graph.contraction_step_metrics is None


def test_tensorkrowch_graph_skips_auto_scheme_when_parent_lineage_is_ambiguous() -> None:
    left = _DummyTensorKrowchNode("A", ("a",))
    helper = _DummyTensorKrowchNode("stack_0", ("a",))
    right = _DummyTensorKrowchNode("B", ("a",))
    result = _DummyTensorKrowchNode("contract_edges", ())
    _connect(left, 0, helper, 0, name="lift")
    _connect(helper, 0, right, 0, name="bond")
    _attach_successor(helper, op_name="contract_edges", parents=(helper, right), child=result)
    network = _DummyTensorKrowchNetwork(
        nodes={"A": left, "B": right, "stack_0": helper, "contract_edges": result},
        leaf_nodes={"A": left, "B": right},
        resultant_nodes={"contract_edges": result},
    )

    graph = _build_tensorkrowch_graph(network)

    assert graph.contraction_steps is None
    assert graph.contraction_step_metrics is None
