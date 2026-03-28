import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.quimb.graph import _build_graph as _build_quimb_graph
from tensor_network_viz.quimb.graph import _get_network_tensors as _get_quimb_tensors

qtn = pytest.importorskip("quimb.tensor")


def _make_tensor(*, inds: tuple[str, ...], tag: str | None = None):
    kwargs = {"tags": {tag}} if tag is not None else {}
    shape = tuple(2 for _ in inds) or (1,)
    return qtn.Tensor(data=np.ones(shape, dtype=float), inds=inds, **kwargs)


def test_build_quimb_graph_accepts_tensor_network() -> None:
    left = _make_tensor(inds=("left", "bond"), tag="A")
    right = _make_tensor(inds=("bond", "right"), tag="B")

    graph = _build_quimb_graph(qtn.TensorNetwork([left, right]))

    assert {node.name for node in graph.nodes.values()} == {"A", "B"}
    assert {edge.kind for edge in graph.edges} == {"contraction", "dangling"}
    assert {edge.label for edge in graph.edges if edge.label} >= {"left", "right"}
    assert any(e.kind == "contraction" and e.name == "bond" for e in graph.edges)


def test_get_quimb_tensors_sorts_unordered_iterables_stably() -> None:
    left = _make_tensor(inds=("left",), tag="B")
    right = _make_tensor(inds=("right",), tag="A")

    tensors = _get_quimb_tensors({left, right})

    assert [next(iter(tensor.tags)) for tensor in tensors] == ["A", "B"]


def test_build_quimb_graph_assigns_fallback_tensor_names() -> None:
    left = _make_tensor(inds=("bond",))
    right = _make_tensor(inds=("bond",))

    graph = _build_quimb_graph([left, right])

    assert [node.name for node in graph.nodes.values()] == ["T0", "T1"]


def test_build_quimb_graph_rewrites_hyperedges_through_virtual_hub() -> None:
    a = _make_tensor(inds=("bond",), tag="A")
    b = _make_tensor(inds=("bond",), tag="B")
    c = _make_tensor(inds=("bond",), tag="C")

    graph = _build_quimb_graph(qtn.TensorNetwork([a, b, c]))

    virtual_nodes = [(node_id, node) for node_id, node in graph.nodes.items() if node.is_virtual]
    assert len(virtual_nodes) == 1

    hub_id, hub = virtual_nodes[0]
    assert hub.name == ""
    assert hub.label == "bond"
    assert hub.degree == 3
    assert {node.name for node in graph.nodes.values() if not node.is_virtual} == {"A", "B", "C"}

    contraction_edges = [edge for edge in graph.edges if edge.kind == "contraction"]
    assert len(contraction_edges) == 3
    assert all(hub_id in edge.node_ids for edge in contraction_edges)
    assert all(edge.label is None for edge in contraction_edges)


def test_plot_quimb_network_2d_draws_simple_contraction() -> None:
    left = _make_tensor(inds=("bond",), tag="A")
    right = _make_tensor(inds=("bond",), tag="B")

    fig, ax = plot_quimb_network_2d([left, right])

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "bond"}
    from plotting_helpers import line_collection_segment_count

    assert line_collection_segment_count(ax) == 1


def test_plot_quimb_network_3d_returns_3d_axes() -> None:
    left = _make_tensor(inds=("bond",), tag="A")
    right = _make_tensor(inds=("bond",), tag="B")

    fig, ax = plot_quimb_network_3d([left, right])

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 1


def test_plot_quimb_network_2d_draws_hypergraph_without_showing_virtual_hub() -> None:
    from plotting_helpers import line_collection_segment_count

    a = _make_tensor(inds=("bond",), tag="A")
    b = _make_tensor(inds=("bond",), tag="B")
    c = _make_tensor(inds=("bond",), tag="C")

    fig, ax = plot_quimb_network_2d(qtn.TensorNetwork([a, b, c]))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels == {"A", "B", "C", "bond"}
    assert sum(1 for t in ax.texts if t.get_text() == "bond") == 6
    assert line_collection_segment_count(ax) == 3


def test_plot_quimb_network_3d_draws_hypergraph() -> None:
    a = _make_tensor(inds=("bond",), tag="A")
    b = _make_tensor(inds=("bond",), tag="B")
    c = _make_tensor(inds=("bond",), tag="C")

    fig, ax = plot_quimb_network_3d(qtn.TensorNetwork([a, b, c]))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert ax.name == "3d"
    assert labels == {"A", "B", "C", "bond"}
    assert sum(1 for t in ax.texts if t.get_text() == "bond") == 6
    assert len(ax.lines) == 3
