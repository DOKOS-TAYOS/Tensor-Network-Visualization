import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tensor_network_viz.viewer as viewer_module
from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.tensorkrowch import (
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
)
from tensor_network_viz.tensorkrowch.graph import _build_graph


class DummyEdge:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.node1 = None
        self.node2 = None


class DummyNode:
    def __init__(self, name: str, axes_names: list[str]) -> None:
        self.name = name
        self.axes_names = list(axes_names)
        self.edges = [None] * len(axes_names)


class DummyNetwork:
    def __init__(self, *, nodes=None, leaf_nodes=None) -> None:
        if nodes is not None:
            self.nodes = nodes
        if leaf_nodes is not None:
            self.leaf_nodes = leaf_nodes


def connect(
    node1: DummyNode,
    axis1: int,
    node2: DummyNode | None = None,
    axis2: int | None = None,
    *,
    name: str | None = None,
) -> DummyEdge:
    edge = DummyEdge(name=name)
    edge.node1 = node1
    edge.node2 = node2
    node1.edges[axis1] = edge
    if node2 is not None and axis2 is not None:
        node2.edges[axis2] = edge
    return edge


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_build_graph_disconnected_components() -> None:
    """Disconnected graph: two components (A-B and C-D) in one list."""
    a = DummyNode("A", ["bond"])
    b = DummyNode("B", ["bond"])
    connect(a, 0, b, 0)
    c = DummyNode("C", ["bond"])
    d = DummyNode("D", ["bond"])
    connect(c, 0, d, 0)

    graph = _build_graph([a, b, c, d])
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 2  # A-B and C-D, no cross-edges
    kinds = {e.kind for e in graph.edges}
    assert kinds == {"contraction"}


def test_build_graph_accepts_list_of_nodes() -> None:
    """List of nodes works the same as TensorNetwork with nodes."""
    left = DummyNode("A", ["left"])
    right = DummyNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    graph = _build_graph([left, right])
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "contraction"


def test_build_graph_uses_leaf_nodes_for_dangling_edge() -> None:
    node = DummyNode("A", ["left"])
    connect(node, 0, name="edge")
    graph = _build_graph(DummyNetwork(leaf_nodes=[node]))

    assert list(graph.nodes.values())[0].name == "A"
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "dangling"
    assert graph.edges[0].label == "left"


def test_plot_tensorkrowch_network_2d_draws_simple_contraction() -> None:
    left = DummyNode("A", ["left"])
    right = DummyNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left<->right"}
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_2d_accepts_list_of_nodes() -> None:
    """plot_tensorkrowch_network_2d accepts list of nodes directly."""
    left = DummyNode("A", ["left"])
    right = DummyNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensorkrowch_network_2d([left, right])

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left<->right"}
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_2d_offsets_multiple_edges() -> None:
    left = DummyNode("A", ["l0", "l1"])
    right = DummyNode("B", ["r0", "r1"])
    connect(left, 0, right, 0)
    connect(left, 1, right, 1)

    _, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]))

    assert len(ax.lines) == 2
    y0 = ax.lines[0].get_ydata()
    y1 = ax.lines[1].get_ydata()
    assert not np.allclose(y0, y1)


def test_build_graph_detects_self_edge() -> None:
    node = DummyNode("A", ["left", "right"])
    edge = DummyEdge(name="trace")
    edge.node1 = node
    edge.node2 = node
    node.edges[0] = edge
    node.edges[1] = edge

    graph = _build_graph(DummyNetwork(nodes=[node]))

    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "self"
    assert graph.edges[0].label == "left<->right"


def test_plot_tensorkrowch_network_3d_returns_3d_axes() -> None:
    left = DummyNode("A", ["left"])
    right = DummyNode("B", ["right"])
    connect(left, 0, right, 0)

    fig, ax = plot_tensorkrowch_network_3d(DummyNetwork(nodes=[left, right]))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_3d_rejects_2d_axis() -> None:
    node = DummyNode("A", ["left"])
    connect(node, 0, name="edge")
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="3D"):
        plot_tensorkrowch_network_3d(DummyNetwork(nodes=[node]), ax=ax)

    plt.close(fig)


def test_show_tensor_network_displays_selected_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DummyNode("A", ["left"])
    connect(node, 0, name="edge")
    shown = {"value": False}

    def fake_show() -> None:
        shown["value"] = True

    monkeypatch.setattr(viewer_module.plt, "show", fake_show)
    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[node]),
        engine="tensorkrowch",
        view="2d",
        config=PlotConfig(figsize=(4, 3)),
    )

    assert shown["value"] is True
    assert fig is ax.figure
