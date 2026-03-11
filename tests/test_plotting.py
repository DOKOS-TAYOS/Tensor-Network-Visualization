from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tensor_network_viz._core.renderer as core_renderer_module
import tensor_network_viz.tensorkrowch.graph as tk_graph_module
import tensor_network_viz.tensorkrowch.renderer as tk_renderer_module
import tensor_network_viz.tensornetwork.graph as tn_graph_module
import tensor_network_viz.tensornetwork.renderer as tn_renderer_module
import tensor_network_viz.viewer as viewer_module
from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.tensorkrowch import (
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
)
from tensor_network_viz.tensorkrowch.graph import (
    _build_graph as _build_tensorkrowch_graph,
)
from tensor_network_viz.tensornetwork import (
    plot_tensornetwork_network_2d,
    plot_tensornetwork_network_3d,
)
from tensor_network_viz.tensornetwork.graph import (
    _build_graph as _build_tensornetwork_graph,
)


class DummyEdge:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.node1: Any = None
        self.node2: Any = None


class DummyTensorKrowchNode:
    def __init__(self, name: str, axes_names: list[str]) -> None:
        self.name = name
        self.axes_names = list(axes_names)
        self.edges: list[DummyEdge | None] = [None] * len(axes_names)


class DummyTensorNetworkNode:
    def __init__(self, name: str, axis_names: list[str]) -> None:
        self.name = name
        self.axis_names = list(axis_names)
        self.edges: list[DummyEdge | None] = [None] * len(axis_names)


class DummyNetwork:
    def __init__(self, *, nodes=None, leaf_nodes=None) -> None:
        if nodes is not None:
            self.nodes = nodes
        if leaf_nodes is not None:
            self.leaf_nodes = leaf_nodes


def connect(
    node1: Any,
    axis1: int,
    node2: Any | None = None,
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


def test_build_tensorkrowch_graph_disconnected_components() -> None:
    a = DummyTensorKrowchNode("A", ["bond"])
    b = DummyTensorKrowchNode("B", ["bond"])
    connect(a, 0, b, 0)
    c = DummyTensorKrowchNode("C", ["bond"])
    d = DummyTensorKrowchNode("D", ["bond"])
    connect(c, 0, d, 0)

    graph = _build_tensorkrowch_graph([a, b, c, d])

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 2
    assert {edge.kind for edge in graph.edges} == {"contraction"}


def test_build_tensorkrowch_graph_accepts_list_of_nodes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    graph = _build_tensorkrowch_graph([left, right])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "contraction"


def test_build_tensorkrowch_graph_sorts_unordered_iterables_stably() -> None:
    left = DummyTensorKrowchNode("B", ["left"])
    right = DummyTensorKrowchNode("A", ["right"])

    graph = _build_tensorkrowch_graph({left, right})

    assert [node.name for node in graph.nodes.values()] == ["A", "B"]


def test_build_tensorkrowch_graph_accepts_generic_iterables_and_deduplicates_nodes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    graph = _build_tensorkrowch_graph(node for node in [left, right, left])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "contraction"


def test_build_tensorkrowch_graph_uses_leaf_nodes_for_dangling_edge() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="edge")

    graph = _build_tensorkrowch_graph(DummyNetwork(leaf_nodes=[node]))

    assert list(graph.nodes.values())[0].name == "A"
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "dangling"
    assert graph.edges[0].label == "left"


def test_tensorkrowch_backend_requires_axes_names() -> None:
    node = DummyTensorNetworkNode("A", ["left"])
    connect(node, 0, name="edge")

    with pytest.raises(TypeError, match="axes_names"):
        _build_tensorkrowch_graph([node])


def test_build_tensornetwork_graph_accepts_iterable_of_nodes() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    graph = _build_tensornetwork_graph([left, right])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "contraction"


def test_build_tensornetwork_graph_sorts_unordered_iterables_stably() -> None:
    left = DummyTensorNetworkNode("B", ["left"])
    right = DummyTensorNetworkNode("A", ["right"])

    graph = _build_tensornetwork_graph({left, right})

    assert [node.name for node in graph.nodes.values()] == ["A", "B"]


def test_build_tensornetwork_graph_accepts_generic_iterables_and_deduplicates_nodes() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    graph = _build_tensornetwork_graph(node for node in [left, right, left])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "contraction"


def test_tensornetwork_backend_requires_axis_names() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="edge")

    with pytest.raises(TypeError, match="axis_names"):
        _build_tensornetwork_graph([node])


def test_plot_tensorkrowch_network_2d_draws_simple_contraction() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left<->right"}
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_2d_accepts_list_of_nodes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensorkrowch_network_2d([left, right])

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left<->right"}
    assert len(ax.lines) == 1


def test_plot_tensornetwork_network_2d_accepts_node_collection() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d({left, right})

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left<->right"}
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_2d_uses_layout_iterations_for_missing_custom_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")
    captured: dict[str, int] = {}

    def fake_compute_layout(graph, dimensions: int, seed: int, *, iterations: int = 220):
        captured["iterations"] = iterations
        assert dimensions == 2
        assert seed == 0
        return {
            node_id: np.array([float(index), float(index)], dtype=float)
            for index, node_id in enumerate(graph.nodes)
        }

    monkeypatch.setattr(core_renderer_module, "_compute_layout", fake_compute_layout)
    config = PlotConfig(
        positions={id(left): (0.0, 0.0)},
        layout_iterations=17,
    )

    plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]), config=config)

    assert captured["iterations"] == 17


def test_plot_tensorkrowch_network_2d_offsets_multiple_edges() -> None:
    left = DummyTensorKrowchNode("A", ["l0", "l1"])
    right = DummyTensorKrowchNode("B", ["r0", "r1"])
    connect(left, 0, right, 0)
    connect(left, 1, right, 1)

    _, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]))

    assert len(ax.lines) == 2
    y0 = ax.lines[0].get_ydata()
    y1 = ax.lines[1].get_ydata()
    assert not np.allclose(y0, y1)


def test_build_tensorkrowch_graph_detects_self_edge() -> None:
    node = DummyTensorKrowchNode("A", ["left", "right"])
    edge = DummyEdge(name="trace")
    edge.node1 = node
    edge.node2 = node
    node.edges[0] = edge
    node.edges[1] = edge

    graph = _build_tensorkrowch_graph(DummyNetwork(nodes=[node]))

    assert len(graph.edges) == 1
    assert graph.edges[0].kind == "self"
    assert graph.edges[0].label == "left<->right"


def test_plot_tensorkrowch_network_3d_returns_3d_axes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0)

    fig, ax = plot_tensorkrowch_network_3d(DummyNetwork(nodes=[left, right]))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 1


def test_plot_tensornetwork_network_3d_returns_3d_axes() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0)

    fig, ax = plot_tensornetwork_network_3d([left, right])

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 1


def test_plot_tensorkrowch_network_3d_rejects_2d_axis() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="edge")
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="3D"):
        plot_tensorkrowch_network_3d(DummyNetwork(nodes=[node]), ax=ax)

    plt.close(fig)


def test_plot_tensornetwork_network_3d_rejects_2d_axis() -> None:
    node = DummyTensorNetworkNode("A", ["left"])
    connect(node, 0, name="edge")
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="plot_tensornetwork_network_3d"):
        plot_tensornetwork_network_3d([node], ax=ax)

    plt.close(fig)


def test_show_tensor_network_displays_selected_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DummyTensorKrowchNode("A", ["left"])
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


def test_show_tensor_network_supports_tensornetwork_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = DummyTensorNetworkNode("A", ["left"])
    connect(node, 0, name="edge")
    called = {"value": False}

    def fake_plot(network, *, config, **kwargs):
        called["value"] = True
        assert network == [node]
        fig, ax = plt.subplots()
        return fig, ax

    import tensor_network_viz.tensornetwork as tensornetwork_module

    monkeypatch.setattr(tensornetwork_module, "plot_tensornetwork_network_2d", fake_plot)
    fig, ax = show_tensor_network(
        [node],
        engine="tensornetwork",
        view="2d",
        config=PlotConfig(figsize=(4, 3)),
        show=False,
    )

    assert called["value"] is True
    assert fig is ax.figure


def test_show_tensor_network_supports_quimb_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    def fake_plot(network, *, config, **kwargs):
        called["value"] = True
        assert network == "quimb-network"
        fig, ax = plt.subplots()
        return fig, ax

    import tensor_network_viz.quimb as quimb_module

    monkeypatch.setattr(quimb_module, "plot_quimb_network_2d", fake_plot)
    fig, ax = show_tensor_network(
        "quimb-network",
        engine="quimb",
        view="2d",
        config=PlotConfig(figsize=(4, 3)),
        show=False,
    )

    assert called["value"] is True
    assert fig is ax.figure


def test_show_tensor_network_supports_tenpy_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    def fake_plot(network, *, config, **kwargs):
        called["value"] = True
        assert network == "tenpy-network"
        fig, ax = plt.subplots()
        return fig, ax

    import tensor_network_viz.tenpy as tenpy_module

    monkeypatch.setattr(tenpy_module, "plot_tenpy_network_2d", fake_plot)
    fig, ax = show_tensor_network(
        "tenpy-network",
        engine="tenpy",
        view="2d",
        config=PlotConfig(figsize=(4, 3)),
        show=False,
    )

    assert called["value"] is True
    assert fig is ax.figure


def test_show_tensor_network_supports_einsum_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = ["pair"]
    called = {"value": False}

    def fake_plot(network, *, config, **kwargs):
        called["value"] = True
        assert network == trace
        fig, ax = plt.subplots()
        return fig, ax

    einsum_module = importlib.import_module("tensor_network_viz.einsum_module")

    monkeypatch.setattr(einsum_module, "plot_einsum_network_2d", fake_plot)
    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        view="2d",
        config=PlotConfig(figsize=(4, 3)),
        show=False,
    )

    assert called["value"] is True
    assert fig is ax.figure


def test_tensornetwork_renderer_does_not_import_tensorkrowch_private_modules() -> None:
    source = Path(tn_renderer_module.__file__).read_text(encoding="utf-8")

    assert "tensorkrowch" not in source


def test_tensorkrowch_renderer_does_not_import_tensornetwork_private_modules() -> None:
    source = Path(tk_renderer_module.__file__).read_text(encoding="utf-8")

    assert "tensornetwork" not in source


def test_tensornetwork_graph_does_not_import_tensorkrowch_private_modules() -> None:
    source = Path(tn_graph_module.__file__).read_text(encoding="utf-8")

    assert "tensorkrowch" not in source


def test_tensorkrowch_graph_does_not_import_tensornetwork_private_modules() -> None:
    source = Path(tk_graph_module.__file__).read_text(encoding="utf-8")

    assert "tensornetwork" not in source
