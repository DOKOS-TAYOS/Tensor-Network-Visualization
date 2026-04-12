import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz import PlotConfig
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

    fig, ax = plot_quimb_network_2d(
        [left, right],
        config=PlotConfig(show_tensor_labels=True, show_index_labels=True),
    )

    labels = {text.get_text() for text in ax.texts if text.get_text()}
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


def test_plot_quimb_network_2d_draws_hypergraph_with_virtual_hub_marker() -> None:
    from matplotlib.colors import to_rgba

    from plotting_helpers import (
        line_collection_segment_count,
        point_collection_facecolors,
        triangle_marker_point_count,
    )

    a = _make_tensor(inds=("bond",), tag="A")
    b = _make_tensor(inds=("bond",), tag="B")
    c = _make_tensor(inds=("bond",), tag="C")
    config = PlotConfig(show_tensor_labels=True, show_index_labels=True)

    fig, ax = plot_quimb_network_2d(
        qtn.TensorNetwork([a, b, c]),
        config=config,
    )

    labels = {text.get_text() for text in ax.texts if text.get_text()}
    assert fig is ax.figure
    assert labels == {"A", "B", "C", "bond"}
    assert sum(1 for t in ax.texts if t.get_text() == "bond") == 6
    assert line_collection_segment_count(ax) == 3
    assert triangle_marker_point_count(ax) == 1
    assert point_collection_facecolors(ax) == [
        tuple(float(value) for value in to_rgba(config.dangling_edge_color))
    ]


def test_plot_quimb_network_3d_draws_hypergraph() -> None:
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

    from plotting_helpers import path3d_collection_facecolors, path3d_triangle_marker_point_count

    a = _make_tensor(inds=("bond",), tag="A")
    b = _make_tensor(inds=("bond",), tag="B")
    c = _make_tensor(inds=("bond",), tag="C")
    config = PlotConfig(show_tensor_labels=True, show_index_labels=True)

    fig, ax = plot_quimb_network_3d(
        qtn.TensorNetwork([a, b, c]),
        config=config,
    )

    labels = {text.get_text() for text in ax.texts if text.get_text()}
    assert fig is ax.figure
    assert ax.name == "3d"
    assert labels == {"A", "B", "C", "bond"}
    assert sum(1 for t in ax.texts if t.get_text() == "bond") == 6
    assert len(ax.lines) == 3
    assert path3d_triangle_marker_point_count(ax) == 1
    assert path3d_collection_facecolors(ax) == [
        tuple(float(value) for value in to_rgba(config.dangling_edge_color))
    ]
    triangle_zorders = [
        float(collection.get_zorder())
        for collection in ax.collections
        if isinstance(collection, Path3DCollection)
        and any(
            np.asarray(path.vertices, dtype=float).shape[0] == 4 for path in collection.get_paths()
        )
    ]
    node_zorders = [
        float(collection.get_zorder())
        for collection in ax.collections
        if isinstance(collection, Poly3DCollection)
    ]
    assert triangle_zorders
    assert node_zorders
    assert min(triangle_zorders) > max(node_zorders)
