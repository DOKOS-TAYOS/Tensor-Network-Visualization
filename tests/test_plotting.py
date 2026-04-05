from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent, MouseButton, MouseEvent
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

import tensor_network_viz._core.renderer as core_renderer_module
import tensor_network_viz._interaction.controller as interaction_controller_module
import tensor_network_viz.tensorkrowch.graph as tk_graph_module
import tensor_network_viz.tensorkrowch.renderer as tk_renderer_module
import tensor_network_viz.tensornetwork.graph as tn_graph_module
import tensor_network_viz.tensornetwork.renderer as tn_renderer_module
import tensor_network_viz.viewer as viewer_module
from plotting_helpers import (
    line_collection_segment_count,
    line_collection_segments,
    patch_collection_circle_count,
    path3d_collection_facecolors,
    path3d_collection_point_count,
    path3d_collection_sizes,
    path_collection_point_count,
    point_collection_facecolors,
    point_collection_sizes,
    poly3d_node_collection_count,
)
from tensor_network_viz import EinsumTrace, PlotConfig, einsum, show_tensor_network
from tensor_network_viz._core import _draw_common
from tensor_network_viz._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.einsum_module.trace import pair_tensor
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


def _widget_center_event(fig: matplotlib.figure.Figure, artist: object) -> MouseEvent:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer)  # type: ignore[attr-defined]
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    return MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)


def _click_checkbutton(checkbuttons: Any, index: int) -> None:
    event = _widget_center_event(checkbuttons.ax.figure, checkbuttons.labels[index])
    checkbuttons._clicked(event)


def _checkbutton_index(checkbuttons: Any, label_text: str) -> int:
    labels = [label.get_text() for label in checkbuttons.labels]
    return labels.index(label_text)


def _fire_close_event(fig: matplotlib.figure.Figure) -> None:
    fig.canvas.callbacks.process("close_event", CloseEvent("close_event", fig.canvas))


def _build_einsum_trace_for_inspector(*, keep_intermediates: bool = True) -> EinsumTrace:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    mid = np.arange(12, dtype=float).reshape(3, 4)
    right = np.arange(8, dtype=float).reshape(4, 2)

    trace.bind("Left", left)
    trace.bind("Mid", mid)
    trace.bind("Right", right)
    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    keepalive = [left, mid, right, r1]
    if keep_intermediates:
        keepalive.append(r0)
    trace._test_keepalive = keepalive  # type: ignore[attr-defined]
    return trace


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


def test_plot_config_show_nodes_defaults_to_true() -> None:
    assert PlotConfig().show_nodes is True


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

    fig, ax = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_tensor_labels=True, show_index_labels=True),
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left", "right"}
    assert line_collection_segment_count(ax) == 1


def test_plot_tensorkrowch_network_2d_accepts_list_of_nodes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensorkrowch_network_2d(
        [left, right],
        config=PlotConfig(show_tensor_labels=True, show_index_labels=True),
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left", "right"}
    assert line_collection_segment_count(ax) == 1


def test_plot_tensornetwork_network_2d_accepts_node_collection() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d(
        {left, right},
        config=PlotConfig(show_tensor_labels=True, show_index_labels=True),
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A", "B", "left", "right"}
    assert line_collection_segment_count(ax) == 1


def test_plot_tensornetwork_network_2d_edges_have_black_outline() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d({left, right})

    line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(line_collections) == 1
    path_effects = line_collections[0].get_path_effects()
    assert path_effects
    stroke = path_effects[0]
    assert getattr(stroke, "_gc", {}).get("foreground") == "black"


def test_plot_graph_2d_keeps_virtual_dangling_index_labels_in_view() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("left",)),
            1: _make_node("hub", ("left", "out"), is_virtual=True),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "left"),
                _EdgeEndpoint(1, 0, "left"),
                name="bond",
            ),
            _make_dangling_edge(
                _EdgeEndpoint(1, 1, "out"),
                name="out",
            ),
        ),
    )

    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            positions={
                0: (0.0, 0.0),
                1: (4.0, 0.0),
            },
        ),
        show_tensor_labels=False,
        show_index_labels=True,
        renderer_name="test_virtual_viewport",
    )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    label_positions = [text.get_position() for text in ax.texts if text.get_text() == "out"]
    assert fig is ax.figure
    assert label_positions
    label_x, label_y = label_positions[0]
    assert x0 <= label_x <= x1
    assert y0 <= label_y <= y1


def test_plot_tensornetwork_network_2d_hover_labels_skips_static_label_artists() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d(
        {left, right},
        config=PlotConfig(figsize=(4, 3), hover_labels=True),
    )
    try:
        gids = {t.get_gid() for t in ax.texts if t.get_gid()}
        assert _draw_common._TENSOR_LABEL_GID not in gids
        assert _draw_common._EDGE_INDEX_LABEL_GID not in gids
        assert getattr(fig, "_tensor_network_viz_hover_cid", None) is not None
    finally:
        plt.close(fig)


def test_plot_tensornetwork_network_3d_hover_labels_skips_static_label_artists() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_3d(
        {left, right},
        config=PlotConfig(figsize=(4, 3), hover_labels=True),
    )
    try:
        gids = {t.get_gid() for t in ax.texts if t.get_gid()}
        assert _draw_common._TENSOR_LABEL_GID not in gids
        assert _draw_common._EDGE_INDEX_LABEL_GID not in gids
        assert getattr(fig, "_tensor_network_viz_hover_cid", None) is not None
    finally:
        plt.close(fig)


def test_plot_tensornetwork_network_2d_hover_labels_can_coexist_with_static_labels() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d(
        {left, right},
        config=PlotConfig(
            figsize=(4, 3),
            hover_labels=True,
            show_tensor_labels=True,
            show_index_labels=True,
        ),
    )
    try:
        gids = {t.get_gid() for t in ax.texts if t.get_gid()}
        assert _draw_common._TENSOR_LABEL_GID in gids
        assert _draw_common._EDGE_INDEX_LABEL_GID in gids
        assert getattr(fig, "_tensor_network_viz_hover_cid", None) is not None
    finally:
        plt.close(fig)


def test_show_tensor_network_default_interactive_controls_start_in_2d() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert fig is ax.figure
    assert ax.name != "3d"
    assert controls.current_view == "2d"
    assert controls.hover_on is True
    assert controls.tensor_labels_on is False
    assert controls.edge_labels_on is False


def test_show_tensor_network_interactive_controls_include_nodes_toggle() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    assert [label.get_text() for label in controls._checkbuttons.labels][:4] == [
        "Hover",
        "Nodes",
        "Tensor labels",
        "Edge labels",
    ]
    assert controls.nodes_on is True


def test_show_tensor_network_builds_3d_view_lazily_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz.tensorkrowch as tensorkrowch_module

    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    calls = {"3d": 0}
    original = tensorkrowch_module.plot_tensorkrowch_network_3d

    def counting_plot(*args: Any, **kwargs: Any):
        calls["3d"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tensorkrowch_module, "plot_tensorkrowch_network_3d", counting_plot)

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert calls["3d"] == 0

    controls.set_view("3d")
    assert calls["3d"] == 1

    controls.set_view("2d")
    controls.set_view("3d")
    assert calls["3d"] == 1


def test_show_tensor_network_reuses_cached_axes_without_creating_empty_overlays() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._radio_ax is not None
    assert controls._check_ax is not None

    initial_axes_count = len(fig.axes)
    cached_2d_ax = controls._view_caches["2d"].ax
    assert cached_2d_ax is not None

    controls.set_view("3d")
    cached_3d_ax = controls._view_caches["3d"].ax
    assert cached_3d_ax is not None
    after_first_switch_count = len(fig.axes)
    assert after_first_switch_count == initial_axes_count + 1
    assert len(cached_3d_ax.lines) >= 1

    controls.set_view("2d")

    assert len(fig.axes) == after_first_switch_count
    assert getattr(fig, "_tensor_network_viz_active_axes", None) is cached_2d_ax
    assert line_collection_segment_count(cached_2d_ax) == 1

    controls.set_view("3d")

    assert len(fig.axes) == after_first_switch_count
    assert getattr(fig, "_tensor_network_viz_active_axes", None) is cached_3d_ax
    assert len(cached_3d_ax.lines) >= 1


def test_show_tensor_network_reuses_tensor_and_edge_label_artists_when_toggled() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)
    tensor_label_ids_before = tuple(
        id(text) for text in controls.current_scene.tensor_label_artists
    )
    edge_label_ids_before = tuple(id(text) for text in controls.current_scene.edge_label_artists)
    assert tensor_label_ids_before
    assert edge_label_ids_before
    assert getattr(fig, "_tensor_network_viz_hover_cid", None) is not None

    controls.set_tensor_labels_enabled(False)
    controls.set_edge_labels_enabled(False)
    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)

    assert tensor_label_ids_before == tuple(
        id(text) for text in controls.current_scene.tensor_label_artists
    )
    assert edge_label_ids_before == tuple(
        id(text) for text in controls.current_scene.edge_label_artists
    )


def test_show_tensor_network_nodes_toggle_reuses_cached_view_and_label_artists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._interactive_scene as interactive_scene_module

    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)
    scene_before = controls.current_scene
    tensor_label_ids_before = tuple(id(text) for text in scene_before.tensor_label_artists)
    edge_label_ids_before = tuple(id(text) for text in scene_before.edge_label_artists)

    def _unexpected_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("node toggle should not rebuild label descriptors")

    def _unexpected_render(*args: object, **kwargs: object) -> object:
        raise AssertionError("node toggle should not rerender the view")

    monkeypatch.setattr(
        interactive_scene_module,
        "_build_tensor_label_descriptors",
        _unexpected_rebuild,
    )
    monkeypatch.setattr(
        interactive_scene_module,
        "_build_edge_label_descriptors",
        _unexpected_rebuild,
    )
    monkeypatch.setattr(controls, "_render_view", _unexpected_render)

    controls.set_nodes_enabled(False)
    controls.set_nodes_enabled(True)

    assert controls.current_scene is scene_before
    assert tensor_label_ids_before == tuple(
        id(text) for text in controls.current_scene.tensor_label_artists
    )
    assert edge_label_ids_before == tuple(
        id(text) for text in controls.current_scene.edge_label_artists
    )


def test_show_tensor_network_builds_compact_node_artists_once_per_view() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_nodes_enabled(False)
    scene = controls.current_scene
    compact_bundle = scene.node_artist_bundles["compact"]
    assert scene.active_node_mode == "compact"

    controls.set_nodes_enabled(True)
    controls.set_nodes_enabled(False)

    assert scene.node_artist_bundles["compact"] is compact_bundle
    assert scene.active_node_mode == "compact"


def test_show_tensor_network_nodes_toggle_persists_across_2d_and_3d_views() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_nodes_enabled(False)
    controls.set_view("3d")
    scene_3d = controls.current_scene
    compact_bundle_3d = scene_3d.node_artist_bundles["compact"]
    assert scene_3d.active_node_mode == "compact"

    controls.set_nodes_enabled(True)
    controls.set_nodes_enabled(False)
    assert scene_3d.node_artist_bundles["compact"] is compact_bundle_3d

    controls.set_view("2d")
    assert controls.current_scene.active_node_mode == "compact"


def test_show_tensor_network_discards_stale_hover_annotations_when_switching_views() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    cached_2d_ax = controls._view_caches["2d"].ax
    assert cached_2d_ax is not None

    ann_2d = getattr(fig, "_tensor_network_viz_hover_ann", None)
    assert ann_2d is not None
    ann_2d.set_text("stale-2d")
    ann_2d.set_visible(True)

    controls.set_view("3d")
    cached_3d_ax = controls._view_caches["3d"].ax
    assert cached_3d_ax is not None
    controls.set_view("2d")

    texts_2d = {text.get_text() for text in cached_2d_ax.texts if text.get_visible()}
    assert "stale-2d" not in texts_2d

    ann_3d = getattr(fig, "_tensor_network_viz_hover_ann", None)
    assert ann_3d is not None
    ann_3d.set_text("stale-3d")
    ann_3d.set_visible(True)

    controls.set_view("3d")

    texts_3d = {text.get_text() for text in cached_3d_ax.texts if text.get_visible()}
    assert "stale-3d" not in texts_3d


def test_show_tensor_network_places_view_selector_between_options_and_playback_slider() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_playback=True,
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._radio_ax is not None
    assert controls._check_ax is not None
    scene_controls = controls.current_scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None
    assert scene_controls._viewer.slider is not None
    assert scene_controls._viewer._btn_play is not None

    radio_bounds = controls._radio_ax.get_position().bounds
    check_bounds = controls._check_ax.get_position().bounds
    slider_bounds = scene_controls._viewer.slider.ax.get_position().bounds
    play_bounds = scene_controls._viewer._btn_play.ax.get_position().bounds

    radio_right = radio_bounds[0] + radio_bounds[2]
    check_right = check_bounds[0] + check_bounds[2]
    slider_right = slider_bounds[0] + slider_bounds[2]

    assert check_bounds[2] <= 0.21
    assert radio_bounds[0] >= check_right - 0.02
    assert radio_bounds[2] <= 0.09
    assert slider_bounds[0] >= radio_right - 0.02
    assert abs(radio_bounds[1] - check_bounds[1]) < 0.02
    assert play_bounds[0] > slider_right


def test_show_tensor_network_disables_hidden_view_slider_widgets_after_switch() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_playback=True,
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    controls.set_view("3d")

    current_scene_controls = controls.current_scene.contraction_controls
    other_scene = controls._view_caches["2d"].scene
    assert current_scene_controls is not None
    assert current_scene_controls._viewer is not None
    assert current_scene_controls._viewer.slider is not None
    assert other_scene is not None
    assert other_scene.contraction_controls is not None
    assert other_scene.contraction_controls._viewer is not None
    assert other_scene.contraction_controls._viewer.slider is not None

    hidden_slider = other_scene.contraction_controls._viewer.slider
    visible_slider = current_scene_controls._viewer.slider

    assert hidden_slider.active is False
    assert visible_slider.active is True

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = visible_slider.ax.get_window_extent(renderer)
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=MouseButton.LEFT)

    fig.canvas.callbacks.process("button_press_event", press)
    fig.canvas.callbacks.process("button_release_event", release)


def test_show_tensor_network_hides_cost_panel_for_inactive_view_after_switch() -> None:
    trace = [
        pair_tensor(
            "A0",
            "x0",
            "r0",
            "pa,p->a",
            metadata={
                "left_shape": (5, 7),
                "right_shape": (5,),
                "result_shape": (7,),
            },
        ),
        pair_tensor(
            "r0",
            "A1",
            "r1",
            "a,apb->pb",
            metadata={
                "left_shape": (7,),
                "right_shape": (7, 11, 13),
                "result_shape": (11, 13),
            },
        ),
    ]

    fig, _ax = show_tensor_network(
        trace,
        engine="einsum",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_playback=True,
            contraction_scheme_cost_hover=True,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    current_scene_controls = controls.current_scene.contraction_controls
    assert current_scene_controls is not None
    assert current_scene_controls._viewer is not None
    assert current_scene_controls._viewer._cost_panel_ax is not None
    assert current_scene_controls._viewer._cost_panel_ax.get_visible()

    controls.set_view("3d")

    current_scene_controls = controls.current_scene.contraction_controls
    other_scene = controls._view_caches["2d"].scene
    assert current_scene_controls is not None
    assert current_scene_controls._viewer is not None
    assert current_scene_controls._viewer._cost_panel_ax is not None
    assert current_scene_controls._viewer._cost_panel_ax.get_visible()
    assert other_scene is not None
    assert other_scene.contraction_controls is not None
    assert other_scene.contraction_controls._viewer is not None
    assert other_scene.contraction_controls._viewer._cost_panel_ax is not None
    assert not other_scene.contraction_controls._viewer._cost_panel_ax.get_visible()


def test_show_tensor_network_scheme_checkbox_visual_state_matches_scheme_toggle() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")

    _click_checkbutton(controls._checkbuttons, scheme_index)

    status_after_enable = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_enable[scheme_index] is True
    assert controls.scheme_on is True
    assert controls.current_scene.contraction_controls is not None
    assert controls.current_scene.contraction_controls.scheme_on is True

    _click_checkbutton(controls._checkbuttons, scheme_index)

    status_after_disable = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_disable[scheme_index] is False
    assert controls.scheme_on is False
    assert controls.current_scene.contraction_controls is not None
    assert controls.current_scene.contraction_controls.scheme_on is False


def test_show_tensor_network_playback_and_cost_hover_keep_visual_checkboxes_in_sync() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    assert [label.get_text() for label in controls._checkbuttons.labels][-3:] == [
        "Scheme",
        "Playback",
        "Costs",
    ]
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    playback_index = _checkbutton_index(controls._checkbuttons, "Playback")
    cost_index = _checkbutton_index(controls._checkbuttons, "Costs")

    _click_checkbutton(controls._checkbuttons, playback_index)

    status_after_playback = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_playback[scheme_index] is True
    assert status_after_playback[playback_index] is True
    assert controls.scheme_on is True
    assert controls.playback_on is True

    _click_checkbutton(controls._checkbuttons, cost_index)

    status_after_cost_hover = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_cost_hover[scheme_index] is True
    assert status_after_cost_hover[playback_index] is True
    assert status_after_cost_hover[cost_index] is True
    assert controls.scheme_on is True
    assert controls.playback_on is True
    assert controls.cost_hover_on is True


def test_show_tn_einsum_trace_inspector_checkbox_auto_enables_playback() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(
            contraction_tensor_inspector=False,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    assert [label.get_text() for label in controls._checkbuttons.labels][-4:] == [
        "Scheme",
        "Playback",
        "Costs",
        "Tensor inspector",
    ]
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    playback_index = _checkbutton_index(controls._checkbuttons, "Playback")
    inspector_index = _checkbutton_index(controls._checkbuttons, "Tensor inspector")

    _click_checkbutton(controls._checkbuttons, inspector_index)

    status_after_enable = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_enable[scheme_index] is True
    assert status_after_enable[playback_index] is True
    assert status_after_enable[inspector_index] is True
    assert controls.scheme_on is True
    assert controls.playback_on is True
    assert controls.tensor_inspector_on is True
    assert getattr(fig, "_tensor_network_viz_tensor_inspector", None) is not None


def test_show_tn_reenabling_tensor_inspector_reveals_auxiliary_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _build_einsum_trace_for_inspector()
    revealed: list[matplotlib.figure.Figure] = []

    monkeypatch.setattr(
        interaction_controller_module,
        "_reveal_auxiliary_figure",
        lambda figure: revealed.append(figure),
    )

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(
            contraction_tensor_inspector=False,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    inspector_index = _checkbutton_index(controls._checkbuttons, "Tensor inspector")

    _click_checkbutton(controls._checkbuttons, inspector_index)

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    assert revealed == [inspector._figure]

    _click_checkbutton(controls._checkbuttons, inspector_index)
    assert inspector._figure is None

    _click_checkbutton(controls._checkbuttons, inspector_index)

    assert inspector._figure is not None
    assert revealed == [revealed[0], inspector._figure]


def test_show_tensor_network_non_einsum_inputs_do_not_expose_tensor_inspector_checkbox() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    assert [label.get_text() for label in controls._checkbuttons.labels][-3:] == [
        "Scheme",
        "Playback",
        "Costs",
    ]
    assert "Tensor inspector" not in [label.get_text() for label in controls._checkbuttons.labels]


def test_show_tensor_network_show_controls_false_does_not_create_tensor_inspector_window() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show_controls=False,
        show=False,
    )

    assert getattr(fig, "_tensor_network_viz_interactive_controls", None) is None
    assert getattr(fig, "_tensor_network_viz_tensor_inspector", None) is None


def test_show_tensor_network_einsum_tensor_inspector_tracks_playback_and_view_switch() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert "r1" in inspector_controls._panel.main_ax.get_title()

    assert controls.current_scene.contraction_controls is not None
    viewer = controls.current_scene.contraction_controls._viewer
    assert viewer is not None
    viewer.set_step(1)
    assert "r0" in inspector_controls._panel.main_ax.get_title()

    viewer.reset()
    assert inspector_controls._panel.main_ax.texts
    assert inspector_controls._panel.main_ax.texts[0].get_text() == "No contraction selected yet."

    viewer.play()
    viewer._tick_playback()
    assert "r0" in inspector_controls._panel.main_ax.get_title()
    viewer._tick_playback()
    assert "r1" in inspector_controls._panel.main_ax.get_title()

    controls.set_view("3d")
    assert controls.current_scene.contraction_controls is not None
    viewer_3d = controls.current_scene.contraction_controls._viewer
    assert viewer_3d is not None
    viewer_3d.set_step(1)
    assert "r0" in inspector_controls._panel.main_ax.get_title()


def test_show_tensor_network_manual_close_of_tensor_inspector_clears_toggle_state() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None

    _fire_close_event(inspector._figure)

    status_after_close = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_close[6] is False
    assert controls.tensor_inspector_on is False


def test_show_tensor_network_costs_and_tensor_inspector_can_coexist() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(
            contraction_scheme_cost_hover=True,
            contraction_tensor_inspector=True,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.cost_hover_on is True
    assert controls.tensor_inspector_on is True

    assert controls.current_scene.contraction_controls is not None
    viewer = controls.current_scene.contraction_controls._viewer
    assert viewer is not None
    assert viewer._cost_panel_ax is not None
    assert viewer._cost_text_artist is not None
    assert viewer._cost_panel_ax.get_visible()
    assert "Contraction:" in viewer._cost_text_artist.get_text()

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert "r1" in inspector_controls._panel.main_ax.get_title()


def test_show_tensor_network_main_figure_close_closes_tensor_inspector() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None

    _fire_close_event(fig)

    assert inspector._figure is None


def test_show_tensor_network_with_external_ax_hides_view_selector() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")
    fig, ax = plt.subplots()

    fig_out, ax_out = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        ax=ax,
        show=False,
    )

    controls = getattr(fig_out, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert fig_out is fig
    assert ax_out is ax
    assert controls._radio is None


def test_show_tensor_network_rejects_mismatched_external_ax_and_view() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")
    _fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Provided ax is 2d"):
        show_tensor_network(
            DummyNetwork(nodes=[left, right]),
            engine="tensorkrowch",
            ax=ax,
            view="3d",
            show=False,
        )


def test_show_tensor_network_show_controls_false_hides_all_figure_controls() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_playback=True,
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show_controls=False,
        show=False,
    )

    assert getattr(fig, "_tensor_network_viz_interactive_controls", None) is None
    assert getattr(fig, "_tensor_network_viz_contraction_controls", None) is None
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is None
    assert len(fig.axes) == 1
    assert getattr(ax, "_tensor_network_viz_scene", None) is None


def test_viewer_static_render_keeps_interactive_viewer_module_lazy() -> None:
    sys.modules.pop("tensor_network_viz.viewer", None)
    sys.modules.pop("tensor_network_viz.interactive_viewer", None)

    viewer_runtime = importlib.import_module("tensor_network_viz.viewer")

    assert "tensor_network_viz.interactive_viewer" not in sys.modules

    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig_static, _ax_static = viewer_runtime.show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show_controls=False,
        show=False,
    )
    try:
        assert "tensor_network_viz.interactive_viewer" not in sys.modules
    finally:
        plt.close(fig_static)

    fig_interactive, _ax_interactive = viewer_runtime.show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show_controls=True,
        show=False,
    )
    try:
        assert "tensor_network_viz.interactive_viewer" in sys.modules
    finally:
        plt.close(fig_interactive)


def test_show_tensor_network_precomputes_label_descriptors_for_menu_toggles() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    scene = controls.current_scene

    assert scene.tensor_label_descriptors is not None
    assert len(scene.tensor_label_descriptors) == 2
    assert scene.edge_label_descriptors is not None
    assert len(scene.edge_label_descriptors) == 2
    assert scene.tensor_label_artists == []
    assert scene.edge_label_artists == []


def test_show_tensor_network_menu_toggles_reuse_precomputed_label_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._interactive_scene as interactive_scene_module

    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    def _unexpected_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("label descriptors should already be cached in the scene")

    monkeypatch.setattr(
        interactive_scene_module,
        "_build_tensor_label_descriptors",
        _unexpected_rebuild,
    )
    monkeypatch.setattr(
        interactive_scene_module,
        "_build_edge_label_descriptors",
        _unexpected_rebuild,
    )

    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)

    assert len(controls.current_scene.tensor_label_artists) == 2
    assert len(controls.current_scene.edge_label_artists) == 2


def test_plot_tensorkrowch_network_2d_draws_tensor_nodes_as_circle_patches() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    _, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]))

    assert patch_collection_circle_count(ax) == 2


def test_plot_tensorkrowch_network_2d_show_nodes_false_draws_points() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    _, ax = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False),
    )

    assert patch_collection_circle_count(ax) == 0
    assert path_collection_point_count(ax) == 2


def test_plot_tensorkrowch_network_2d_show_nodes_false_ignores_node_radius() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    _, ax_small = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.04),
    )
    _, ax_large = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.5),
    )

    assert path_collection_point_count(ax_small) == 2
    assert path_collection_point_count(ax_large) == 2
    assert point_collection_sizes(ax_small) == point_collection_sizes(ax_large)


def test_plot_tensorkrowch_network_2d_show_nodes_false_keeps_degree_one_color() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="left")
    config = PlotConfig(show_nodes=False)

    _, ax = plot_tensorkrowch_network_2d(DummyNetwork(leaf_nodes=[node]), config=config)

    facecolors = point_collection_facecolors(ax)
    assert facecolors == [tuple(float(value) for value in to_rgba(config.node_color_degree_one))]


def test_plot_tensorkrowch_network_2d_show_nodes_false_dangling_reaches_node_center() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="left")
    center = np.array([0.0, 0.0], dtype=float)

    _, ax = plot_tensorkrowch_network_2d(
        DummyNetwork(leaf_nodes=[node]),
        config=PlotConfig(
            show_nodes=False,
            positions={id(node): (float(center[0]), float(center[1]))},
        ),
    )

    segs = line_collection_segments(ax)
    assert len(segs) == 1
    distances = np.linalg.norm(np.asarray(segs[0], dtype=float) - center, axis=1)
    assert float(np.min(distances)) == pytest.approx(0.0)


def test_extent_scale_factor_reflects_long_dense_chain_vs_pair() -> None:
    """Large span with small nearest-neighbor spacing should shrink glyphs vs a loose pair."""
    long_dense = np.array([[i * 0.2, 0.0] for i in range(20)], dtype=float)
    pair = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    s_long = core_renderer_module._extent_scale_factor(long_dense)
    s_pair = core_renderer_module._extent_scale_factor(pair)
    assert s_long < s_pair


def test_plot_tensorkrowch_network_2d_warns_on_unknown_position_keys_when_validate_true() -> None:
    """validate_positions=True warns when custom positions have unknown node ids."""
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    config = PlotConfig(
        positions={
            id(left): (0.0, 0.0),
            999999: (1.0, 1.0),  # unknown id
        },
        validate_positions=True,
    )

    with pytest.warns(UserWarning, match="does not match any node id"):
        plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]), config=config)


def test_plot_tensorkrowch_network_2d_accepts_full_custom_positions() -> None:
    """Full custom positions bypass automatic layout; figure must render."""
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    config = PlotConfig(
        positions={
            id(left): (0.0, 0.0),
            id(right): (2.0, 0.0),
        },
    )

    fig, ax = plot_tensorkrowch_network_2d(DummyNetwork(nodes=[left, right]), config=config)

    assert fig is ax.figure
    assert line_collection_segment_count(ax) == 1


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

    segs = line_collection_segments(ax)
    assert len(segs) == 2
    y0 = segs[0][:, 1]
    y1 = segs[1][:, 1]
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
    assert graph.edges[0].label is None


def test_plot_tensorkrowch_network_3d_returns_3d_axes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0)

    fig, ax = plot_tensorkrowch_network_3d(DummyNetwork(nodes=[left, right]))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 1
    assert len(ax.collections) >= 1


def test_plot_tensorkrowch_network_3d_show_nodes_false_draws_marker_nodes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0)

    _, ax = plot_tensorkrowch_network_3d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False),
    )

    assert poly3d_node_collection_count(ax) == 0
    assert path3d_collection_point_count(ax) == 2


def test_plot_tensorkrowch_network_3d_show_nodes_false_ignores_node_radius() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0)

    _, ax_small = plot_tensorkrowch_network_3d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.04),
    )
    _, ax_large = plot_tensorkrowch_network_3d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.5),
    )

    assert path3d_collection_point_count(ax_small) == 2
    assert path3d_collection_point_count(ax_large) == 2
    assert path3d_collection_sizes(ax_small) == path3d_collection_sizes(ax_large)


def test_plot_tensorkrowch_network_3d_show_nodes_false_keeps_degree_one_color() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="left")
    config = PlotConfig(show_nodes=False)

    _, ax = plot_tensorkrowch_network_3d(DummyNetwork(leaf_nodes=[node]), config=config)

    facecolors = path3d_collection_facecolors(ax)
    assert facecolors == [tuple(float(value) for value in to_rgba(config.node_color_degree_one))]


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


def test_show_figure_uses_ipython_display_in_jupyter_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("IPython")
    fig, _ax = plt.subplots()
    displayed: list[object] = []

    class _FakeKernel:
        pass

    class _FakeIPython:
        kernel = _FakeKernel()

    def _fake_get_ipython() -> _FakeIPython:
        return _FakeIPython()

    def _fake_display(obj: object) -> None:
        displayed.append(obj)

    monkeypatch.setattr("IPython.core.getipython.get_ipython", _fake_get_ipython)
    monkeypatch.setattr("IPython.display.display", _fake_display)
    viewer_module._show_figure(fig)
    assert displayed == [fig]
    plt.close(fig)


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
