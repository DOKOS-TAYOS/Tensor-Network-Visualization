from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent, MouseButton, MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import tensor_network_viz._core.renderer as core_renderer_module
import tensor_network_viz._interaction.controller as interaction_controller_module
import tensor_network_viz._interaction.tensor_inspector as tensor_inspector_module
import tensor_network_viz.tensorkrowch.graph as tk_graph_module
import tensor_network_viz.tensorkrowch.renderer as tk_renderer_module
import tensor_network_viz.tensornetwork.graph as tn_graph_module
import tensor_network_viz.tensornetwork.renderer as tn_renderer_module
import tensor_network_viz.viewer as viewer_module
from plotting_helpers import (
    assert_rendered_figure,
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
from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    TensorNetworkDiagnosticsConfig,
    TensorNetworkFocus,
    einsum,
    show_tensor_network,
)
from tensor_network_viz._contraction_viewer_ui import _PLAYBACK_SLIDER_BOUNDS
from tensor_network_viz._core import _draw_common
from tensor_network_viz._core.draw.labels_misc import _edge_index_text_kwargs
from tensor_network_viz._core.focus import filter_graph_for_focus
from tensor_network_viz._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz._interaction.tensor_inspector import _INSPECTOR_TENSOR_ELEMENTS_LAYOUT
from tensor_network_viz._matplotlib_state import get_scene
from tensor_network_viz._tensor_elements_support import _TensorRecord
from tensor_network_viz.einsum_module.trace import pair_tensor
from tensor_network_viz.tensor_elements import _show_tensor_records
from tensor_network_viz.tensor_elements_config import TensorElementsConfig
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


def _line3d_collections(ax: Any) -> list[Line3DCollection]:
    return [collection for collection in ax.collections if isinstance(collection, Line3DCollection)]


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


def test_default_edge_index_label_boxes_are_tinted_from_index_edge_colors() -> None:
    config = PlotConfig()

    bond_kwargs = _edge_index_text_kwargs(
        config,
        fontsize=10.0,
        stub_kind="bond",
        bbox_pad=0.18,
    )
    dangling_kwargs = _edge_index_text_kwargs(
        config,
        fontsize=10.0,
        stub_kind="dangling",
        bbox_pad=0.18,
    )

    assert bond_kwargs["bbox"]["facecolor"] == pytest.approx(
        (0.907, 0.861, 0.987, 0.34),
        abs=0.001,
    )
    assert dangling_kwargs["bbox"]["facecolor"] == pytest.approx(
        (0.831, 0.951, 0.911, 0.34),
        abs=0.001,
    )
    assert bond_kwargs["bbox"]["edgecolor"] == pytest.approx(
        (0.486, 0.227, 0.929, 0.38),
        abs=0.001,
    )
    assert dangling_kwargs["bbox"]["edgecolor"] == pytest.approx(
        (0.063, 0.725, 0.506, 0.38),
        abs=0.001,
    )


class DummyNetwork:
    def __init__(self, *, nodes=None, leaf_nodes=None) -> None:
        if nodes is not None:
            self.nodes = nodes
        if leaf_nodes is not None:
            self.leaf_nodes = leaf_nodes


def _einsum_trace_with_three_tensors() -> EinsumTrace:
    trace = EinsumTrace()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    mid = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)

    trace.bind("A", left)
    trace.bind("B", mid)
    trace.bind("C", right)
    result_ab = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    result_abc = einsum("ac,cd->ad", result_ab, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, result_ab, result_abc]  # type: ignore[attr-defined]
    return trace


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


def _click_button(button: Any) -> None:
    fig = button.ax.figure
    fig.canvas.draw()
    bbox = button.ax.get_window_extent(fig.canvas.get_renderer())
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", press)
    fig.canvas.callbacks.process("button_release_event", release)


def _drag_slider_to_value(slider: Any, value: float) -> None:
    fig = slider.ax.figure
    fig.canvas.draw()
    bbox = slider.ax.get_window_extent(fig.canvas.get_renderer())
    span = float(slider.valmax) - float(slider.valmin)
    fraction = 0.0 if span <= 0.0 else (float(value) - float(slider.valmin)) / span
    fraction = min(max(fraction, 0.0), 1.0)
    x = int(round(bbox.x0 + fraction * (bbox.x1 - bbox.x0)))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    motion = MouseEvent("motion_notify_event", fig.canvas, x, y, button=MouseButton.LEFT)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", press)
    fig.canvas.callbacks.process("motion_notify_event", motion)
    fig.canvas.callbacks.process("button_release_event", release)


def _checkbutton_index(checkbuttons: Any, label_text: str) -> int:
    labels = [label.get_text() for label in checkbuttons.labels]
    return labels.index(label_text)


def _visible_axes_at_bounds(
    fig: matplotlib.figure.Figure,
    bounds: tuple[float, float, float, float],
    *,
    abs_tol: float = 0.005,
) -> list[matplotlib.axes.Axes]:
    return [
        ax
        for ax in fig.axes
        if ax.get_visible()
        and all(
            abs(float(actual) - float(expected)) <= abs_tol
            for actual, expected in zip(ax.get_position().bounds, bounds, strict=True)
        )
    ]


def _fire_close_event(fig: matplotlib.figure.Figure) -> None:
    fig.canvas.callbacks.process("close_event", CloseEvent("close_event", fig.canvas))


def _dispatch_motion_event_at_data(
    ax: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
) -> MouseEvent:
    ax.figure.canvas.draw()
    x_display, y_display = ax.transData.transform((x, y))
    event = MouseEvent(
        "motion_notify_event",
        ax.figure.canvas,
        int(round(x_display)),
        int(round(y_display)),
    )
    ax.figure.canvas.callbacks.process("motion_notify_event", event)
    return event


def _dispatch_motion_event_at_widget(widget: Any) -> MouseEvent:
    fig = widget.ax.figure
    fig.canvas.draw()
    bbox = widget.ax.get_window_extent(fig.canvas.get_renderer())
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    event = MouseEvent("motion_notify_event", fig.canvas, x, y)
    fig.canvas.callbacks.process("motion_notify_event", event)
    return event


def _dispatch_button_event_at_data(
    ax: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
) -> MouseEvent:
    ax.figure.canvas.draw()
    x_display, y_display = ax.transData.transform((x, y))
    event = MouseEvent(
        "button_press_event",
        ax.figure.canvas,
        int(round(x_display)),
        int(round(y_display)),
        button=MouseButton.LEFT,
    )
    ax.figure.canvas.callbacks.process("button_press_event", event)
    return event


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


def _build_einsum_trace_for_comparison_inspector() -> tuple[EinsumTrace, np.ndarray, np.ndarray]:
    trace = EinsumTrace()
    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    mid = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=float)
    right = np.array([[1.0, 1.0], [0.0, 2.0]], dtype=float)

    trace.bind("Left", left)
    trace.bind("Mid", mid)
    trace.bind("Right", right)
    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, r0, r1]  # type: ignore[attr-defined]
    return trace, np.asarray(r0), np.asarray(r1)


def _records_for_inspector_slider_layout() -> list[_TensorRecord]:
    return [
        _TensorRecord(
            array=np.arange(6, dtype=float).reshape(2, 3),
            name="Tensor 1",
            axis_names=("left", "right"),
            engine="tensornetwork",
        ),
        _TensorRecord(
            array=(np.arange(6, dtype=float).reshape(2, 3) + 10.0),
            name="Tensor 2",
            axis_names=("left", "right"),
            engine="tensornetwork",
        ),
    ]


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
    assert_rendered_figure(fig, ax)
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
    assert_rendered_figure(fig, ax)
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
    assert_rendered_figure(fig, ax)
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
    assert_rendered_figure(fig, ax)
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
        hover_cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
        assert isinstance(hover_cid, int)
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
        hover_cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
        assert isinstance(hover_cid, int)
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
        hover_cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
        assert isinstance(hover_cid, int)
    finally:
        plt.close(fig)


def test_plot_tensornetwork_network_2d_applies_optional_fontsize_overrides() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = plot_tensornetwork_network_2d(
        {left, right},
        config=PlotConfig(
            show_tensor_labels=True,
            show_index_labels=True,
            tensor_label_fontsize=13.0,
            edge_label_fontsize=11.0,
        ),
    )
    try:
        tensor_label_sizes = [
            text.get_fontsize()
            for text in ax.texts
            if text.get_gid() == _draw_common._TENSOR_LABEL_GID
        ]
        edge_label_sizes = [
            text.get_fontsize()
            for text in ax.texts
            if text.get_gid() == _draw_common._EDGE_INDEX_LABEL_GID
        ]

        assert tensor_label_sizes
        assert edge_label_sizes
        assert tensor_label_sizes == pytest.approx([13.0, 13.0])
        assert edge_label_sizes == pytest.approx([11.0, 11.0])
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
    assert_rendered_figure(fig, ax)
    assert ax.name != "3d"
    assert controls.current_view == "2d"
    assert controls.hover_on is True
    assert controls.tensor_labels_on is False
    assert controls.edge_labels_on is False
    assert controls._view_toggle_ax is not None and controls._view_toggle_ax in fig.axes
    assert controls._view_toggle_button is not None
    assert controls._view_toggle_button.label.get_text() == "3D"
    assert controls._controls_panel is not None
    assert controls._controls_panel.focus_mode_button is not None
    assert controls._controls_panel.focus_mode_button.label.get_text() == "Off"
    assert controls._controls_panel.focus_radius_button is not None
    assert controls._controls_panel.focus_radius_button.ax.get_visible() is False
    assert controls._controls_panel.focus_clear_button is not None
    assert controls._controls_panel.focus_clear_button.ax.get_visible() is False
    assert controls._check_ax is not None and controls._check_ax in fig.axes
    assert len(fig.axes) >= 3


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
    assert controls._checkbuttons.ax in fig.axes


def test_filter_graph_for_path_focus_keeps_cut_bonds_as_dangling_stubs() -> None:
    left = DummyTensorKrowchNode("A", ["left", "ab"])
    middle = DummyTensorKrowchNode("B", ["ab", "bc", "bd"])
    right = DummyTensorKrowchNode("C", ["bc", "right"])
    branch = DummyTensorKrowchNode("D", ["bd", "dangling"])
    connect(left, 1, middle, 0, name="AB")
    connect(middle, 1, right, 0, name="BC")
    connect(middle, 2, branch, 0, name="BD")

    graph = _build_tensorkrowch_graph(DummyNetwork(nodes=[left, middle, right, branch]))
    focused = filter_graph_for_focus(
        graph,
        TensorNetworkFocus(kind="path", endpoints=("A", "C")),
    )

    focused_names = {node.name for node in focused.nodes.values() if not node.is_virtual}
    middle_id = next(node_id for node_id, node in focused.nodes.items() if node.name == "B")

    assert focused_names == {"A", "B", "C"}
    assert any(
        edge.kind == "dangling" and edge.node_ids == (middle_id,) and edge.label == "bd"
        for edge in focused.edges
    )


def test_filter_graph_for_disconnected_path_focus_keeps_both_endpoints_with_cut_stubs() -> None:
    left_a = DummyTensorKrowchNode("A", ["left", "ax"])
    right_a = DummyTensorKrowchNode("x", ["ax", "right"])
    left_b = DummyTensorKrowchNode("B", ["left", "by"])
    right_b = DummyTensorKrowchNode("y", ["by", "right"])
    connect(left_a, 1, right_a, 0, name="AX")
    connect(left_b, 1, right_b, 0, name="BY")

    graph = _build_tensorkrowch_graph(DummyNetwork(nodes=[left_a, right_a, left_b, right_b]))
    focused = filter_graph_for_focus(
        graph,
        TensorNetworkFocus(kind="path", endpoints=("A", "B")),
    )

    focused_names = {node.name for node in focused.nodes.values() if not node.is_virtual}
    focused_id_a = next(node_id for node_id, node in focused.nodes.items() if node.name == "A")
    focused_id_b = next(node_id for node_id, node in focused.nodes.items() if node.name == "B")

    assert focused_names == {"A", "B"}
    assert any(
        edge.kind == "dangling" and edge.node_ids == (focused_id_a,) and edge.label == "ax"
        for edge in focused.edges
    )
    assert any(
        edge.kind == "dangling" and edge.node_ids == (focused_id_b,) and edge.label == "by"
        for edge in focused.edges
    )


def test_filter_graph_for_neighborhood_focus_keeps_cut_bonds_as_dangling_stubs() -> None:
    nodes = [
        DummyTensorKrowchNode("A", ["right"]),
        DummyTensorKrowchNode("B", ["left", "right"]),
        DummyTensorKrowchNode("C", ["left", "right"]),
        DummyTensorKrowchNode("D", ["left", "right"]),
        DummyTensorKrowchNode("E", ["left"]),
    ]
    connect(nodes[0], 0, nodes[1], 0, name="AB")
    connect(nodes[1], 1, nodes[2], 0, name="BC")
    connect(nodes[2], 1, nodes[3], 0, name="CD")
    connect(nodes[3], 1, nodes[4], 0, name="DE")

    graph = _build_tensorkrowch_graph(DummyNetwork(nodes=nodes))
    focused = filter_graph_for_focus(
        graph,
        TensorNetworkFocus(kind="neighborhood", center="C", radius=1),
    )

    focused_names = {node.name for node in focused.nodes.values() if not node.is_virtual}
    left_cut_id = next(node_id for node_id, node in focused.nodes.items() if node.name == "B")
    right_cut_id = next(node_id for node_id, node in focused.nodes.items() if node.name == "D")

    assert focused_names == {"B", "C", "D"}
    assert any(
        edge.kind == "dangling" and edge.node_ids == (left_cut_id,) and edge.label == "left"
        for edge in focused.edges
    )
    assert any(
        edge.kind == "dangling" and edge.node_ids == (right_cut_id,) and edge.label == "right"
        for edge in focused.edges
    )


def test_show_tensor_network_path_focus_keeps_cut_bond_labels_as_stubs() -> None:
    left = DummyTensorKrowchNode("A", ["left", "ab"])
    middle = DummyTensorKrowchNode("B", ["ab", "bc", "bd"])
    right = DummyTensorKrowchNode("C", ["bc", "right"])
    branch = DummyTensorKrowchNode("D", ["bd", "dangling"])
    connect(left, 1, middle, 0, name="AB")
    connect(middle, 1, right, 0, name="BC")
    connect(middle, 2, branch, 0, name="BD")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, middle, right, branch]),
        engine="tensorkrowch",
        config=PlotConfig(
            show_tensor_labels=True,
            show_index_labels=True,
            focus=TensorNetworkFocus(kind="path", endpoints=("A", "C")),
        ),
        show=False,
    )

    labels = [text.get_text() for text in ax.texts if text.get_visible()]
    assert_rendered_figure(fig, ax)
    assert "D" not in labels
    assert labels.count("bd") == 1
    assert labels.count("ab") == 2
    assert labels.count("bc") == 2


def test_show_tensor_network_disconnected_path_focus_renders_both_endpoints() -> None:
    trace = [
        pair_tensor("A", "x", "r0", "ab,b->a"),
        pair_tensor("B", "y", "r1", "cd,d->c"),
    ]

    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        config=PlotConfig(
            focus=TensorNetworkFocus(kind="path", endpoints=("A", "B")),
        ),
        show=False,
    )

    scene = get_scene(ax)

    assert scene is not None
    assert {scene.graph.nodes[node_id].name for node_id in scene.visible_node_ids} == {"A", "B"}
    assert scene.focus_feedback is not None
    assert scene.focus_feedback.disconnected_endpoints == ("A", "B")
    assert_rendered_figure(fig, ax)


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


def test_show_tensor_network_uses_2d_custom_positions_in_3d_without_coordinate_warnings() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            positions={
                id(left): (0.0, 0.0),
                id(right): (2.0, 0.0),
            },
            validate_positions=True,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controls.set_view("3d")

    assert not [
        warning
        for warning in caught
        if "missing coords will be zero-filled" in str(warning.message)
    ]


def test_show_tensor_network_view_toggle_button_switches_between_2d_and_3d() -> None:
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
    assert controls._view_toggle_button is not None
    assert controls.current_view == "2d"
    assert controls._view_toggle_button.label.get_text() == "3D"

    _click_button(controls._view_toggle_button)
    assert controls.current_view == "3d"
    assert controls._view_toggle_button.label.get_text() == "2D"

    _click_button(controls._view_toggle_button)
    assert controls.current_view == "2d"
    assert controls._view_toggle_button.label.get_text() == "3D"


def test_show_tensor_network_grid3d_switch_from_3d_to_2d_uses_projected_offsets() -> None:
    front = DummyTensorNetworkNode("front", ["a"])
    back = DummyTensorNetworkNode("back", ["b"])

    fig, _ax = show_tensor_network([[[front]], [[back]]], show=False)

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.current_view == "3d"

    controls.set_view("2d")
    scene = controls.current_scene
    front_pos = scene.positions[id(front)]
    back_pos = scene.positions[id(back)]

    assert scene.dimensions == 2
    assert float(back_pos[0]) < float(front_pos[0])
    assert float(back_pos[1]) < float(front_pos[1])


def test_show_tensor_network_grid3d_switch_from_2d_to_3d_uses_layer_depth() -> None:
    front = DummyTensorNetworkNode("front", ["a"])
    back = DummyTensorNetworkNode("back", ["b"])

    fig, _ax = show_tensor_network([[[front]], [[back]]], view="2d", show=False)

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.current_view == "2d"

    controls.set_view("3d")
    scene = controls.current_scene
    front_pos = scene.positions[id(front)]
    back_pos = scene.positions[id(back)]

    assert scene.dimensions == 3
    assert abs(float(back_pos[2]) - float(front_pos[2])) > 1e-9


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
    assert controls._view_toggle_ax is not None
    assert controls._check_ax is not None

    initial_axes_count = len(fig.axes)
    cached_2d_ax = controls._view_caches["2d"].ax
    assert cached_2d_ax is not None

    controls.set_view("3d")
    cached_3d_ax = controls._view_caches["3d"].ax
    assert cached_3d_ax is not None
    after_first_switch_count = len(fig.axes)
    assert after_first_switch_count == initial_axes_count + 1
    assert _line3d_collections(cached_3d_ax)

    controls.set_view("2d")

    assert len(fig.axes) == after_first_switch_count
    assert getattr(fig, "_tensor_network_viz_active_axes", None) is cached_2d_ax
    assert line_collection_segment_count(cached_2d_ax) == 1

    controls.set_view("3d")

    assert len(fig.axes) == after_first_switch_count
    assert getattr(fig, "_tensor_network_viz_active_axes", None) is cached_3d_ax
    assert _line3d_collections(cached_3d_ax)


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
    hover_cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
    assert isinstance(hover_cid, int)

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


def test_show_tensor_network_offscreen_agg_skips_draw_idle_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    draw_idle_calls: list[FigureCanvasAgg] = []
    original_draw_idle = FigureCanvasAgg.draw_idle

    def counting_draw_idle(self: FigureCanvasAgg) -> None:
        draw_idle_calls.append(self)
        original_draw_idle(self)

    monkeypatch.setattr(FigureCanvasAgg, "draw_idle", counting_draw_idle)

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)
    controls.set_view("3d")

    assert draw_idle_calls == []


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


def test_show_tensor_network_nodes_toggle_updates_2d_dangling_anchor() -> None:
    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="left")
    center = np.array([0.0, 0.0], dtype=float)

    fig, _ax = show_tensor_network(
        DummyNetwork(leaf_nodes=[node]),
        engine="tensorkrowch",
        config=PlotConfig(
            stub_length=0.2,
            positions={id(node): (float(center[0]), float(center[1]))},
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    ax_2d = controls._view_caches["2d"].ax
    assert ax_2d is not None

    seg_before = np.asarray(line_collection_segments(ax_2d)[0], dtype=float)
    length_before = float(np.linalg.norm(seg_before[1] - seg_before[0]))
    assert float(np.min(np.linalg.norm(seg_before - center, axis=1))) > 0.0

    controls.set_nodes_enabled(False)

    seg_after = np.asarray(line_collection_segments(ax_2d)[0], dtype=float)
    assert float(np.min(np.linalg.norm(seg_after - center, axis=1))) == pytest.approx(0.0)
    assert float(np.linalg.norm(seg_after[1] - seg_after[0])) == pytest.approx(
        length_before,
        rel=1e-6,
    )


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


def test_show_tensor_network_places_compact_top_row_above_options_and_before_playback_slider() -> (
    None
):
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._view_toggle_ax is not None
    assert controls._check_ax is not None
    assert controls._controls_panel is not None
    assert controls._controls_panel.focus_mode_button is not None
    assert controls._controls_panel.focus_radius_button is not None
    assert controls._controls_panel.focus_clear_button is not None
    scene_controls = controls.current_scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None
    assert scene_controls._viewer.slider is not None
    assert scene_controls._viewer._btn_play is not None

    view_toggle_bounds = controls._view_toggle_ax.get_position().bounds
    focus_mode_bounds = controls._controls_panel.focus_mode_button.ax.get_position().bounds
    focus_radius_bounds = controls._controls_panel.focus_radius_button.ax.get_position().bounds
    focus_clear_bounds = controls._controls_panel.focus_clear_button.ax.get_position().bounds
    check_bounds = controls._check_ax.get_position().bounds
    slider_bounds = scene_controls._viewer.slider.ax.get_position().bounds
    play_bounds = scene_controls._viewer._btn_play.ax.get_position().bounds

    view_toggle_right = view_toggle_bounds[0] + view_toggle_bounds[2]
    focus_mode_right = focus_mode_bounds[0] + focus_mode_bounds[2]
    focus_radius_right = focus_radius_bounds[0] + focus_radius_bounds[2]
    focus_clear_right = focus_clear_bounds[0] + focus_clear_bounds[2]
    check_top = check_bounds[1] + check_bounds[3]
    slider_right = slider_bounds[0] + slider_bounds[2]

    assert check_bounds[2] <= 0.21
    assert view_toggle_bounds[0] == pytest.approx(check_bounds[0], abs=0.005)
    assert view_toggle_bounds[1] > check_top
    assert view_toggle_bounds[2] <= 0.06
    assert view_toggle_bounds[3] <= 0.05
    assert focus_mode_bounds[0] > view_toggle_right
    assert focus_mode_bounds[1] == pytest.approx(view_toggle_bounds[1], abs=0.005)
    assert focus_radius_bounds[0] > focus_mode_right
    assert focus_radius_bounds[1] == pytest.approx(view_toggle_bounds[1], abs=0.005)
    assert focus_clear_bounds[0] > focus_radius_right
    assert focus_clear_bounds[1] == pytest.approx(view_toggle_bounds[1], abs=0.005)
    assert slider_bounds[0] >= focus_clear_right - 0.005
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
    assert controls.current_scene.contraction_controls._viewer is not None
    assert controls.current_scene.contraction_controls._viewer.slider is not None
    assert controls.current_scene.contraction_controls._viewer.slider.ax.get_visible() is True

    _click_checkbutton(controls._checkbuttons, scheme_index)

    status_after_disable = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_disable[scheme_index] is False
    assert controls.scheme_on is False
    assert controls.current_scene.contraction_controls is not None
    assert controls.current_scene.contraction_controls.scheme_on is False
    assert controls.current_scene.contraction_controls._viewer is not None
    assert controls.current_scene.contraction_controls._viewer.slider is not None
    assert controls.current_scene.contraction_controls._viewer.slider.ax.get_visible() is False


def test_neighbor_focus_after_scheme_hides_stale_playback_slider_axes() -> None:
    nodes = [
        DummyTensorKrowchNode("A", ["phys", "right"]),
        DummyTensorKrowchNode("B", ["left", "phys", "right"]),
        DummyTensorKrowchNode("C", ["left", "phys", "right"]),
        DummyTensorKrowchNode("D", ["left", "phys"]),
    ]
    for index in range(len(nodes) - 1):
        connect(nodes[index], len(nodes[index].edges) - 1, nodes[index + 1], 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=nodes),
        engine="tensorkrowch",
        config=PlotConfig(
            contraction_scheme_by_name=(
                ("A", "B"),
                ("A", "B", "C"),
                ("A", "B", "C", "D"),
            ),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._controls_panel is not None
    assert controls._controls_panel.focus_mode_button is not None
    assert controls._checkbuttons is not None

    _click_button(controls._controls_panel.focus_mode_button)
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    _click_checkbutton(controls._checkbuttons, scheme_index)
    assert _visible_axes_at_bounds(fig, _PLAYBACK_SLIDER_BOUNDS)

    node_a_id = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "A"
    )
    node_a_position = np.asarray(controls.current_scene.positions[node_a_id], dtype=float)
    _dispatch_button_event_at_data(
        ax,
        x=float(node_a_position[0]),
        y=float(node_a_position[1]),
    )

    scene_controls = controls.current_scene.contraction_controls
    viewer = None if scene_controls is None else scene_controls._viewer
    expected_slider_ax = (
        viewer.slider.ax
        if viewer is not None and viewer.slider is not None and viewer.slider.ax.get_visible()
        else None
    )
    expected_visible_axes = [] if expected_slider_ax is None else [expected_slider_ax]

    assert _visible_axes_at_bounds(fig, _PLAYBACK_SLIDER_BOUNDS) == expected_visible_axes


def test_show_tensor_network_scheme_and_cost_hover_keep_visual_checkboxes_in_sync() -> None:
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
        "Costs",
        "Dimensions",
    ]
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    cost_index = _checkbutton_index(controls._checkbuttons, "Costs")

    _click_checkbutton(controls._checkbuttons, scheme_index)

    status_after_scheme = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_scheme[scheme_index] is True
    assert controls.scheme_on is True

    _click_checkbutton(controls._checkbuttons, cost_index)

    status_after_cost_hover = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_cost_hover[scheme_index] is True
    assert status_after_cost_hover[cost_index] is True
    assert controls.scheme_on is True
    assert controls.cost_hover_on is True


def test_show_tensor_network_cost_panel_touches_hover_menu_right_edge() -> None:
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
    assert controls._check_ax is not None
    assert controls._checkbuttons is not None
    cost_index = _checkbutton_index(controls._checkbuttons, "Costs")

    _click_checkbutton(controls._checkbuttons, cost_index)

    scene_controls = controls.current_scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None
    assert scene_controls._viewer._cost_panel_ax is not None

    hover_menu_bounds = controls._check_ax.get_position().bounds
    cost_panel_bounds = scene_controls._viewer._cost_panel_ax.get_position().bounds
    hover_menu_right = hover_menu_bounds[0] + hover_menu_bounds[2]

    assert hover_menu_bounds[2] >= 0.21
    assert cost_panel_bounds[0] == pytest.approx(hover_menu_right)


def test_show_tensor_network_3d_dimensions_rerender_keeps_scheme_slider_interactive() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        view="3d",
        config=PlotConfig(
            diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=True),
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    assert getattr(controls.current_scene.ax, "name", None) == "3d"
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    dimensions_index = _checkbutton_index(controls._checkbuttons, "Dimensions")

    fig.canvas.grab_mouse(controls.current_scene.ax)

    _click_checkbutton(controls._checkbuttons, dimensions_index)
    assert getattr(fig.canvas, "mouse_grabber", None) is None
    _click_checkbutton(controls._checkbuttons, scheme_index)

    scene_controls = controls.current_scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None
    assert scene_controls._viewer.slider is not None

    _drag_slider_to_value(scene_controls._viewer.slider, 1.0)
    _drag_slider_to_value(scene_controls._viewer.slider, 0.0)

    assert scene_controls._viewer.current_step == 0


def test_show_tensor_network_keeps_diagnostics_labels_last_after_scheme_updates() -> None:
    fig, ax = show_tensor_network(
        _einsum_trace_with_three_tensors(),
        engine="einsum",
        config=PlotConfig(
            show_tensor_labels=True,
            show_index_labels=True,
            diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=True),
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    scene = get_scene(ax)
    assert scene is not None
    label_artists = (
        tuple(scene.tensor_label_artists)
        + tuple(scene.edge_label_artists)
        + tuple(scene.diagnostic_artists)
    )
    visible_texts = tuple(text for text in scene.ax.texts if text.get_visible())
    assert label_artists
    assert scene.diagnostic_artists
    assert tuple(visible_texts[-len(label_artists) :]) == label_artists
    assert tuple(visible_texts[-len(scene.diagnostic_artists) :]) == tuple(scene.diagnostic_artists)

    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    _click_checkbutton(controls._checkbuttons, scheme_index)

    scene_controls = scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None
    assert scene_controls._viewer.slider is not None

    scene_controls._viewer.slider.set_val(1.0)

    scene = get_scene(ax)
    assert scene is not None
    label_artists = (
        tuple(scene.tensor_label_artists)
        + tuple(scene.edge_label_artists)
        + tuple(scene.diagnostic_artists)
    )
    visible_texts = tuple(text for text in scene.ax.texts if text.get_visible())
    assert tuple(visible_texts[-len(label_artists) :]) == label_artists
    assert tuple(visible_texts[-len(scene.diagnostic_artists) :]) == tuple(scene.diagnostic_artists)


def test_show_tensor_network_scheme_keeps_node_hover_active_after_step_changes() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        config=PlotConfig(
            hover_labels=True,
            show_contraction_scheme=True,
            contraction_scheme_by_name=(("A", "B"),),
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    scene_controls = controls.current_scene.contraction_controls
    assert scene_controls is not None
    assert scene_controls._viewer is not None

    scene_controls._viewer.set_step(0)
    scene_controls._viewer.set_step(1)
    visible_node_ids = tuple(int(node_id) for node_id in controls.current_scene.visible_node_ids)
    hovered_node_id = next(
        node_id
        for node_id in visible_node_ids
        if controls.current_scene.graph.nodes[node_id].name == "A"
    )
    hovered_position = np.asarray(controls.current_scene.positions[hovered_node_id], dtype=float)
    _dispatch_motion_event_at_data(
        ax,
        x=float(hovered_position[0]),
        y=float(hovered_position[1]),
    )

    hover_annotation = getattr(fig, "_tensor_network_viz_hover_ann", None)
    assert hover_annotation is not None
    assert hover_annotation.get_visible() is True
    assert hover_annotation.get_text() == "A"


def test_show_tn_einsum_trace_inspector_checkbox_keeps_scheme_off() -> None:
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
        "Costs",
        "Tensor inspector",
        "Dimensions",
    ]
    scheme_index = _checkbutton_index(controls._checkbuttons, "Scheme")
    inspector_index = _checkbutton_index(controls._checkbuttons, "Tensor inspector")

    _click_checkbutton(controls._checkbuttons, inspector_index)

    status_after_enable = tuple(bool(v) for v in controls._checkbuttons.get_status())
    assert status_after_enable[scheme_index] is False
    assert status_after_enable[inspector_index] is True
    assert controls.scheme_on is False
    assert controls.tensor_inspector_on is True
    assert getattr(fig, "_tensor_network_viz_tensor_inspector", None) is not None


def test_show_tn_reenabling_tensor_inspector_reveals_auxiliary_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _build_einsum_trace_for_inspector()
    revealed: list[matplotlib.figure.Figure] = []

    monkeypatch.setattr(
        tensor_inspector_module,
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


def test_non_playback_tensorkrowch_inputs_hide_tensor_inspector_checkbox() -> None:
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
        "Costs",
        "Dimensions",
    ]
    assert "Tensor inspector" not in [label.get_text() for label in controls._checkbuttons.labels]


def test_non_playback_tensorkrowch_inputs_show_tensor_inspector_when_tensors_are_materialized() -> (
    None
):
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    left.tensor = np.array([1.0, 2.0], dtype=float)  # type: ignore[attr-defined]
    right.tensor = np.array([3.0, 4.0], dtype=float)  # type: ignore[attr-defined]
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
    assert [label.get_text() for label in controls._checkbuttons.labels][-4:] == [
        "Scheme",
        "Costs",
        "Tensor inspector",
        "Dimensions",
    ]


def test_show_tensor_network_surfaces_unexpected_tensor_record_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    left.tensor = np.array([1.0, 2.0], dtype=float)  # type: ignore[attr-defined]
    right.tensor = np.array([3.0, 4.0], dtype=float)  # type: ignore[attr-defined]
    connect(left, 0, right, 0, name="bond")

    def _raise_unexpected_error(_network: Any, *, engine: Any) -> Any:
        raise RuntimeError(f"unexpected extraction failure for {engine}")

    monkeypatch.setattr(
        interaction_controller_module,
        "_extract_tensor_records",
        _raise_unexpected_error,
    )

    with pytest.raises(RuntimeError, match="unexpected extraction failure"):
        show_tensor_network(
            DummyNetwork(nodes=[left, right]),
            engine="tensorkrowch",
            show=False,
        )


def test_clicking_a_visible_tensor_does_not_open_inspector_when_checkbox_is_off() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    left.tensor = np.array([1.0, 2.0], dtype=float)  # type: ignore[attr-defined]
    right.tensor = np.array([3.0, 4.0], dtype=float)  # type: ignore[attr-defined]
    connect(left, 0, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert controls._checkbuttons is not None
    assert controls.scheme_on is False
    assert controls.tensor_inspector_on is False
    node_id = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "A"
    )
    node_position = np.asarray(controls.current_scene.positions[node_id], dtype=float)

    _dispatch_button_event_at_data(
        ax,
        x=float(node_position[0]),
        y=float(node_position[1]),
    )

    assert inspector._figure is None
    assert controls.tensor_inspector_on is False


def test_clicking_a_visible_tensor_opens_shared_inspector_when_checkbox_is_on() -> None:
    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    left.tensor = np.array([1.0, 2.0], dtype=float)  # type: ignore[attr-defined]
    right.tensor = np.array([3.0, 4.0], dtype=float)  # type: ignore[attr-defined]
    connect(left, 0, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    inspector_index = _checkbutton_index(controls._checkbuttons, "Tensor inspector")
    _click_checkbutton(controls._checkbuttons, inspector_index)

    node_id = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "A"
    )
    node_position = np.asarray(controls.current_scene.positions[node_id], dtype=float)
    _dispatch_button_event_at_data(
        ax,
        x=float(node_position[0]),
        y=float(node_position[1]),
    )

    assert inspector._figure is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert "a" in inspector_controls._panel.main_ax.get_title().lower()
    assert controls.scheme_on is False
    assert controls.tensor_inspector_on is True


def test_tensor_inspector_uses_wider_left_compare_layout() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert inspector_controls._group_radio_ax is not None
    assert inspector_controls._mode_radio_ax is not None
    assert inspector._compare_radio is not None
    assert inspector._figure.get_size_inches()[0] > 7.2
    assert inspector._compare_toggle_button is not None
    assert inspector._capture_reference_button is not None
    assert inspector._clear_reference_button is not None

    compare_selector_bounds = inspector._compare_radio.ax.get_position().bounds
    compare_bounds = inspector._compare_toggle_button.ax.get_position().bounds
    capture_bounds = inspector._capture_reference_button.ax.get_position().bounds
    clear_bounds = inspector._clear_reference_button.ax.get_position().bounds
    group_bounds = inspector_controls._group_radio_ax.get_position().bounds
    mode_bounds = inspector_controls._mode_radio_ax.get_position().bounds
    compare_top = compare_selector_bounds[1] + compare_selector_bounds[3]
    compare_right = compare_selector_bounds[0] + compare_selector_bounds[2]

    assert compare_right <= group_bounds[0] + 0.01
    assert mode_bounds[0] > group_bounds[0]
    assert compare_bounds[0] == pytest.approx(compare_selector_bounds[0], abs=0.005)
    assert compare_bounds[1] >= compare_top
    assert compare_bounds[1] - compare_top <= 0.02
    assert compare_bounds[2] <= 0.12
    assert compare_bounds[3] <= 0.06
    assert capture_bounds[1] == pytest.approx(compare_bounds[1], abs=0.005)
    assert clear_bounds[1] == pytest.approx(compare_bounds[1], abs=0.005)
    assert capture_bounds[0] > compare_bounds[0] + compare_bounds[2] - 0.01
    assert clear_bounds[0] > capture_bounds[0] + capture_bounds[2] - 0.01
    assert inspector._capture_reference_button.label.get_text() == "⌖"
    assert inspector._clear_reference_button.label.get_text() == "x"


def test_tensor_inspector_layout_moves_tensor_slider_lower_with_more_label_gap() -> None:
    fig, _ax, controller = _show_tensor_records(
        _records_for_inspector_slider_layout(),
        config=TensorElementsConfig(figsize=(8.8, 6.4)),
        controls_layout=_INSPECTOR_TENSOR_ELEMENTS_LAYOUT,
        ax=None,
        show_controls=True,
        show=False,
    )

    assert controller._mode_radio_ax is not None
    assert controller._slider_ax is not None
    assert controller._slider is not None

    mode_bounds = controller._mode_radio_ax.get_position().bounds
    slider_bounds = controller._slider_ax.get_position().bounds

    assert slider_bounds[1] <= mode_bounds[1] - 0.015
    assert controller._slider.label.get_position()[0] <= -0.075

    plt.close(fig)


def test_tensor_inspector_analysis_axis_layout_expands_left_and_down_without_hitting_slider() -> (
    None
):
    fig, _ax, controller = _show_tensor_records(
        _records_for_inspector_slider_layout(),
        config=TensorElementsConfig(figsize=(8.8, 6.4)),
        controls_layout=_INSPECTOR_TENSOR_ELEMENTS_LAYOUT,
        ax=None,
        show_controls=True,
        show=False,
    )

    controller.set_group("analysis")
    controller.set_mode("slice")

    assert controller._mode_radio_ax is not None
    assert controller._analysis_axis_ax is not None
    assert controller._slider_ax is not None

    mode_bounds = controller._mode_radio_ax.get_position().bounds
    analysis_bounds = controller._analysis_axis_ax.get_position().bounds
    slider_bounds = controller._slider_ax.get_position().bounds
    mode_right = mode_bounds[0] + mode_bounds[2]
    slider_top = slider_bounds[1] + slider_bounds[3]

    assert analysis_bounds[0] >= mode_right - 0.005
    assert analysis_bounds[0] - mode_right <= 0.01
    assert analysis_bounds[1] >= slider_top
    assert analysis_bounds[1] - slider_top <= 0.02
    assert analysis_bounds[2] >= 0.20
    assert analysis_bounds[3] >= 0.159

    plt.close(fig)


def test_tensor_inspector_reference_action_buttons_show_hover_tooltips() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    assert inspector._capture_reference_button is not None
    assert inspector._clear_reference_button is not None
    assert inspector._button_hover_text is not None

    hover_text = inspector._button_hover_text
    assert hover_text.get_visible() is False

    _dispatch_motion_event_at_widget(inspector._capture_reference_button)
    assert hover_text.get_visible() is True
    assert hover_text.get_text() == "Capture reference"
    capture_bounds = inspector._capture_reference_button.ax.get_position().bounds
    capture_top = capture_bounds[1] + capture_bounds[3]
    capture_x, capture_y = hover_text.get_position()
    assert capture_x == pytest.approx(capture_bounds[0], abs=0.01)
    assert capture_y >= capture_top
    assert capture_y - capture_top <= 0.02

    _dispatch_motion_event_at_widget(inspector._clear_reference_button)
    assert hover_text.get_visible() is True
    assert hover_text.get_text() == "Clear reference"
    clear_bounds = inspector._clear_reference_button.ax.get_position().bounds
    clear_top = clear_bounds[1] + clear_bounds[3]
    clear_x, clear_y = hover_text.get_position()
    assert clear_x == pytest.approx(clear_bounds[0], abs=0.01)
    assert clear_y >= clear_top
    assert clear_y - clear_top <= 0.02

    event = MouseEvent("motion_notify_event", inspector._figure.canvas, 1, 1)
    inspector._figure.canvas.callbacks.process("motion_notify_event", event)
    assert hover_text.get_visible() is False


def test_tensor_inspector_manual_node_selection_takes_precedence_until_cleared() -> None:
    trace = _build_einsum_trace_for_inspector()

    fig, ax = show_tensor_network(
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

    viewer = controls.current_scene.contraction_controls._viewer
    assert viewer is not None
    viewer.set_step(1)
    assert "r0" in inspector_controls._panel.main_ax.get_title().lower()

    left_node_id = next(
        node_id
        for node_id, node in controls.current_scene.graph.nodes.items()
        if node.name == "Left"
    )
    left_position = np.asarray(controls.current_scene.positions[left_node_id], dtype=float)
    _dispatch_button_event_at_data(
        ax,
        x=float(left_position[0]),
        y=float(left_position[1]),
    )
    assert "left" in inspector_controls._panel.main_ax.get_title().lower()

    viewer.set_step(2)
    assert "left" in inspector_controls._panel.main_ax.get_title().lower()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _dispatch_button_event_at_data(
        ax,
        x=float(xlim[0]),
        y=float(ylim[0]),
    )
    assert "r1" in inspector_controls._panel.main_ax.get_title().lower()


def test_tensor_inspector_can_compare_against_previous_playback_step() -> None:
    trace, r0, r1 = _build_einsum_trace_for_comparison_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None

    viewer = controls.current_scene.contraction_controls._viewer
    assert viewer is not None
    viewer.set_step(2)
    inspector.set_compare_mode("abs_diff")

    expected = np.abs(r1 - r0)
    assert (
        np.asarray(
            inspector_controls._panel.main_ax.images[0].get_array(),
            dtype=float,
        ).tolist()
        == expected.tolist()
    )


def test_tensor_inspector_can_compare_against_captured_reference() -> None:
    left = DummyTensorKrowchNode("A", ["left", "bond"])
    right = DummyTensorKrowchNode("B", ["bond", "right"])
    left.tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)  # type: ignore[attr-defined]
    right.tensor = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)  # type: ignore[attr-defined]
    connect(left, 1, right, 0, name="bond")

    fig, ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    inspector_index = _checkbutton_index(controls._checkbuttons, "Tensor inspector")
    _click_checkbutton(controls._checkbuttons, inspector_index)

    node_id_a = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "A"
    )
    pos_a = np.asarray(controls.current_scene.positions[node_id_a], dtype=float)
    _dispatch_button_event_at_data(
        ax,
        x=float(pos_a[0]),
        y=float(pos_a[1]),
    )
    inspector.capture_reference()

    node_id_b = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "B"
    )
    pos_b = np.asarray(controls.current_scene.positions[node_id_b], dtype=float)
    _dispatch_button_event_at_data(
        ax,
        x=float(pos_b[0]),
        y=float(pos_b[1]),
    )
    inspector.set_compare_mode("ratio")

    inspector_controls = getattr(
        inspector._figure,
        "_tensor_network_viz_tensor_elements_controls",
        None,
    )
    assert inspector_controls is not None
    assert np.asarray(
        inspector_controls._panel.main_ax.images[0].get_array(),
        dtype=float,
    ).tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_tensor_inspector_current_reference_toggle_button_switches_base_modes() -> None:
    trace, _r0, _r1 = _build_einsum_trace_for_comparison_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    assert inspector._compare_toggle_button is not None
    assert inspector._compare_toggle_button.label.get_text() == "Reference"

    _click_button(inspector._compare_toggle_button)
    assert inspector._compare_mode == "reference"
    assert inspector._compare_toggle_button.label.get_text() == "Current"

    inspector.set_compare_mode("abs_diff")
    assert inspector._compare_mode == "abs_diff"
    assert inspector._compare_toggle_button.label.get_text() == "Current"

    _click_button(inspector._compare_toggle_button)
    assert inspector._compare_mode == "current"
    assert inspector._compare_toggle_button.label.get_text() == "Reference"


def test_tensor_inspector_buttons_release_stale_mouse_grabber_before_click() -> None:
    trace, _r0, _r1 = _build_einsum_trace_for_comparison_inspector()

    fig, _ax = show_tensor_network(
        trace,
        config=PlotConfig(contraction_tensor_inspector=True),
        show=False,
    )

    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)
    assert inspector is not None
    assert inspector._figure is not None
    assert inspector._compare_toggle_button is not None
    stale_ax = inspector._figure.add_axes((0.9, 0.9, 0.05, 0.05))
    stale_ax.set_visible(False)

    inspector._figure.canvas.grab_mouse(stale_ax)
    assert inspector._figure.canvas.mouse_grabber is stale_ax

    _click_button(inspector._compare_toggle_button)

    assert inspector._compare_mode == "reference"
    assert inspector._figure.canvas.mouse_grabber is None


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
    assert controls._view_toggle_button is None


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

    with pytest.raises(ValueError, match="show_controls=True"):
        show_tensor_network(
            DummyNetwork(nodes=[left, right]),
            engine="tensorkrowch",
            config=PlotConfig(
                show_contraction_scheme=True,
                contraction_scheme_by_name=(("A", "B"),),
            ),
            show_controls=False,
            show=False,
        )


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


def test_show_tensor_network_defers_label_descriptors_until_menu_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._interactive_scene as interactive_scene_module

    left = DummyTensorKrowchNode("A", ["left"])
    right = DummyTensorKrowchNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")

    calls = {"edge": 0, "tensor": 0}
    original_build_tensor = interactive_scene_module._build_tensor_label_descriptors
    original_build_edge = interactive_scene_module._build_edge_label_descriptors

    def counting_build_tensor(*args: object, **kwargs: object) -> object:
        calls["tensor"] += 1
        return original_build_tensor(*args, **kwargs)

    def counting_build_edge(*args: object, **kwargs: object) -> object:
        calls["edge"] += 1
        return original_build_edge(*args, **kwargs)

    monkeypatch.setattr(
        interactive_scene_module,
        "_build_tensor_label_descriptors",
        counting_build_tensor,
    )
    monkeypatch.setattr(
        interactive_scene_module,
        "_build_edge_label_descriptors",
        counting_build_edge,
    )

    fig, _ax = show_tensor_network(
        DummyNetwork(nodes=[left, right]),
        engine="tensorkrowch",
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    scene = controls.current_scene

    assert calls == {"edge": 0, "tensor": 0}
    assert scene.tensor_label_descriptors is None
    assert scene.edge_label_descriptors is None
    assert scene.tensor_label_artists == []
    assert scene.edge_label_artists == []

    controls.set_tensor_labels_enabled(True)
    controls.set_edge_labels_enabled(True)

    assert calls == {"edge": 1, "tensor": 1}
    assert scene.tensor_label_descriptors is not None
    assert len(scene.tensor_label_descriptors) == 2
    assert scene.edge_label_descriptors is not None
    assert len(scene.edge_label_descriptors) == 2
    assert len(scene.tensor_label_artists) == 2
    assert len(scene.edge_label_artists) == 2


def test_show_tensor_network_menu_toggles_reuse_materialized_label_descriptors(
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

    def _unexpected_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("label descriptors should already be materialized in the scene")

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

    controls.set_tensor_labels_enabled(False)
    controls.set_edge_labels_enabled(False)
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
    stub_length = 0.2

    _, ax_normal = plot_tensorkrowch_network_2d(
        DummyNetwork(leaf_nodes=[node]),
        config=PlotConfig(
            show_nodes=True,
            stub_length=stub_length,
            positions={id(node): (float(center[0]), float(center[1]))},
        ),
    )
    _, ax = plot_tensorkrowch_network_2d(
        DummyNetwork(leaf_nodes=[node]),
        config=PlotConfig(
            show_nodes=False,
            stub_length=stub_length,
            positions={id(node): (float(center[0]), float(center[1]))},
        ),
    )

    segs = line_collection_segments(ax)
    assert len(segs) == 1
    seg = np.asarray(segs[0], dtype=float)
    seg_normal = np.asarray(line_collection_segments(ax_normal)[0], dtype=float)
    distances = np.linalg.norm(seg - center, axis=1)
    assert float(np.min(distances)) == pytest.approx(0.0)
    assert float(np.linalg.norm(seg[1] - seg[0])) == pytest.approx(
        float(np.linalg.norm(seg_normal[1] - seg_normal[0])),
        rel=1e-6,
    )


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

    assert_rendered_figure(fig, ax)
    assert line_collection_segment_count(ax) == 1


def test_show_tensor_network_static_nested_grid_2d_renders_with_holes() -> None:
    left = DummyTensorNetworkNode("L", ["a", "b"])
    right = DummyTensorNetworkNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")

    fig, ax = show_tensor_network(
        [[left, None], [None, right]],
        config=PlotConfig(show_tensor_labels=True),
        show=False,
        show_controls=False,
    )

    labels = {text.get_text() for text in ax.texts}
    assert_rendered_figure(fig, ax)
    assert labels >= {"L", "R"}


def test_show_tensor_network_nested_grid_3d_defaults_to_3d_view() -> None:
    left = DummyTensorNetworkNode("L", ["a", "b"])
    right = DummyTensorNetworkNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")

    fig, ax = show_tensor_network([[[left], [right]]], show=False, show_controls=False)

    assert_rendered_figure(fig, ax)
    assert ax.name == "3d"


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

    assert_rendered_figure(fig, ax)
    assert ax.name == "3d"
    assert len(_line3d_collections(ax)) == 1
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


def test_plot_tensorkrowch_network_3d_draws_nodes_above_dangling_edges() -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    node = DummyTensorKrowchNode("A", ["left"])
    connect(node, 0, name="left")

    _, ax = plot_tensorkrowch_network_3d(
        DummyNetwork(leaf_nodes=[node]),
        config=PlotConfig(show_nodes=True),
    )

    line_zorders = [float(collection.get_zorder()) for collection in _line3d_collections(ax)]
    node_zorders = [
        float(collection.get_zorder())
        for collection in ax.collections
        if isinstance(collection, Poly3DCollection)
    ]

    assert line_zorders
    assert node_zorders
    assert max(line_zorders) < min(node_zorders)


def test_plot_tensornetwork_network_3d_returns_3d_axes() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0)

    fig, ax = plot_tensornetwork_network_3d([left, right])

    assert_rendered_figure(fig, ax)
    assert ax.name == "3d"
    assert len(_line3d_collections(ax)) == 1


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


def test_show_tensor_network_with_external_axis_does_not_relayout_sibling_subplots() -> None:
    left = DummyTensorNetworkNode("A", ["left"])
    right = DummyTensorNetworkNode("B", ["right"])
    connect(left, 0, right, 0, name="bond")
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    sibling_bounds_before = axes[1].get_position().bounds

    _fig, ax = show_tensor_network(
        [left, right],
        ax=axes[0],
        show=False,
        show_controls=False,
    )

    assert ax is axes[0]
    assert np.allclose(axes[1].get_position().bounds, sibling_bounds_before)
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
    assert_rendered_figure(fig, ax)


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
    assert_rendered_figure(fig, ax)


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
    assert_rendered_figure(fig, ax)


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
    assert_rendered_figure(fig, ax)


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
    assert_rendered_figure(fig, ax)


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
