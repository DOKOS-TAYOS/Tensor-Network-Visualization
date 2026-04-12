"""Tests for contraction playback UI and plot pipeline integration."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tensor_network_viz import ContractionViewer2D, PlotConfig, pair_tensor
from tensor_network_viz._contraction_viewer_ui import create_playback_details_panel
from tensor_network_viz._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.contraction_viewer import attach_playback_to_tensor_network_figure
from tensor_network_viz.einsum_module.graph import _build_graph


def _widget_center_event(fig: matplotlib.figure.Figure, artist: object) -> MouseEvent:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer)  # type: ignore[attr-defined]
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    return MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)


def _click_checkbutton(checkbuttons: CheckButtons, index: int) -> None:
    event = _widget_center_event(checkbuttons.ax.figure, checkbuttons.labels[index])
    checkbuttons._clicked(event)


def _click_button(button: Button) -> None:
    fig = button.ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = button.ax.get_window_extent(renderer)
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=MouseButton.LEFT)
    button._click(press)
    button._release(release)


def _press_slider(slider: Slider, *, fraction: float = 0.5) -> None:
    fig = slider.ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = slider.ax.get_window_extent(renderer)
    x = int(round(bbox.x0 + float(fraction) * (bbox.x1 - bbox.x0)))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", press)


def _collections_with_gid(
    ax: matplotlib.axes.Axes | matplotlib.axes.Axes,
    gid: str,
) -> list[Any]:
    return [
        artist for artist in ax.collections if getattr(artist, "get_gid", lambda: None)() == gid
    ]


def _first_rgba(artist: Any) -> np.ndarray:
    getter = getattr(artist, "get_facecolor", None)
    if callable(getter):
        data = np.asarray(getter(), dtype=float)
    else:
        data = np.asarray(artist.get_facecolors(), dtype=float)
    return np.asarray(data[0], dtype=float)


def test_contraction_viewer_set_step_clamps_and_updates_visibility() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    rects = [
        Rectangle((0, 0), 1, 1),
        Rectangle((2, 0), 1, 1),
    ]
    for r in rects:
        ax.add_patch(r)
    v = ContractionViewer2D(
        rects,
        fig=fig,
        ax=ax,
        enable_playback=False,
        mode="cumulative",
    )
    v.set_step(0)
    assert not rects[0].get_visible()
    v.set_step(1)
    assert rects[0].get_visible()
    assert not rects[1].get_visible()
    v.set_step(99)
    assert v.current_step == 2
    for r in rects:
        assert r.get_visible()


def test_playback_details_panel_disables_matplotlib_auto_wrap() -> None:
    fig = matplotlib.pyplot.figure(figsize=(4, 3))

    try:
        _ax_details, text = create_playback_details_panel(fig)

        assert text.get_wrap() is False
    finally:
        matplotlib.pyplot.close(fig)


def test_enable_playback_false_build_ui_is_noop() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    v = ContractionViewer2D([r], fig=fig, ax=ax, enable_playback=False)
    n_axes_before = len(fig.axes)
    v.build_ui()
    assert len(fig.axes) == n_axes_before
    assert v.slider is None


def test_enable_playback_true_creates_slider_and_buttons() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    v = ContractionViewer2D([r], fig=fig, ax=ax, enable_playback=True)
    v.build_ui()
    assert v.slider is not None
    assert isinstance(v.slider, Slider)
    widgets = [axw for axw in fig.axes if axw is not ax]
    assert len(widgets) >= 3
    slider_bounds = v.slider.ax.get_position().bounds
    main_bounds = ax.get_position().bounds
    assert slider_bounds[2] <= 0.48
    assert main_bounds[1] >= 0.22


def test_default_playback_slider_uses_theme_bond_purple() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    config = PlotConfig()
    v = ContractionViewer2D([r], fig=fig, ax=ax, config=config, enable_playback=True)

    v.build_ui()

    assert v.slider is not None
    assert v.slider.poly.get_facecolor() == pytest.approx(to_rgba(config.bond_edge_color))
    assert v.slider.track.get_facecolor() == pytest.approx(
        (0.907, 0.861, 0.987, 1.0),
        abs=0.001,
    )
    assert v.slider._handle.get_markerfacecolor() == config.bond_edge_color
    assert v.slider._handle.get_markeredgecolor() == config.node_edge_color


def test_playback_buttons_use_polished_state_styles() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    rects = [Rectangle((index, 0), 1, 1) for index in range(2)]
    for rect in rects:
        ax.add_patch(rect)
    v = ContractionViewer2D(rects, fig=fig, ax=ax, enable_playback=True)

    v.build_ui()

    assert v._btn_play is not None
    assert v._btn_pause is not None
    assert v._btn_reset is not None
    assert to_hex(v._btn_play.ax.patch.get_facecolor()).lower() == "#f8fafc"
    assert to_hex(v._btn_pause.ax.patch.get_facecolor()).lower() == "#dbeafe"
    assert to_hex(v._btn_reset.ax.patch.get_facecolor()).lower() == "#f8fafc"

    v.play()
    assert to_hex(v._btn_play.ax.patch.get_facecolor()).lower() == "#dbeafe"
    assert to_hex(v._btn_pause.ax.patch.get_facecolor()).lower() == "#f8fafc"

    v.pause()
    assert to_hex(v._btn_play.ax.patch.get_facecolor()).lower() == "#f8fafc"
    assert to_hex(v._btn_pause.ax.patch.get_facecolor()).lower() == "#dbeafe"

    v.reset()
    assert to_hex(v._btn_reset.ax.patch.get_facecolor()).lower() == "#e5e7eb"
    assert to_hex(v._btn_reset.label.get_color()).lower() == "#64748b"


def test_playback_slider_releases_stale_mouse_grabber_before_drag() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    stale_ax = fig.add_axes((0.9, 0.9, 0.05, 0.05))
    stale_ax.set_visible(False)
    v = ContractionViewer2D([r], fig=fig, ax=ax, enable_playback=True)
    v.build_ui()

    assert v.slider is not None
    fig.canvas.grab_mouse(stale_ax)
    assert fig.canvas.mouse_grabber is stale_ax

    _press_slider(v.slider)

    assert fig.canvas.mouse_grabber is v.slider.ax


def test_playback_buttons_release_stale_mouse_grabber_before_click() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    stale_ax = fig.add_axes((0.9, 0.9, 0.05, 0.05))
    stale_ax.set_visible(False)
    v = ContractionViewer2D([r], fig=fig, ax=ax, enable_playback=True)
    v.build_ui()
    v.set_step(1)

    assert v._btn_reset is not None
    fig.canvas.grab_mouse(stale_ax)
    assert fig.canvas.mouse_grabber is stale_ax

    _click_button(v._btn_reset)

    assert v.current_step == 0
    assert fig.canvas.mouse_grabber is None


def test_playback_slider_snaps_fractional_values_to_discrete_steps() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    rects = [Rectangle((index, 0), 1, 1) for index in range(3)]
    for rect in rects:
        ax.add_patch(rect)
    v = ContractionViewer2D(rects, fig=fig, ax=ax, enable_playback=True)
    v.build_ui()

    assert v.slider is not None
    v.slider.set_val(2.6)

    assert v.current_step == 3
    assert float(v.slider.val) == pytest.approx(3.0)


def test_plot_graph_contraction_scheme_adds_widgets() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_playback",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is viewer
    assert viewer.slider is not None
    assert isinstance(viewer.slider, Slider)
    assert viewer.slider.valmin == 0
    assert viewer.slider.valmax == len(trace)
    assert viewer.current_step == len(trace)
    assert viewer.slider.ax.get_visible()
    assert viewer._btn_play is not None
    assert viewer._btn_play.label.get_text() == "Play"
    assert viewer._btn_pause is not None
    assert viewer._btn_pause.label.get_text() == "Pause"
    assert viewer._btn_reset is not None
    assert viewer._btn_reset.label.get_text() == "Reset"
    assert viewer._cost_panel_ax is not None
    assert viewer._cost_panel_ax in fig.axes
    assert not viewer._cost_panel_ax.get_visible()
    assert len(fig.axes) >= 6


def test_attach_playback_registers_viewer_and_builds_2d_controls() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    cfg = PlotConfig()
    viewer = attach_playback_to_tensor_network_figure(
        artists_by_step=[r],
        fig=fig,
        ax=ax,
        config=cfg,
    )
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is viewer
    assert viewer.slider is not None
    assert viewer.slider.valmin == 0
    assert viewer.slider.valmax == 1
    assert viewer.current_step == 1
    assert viewer._btn_play is not None
    assert viewer._btn_pause is not None
    assert viewer._btn_reset is not None
    viewer.reset()
    assert viewer.current_step == 0
    assert viewer.slider.val == 0


def test_attach_playback_uses_3d_viewer_for_3d_axes() -> None:
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    artist = Poly3DCollection([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]])
    ax.add_collection3d(artist)

    viewer = attach_playback_to_tensor_network_figure(
        artists_by_step=[artist],
        fig=fig,
        ax=ax,
        config=PlotConfig(),
    )

    assert viewer.__class__.__name__ == "ContractionViewer3D"
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is viewer
    assert viewer.slider is not None
    assert viewer.slider.valmax == 1
    assert viewer.current_step == 1


def test_highlight_current_2d_scheme_patch_faint_fill_opaque_edge() -> None:
    """FancyBboxPatch + scheme gid uses tint+edge in playback (not gray-filled past)."""
    fig, ax = matplotlib.pyplot.subplots()
    p0 = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.02",
        facecolor=(0.2, 0.4, 0.9, 0.5),
        edgecolor=(0.1, 0.1, 0.2, 1.0),
        gid="tnv_contraction_scheme",
    )
    p1 = FancyBboxPatch(
        (2, 0),
        1,
        1,
        boxstyle="round,pad=0.02",
        facecolor=(0.9, 0.2, 0.2, 0.5),
        edgecolor=(0.2, 0.1, 0.1, 1.0),
        gid="tnv_contraction_scheme",
    )
    ax.add_patch(p0)
    ax.add_patch(p1)
    v = ContractionViewer2D(
        [p0, p1],
        fig=fig,
        ax=ax,
        enable_playback=False,
        mode="highlight_current",
        current_color="tab:red",
        scheme_2d_highlight_fill_alpha=0.08,
    )
    v.set_step(2)
    past_fc = p0.get_facecolor()
    assert float(np.ravel(past_fc)[-1]) == pytest.approx(0.0)
    cur_fc = p1.get_facecolor()
    assert float(np.ravel(cur_fc)[-1]) == pytest.approx(0.08)
    cur_ec = p1.get_edgecolor()
    assert float(np.ravel(cur_ec)[-1]) == pytest.approx(1.0)


def test_pause_stops_play_without_timer_crash() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    v = ContractionViewer2D([r], fig=fig, ax=ax, enable_playback=True)
    v.build_ui()
    v.play()
    v.pause()
    assert not v._is_playing


def test_plot_graph_contraction_scheme_2d_uses_real_square_and_circle_nodes() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_real_nodes_2d",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None

    viewer.set_step(2)

    scheme_artists = [
        artist
        for artist in ax.collections
        if getattr(artist, "get_gid", lambda: None)() == "tnv_contraction_scheme"
    ]
    assert scheme_artists == []
    circles = _collections_with_gid(ax, "tnv_tensor_nodes_circle")
    squares = _collections_with_gid(ax, "tnv_tensor_nodes_square")
    assert len(circles) == 1
    assert len(squares) == 1
    assert getattr(circles[0], "_tnv_node_count", None) == 1
    assert getattr(squares[0], "_tnv_node_count", None) == 2
    assert np.allclose(_first_rgba(circles[0]), _first_rgba(squares[0]))


def test_plot_graph_contraction_scheme_3d_colors_current_octahedra_and_past_cubes() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=3,
        config=PlotConfig(
            figsize=(5, 4),
            show_contraction_scheme=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_real_nodes_3d",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None

    viewer.set_step(2)

    cubes = _collections_with_gid(ax, "tnv_tensor_nodes_cube")
    octahedra = _collections_with_gid(ax, "tnv_tensor_nodes_octahedron")
    assert len(cubes) == 1
    assert len(octahedra) == 1
    assert getattr(cubes[0], "_tnv_node_count", None) == 2
    assert getattr(octahedra[0], "_tnv_node_count", None) == 1
    assert np.allclose(_first_rgba(cubes[0]), _first_rgba(octahedra[0]))


def test_plot_graph_contraction_scheme_merge_keeps_oldest_group_color() -> None:
    trace = [
        pair_tensor("A", "B", "ab", "i,j->ij"),
        pair_tensor("C", "D", "cd", "k,l->kl"),
        pair_tensor("ab", "cd", "out", "ij,kl->ijkl"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_group_color_merge",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None

    viewer.set_step(1)
    first_step_color = _first_rgba(_collections_with_gid(ax, "tnv_tensor_nodes_circle")[0])
    viewer.set_step(2)
    second_step_color = _first_rgba(_collections_with_gid(ax, "tnv_tensor_nodes_circle")[0])
    assert not np.allclose(first_step_color, second_step_color)

    viewer.set_step(3)
    merged_color = _first_rgba(_collections_with_gid(ax, "tnv_tensor_nodes_square")[0])
    assert np.allclose(merged_color, first_step_color)


def test_plot_graph_lazy_scheme_controls_render_without_initial_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.draw.graph_pipeline as graph_pipeline

    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    calls: list[str] = []

    original_steps = graph_pipeline._effective_contraction_steps
    original_states = graph_pipeline._build_contraction_playback_states
    original_metrics = graph_pipeline._contraction_step_metrics_for_draw

    def counting_steps(*args: Any, **kwargs: Any) -> Any:
        calls.append("steps")
        return original_steps(*args, **kwargs)

    def counting_states(*args: Any, **kwargs: Any) -> Any:
        calls.append("states")
        return original_states(*args, **kwargs)

    def counting_metrics(*args: Any, **kwargs: Any) -> Any:
        calls.append("metrics")
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(graph_pipeline, "_effective_contraction_steps", counting_steps)
    monkeypatch.setattr(graph_pipeline, "_build_contraction_playback_states", counting_states)
    monkeypatch.setattr(graph_pipeline, "_contraction_step_metrics_for_draw", counting_metrics)

    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_lazy_scheme_controls",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert calls == []
    assert controls._checkbuttons is not None
    assert tuple(bool(v) for v in controls._checkbuttons.get_status()) == (False, False)


def test_lazy_scheme_click_builds_bundle_once_and_costs_reuse_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.draw.graph_pipeline as graph_pipeline

    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    counts = {"steps": 0, "states": 0, "metrics": 0}

    original_steps = graph_pipeline._effective_contraction_steps
    original_states = graph_pipeline._build_contraction_playback_states
    original_metrics = graph_pipeline._contraction_step_metrics_for_draw

    def counting_steps(*args: Any, **kwargs: Any) -> Any:
        counts["steps"] += 1
        return original_steps(*args, **kwargs)

    def counting_states(*args: Any, **kwargs: Any) -> Any:
        counts["states"] += 1
        return original_states(*args, **kwargs)

    def counting_metrics(*args: Any, **kwargs: Any) -> Any:
        counts["metrics"] += 1
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(graph_pipeline, "_effective_contraction_steps", counting_steps)
    monkeypatch.setattr(graph_pipeline, "_build_contraction_playback_states", counting_states)
    monkeypatch.setattr(graph_pipeline, "_contraction_step_metrics_for_draw", counting_metrics)

    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_lazy_scheme_click",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None

    _click_checkbutton(controls._checkbuttons, 0)

    assert counts == {"steps": 1, "states": 1, "metrics": 1}
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is not None

    _click_checkbutton(controls._checkbuttons, 1)

    assert counts == {"steps": 1, "states": 1, "metrics": 1}
    assert controls._viewer is not None
    assert controls._viewer.slider is not None
    assert ax.get_position().bounds[1] >= 0.22


def test_scheme_play_button_and_toggle_pause_and_hide_controls() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_lazy_playback_pause",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None

    _click_checkbutton(controls._checkbuttons, 0)

    assert controls._viewer is not None
    assert controls._viewer._btn_play is not None
    _click_button(controls._viewer._btn_play)
    assert controls._viewer._is_playing

    _click_checkbutton(controls._checkbuttons, 0)

    assert not controls._viewer._is_playing
    assert controls._viewer.slider is not None
    assert not controls._viewer.slider.ax.get_visible()
    assert controls._viewer._btn_play is not None
    assert not controls._viewer._btn_play.ax.get_visible()


def test_cost_hover_click_auto_enables_scheme_and_registers_hover() -> None:
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
    graph = _build_graph(trace)
    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
            hover_labels=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_lazy_cost_hover",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None

    _click_checkbutton(controls._checkbuttons, 1)

    assert controls.scheme_on
    assert controls.cost_hover_on
    assert getattr(fig, "_tensor_network_viz_hover_cid", None) is None
    assert controls._viewer is not None
    assert controls._viewer._cost_panel_ax is not None
    assert controls._viewer._cost_text_artist is not None
    assert controls._viewer._cost_panel_ax.get_visible()
    assert "Contraction:" in controls._viewer._cost_text_artist.get_text()
    controls._viewer.set_step(0)
    assert not controls._viewer._cost_panel_ax.get_visible()


def test_cost_hover_truncates_overflow_and_shows_full_text_on_hover() -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3))
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    long_hover_line = "Long context: " + " ".join(f"label_{i}" for i in range(30))
    full_text = "\n".join(
        [
            "Contraction: A_with_a_long_name x B_with_a_long_name -> C_with_a_long_name",
            "Equation: abcdefghijklmnopqrstuvwxyz,abcdefghijklmnopqrstuvwxyz->long_output",
            "MACs: 123456789",
            "FLOPs: 246913578",
            "Peak tensor size: 987654321",
            long_hover_line,
        ]
    )
    viewer = ContractionViewer2D(
        [rect],
        fig=fig,
        ax=ax,
        enable_playback=True,
        step_details_by_step=[full_text],
        initial_step=1,
    )
    try:
        viewer.build_ui()
        viewer.set_step_details_enabled(True)
        viewer.set_step(1)
        fig.canvas.draw()

        assert viewer._cost_panel_ax is not None
        assert viewer._cost_text_artist is not None
        assert viewer._cost_panel_hover_annotation is not None
        panel_text = viewer._cost_text_artist.get_text()
        assert panel_text != full_text
        assert panel_text.endswith("...")

        renderer = fig.canvas.get_renderer()
        bbox = viewer._cost_panel_ax.get_window_extent(renderer)
        event = MouseEvent(
            "motion_notify_event",
            fig.canvas,
            int(round((bbox.x0 + bbox.x1) / 2.0)),
            int(round((bbox.y0 + bbox.y1) / 2.0)),
        )
        fig.canvas.callbacks.process("motion_notify_event", event)

        assert viewer._cost_panel_hover_annotation.get_visible()
        hover_text = viewer._cost_panel_hover_annotation.get_text()
        assert "label_29" in hover_text
        assert max(len(line) for line in hover_text.splitlines()) < len(long_hover_line)
        assert viewer._cost_panel_hover_annotation.axes is None
    finally:
        matplotlib.pyplot.close(fig)


def test_cost_hover_truncation_does_not_flash_full_panel_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3))
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    full_text = (
        "Contraction: "
        "A_with_an_extremely_long_name_and_many_labels x "
        "B_with_an_extremely_long_name_and_many_labels -> "
        "C_with_an_extremely_long_name_and_many_labels"
    )
    viewer = ContractionViewer2D(
        [rect],
        fig=fig,
        ax=ax,
        enable_playback=True,
        step_details_by_step=[full_text],
        initial_step=0,
    )
    try:
        viewer.build_ui()
        assert viewer._cost_text_artist is not None
        recorded_text: list[str] = []
        original_set_text = viewer._cost_text_artist.set_text

        def recording_set_text(text: object) -> None:
            recorded_text.append(str(text))
            original_set_text(text)

        monkeypatch.setattr(viewer._cost_text_artist, "set_text", recording_set_text)

        viewer.set_step_details_enabled(True)
        viewer.set_step(1)

        assert full_text not in recorded_text
        assert viewer._cost_text_artist.get_text().endswith("...")
    finally:
        matplotlib.pyplot.close(fig)


def test_cost_hover_keeps_multiline_text_when_each_line_fits() -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(9.88, 7.68))
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    full_text = "\n".join(
        [
            "Contraction: ab,bc->ac (contracts: b)",
            "Index sizes: a=2, b=16, c=2",
            "Tensor shapes: MPS=[2, 16], MPO=[16, 2]",
            "Naive operations: 64 MACs (approx 128 FLOPs)",
            "Complexity: O(N_a N_b N_c)",
        ]
    )
    viewer = ContractionViewer2D(
        [rect],
        fig=fig,
        ax=ax,
        enable_playback=True,
        step_details_by_step=[full_text],
        initial_step=1,
    )
    try:
        viewer.build_ui()
        viewer.set_step_details_enabled(True)
        viewer.set_step(1)
        fig.canvas.draw()

        assert viewer._cost_text_artist is not None
        assert viewer._cost_text_artist.get_text() == full_text
        assert viewer._cost_panel_hover_text is None
    finally:
        matplotlib.pyplot.close(fig)


def test_cost_hover_shrinks_text_before_enabling_hover() -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(9.88, 7.68))
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    full_text = "\n".join(
        [
            "Contraction: ab,bc->ac",
            "Index sizes: a=2, b=16, c=2",
            "Tensor shapes: A=[2, 16], B=[16, 2]",
            "Naive operations: 64 MACs",
            "Complexity: O(N_a N_b N_c)",
            "Output labels: a, c",
            "Contracted labels: b",
        ]
    )
    viewer = ContractionViewer2D(
        [rect],
        fig=fig,
        ax=ax,
        enable_playback=True,
        step_details_by_step=[full_text],
        initial_step=1,
    )
    try:
        viewer.build_ui()
        assert viewer._cost_text_artist is not None
        base_fontsize = float(viewer._cost_text_artist.get_fontsize())

        viewer.set_step_details_enabled(True)
        viewer.set_step(1)
        fig.canvas.draw()

        assert viewer._cost_text_artist.get_text() == full_text
        assert float(viewer._cost_text_artist.get_fontsize()) == pytest.approx(base_fontsize * 0.8)
        assert viewer._cost_panel_hover_text is None
    finally:
        matplotlib.pyplot.close(fig)


def test_cost_hover_leaves_short_text_without_panel_hover() -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3))
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    full_text = "Contraction: A x B -> C"
    viewer = ContractionViewer2D(
        [rect],
        fig=fig,
        ax=ax,
        enable_playback=True,
        step_details_by_step=[full_text],
        initial_step=1,
    )
    try:
        viewer.build_ui()
        viewer.set_step_details_enabled(True)
        viewer.set_step(1)
        fig.canvas.draw()

        assert viewer._cost_panel_ax is not None
        assert viewer._cost_text_artist is not None
        assert viewer._cost_panel_hover_annotation is not None
        assert viewer._cost_text_artist.get_text() == full_text

        renderer = fig.canvas.get_renderer()
        bbox = viewer._cost_panel_ax.get_window_extent(renderer)
        event = MouseEvent(
            "motion_notify_event",
            fig.canvas,
            int(round((bbox.x0 + bbox.x1) / 2.0)),
            int(round((bbox.y0 + bbox.y1) / 2.0)),
        )
        fig.canvas.callbacks.process("motion_notify_event", event)

        assert not viewer._cost_panel_hover_annotation.get_visible()
    finally:
        matplotlib.pyplot.close(fig)


def test_cost_hover_with_manual_scheme_and_no_metrics_does_not_crash() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("left",)),
            1: _make_node("B", ("right",)),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "left"),
                _EdgeEndpoint(1, 0, "right"),
                name="bond",
            ),
        ),
    )

    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
            contraction_scheme_by_name=(("A", "B"),),
            hover_labels=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_manual_scheme_cost_hover",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None

    _click_checkbutton(controls._checkbuttons, 1)

    assert controls.scheme_on
    assert controls.cost_hover_on
    assert getattr(fig, "_tensor_network_viz_hover_cid", None) is None
    assert controls._viewer is not None
    assert controls._viewer._cost_panel_ax is not None
    assert not controls._viewer._cost_panel_ax.get_visible()


def test_scheme_reenable_restores_recorded_playback_without_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.draw.graph_pipeline as graph_pipeline

    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    calls = {"steps": 0}
    original_steps = graph_pipeline._effective_contraction_steps

    def counting_steps(*args: Any, **kwargs: Any) -> Any:
        calls["steps"] += 1
        return original_steps(*args, **kwargs)

    monkeypatch.setattr(graph_pipeline, "_effective_contraction_steps", counting_steps)

    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=False,
            contraction_scheme_cost_hover=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_restore_playback",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._checkbuttons is not None

    _click_checkbutton(controls._checkbuttons, 0)
    assert calls["steps"] == 1
    assert controls._viewer is not None
    assert controls._viewer.slider is not None
    assert controls._viewer.slider.ax.get_visible()

    _click_checkbutton(controls._checkbuttons, 0)
    assert not controls._viewer.slider.ax.get_visible()

    _click_checkbutton(controls._checkbuttons, 0)
    assert calls["steps"] == 1
    assert controls._viewer.slider.ax.get_visible()


def test_scheme_step_ignores_unremovable_artists_in_scene() -> None:
    graph = _GraphData(
        nodes={
            0: _make_node("A", ("left",)),
            1: _make_node("B", ("right",)),
        },
        edges=(
            _make_contraction_edge(
                _EdgeEndpoint(0, 0, "left"),
                _EdgeEndpoint(1, 0, "right"),
                name="bond",
            ),
        ),
    )

    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(
            figsize=(4, 3),
            show_contraction_scheme=True,
            contraction_scheme_by_name=(("A", "B"),),
            hover_labels=False,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_scheme_unremovable_artist",
    )

    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)
    assert controls is not None
    assert controls._viewer is not None
    assert controls._scene is not None

    stuck_artist = Artist()
    controls._scene.scheme_artists.append(stuck_artist)

    controls._viewer.set_step(0)

    assert stuck_artist.get_visible() is False


def test_plot_graph_without_scheme_source_does_not_create_lazy_controls() -> None:
    graph = _GraphData(
        nodes={0: _make_node("A", ("i",))},
        edges=(),
    )

    fig, _ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(figsize=(4, 3)),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_no_scheme_source_controls",
    )

    assert getattr(fig, "_tensor_network_viz_contraction_controls", None) is None
