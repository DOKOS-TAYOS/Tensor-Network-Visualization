"""Tests for contraction playback UI and plot pipeline integration."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Slider

from tensor_network_viz import ContractionViewer2D, PlotConfig, pair_tensor
from tensor_network_viz._core.renderer import _plot_graph
from tensor_network_viz.contraction_viewer import attach_playback_to_tensor_network_figure
from tensor_network_viz.einsum_module.graph import _build_graph


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


def test_plot_graph_contraction_playback_adds_widgets() -> None:
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
            contraction_playback=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_playback",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None
    assert viewer.slider is not None
    assert len(fig.axes) >= 2


def test_plot_graph_scheme_without_playback_no_extra_slider_axes() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=2,
        config=PlotConfig(figsize=(4, 3), show_contraction_scheme=True),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_no_playback",
    )
    assert getattr(fig, "_tensor_network_viz_contraction_viewer", None) is None
    assert len(fig.axes) == 1


def test_contraction_playback_without_scheme_raises() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    graph = _build_graph(trace)
    with pytest.raises(ValueError, match="show_contraction_scheme=True"):
        _plot_graph(
            graph,
            dimensions=2,
            config=PlotConfig(
                figsize=(4, 3),
                show_contraction_scheme=False,
                contraction_playback=True,
            ),
            show_tensor_labels=False,
            show_index_labels=False,
            renderer_name="test_playback_bad",
        )


def test_attach_playback_requires_drawable_artists() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    r = Rectangle((0, 0), 1, 1)
    ax.add_patch(r)
    cfg = PlotConfig()
    attach_playback_to_tensor_network_figure(
        artists_by_step=[r],
        fig=fig,
        ax=ax,
        config=cfg,
    )
    assert fig.axes  # main + widget axes


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


def test_plot_graph_contraction_playback_3d_hides_past_scheme_steps() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
        pair_tensor("r2", "A2", "r3", "b,bqc->qc"),
        pair_tensor("r3", "x2", "r4", "qc,q->c"),
        pair_tensor("r4", "A3", "r5", "c,crd->rd"),
        pair_tensor("r5", "x3", "r6", "rd,r->d"),
        pair_tensor("r6", "A4", "r7", "d,dse->se"),
    ]
    graph = _build_graph(trace)
    fig, ax = _plot_graph(
        graph,
        dimensions=3,
        config=PlotConfig(
            figsize=(5, 4),
            show_contraction_scheme=True,
            contraction_playback=True,
        ),
        show_tensor_labels=False,
        show_index_labels=False,
        renderer_name="test_playback_3d_scheme_visibility",
    )
    viewer = getattr(fig, "_tensor_network_viz_contraction_viewer", None)
    assert viewer is not None

    viewer.set_step(6)

    visible_steps = [
        artist.get_visible() for artist in viewer._artists if artist is not None
    ]
    assert len(visible_steps) >= 6
    assert sum(1 for visible in visible_steps if visible) == 1
    assert viewer._artists[5] is not None
    assert viewer._artists[5].get_visible()
