from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensor_network_viz import PlotConfig
from tensor_network_viz._core.draw.fonts_and_scale import _draw_scale_params
from tensor_network_viz._core.draw.plotter import _make_plotter
from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)
from tensor_network_viz._core.renderer import _plot_graph, _resolve_positions


def _chain_graph(n_nodes: int) -> _GraphData:
    nodes = {i: _make_node(f"T{i}", ("L", "R")) for i in range(n_nodes)}
    edges: list[_EdgeData] = []
    for i in range(n_nodes - 1):
        edges.append(
            _make_contraction_edge(
                _EdgeEndpoint(i, 1, f"b{i}"),
                _EdgeEndpoint(i + 1, 0, f"b{i}"),
                name=f"b{i}",
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def test_auto_tensor_label_refinement_refines_below_threshold(monkeypatch) -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    calls: list[int] = []

    def fake_refit(**_: object) -> None:
        calls.append(1)

    monkeypatch.setattr(render_prep, "_refit_tensor_labels_to_disks", fake_refit)

    _plot_graph(
        _chain_graph(39),
        dimensions=2,
        config=PlotConfig(tensor_label_refinement="auto"),
        renderer_name="test_auto_below",
    )

    assert calls == [1]


def test_auto_tensor_label_refinement_skips_refine_at_threshold(monkeypatch) -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    calls: list[int] = []

    def fake_refit(**_: object) -> None:
        calls.append(1)

    monkeypatch.setattr(render_prep, "_refit_tensor_labels_to_disks", fake_refit)

    _plot_graph(
        _chain_graph(40),
        dimensions=2,
        config=PlotConfig(tensor_label_refinement="auto"),
        renderer_name="test_auto_threshold",
    )

    assert calls == []


def test_always_tensor_label_refinement_forces_refine_at_threshold(monkeypatch) -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    calls: list[int] = []

    def fake_refit(**_: object) -> None:
        calls.append(1)

    monkeypatch.setattr(render_prep, "_refit_tensor_labels_to_disks", fake_refit)

    _plot_graph(
        _chain_graph(40),
        dimensions=2,
        config=PlotConfig(tensor_label_refinement="always"),
        renderer_name="test_quality_threshold",
    )

    assert calls == [1]


def test_never_tensor_label_refinement_skips_refine_even_below_threshold(monkeypatch) -> None:
    import tensor_network_viz._core.draw.render_prep as render_prep

    calls: list[int] = []

    def fake_refit(**_: object) -> None:
        calls.append(1)

    monkeypatch.setattr(render_prep, "_refit_tensor_labels_to_disks", fake_refit)

    _plot_graph(
        _chain_graph(12),
        dimensions=2,
        config=PlotConfig(tensor_label_refinement="never"),
        renderer_name="test_fast_small",
    )

    assert calls == []


def test_resolve_positions_reuses_cached_layout_for_same_key(monkeypatch) -> None:
    import tensor_network_viz._core.renderer as renderer

    graph = _chain_graph(8)
    calls: list[int] = []
    original: Callable[..., object] = renderer._compute_layout_from_components

    def counting_compute_layout_from_components(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        renderer,
        "_compute_layout_from_components",
        counting_compute_layout_from_components,
    )

    config = PlotConfig(layout_iterations=77)
    positions_a, components_a = _resolve_positions(graph, config, dimensions=2, seed=5)
    positions_b, components_b = _resolve_positions(graph, config, dimensions=2, seed=5)

    assert calls == [1]
    assert positions_a is positions_b
    assert components_a is components_b


def test_resolve_positions_cache_key_tracks_dimensions_seed_and_iterations(monkeypatch) -> None:
    import tensor_network_viz._core.renderer as renderer

    graph = _chain_graph(8)
    calls: list[int] = []
    original: Callable[..., object] = renderer._compute_layout_from_components

    def counting_compute_layout_from_components(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        renderer,
        "_compute_layout_from_components",
        counting_compute_layout_from_components,
    )

    _resolve_positions(graph, PlotConfig(layout_iterations=77), dimensions=2, seed=5)
    _resolve_positions(graph, PlotConfig(layout_iterations=77), dimensions=2, seed=6)
    _resolve_positions(graph, PlotConfig(layout_iterations=77), dimensions=3, seed=5)
    _resolve_positions(graph, PlotConfig(layout_iterations=88), dimensions=2, seed=5)

    assert calls == [1, 1, 1, 1]


def test_plot_graph_reuses_cached_geometry_for_same_key(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    import tensor_network_viz._core.renderer as renderer

    graph = _chain_graph(8)
    config = PlotConfig(layout_iterations=77)

    direction_calls: list[int] = []
    scale_calls: list[int] = []
    original_compute_axis_directions: Callable[..., Any] = renderer._compute_axis_directions
    original_resolve_scale: Callable[..., Any] = renderer._resolve_draw_scale_and_bond_curve_pad

    def counting_compute_axis_directions(*args: Any, **kwargs: Any) -> Any:
        direction_calls.append(1)
        return original_compute_axis_directions(*args, **kwargs)

    def counting_resolve_scale(*args: Any, **kwargs: Any) -> Any:
        scale_calls.append(1)
        return original_resolve_scale(*args, **kwargs)

    monkeypatch.setattr(renderer, "_compute_axis_directions", counting_compute_axis_directions)
    monkeypatch.setattr(renderer, "_resolve_draw_scale_and_bond_curve_pad", counting_resolve_scale)

    fig_a, _ax_a = renderer._plot_graph(
        graph,
        dimensions=2,
        config=config,
        seed=5,
        renderer_name="test_cached_geometry_a",
    )
    fig_b, _ax_b = renderer._plot_graph(
        graph,
        dimensions=2,
        config=config,
        seed=5,
        renderer_name="test_cached_geometry_b",
    )
    try:
        assert direction_calls == [1]
        assert scale_calls == [1]
    finally:
        plt.close(fig_a)
        plt.close(fig_b)


def test_2d_plotter_collections_disable_autolim(monkeypatch) -> None:
    fig, ax = plt.subplots()
    autolim_values: list[bool | None] = []
    original_add_collection: Callable[..., Any] = ax.add_collection

    def recording_add_collection(*args: Any, **kwargs: Any) -> Any:
        autolim_values.append(kwargs.get("autolim"))
        return original_add_collection(*args, **kwargs)

    monkeypatch.setattr(ax, "add_collection", recording_add_collection)

    plotter = _make_plotter(ax, dimensions=2)
    params = _draw_scale_params(
        PlotConfig(),
        1.0,
        fig=fig,
        is_3d=False,
    )
    plotter.plot_line(
        np.array([0.0, 0.0], dtype=float),
        np.array([1.0, 0.0], dtype=float),
        color="#123456",
        linewidth=1.5,
        zorder=1.0,
    )
    flush = getattr(plotter, "flush_edge_collections", None)
    assert callable(flush)
    flush()
    draw_one = getattr(plotter, "draw_tensor_node", None)
    assert callable(draw_one)
    draw_one(
        [0.0, 0.0],
        config=PlotConfig(),
        p=params,
        degree_one=False,
        mode="normal",
        zorder=2.0,
    )

    plt.close(fig)

    assert autolim_values
    assert all(value is False for value in autolim_values)
