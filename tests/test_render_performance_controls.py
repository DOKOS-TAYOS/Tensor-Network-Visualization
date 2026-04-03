from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

from tensor_network_viz import PlotConfig
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
