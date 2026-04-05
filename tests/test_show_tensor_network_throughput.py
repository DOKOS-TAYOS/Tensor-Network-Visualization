"""Focused throughput regression check for Quimb graph caching."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tensor_network_viz import PlotConfig, clear_tensor_network_graph_cache, show_tensor_network


def _quimb_linear_chain_network(n: int) -> object:
    qtn = pytest.importorskip("quimb.tensor")
    tensors: list[object] = []
    for i in range(n):
        if i == 0:
            tensors.append(
                qtn.Tensor(data=np.ones((2, 3)), inds=(f"i{i}", f"b{i}"), tags={f"T{i}"})
            )
        elif i == n - 1:
            tensors.append(
                qtn.Tensor(data=np.ones((3, 2)), inds=(f"b{i - 1}", f"o{i}"), tags={f"T{i}"})
            )
        else:
            tensors.append(
                qtn.Tensor(
                    data=np.ones((3, 3, 3)),
                    inds=(f"b{i - 1}", f"b{i}", f"aux{i}"),
                    tags={f"T{i}"},
                )
            )
    return qtn.TensorNetwork(tensors)


def _measure_render_seconds(
    network: object,
    *,
    show_controls: bool,
) -> float:
    started = time.perf_counter()
    fig, _ax = show_tensor_network(
        network,
        config=PlotConfig(figsize=(9, 5)),
        show_controls=show_controls,
        show=False,
    )
    elapsed = time.perf_counter() - started
    plt.close(fig)
    return elapsed


@pytest.mark.perf
def test_quimb_graph_cache_second_lookup_is_cheap() -> None:
    """Extraction time: second ``_get_or_build_graph`` hit should be negligible vs cold build."""
    pytest.importorskip("quimb.tensor")
    from tensor_network_viz._core.graph_cache import _get_or_build_graph
    from tensor_network_viz.quimb.graph import _build_graph

    nw = _quimb_linear_chain_network(400)
    clear_tensor_network_graph_cache(nw)
    t0 = time.perf_counter()
    g1 = _get_or_build_graph(nw, _build_graph)
    cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    g2 = _get_or_build_graph(nw, _build_graph)
    warm = time.perf_counter() - t1

    assert g1 is g2
    assert warm < cold * 0.05, f"expected cache hit cold={cold:.6f}s warm={warm:.6f}s"


@pytest.mark.perf
def test_first_and_repeated_public_render_paths_stay_bounded() -> None:
    warmup_network = _quimb_linear_chain_network(8)
    _measure_render_seconds(warmup_network, show_controls=True)

    network = _quimb_linear_chain_network(160)

    clear_tensor_network_graph_cache(network)
    static_cold = _measure_render_seconds(network, show_controls=False)
    static_warm = _measure_render_seconds(network, show_controls=False)

    clear_tensor_network_graph_cache(network)
    interactive_cold = _measure_render_seconds(network, show_controls=True)

    assert static_cold < 2.5, f"static first render took {static_cold:.4f}s"
    assert static_warm < 0.9, f"static repeated render took {static_warm:.4f}s"
    assert static_warm < static_cold * 0.65, (
        f"expected warm static render << cold (cold={static_cold:.4f}s warm={static_warm:.4f}s)"
    )
    # Cold interactive startup still includes Matplotlib widget/font initialization, so keep this
    # as a broad regression guard rather than a machine-tight benchmark.
    assert interactive_cold < 8.0, f"interactive first render took {interactive_cold:.4f}s"


@pytest.mark.perf
def test_menu_toggles_and_first_3d_switch_stay_bounded() -> None:
    # Warm up text/layout and 3D view initialization so the assertion tracks regressions in the
    # interactive toggle path instead of one-time backend cold-start costs on slower machines.
    warmup_network = _quimb_linear_chain_network(8)
    warmup_fig, _warmup_ax = show_tensor_network(
        warmup_network,
        config=PlotConfig(figsize=(9, 5)),
        show=False,
    )
    try:
        warmup_controls = getattr(warmup_fig, "_tensor_network_viz_interactive_controls", None)
        assert warmup_controls is not None
        warmup_controls.set_tensor_labels_enabled(True)
        warmup_controls.set_edge_labels_enabled(True)
        warmup_controls.set_view("3d")
    finally:
        plt.close(warmup_fig)

    network = _quimb_linear_chain_network(160)
    fig, _ax = show_tensor_network(
        network,
        config=PlotConfig(figsize=(9, 5)),
        show=False,
    )
    try:
        controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
        assert controls is not None

        t0 = time.perf_counter()
        controls.set_tensor_labels_enabled(True)
        tensor_toggle = time.perf_counter() - t0

        t1 = time.perf_counter()
        controls.set_edge_labels_enabled(True)
        edge_toggle = time.perf_counter() - t1

        t2 = time.perf_counter()
        controls.set_view("3d")
        first_switch_3d = time.perf_counter() - t2

        assert tensor_toggle < 3.2, f"Tensor labels toggle took {tensor_toggle:.4f}s"
        assert edge_toggle < 3.5, f"Edge labels toggle took {edge_toggle:.4f}s"
        assert first_switch_3d < 5.5, f"first 2D->3D switch took {first_switch_3d:.4f}s"
    finally:
        plt.close(fig)


@pytest.mark.perf
def test_switch_to_3d_limits_exact_segment_distance_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz._core.layout.free_directions_3d as directions_3d

    network = _quimb_linear_chain_network(160)
    fig, _ax = show_tensor_network(
        network,
        config=PlotConfig(figsize=(9, 5)),
        show=False,
    )
    try:
        controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
        assert controls is not None

        calls = {"count": 0}
        original = directions_3d._segment_segment_min_distance_sq_3d

        def counting_distance(
            start_a: np.ndarray,
            end_a: np.ndarray,
            start_b: np.ndarray,
            end_b: np.ndarray,
        ) -> float:
            calls["count"] += 1
            return float(original(start_a, end_a, start_b, end_b))

        monkeypatch.setattr(
            directions_3d,
            "_segment_segment_min_distance_sq_3d",
            counting_distance,
        )

        controls.set_view("3d")

        assert calls["count"] < 60_000, f"too many exact 3D segment checks: {calls['count']}"
    finally:
        plt.close(fig)
