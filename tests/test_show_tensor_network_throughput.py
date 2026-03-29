"""Timing smoke tests for full ``show_tensor_network`` flow (includes Matplotlib draw).

**Full pipeline:** for moderate Quimb networks, Matplotlib layout/draw dominates; two consecutive
``show_tensor_network`` runs on the same object stay within a few percent (e.g. ~2.93s / ~2.90s
before and after graph caching for a chain of 32 tensors, Agg, ``refine_tensor_labels=False``).

**Extraction:** the normalized graph is cached per network object;
``test_quimb_graph_cache_second_lookup_is_cheap`` asserts a second ``_get_or_build_graph`` hit is
much cheaper than a cold Quimb build (large chain). Set ``TNV_SHOW_FLOW_TIMING=1`` to print
first/second full-flow seconds during pytest.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz import PlotConfig, clear_tensor_network_graph_cache, show_tensor_network


@pytest.fixture(scope="module")
def quimb_chain_network() -> object:
    qtn = pytest.importorskip("quimb.tensor")

    def chain(n: int) -> object:
        tensors: list[object] = []
        for i in range(n):
            if i == 0:
                tensors.append(
                    qtn.Tensor(data=np.ones((2, 3)), inds=(f"i{i}", f"b{i}"), tags={f"T{i}"})
                )
            elif i == n - 1:
                tensors.append(
                    qtn.Tensor(data=np.ones((3, 2)), inds=(f"b{i-1}", f"o{i}"), tags={f"T{i}"})
                )
            else:
                tensors.append(
                    qtn.Tensor(
                        data=np.ones((3, 3, 3)),
                        inds=(f"b{i-1}", f"b{i}", f"aux{i}"),
                        tags={f"T{i}"},
                    )
                )
        return qtn.TensorNetwork(tensors)

    return chain(40)


def test_show_tensor_network_twice_full_flow(quimb_chain_network: object) -> None:
    cfg = PlotConfig(figsize=(8, 6), refine_tensor_labels=False)

    t0 = time.perf_counter()
    show_tensor_network(quimb_chain_network, engine="quimb", view="2d", config=cfg, show=False)
    first = time.perf_counter() - t0

    t1 = time.perf_counter()
    show_tensor_network(quimb_chain_network, engine="quimb", view="2d", config=cfg, show=False)
    second = time.perf_counter() - t1

    if os.environ.get("TNV_SHOW_FLOW_TIMING", "").strip() in ("1", "true", "yes"):
        print(f"\n[tnv timing] show_tensor_network first={first:.4f}s second={second:.4f}s\n")

    assert first > 0.0 and second > 0.0


def test_clear_cache_then_show_rebuilds_graph(quimb_chain_network: object) -> None:
    cfg = PlotConfig(figsize=(8, 6), refine_tensor_labels=False)
    show_tensor_network(quimb_chain_network, engine="quimb", view="2d", config=cfg, show=False)
    clear_tensor_network_graph_cache(quimb_chain_network)
    show_tensor_network(quimb_chain_network, engine="quimb", view="2d", config=cfg, show=False)


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
                qtn.Tensor(data=np.ones((3, 2)), inds=(f"b{i-1}", f"o{i}"), tags={f"T{i}"})
            )
        else:
            tensors.append(
                qtn.Tensor(
                    data=np.ones((3, 3, 3)),
                    inds=(f"b{i-1}", f"b{i}", f"aux{i}"),
                    tags={f"T{i}"},
                )
            )
    return qtn.TensorNetwork(tensors)


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
