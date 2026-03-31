"""Focused throughput regression check for Quimb graph caching."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz import clear_tensor_network_graph_cache


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
