"""Tests for normalized graph reuse across repeated draws."""

from __future__ import annotations

from tensor_network_viz._core.graph import (
    _GraphData,
    _make_node,
)
from tensor_network_viz._core.graph_cache import (
    _get_or_build_graph,
    clear_tensor_network_graph_cache,
)


class _DummyNetwork:
    """Weak-key-friendly stand-in for a tensor network object."""


def test_get_or_build_graph_reuses_instance() -> None:
    calls: list[int] = []
    nw = _DummyNetwork()

    def build(_: object) -> _GraphData:
        calls.append(1)
        return _GraphData(nodes={0: _make_node("a", ())}, edges=())

    g1 = _get_or_build_graph(nw, build)
    g2 = _get_or_build_graph(nw, build)
    assert g1 is g2
    assert calls == [1]


def test_clear_tensor_network_graph_cache_forces_rebuild() -> None:
    calls: list[int] = []
    nw = _DummyNetwork()

    def build(_: object) -> _GraphData:
        calls.append(1)
        return _GraphData(nodes={0: _make_node("a", ())}, edges=())

    g1 = _get_or_build_graph(nw, build)
    clear_tensor_network_graph_cache(nw)
    g2 = _get_or_build_graph(nw, build)
    assert g1 is not g2
    assert calls == [1, 1]


def test_builder_identity_partitions_cache() -> None:
    calls_a: list[int] = []
    calls_b: list[int] = []
    nw = _DummyNetwork()

    def build_a(_: object) -> _GraphData:
        calls_a.append(1)
        return _GraphData(nodes={0: _make_node("a", ())}, edges=())

    def build_b(_: object) -> _GraphData:
        calls_b.append(1)
        return _GraphData(nodes={1: _make_node("b", ("i",))}, edges=())

    ga = _get_or_build_graph(nw, build_a)
    gb = _get_or_build_graph(nw, build_b)
    assert ga is not gb
    assert calls_a == [1] and calls_b == [1]
    assert _get_or_build_graph(nw, build_a) is ga
