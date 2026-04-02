"""Tests for automatic engine detection in ``show_tensor_network``."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz import pair_tensor, show_tensor_network
from tensor_network_viz.viewer import _detect_engine


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


class DummyTensorKrowchNetwork:
    def __init__(self, *, nodes: list[DummyTensorKrowchNode]) -> None:
        self.nodes = nodes


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
def close_figures() -> None:
    yield
    plt.close("all")


def test_detect_engine_for_tensorkrowch_network() -> None:
    left = DummyTensorKrowchNode("L", ["a", "b"])
    right = DummyTensorKrowchNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")
    network = DummyTensorKrowchNetwork(nodes=[left, right])

    assert _detect_engine(network) == "tensorkrowch"


def test_detect_engine_for_tensornetwork_node_iterable() -> None:
    left = DummyTensorNetworkNode("L", ["a", "b"])
    right = DummyTensorNetworkNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")

    assert _detect_engine([left, right]) == "tensornetwork"


def test_detect_engine_for_einsum_trace_iterable() -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    assert _detect_engine(trace) == "einsum"


def test_show_tensor_network_autodetects_tensorkrowch_engine() -> None:
    left = DummyTensorKrowchNode("L", ["a", "b"])
    right = DummyTensorKrowchNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")
    network = DummyTensorKrowchNetwork(nodes=[left, right])

    fig, ax = show_tensor_network(network, show=False, interactive_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_autodetects_einsum_engine() -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = show_tensor_network(trace, show=False, interactive_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_autodetects_single_pass_iterable() -> None:
    trace = (step for step in [pair_tensor("A", "x", "r0", "ab,b->a")])

    fig, ax = show_tensor_network(trace, show=False, interactive_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_rejects_unknown_input_when_engine_is_omitted() -> None:
    with pytest.raises(ValueError, match="Could not infer tensor network engine"):
        show_tensor_network(object(), show=False)
