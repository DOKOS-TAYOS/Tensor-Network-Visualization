"""Tests for automatic engine detection in ``show_tensor_network``."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tensor_network_viz import (
    PlotConfig,
    TenPyTensorNetwork,
    make_tenpy_tensor_network,
    pair_tensor,
    show_tensor_network,
)
from tensor_network_viz._input_inspection import (
    _grid_positions_for_network_input,
    _prepare_network_input,
)
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
    def __init__(self, *, nodes: Iterable[DummyTensorKrowchNode]) -> None:
        self.nodes = nodes


class DummyQuimbTensor:
    def __init__(
        self,
        name: str,
        inds: tuple[str, ...],
    ) -> None:
        self.name = name
        self.inds = inds
        self.tags = {name}


class DummyQuimbNetwork:
    def __init__(self, *, tensors: Iterable[DummyQuimbTensor]) -> None:
        self.tensors = tensors


class DummyTenPyTensor:
    def __init__(
        self, array: np.ndarray[Any, np.dtype[np.float64]], labels: tuple[str, ...]
    ) -> None:
        self._array = array
        self._labels = labels

    def to_ndarray(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._array

    def get_leg_labels(self) -> tuple[str, ...]:
        return self._labels


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
def close_figures() -> Iterator[None]:
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

    fig, ax = show_tensor_network(network, show=False, show_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_autodetects_einsum_engine() -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = show_tensor_network(trace, show=False, show_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_autodetects_tenpy_subclass_network() -> None:
    class DerivedTenPyTensorNetwork(TenPyTensorNetwork):
        pass

    base = make_tenpy_tensor_network(
        [
            (
                "A",
                DummyTenPyTensor(
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                    ("left", "right"),
                ),
            )
        ],
        [],
    )
    network = DerivedTenPyTensorNetwork(nodes=base.nodes, bonds=base.bonds)

    assert _detect_engine(network) == "tenpy"

    fig, ax = show_tensor_network(network, show=False, show_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_show_tensor_network_autodetects_single_pass_iterable() -> None:
    trace = (step for step in [pair_tensor("A", "x", "r0", "ab,b->a")])

    fig, ax = show_tensor_network(trace, show=False, show_controls=False)

    assert fig is ax.figure
    assert ax.name != "3d"


def test_detect_engine_for_nested_tensornetwork_grid() -> None:
    left = DummyTensorNetworkNode("L", ["a", "b"])
    right = DummyTensorNetworkNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")

    assert _detect_engine([[left, None], [None, right]]) == "tensornetwork"


def test_show_tensor_network_autodetects_nested_tensornetwork_grid_3d() -> None:
    left = DummyTensorNetworkNode("L", ["a", "b"])
    right = DummyTensorNetworkNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")

    fig, ax = show_tensor_network([[[left, right]]], show=False, show_controls=False)

    assert fig is ax.figure
    assert ax.name == "3d"


def test_nested_grid_3d_2d_projection_offsets_deeper_layers_negative_x_negative_y() -> None:
    front = DummyTensorNetworkNode("front", ["a"])
    middle = DummyTensorNetworkNode("middle", ["b"])
    back = DummyTensorNetworkNode("back", ["c"])
    prepared = _prepare_network_input([[[front]], [[middle]], [[back]]])

    positions = _grid_positions_for_network_input(prepared, dimensions=2)

    assert positions is not None
    front_pos = positions[id(front)]
    middle_pos = positions[id(middle)]
    back_pos = positions[id(back)]
    assert middle_pos[0] < front_pos[0]
    assert middle_pos[1] < front_pos[1]
    assert back_pos[0] < middle_pos[0]
    assert back_pos[1] < middle_pos[1]


def test_show_tensor_network_keeps_first_node_for_single_pass_nodes_attribute() -> None:
    left = DummyTensorKrowchNode("L", ["a", "b"])
    right = DummyTensorKrowchNode("R", ["b", "c"])
    connect(left, 1, right, 0, name="bond")
    network = DummyTensorKrowchNetwork(nodes=iter([left, right]))

    fig, ax = show_tensor_network(
        network,
        config=PlotConfig(show_tensor_labels=True),
        show=False,
        show_controls=False,
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"L", "R"}


def test_show_tensor_network_keeps_first_tensor_for_single_pass_tensors_attribute() -> None:
    left = DummyQuimbTensor("Q0", ("a", "b"))
    right = DummyQuimbTensor("Q1", ("c", "d"))
    network = DummyQuimbNetwork(tensors=iter([left, right]))

    fig, ax = show_tensor_network(
        network,
        config=PlotConfig(show_tensor_labels=True),
        show=False,
        show_controls=False,
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"Q0", "Q1"}


def test_show_tensor_network_rejects_unknown_input_when_engine_is_omitted() -> None:
    with pytest.raises(ValueError, match="Could not infer tensor network engine"):
        show_tensor_network(object(), show=False)


def test_show_tensor_network_rejects_mixed_depth_grid_input() -> None:
    left = DummyTensorNetworkNode("L", ["a"])

    with pytest.raises(
        ValueError, match="Mixed nested tensor-network grid depths are not supported"
    ):
        show_tensor_network([[left], [[left]]], show=False, show_controls=False)


def test_show_tensor_network_rejects_duplicate_object_in_grid_input() -> None:
    left = DummyTensorNetworkNode("L", ["a"])

    with pytest.raises(ValueError, match="same tensor or node object appears more than once"):
        show_tensor_network([[left, left]], show=False, show_controls=False)
