"""TeNPy-specific normalization into the shared graph model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from .._core.graph_utils import _stringify

_SUPPORTED_NETWORKS_HINT: str = (
    "Expected a TeNPy tensor chain: MPS or subclasses (e.g. PurificationMPS, UniformMPS), "
    "MPO, or MomentumMPS-like objects (callable get_X and attribute uMPS_GS)."
)


def _boundary_mode_from_geometry(geometry: Any) -> str:
    if not hasattr(geometry, "bc") or not hasattr(geometry, "finite"):
        raise TypeError(
            "TeNPy network geometry must expose boundary metadata via attributes 'bc' and 'finite'."
        )
    boundary_condition = _stringify(getattr(geometry, "bc", None)).lower()
    if boundary_condition in {"finite", "segment"}:
        return "open"
    if boundary_condition == "infinite" and not bool(geometry.finite):
        return "periodic"
    raise ValueError("TeNPy visualization supports finite, segment, and infinite networks.")


def _leg_labels(tensor: Any) -> tuple[str, ...]:
    if not hasattr(tensor, "get_leg_labels"):
        raise TypeError("TeNPy tensors must expose leg labels via 'get_leg_labels()'.")
    return tuple(_stringify(label) for label in tensor.get_leg_labels())


def _find_leg_index(labels: tuple[str, ...], leg_name: str, *, node_name: str) -> int:
    try:
        return labels.index(leg_name)
    except ValueError as exc:
        raise TypeError(f"Tensor {node_name!r} is missing required leg {leg_name!r}.") from exc


def _build_nodes(
    length: int,
    *,
    tensor_at: Callable[[int], Any],
    node_prefix: str,
) -> tuple[dict[int, Any], dict[int, tuple[str, ...]]]:
    nodes: dict[int, Any] = {}
    labels_by_node: dict[int, tuple[str, ...]] = {}

    for index in range(length):
        labels = _leg_labels(tensor_at(index))
        labels_by_node[index] = labels
        nodes[index] = _make_node(
            name=f"{node_prefix}{index}",
            axes_names=labels,
        )

    return nodes, labels_by_node


def _build_chain_edges(
    *,
    length: int,
    labels_by_node: dict[int, tuple[str, ...]],
    left_leg: str,
    right_leg: str,
    boundary_mode: str,
) -> list[Any]:
    edges: list[Any] = []

    neighbor_pairs = [(index, index + 1) for index in range(length - 1)]
    if boundary_mode == "periodic" and length > 0:
        neighbor_pairs.append((length - 1, 0))

    for left_index, right_index in neighbor_pairs:
        left_labels = labels_by_node[left_index]
        right_labels = labels_by_node[right_index]
        left_endpoint = _EdgeEndpoint(
            node_id=left_index,
            axis_index=_find_leg_index(left_labels, right_leg, node_name=f"{left_index}"),
            axis_name=right_leg,
        )
        right_endpoint = _EdgeEndpoint(
            node_id=right_index,
            axis_index=_find_leg_index(right_labels, left_leg, node_name=f"{right_index}"),
            axis_name=left_leg,
        )
        edges.append(_make_contraction_edge(left_endpoint, right_endpoint, name=None))

    for index in range(length):
        labels = labels_by_node[index]
        for axis_index, label in enumerate(labels):
            if boundary_mode == "periodic":
                is_internal_left = label == left_leg
                is_internal_right = label == right_leg
            else:
                is_internal_left = label == left_leg and index > 0
                is_internal_right = label == right_leg and index < length - 1
            if is_internal_left or is_internal_right:
                continue
            endpoint = _EdgeEndpoint(
                node_id=index,
                axis_index=axis_index,
                axis_name=label,
            )
            edges.append(_make_dangling_edge(endpoint, name=label or None, label=label or None))

    return edges


def _build_mps_graph(network: Any) -> _GraphData:
    boundary_mode = _boundary_mode_from_geometry(network)
    nodes, labels_by_node = _build_nodes(
        int(network.L),
        tensor_at=lambda index: network.get_B(index, form=None),
        node_prefix="B",
    )
    edges = _build_chain_edges(
        length=int(network.L),
        labels_by_node=labels_by_node,
        left_leg="vL",
        right_leg="vR",
        boundary_mode=boundary_mode,
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_mpo_graph(network: Any) -> _GraphData:
    boundary_mode = _boundary_mode_from_geometry(network)
    nodes, labels_by_node = _build_nodes(
        int(network.L),
        tensor_at=network.get_W,
        node_prefix="W",
    )
    edges = _build_chain_edges(
        length=int(network.L),
        labels_by_node=labels_by_node,
        left_leg="wL",
        right_leg="wR",
        boundary_mode=boundary_mode,
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _build_momentum_mps_graph(network: Any) -> _GraphData:
    geometry = network.uMPS_GS
    boundary_mode = _boundary_mode_from_geometry(geometry)
    length = int(geometry.L)
    nodes, labels_by_node = _build_nodes(
        length,
        tensor_at=lambda index: network.get_X(index),
        node_prefix="X",
    )
    edges = _build_chain_edges(
        length=length,
        labels_by_node=labels_by_node,
        left_leg="vL",
        right_leg="vR",
        boundary_mode=boundary_mode,
    )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _is_momentum_mps_like(network: Any) -> bool:
    return callable(getattr(network, "get_X", None)) and hasattr(network, "uMPS_GS")


def _build_graph(network: Any) -> _GraphData:
    if callable(getattr(network, "get_W", None)):
        return _build_mpo_graph(network)
    if _is_momentum_mps_like(network):
        return _build_momentum_mps_graph(network)
    if callable(getattr(network, "get_B", None)):
        return _build_mps_graph(network)
    raise TypeError(
        f"Unsupported TeNPy input: {type(network).__name__!r}. {_SUPPORTED_NETWORKS_HINT}"
    )
