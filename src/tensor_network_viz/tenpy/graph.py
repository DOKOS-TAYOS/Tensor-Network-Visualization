"""TeNPy-specific normalization into the shared graph model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._core.graph import (
    _build_edge_label,
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _NodeData,
)
from .._core.graph_utils import _stringify


def _is_supported_tenpy_network(network: Any) -> bool:
    return hasattr(network, "L") and hasattr(network, "finite") and hasattr(network, "bc")


def _boundary_mode(network: Any) -> str:
    if not _is_supported_tenpy_network(network):
        raise TypeError("Input must be a TeNPy MPS or MPO network.")

    boundary_condition = _stringify(getattr(network, "bc", None)).lower()
    if boundary_condition in {"finite", "segment"}:
        return "open"
    if boundary_condition == "infinite" and not bool(network.finite):
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
) -> tuple[dict[int, _NodeData], dict[int, tuple[str, ...]]]:
    nodes: dict[int, _NodeData] = {}
    labels_by_node: dict[int, tuple[str, ...]] = {}

    for index in range(length):
        labels = _leg_labels(tensor_at(index))
        labels_by_node[index] = labels
        nodes[index] = _NodeData(
            name=f"{node_prefix}{index}",
            axes_names=labels,
            degree=len(labels),
        )

    return nodes, labels_by_node


def _build_chain_edges(
    *,
    length: int,
    labels_by_node: dict[int, tuple[str, ...]],
    left_leg: str,
    right_leg: str,
    boundary_mode: str,
) -> list[_EdgeData]:
    edges: list[_EdgeData] = []

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
        endpoints = (left_endpoint, right_endpoint)
        kind = "self" if left_index == right_index else "contraction"
        node_ids = (left_index,) if kind == "self" else (left_index, right_index)
        edges.append(
            _EdgeData(
                name=None,
                kind=kind,
                node_ids=node_ids,
                endpoints=endpoints,
                label=_build_edge_label(kind=kind, endpoints=endpoints, edge_name=None),
            )
        )

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
            edges.append(
                _EdgeData(
                    name=label or None,
                    kind="dangling",
                    node_ids=(index,),
                    endpoints=(endpoint,),
                    label=label or None,
                )
            )

    return edges


def _build_mps_graph(network: Any) -> _GraphData:
    boundary_mode = _boundary_mode(network)
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
    boundary_mode = _boundary_mode(network)
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


def _build_graph(network: Any) -> _GraphData:
    if hasattr(network, "get_B"):
        return _build_mps_graph(network)
    if hasattr(network, "get_W"):
        return _build_mpo_graph(network)
    raise TypeError("Input must be a TeNPy MPS or MPO network.")
