"""Shared logic for building graphs from node/edge-based backends (TensorKrowch, TensorNetwork)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .graph import (
    _build_edge_label,
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _NodeData,
)
from .graph_utils import (
    _is_unordered_collection,
    _iterable_attr,
    _optional_string,
    _require_attr,
    _stringify,
)


def _sortable_node_key(node: Any, *, axis_attr: str) -> tuple[Any, ...]:
    axes = getattr(node, axis_attr, ())
    if isinstance(axes, dict):
        axes = axes.values()
    try:
        normalized_axes = tuple(_stringify(item) for item in axes)
    except TypeError:
        normalized_axes = ()

    edges = getattr(node, "edges", ())
    try:
        degree = len(edges)
    except TypeError:
        degree = 0

    return (
        type(node).__module__,
        type(node).__qualname__,
        _stringify(getattr(node, "name", None)),
        normalized_axes,
        degree,
    )


def _get_network_nodes(
    network: Any,
    *,
    axis_attr: str,
    node_sources: tuple[str, ...],
    backend_name: str,
) -> list[Any]:
    """Extract nodes from a network object or iterable."""
    if isinstance(network, dict):
        raw_nodes = network.values()
        should_sort = False
    elif node_sources and any(hasattr(network, s) for s in node_sources):
        raw_nodes = None
        for src in node_sources:
            if hasattr(network, src):
                raw_nodes = getattr(network, src)
                break
        if raw_nodes is None:
            raise TypeError(
                f"Input must be an iterable of {backend_name} nodes, or an object with "
                f"'{node_sources[0]}' or '{node_sources[1]}' attribute."
            )
        should_sort = _is_unordered_collection(raw_nodes)
    elif isinstance(network, (str, bytes, bytearray)):
        raise TypeError(f"Input must be an iterable of {backend_name} nodes.")
    elif isinstance(network, Iterable):
        raw_nodes = network
        should_sort = _is_unordered_collection(raw_nodes)
    else:
        if node_sources:
            extra = f", or an object with '{node_sources[0]}' or '{node_sources[1]}' attribute."
        else:
            extra = "."
        raise TypeError(f"Input must be an iterable of {backend_name} nodes{extra}")

    iterable = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes

    try:
        items = list(iterable)
    except TypeError as exc:
        raise TypeError(f"{backend_name} nodes must be iterable.") from exc

    unique_nodes: list[Any] = []
    seen: set[int] = set()
    for node in items:
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        unique_nodes.append(node)

    if should_sort:
        unique_nodes.sort(key=lambda n: _sortable_node_key(n, axis_attr=axis_attr))
    return unique_nodes


def _build_graph_from_nodes_edges(
    network: Any,
    *,
    axis_attr: str,
    node_sources: tuple[str, ...],
    backend_name: str,
) -> _GraphData:
    """Build _GraphData from nodes with edges and axis names."""
    node_refs = _get_network_nodes(
        network,
        axis_attr=axis_attr,
        node_sources=node_sources,
        backend_name=backend_name,
    )
    if not node_refs:
        raise ValueError("The tensor network does not expose any nodes to visualize.")

    nodes: dict[int, _NodeData] = {}
    edge_refs: dict[int, Any] = {}
    edge_endpoints: dict[int, list[_EdgeEndpoint]] = {}

    for node in node_refs:
        name = _stringify(_require_attr(node, "name", "node"))
        node_edges = tuple(_iterable_attr(node, "edges", "node"))
        axes_names = tuple(
            _stringify(item) for item in _iterable_attr(node, axis_attr, "node")
        )
        if len(node_edges) != len(axes_names):
            raise TypeError(
                f"Node {name!r} has {len(node_edges)} edges but {len(axes_names)} {axis_attr}."
            )

        node_id = id(node)
        nodes[node_id] = _NodeData(
            name=name,
            axes_names=axes_names,
            degree=len(node_edges),
        )

        for axis_index, edge in enumerate(node_edges):
            if edge is None:
                continue
            edge_id = id(edge)
            edge_refs[edge_id] = edge
            edge_endpoints.setdefault(edge_id, []).append(
                _EdgeEndpoint(
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_name=axes_names[axis_index],
                )
            )

    edges: list[_EdgeData] = []
    for edge_id, edge in edge_refs.items():
        name = _optional_string(_require_attr(edge, "name", "edge"))
        node1 = _require_attr(edge, "node1", "edge")
        node2 = _require_attr(edge, "node2", "edge")

        node1_id = id(node1) if node1 is not None and id(node1) in nodes else None
        node2_id = id(node2) if node2 is not None and id(node2) in nodes else None
        endpoints = tuple(edge_endpoints.get(edge_id, ()))
        if not endpoints:
            continue
        if len(endpoints) > 2:
            raise TypeError("Edges with more than two endpoints are not supported.")

        if node1_id is not None and node2_id is not None:
            if node1_id == node2_id:
                kind = "self"
                node_ids = (node1_id,)
            else:
                kind = "contraction"
                node_ids = (node1_id, node2_id)
        elif node1_id is not None:
            kind = "dangling"
            node_ids = (node1_id,)
        elif node2_id is not None:
            kind = "dangling"
            node_ids = (node2_id,)
        else:
            continue

        edges.append(
            _EdgeData(
                name=name,
                kind=kind,
                node_ids=node_ids,
                endpoints=endpoints,
                label=_build_edge_label(kind=kind, endpoints=endpoints, edge_name=name),
            )
        )

    return _GraphData(nodes=nodes, edges=tuple(edges))
