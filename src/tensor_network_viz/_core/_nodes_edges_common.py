"""Shared logic for building graphs from node/edge-based backends (TensorKrowch, TensorNetwork)."""

from __future__ import annotations

from typing import Any

from .graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from .graph_utils import (
    _extract_unique_items,
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
    return _extract_unique_items(
        network,
        attr_sources=node_sources,
        sort_key=lambda n: _sortable_node_key(n, axis_attr=axis_attr),
        backend_name=backend_name,
        type_name="nodes",
    )


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

    nodes: dict[int, Any] = {}
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
        nodes[node_id] = _make_node(
            name=name,
            axes_names=axes_names,
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

    edges = []
    for edge_id, edge in edge_refs.items():
        name = _optional_string(_require_attr(edge, "name", "edge"))
        node1 = _require_attr(edge, "node1", "edge")
        node2 = _require_attr(edge, "node2", "edge")

        node1_id = id(node1) if node1 is not None and id(node1) in nodes else None
        node2_id = id(node2) if node2 is not None and id(node2) in nodes else None
        endpoint_list = edge_endpoints.get(edge_id)
        if not endpoint_list:
            continue
        endpoints = tuple(endpoint_list)
        if len(endpoints) > 2:
            raise TypeError("Edges with more than two endpoints are not supported.")

        if node1_id is None and node2_id is None:
            continue
        if len(endpoints) == 1:
            edges.append(_make_dangling_edge(endpoints[0], name=name))
            continue
        if len(endpoints) != 2:
            raise TypeError("Edges with more than two endpoints are not supported.")
        left_endpoint, right_endpoint = endpoints
        edges.append(_make_contraction_edge(left_endpoint, right_endpoint, name=name))

    return _GraphData(nodes=nodes, edges=tuple(edges))
