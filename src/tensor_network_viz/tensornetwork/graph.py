"""TensorNetwork-specific normalization into the shared graph model."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from typing import Any

from .._core.graph import (
    _build_edge_label,
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _NodeData,
)


def _is_unordered_collection(value: Any) -> bool:
    return isinstance(value, AbstractSet) and not isinstance(value, (str, bytes, bytearray))


def _stringify(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _sortable_node_key(node: Any) -> tuple[Any, ...]:
    axis_names = getattr(node, "axis_names", ())
    if isinstance(axis_names, dict):
        axis_names = axis_names.values()
    try:
        normalized_axes = tuple(_stringify(item) for item in axis_names)
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


def _require_attr(obj: Any, attr_name: str, object_name: str) -> Any:
    if not hasattr(obj, attr_name):
        raise TypeError(f"{object_name.capitalize()} is missing required attribute '{attr_name}'.")
    return getattr(obj, attr_name)


def _iterable_attr(obj: Any, attr_name: str, object_name: str) -> list[Any]:
    value = _require_attr(obj, attr_name, object_name)
    if isinstance(value, dict):
        return list(value.values())
    try:
        return list(value)
    except TypeError as exc:
        msg = f"{object_name.capitalize()} attribute '{attr_name}' must be iterable."
        raise TypeError(msg) from exc


def _get_network_nodes(nodes: Any) -> list[Any]:
    """Extract TensorNetwork nodes from an iterable node collection."""
    if isinstance(nodes, dict):
        raw_nodes = nodes.values()
        should_sort = False
    elif isinstance(nodes, (str, bytes, bytearray)):
        raise TypeError("Input must be an iterable of TensorNetwork nodes.")
    elif isinstance(nodes, Iterable):
        raw_nodes = nodes
        should_sort = _is_unordered_collection(raw_nodes)
    else:
        raise TypeError("Input must be an iterable of TensorNetwork nodes.")

    try:
        items = list(raw_nodes)
    except TypeError as exc:
        raise TypeError("TensorNetwork nodes must be iterable.") from exc

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
        unique_nodes.sort(key=_sortable_node_key)
    return unique_nodes


def _build_graph(nodes_input: Any) -> _GraphData:
    node_refs = _get_network_nodes(nodes_input)
    if not node_refs:
        raise ValueError("The tensor network does not expose any nodes to visualize.")

    nodes: dict[int, _NodeData] = {}
    edge_refs: dict[int, Any] = {}
    edge_endpoints: dict[int, list[_EdgeEndpoint]] = {}

    for node in node_refs:
        name = _stringify(_require_attr(node, "name", "node"))
        node_edges = tuple(_iterable_attr(node, "edges", "node"))
        axis_names = tuple(_stringify(item) for item in _iterable_attr(node, "axis_names", "node"))
        if len(node_edges) != len(axis_names):
            raise TypeError(
                f"Node {name!r} has {len(node_edges)} edges but {len(axis_names)} axis_names."
            )

        node_id = id(node)
        nodes[node_id] = _NodeData(
            name=name,
            axes_names=axis_names,
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
                    axis_name=axis_names[axis_index],
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
