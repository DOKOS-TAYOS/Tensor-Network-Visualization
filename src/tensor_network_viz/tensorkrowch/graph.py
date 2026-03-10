"""Graph data structures and construction from tensor networks."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Literal

EdgeKind = Literal["contraction", "dangling", "self"]


@dataclass(frozen=True)
class _EdgeEndpoint:
    node_id: int
    axis_index: int
    axis_name: str | None


@dataclass(frozen=True)
class _NodeData:
    name: str
    axes_names: tuple[str, ...]
    degree: int


@dataclass(frozen=True)
class _EdgeData:
    name: str | None
    kind: EdgeKind
    node_ids: tuple[int, ...]
    endpoints: tuple[_EdgeEndpoint, ...]
    label: str | None


@dataclass(frozen=True)
class _GraphData:
    nodes: dict[int, _NodeData]
    edges: tuple[_EdgeData, ...]


def _is_unordered_collection(value: Any) -> bool:
    return isinstance(value, AbstractSet) and not isinstance(value, (str, bytes, bytearray))


def _sortable_node_key(node: Any) -> tuple[Any, ...]:
    axes_names = getattr(node, "axes_names", getattr(node, "axis_names", ()))
    if isinstance(axes_names, dict):
        axes_names = axes_names.values()
    try:
        normalized_axes = tuple(_stringify(item) for item in axes_names)
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


def _get_network_nodes(network: Any) -> list[Any]:
    """Extract node list from a network object or an iterable of nodes."""
    if isinstance(network, dict):
        raw_nodes = network.values()
        should_sort = False
    elif hasattr(network, "nodes"):
        raw_nodes = network.nodes
        should_sort = _is_unordered_collection(raw_nodes)
    elif hasattr(network, "leaf_nodes"):
        raw_nodes = network.leaf_nodes
        should_sort = _is_unordered_collection(raw_nodes)
    elif isinstance(network, (str, bytes, bytearray)):
        raise TypeError(
            "Input must be an iterable of nodes, or an object with "
            "'nodes' or 'leaf_nodes' attribute."
        )
    elif isinstance(network, Iterable):
        raw_nodes = network
        should_sort = _is_unordered_collection(raw_nodes)
    else:
        raise TypeError(
            "Input must be an iterable of nodes, or an object with "
            "'nodes' or 'leaf_nodes' attribute."
        )

    iterable = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes

    try:
        items = list(iterable)
    except TypeError as exc:
        raise TypeError("Tensor network nodes must be iterable.") from exc

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


def _iterable_attr(obj: Any, attr_name: str, object_name: str) -> list[Any]:
    value = _require_attr(obj, attr_name, object_name)
    if isinstance(value, dict):
        return list(value.values())
    try:
        return list(value)
    except TypeError as exc:
        msg = f"{object_name.capitalize()} attribute '{attr_name}' must be iterable."
        raise TypeError(msg) from exc


def _iterable_attr_alias(obj: Any, attr_names: tuple[str, ...], object_name: str) -> list[Any]:
    for attr_name in attr_names:
        if hasattr(obj, attr_name):
            return _iterable_attr(obj, attr_name, object_name)

    names = "', '".join(attr_names)
    raise TypeError(f"{object_name.capitalize()} is missing required attribute one of '{names}'.")


def _require_attr(obj: Any, attr_name: str, object_name: str) -> Any:
    if not hasattr(obj, attr_name):
        raise TypeError(f"{object_name.capitalize()} is missing required attribute '{attr_name}'.")
    return getattr(obj, attr_name)


def _stringify(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _build_edge_label(
    kind: EdgeKind,
    endpoints: tuple[_EdgeEndpoint, ...],
    edge_name: str | None,
) -> str | None:
    axis_names = [endpoint.axis_name for endpoint in endpoints if endpoint.axis_name]
    if kind == "dangling":
        return axis_names[0] if axis_names else edge_name
    if len(axis_names) >= 2:
        return f"{axis_names[0]}<->{axis_names[1]}"
    return edge_name


def _build_graph(network: Any) -> _GraphData:
    node_refs = _get_network_nodes(network)
    if not node_refs:
        raise ValueError("The tensor network does not expose any nodes to visualize.")

    nodes: dict[int, _NodeData] = {}
    edge_refs: dict[int, Any] = {}
    edge_endpoints: dict[int, list[_EdgeEndpoint]] = {}

    for node in node_refs:
        name = _stringify(_require_attr(node, "name", "node"))
        node_edges = tuple(_iterable_attr(node, "edges", "node"))
        axes_names = tuple(
            _stringify(item)
            for item in _iterable_attr_alias(node, ("axes_names", "axis_names"), "node")
        )
        if len(node_edges) != len(axes_names):
            raise TypeError(
                f"Node {name!r} has {len(node_edges)} edges but {len(axes_names)} axes_names."
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
            kind: EdgeKind
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
