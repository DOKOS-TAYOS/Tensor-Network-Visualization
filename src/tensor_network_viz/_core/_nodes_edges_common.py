"""Shared logic for building graphs from node/edge-based backends (TensorKrowch, TensorNetwork)."""

from __future__ import annotations

from typing import Any

from .graph import (
    _coerce_shape,
    _EdgeEndpoint,
    _element_count_for_shape,
    _estimated_nbytes_for_node,
    _finalize_graph_diagnostics,
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


def _node_tensor_payload(node: Any) -> Any | None:
    """Return the tensor-like payload used for diagnostics, when available."""
    for attr in ("tensor", "data"):
        value = getattr(node, attr, None)
        if value is not None:
            return value
    to_ndarray = getattr(node, "to_ndarray", None)
    if callable(to_ndarray):
        return to_ndarray()
    return None


def _node_tensor_metadata(
    node: Any,
) -> tuple[tuple[int, ...] | None, str | None, int | None, int | None]:
    """Extract shape, dtype, element count, and estimated bytes from one backend node."""
    tensor = _node_tensor_payload(node)
    shape = _coerce_shape(getattr(node, "shape", None))
    if shape is None and tensor is not None:
        shape = _coerce_shape(getattr(tensor, "shape", None))
    dtype_attr = None if tensor is None else getattr(tensor, "dtype", None)
    dtype_text = None if dtype_attr is None else str(dtype_attr)
    element_count = _element_count_for_shape(shape)
    estimated_nbytes = _estimated_nbytes_for_node(shape, dtype_text, element_count=element_count)
    return shape, dtype_text, element_count, estimated_nbytes


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
        axes_names = tuple(_stringify(item) for item in _iterable_attr(node, axis_attr, "node"))
        if len(node_edges) != len(axes_names):
            raise TypeError(
                f"Node {name!r} has {len(node_edges)} edges but {len(axes_names)} {axis_attr}."
            )

        node_id = id(node)
        shape, dtype_text, element_count, estimated_nbytes = _node_tensor_metadata(node)
        nodes[node_id] = _make_node(
            name=name,
            axes_names=axes_names,
            shape=shape,
            dtype=dtype_text,
            element_count=element_count,
            estimated_nbytes=estimated_nbytes,
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

    return _finalize_graph_diagnostics(_GraphData(nodes=nodes, edges=tuple(edges)))
