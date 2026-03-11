"""Quimb-specific normalization into the shared graph model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _NodeData,
)
from .._core.graph_utils import (
    _is_unordered_collection,
    _stringify,
)


def _sortable_tensor_key(tensor: Any) -> tuple[Any, ...]:
    tags = getattr(tensor, "tags", ())
    inds = getattr(tensor, "inds", ())
    try:
        normalized_tags = tuple(sorted(_stringify(tag) for tag in tags))
    except TypeError:
        normalized_tags = ()
    try:
        normalized_inds = tuple(_stringify(ind) for ind in inds)
    except TypeError:
        normalized_inds = ()

    return (
        type(tensor).__module__,
        type(tensor).__qualname__,
        normalized_tags,
        normalized_inds,
    )


def _get_network_tensors(network: Any) -> list[Any]:
    """Extract Quimb tensors from a TensorNetwork or iterable tensor collection."""
    if isinstance(network, dict):
        raw_tensors = network.values()
        should_sort = False
    elif hasattr(network, "tensors"):
        raw_tensors = network.tensors
        should_sort = _is_unordered_collection(raw_tensors)
    elif isinstance(network, (str, bytes, bytearray)):
        raise TypeError("Input must be a Quimb TensorNetwork or an iterable of tensors.")
    elif isinstance(network, Iterable):
        raw_tensors = network
        should_sort = _is_unordered_collection(raw_tensors)
    else:
        raise TypeError("Input must be a Quimb TensorNetwork or an iterable of tensors.")

    try:
        items = list(raw_tensors)
    except TypeError as exc:
        raise TypeError("Quimb tensors must be iterable.") from exc

    unique_tensors: list[Any] = []
    seen: set[int] = set()
    for tensor in items:
        if tensor is None:
            continue
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        unique_tensors.append(tensor)

    if should_sort:
        unique_tensors.sort(key=_sortable_tensor_key)
    return unique_tensors


def _tensor_display_name(tensor: Any, fallback_index: int) -> str:
    tags = getattr(tensor, "tags", ())
    try:
        normalized_tags = sorted(_stringify(tag) for tag in tags if _stringify(tag))
    except TypeError:
        normalized_tags = []
    if normalized_tags:
        return ":".join(normalized_tags)
    return f"T{fallback_index}"


def _build_graph(network: Any) -> _GraphData:
    tensor_refs = _get_network_tensors(network)
    if not tensor_refs:
        raise ValueError("The tensor network does not expose any tensors to visualize.")

    nodes: dict[int, _NodeData] = {}
    index_endpoints: dict[str, list[_EdgeEndpoint]] = {}

    for tensor_index, tensor in enumerate(tensor_refs):
        inds = tuple(_stringify(ind) for ind in getattr(tensor, "inds", ()))
        node_id = id(tensor)
        nodes[node_id] = _NodeData(
            name=_tensor_display_name(tensor, tensor_index),
            axes_names=inds,
            degree=len(inds),
        )

        for axis_index, ind in enumerate(inds):
            index_endpoints.setdefault(ind, []).append(
                _EdgeEndpoint(
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_name=ind,
                )
            )

    edges: list[_EdgeData] = []
    for ind_name, endpoints in index_endpoints.items():
        if len(endpoints) > 2:
            raise TypeError("Quimb indices shared by more than two tensors are not supported.")

        if len(endpoints) == 1:
            kind = "dangling"
            node_ids = (endpoints[0].node_id,)
        elif endpoints[0].node_id == endpoints[1].node_id:
            kind = "self"
            node_ids = (endpoints[0].node_id,)
        else:
            kind = "contraction"
            node_ids = (endpoints[0].node_id, endpoints[1].node_id)

        edges.append(
            _EdgeData(
                name=ind_name,
                kind=kind,
                node_ids=node_ids,
                endpoints=tuple(endpoints),
                label=ind_name or None,
            )
        )

    return _GraphData(nodes=nodes, edges=tuple(edges))
