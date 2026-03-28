"""Quimb-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from .._core.graph_utils import (
    _extract_unique_items,
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
    return _extract_unique_items(
        network,
        attr_sources=("tensors",),
        sort_key=_sortable_tensor_key,
        backend_name="Quimb",
        type_name="tensors",
    )


def _tensor_display_name(tensor: Any, fallback_index: int) -> str:
    tags = getattr(tensor, "tags", ())
    try:
        normalized_tags = sorted(_stringify(tag) for tag in tags if _stringify(tag))
    except TypeError:
        normalized_tags = []
    if normalized_tags:
        return ":".join(normalized_tags)
    return f"T{fallback_index}"


def _build_hyperedge_hub(
    *,
    ind_name: str,
    endpoints: list[_EdgeEndpoint],
) -> Any:
    axis_names = tuple(f"{ind_name}__branch_{index}" for index in range(len(endpoints)))
    return _make_node(
        name="",
        axes_names=axis_names,
        label=ind_name or None,
        is_virtual=True,
    )


def _build_graph(network: Any) -> _GraphData:
    tensor_refs = _get_network_tensors(network)
    if not tensor_refs:
        raise ValueError("The tensor network does not expose any tensors to visualize.")

    nodes: dict[int, Any] = {}
    index_endpoints: dict[str, list[_EdgeEndpoint]] = {}

    for tensor_index, tensor in enumerate(tensor_refs):
        inds = tuple(_stringify(ind) for ind in getattr(tensor, "inds", ()))
        node_id = id(tensor)
        nodes[node_id] = _make_node(
            name=_tensor_display_name(tensor, tensor_index),
            axes_names=inds,
        )

        for axis_index, ind in enumerate(inds):
            index_endpoints.setdefault(ind, []).append(
                _EdgeEndpoint(
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_name=ind,
                )
            )

    edges = []
    next_virtual_node_id = -1
    for ind_name, endpoints in index_endpoints.items():
        if len(endpoints) > 2:
            hub_id = next_virtual_node_id
            next_virtual_node_id -= 1
            nodes[hub_id] = _build_hyperedge_hub(
                ind_name=ind_name,
                endpoints=endpoints,
            )

            for branch_index, endpoint in enumerate(endpoints):
                hub_endpoint = _EdgeEndpoint(
                    node_id=hub_id,
                    axis_index=branch_index,
                    axis_name=nodes[hub_id].axes_names[branch_index],
                )
                edges.append(
                    _make_contraction_edge(
                        endpoint,
                        hub_endpoint,
                        name=ind_name,
                    )
                )
            continue

        if len(endpoints) == 1:
            edges.append(_make_dangling_edge(endpoints[0], name=ind_name, label=ind_name or None))
            continue
        edges.append(
            _make_contraction_edge(
                endpoints[0],
                endpoints[1],
                name=ind_name,
            )
        )

    return _GraphData(nodes=nodes, edges=tuple(edges))
