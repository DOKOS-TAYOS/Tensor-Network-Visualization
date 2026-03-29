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


def _quimb_tensor_parsed(tensor: Any) -> tuple[tuple[Any, ...], tuple[str, ...], str | None]:
    """Single pass over tags/inds: sort key, axis name tuple, display name or None for T{{idx}}."""
    tags = getattr(tensor, "tags", ())
    inds_raw = getattr(tensor, "inds", ())
    try:
        tag_strs = [_stringify(tag) for tag in tags]
        normalized_tags = tuple(sorted(tag_strs))
    except TypeError:
        tag_strs = []
        normalized_tags = ()
    try:
        ind_strings = tuple(_stringify(ind) for ind in inds_raw)
    except TypeError:
        ind_strings = ()
    sort_key = (
        type(tensor).__module__,
        type(tensor).__qualname__,
        normalized_tags,
        ind_strings,
    )
    non_empty_tags = sorted(s for s in tag_strs if s)
    if non_empty_tags:
        return sort_key, ind_strings, ":".join(non_empty_tags)
    return sort_key, ind_strings, None


def _tensors_sorted_with_meta(
    network: Any,
) -> tuple[list[Any], list[tuple[tuple[str, ...], str | None]]]:
    """Unique tensors, stable-sorted like the former ``_sortable_tensor_key``, one parse each."""
    tensor_refs = _extract_unique_items(
        network,
        attr_sources=("tensors",),
        sort_key=None,
        backend_name="Quimb",
        type_name="tensors",
    )
    if not tensor_refs:
        return [], []
    parsed = [_quimb_tensor_parsed(t) for t in tensor_refs]
    order = sorted(range(len(tensor_refs)), key=lambda i: parsed[i][0])
    ordered_tensors = [tensor_refs[i] for i in order]
    meta = [(parsed[i][1], parsed[i][2]) for i in order]
    return ordered_tensors, meta


def _get_network_tensors(network: Any) -> list[Any]:
    """Extract Quimb tensors from a TensorNetwork or iterable tensor collection."""
    tensors, _ = _tensors_sorted_with_meta(network)
    return tensors


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
    tensor_refs, meta = _tensors_sorted_with_meta(network)
    if not tensor_refs:
        raise ValueError("The tensor network does not expose any tensors to visualize.")

    nodes: dict[int, Any] = {}
    index_endpoints: dict[str, list[_EdgeEndpoint]] = {}

    for tensor_index, tensor in enumerate(tensor_refs):
        inds, name_opt = meta[tensor_index]
        display_name = name_opt if name_opt is not None else f"T{tensor_index}"
        node_id = id(tensor)
        nodes[node_id] = _make_node(
            name=display_name,
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
