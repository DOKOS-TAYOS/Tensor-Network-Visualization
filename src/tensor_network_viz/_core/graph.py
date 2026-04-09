"""Shared graph model for normalized tensor network visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np

EdgeKind = Literal["contraction", "dangling", "self"]
ContractionNodeIds: TypeAlias = tuple[int, int]
_DEFAULT_LABEL = object()


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
    label: str | None = None
    is_virtual: bool = False
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    element_count: int | None = None
    estimated_nbytes: int | None = None


@dataclass(frozen=True)
class _EdgeData:
    name: str | None
    kind: EdgeKind
    node_ids: tuple[int, ...]
    endpoints: tuple[_EdgeEndpoint, ...]
    label: str | None
    bond_dimension: int | None = None


@dataclass(frozen=True)
class _ContractionStepMetrics:
    """Naive dense cost for one einsum trace step (see ``einsum_module.contraction_cost``)."""

    label_dims: tuple[tuple[str, int], ...]
    multiplicative_cost: int
    flop_mac: int
    equation_snippet: str | None = None
    operand_names: tuple[str, ...] = ()
    operand_shapes: tuple[tuple[int, ...], ...] = ()
    output_labels: tuple[str, ...] = ()
    contracted_labels: tuple[str, ...] = ()
    label_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class _GraphData:
    nodes: dict[int, _NodeData]
    edges: tuple[_EdgeData, ...]
    #: Per execution step, non-virtual node ids involved in that real contraction event.
    #: Result tensors contribute the transitive lineage of their visible operands, so disjoint
    #: branches stay separated until a later merge step actually touches both.
    contraction_steps: tuple[frozenset[int], ...] | None = None
    #: Same length as ``contraction_steps`` when both are set: per-step naive dense metrics from
    #: the einsum parse.
    contraction_step_metrics: tuple[_ContractionStepMetrics | None, ...] | None = None


def _make_node(
    name: str,
    axes_names: tuple[str, ...],
    *,
    label: str | None = None,
    is_virtual: bool = False,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    element_count: int | None = None,
    estimated_nbytes: int | None = None,
) -> _NodeData:
    return _NodeData(
        name=name,
        axes_names=axes_names,
        degree=len(axes_names),
        label=label,
        is_virtual=is_virtual,
        shape=shape,
        dtype=dtype,
        element_count=element_count,
        estimated_nbytes=estimated_nbytes,
    )


def _make_dangling_edge(
    endpoint: _EdgeEndpoint,
    *,
    name: str | None,
    label: str | None | object = _DEFAULT_LABEL,
) -> _EdgeData:
    resolved_label = endpoint.axis_name or name
    edge_label = resolved_label if label is _DEFAULT_LABEL else cast(str | None, label)
    return _EdgeData(
        name=name,
        kind="dangling",
        node_ids=(endpoint.node_id,),
        endpoints=(endpoint,),
        label=edge_label,
    )


def _make_contraction_edge(
    left_endpoint: _EdgeEndpoint,
    right_endpoint: _EdgeEndpoint,
    *,
    name: str | None,
    label: str | None | object = _DEFAULT_LABEL,
) -> _EdgeData:
    kind = "self" if left_endpoint.node_id == right_endpoint.node_id else "contraction"
    node_ids = (
        (left_endpoint.node_id,)
        if kind == "self"
        else (left_endpoint.node_id, right_endpoint.node_id)
    )
    endpoints = (left_endpoint, right_endpoint)
    resolved_label = _build_edge_label(kind=kind, endpoints=endpoints, edge_name=name)
    edge_label = resolved_label if label is _DEFAULT_LABEL else cast(str | None, label)
    return _EdgeData(
        name=name,
        kind=kind,
        node_ids=node_ids,
        endpoints=endpoints,
        label=edge_label,
    )


def _sorted_contraction_node_ids(left_id: int, right_id: int) -> ContractionNodeIds:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _require_contraction_node_ids(edge: _EdgeData) -> ContractionNodeIds:
    if edge.kind != "contraction" or len(edge.node_ids) != 2:
        raise TypeError("Contraction edges must expose exactly two node ids.")
    left_id, right_id = edge.node_ids
    return left_id, right_id


def _require_contraction_endpoints(edge: _EdgeData) -> tuple[_EdgeEndpoint, _EdgeEndpoint]:
    left_id, right_id = _require_contraction_node_ids(edge)
    endpoints_by_node = {endpoint.node_id: endpoint for endpoint in edge.endpoints}
    if left_id not in endpoints_by_node or right_id not in endpoints_by_node:
        raise TypeError("Contraction edges must expose one endpoint per node.")
    return endpoints_by_node[left_id], endpoints_by_node[right_id]


def _endpoint_index_caption(
    endpoint: _EdgeEndpoint,
    edge: _EdgeData,
    graph: _GraphData,
) -> str | None:
    """Human-readable index name at one end of a bond (tensor-side or hyperedge hub)."""
    node = graph.nodes[endpoint.node_id]
    if node.is_virtual:
        if edge.name:
            return str(edge.name)
        axis = endpoint.axis_name
        if axis and "__branch_" in axis:
            return axis.split("__branch_", 1)[0]
        return axis
    if endpoint.axis_name:
        return str(endpoint.axis_name)
    if edge.name:
        return str(edge.name)
    return None


def _build_edge_label(
    kind: EdgeKind,
    endpoints: tuple[_EdgeEndpoint, ...],
    edge_name: str | None,
) -> str | None:
    axis_names = [endpoint.axis_name for endpoint in endpoints if endpoint.axis_name]
    if kind == "dangling":
        return axis_names[0] if axis_names else edge_name
    return None


def _coerce_shape(shape: object) -> tuple[int, ...] | None:
    """Normalize an arbitrary shape-like object into a tuple of ints."""
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)  # type: ignore[arg-type]
    except TypeError:
        return None


def _element_count_for_shape(shape: tuple[int, ...] | None) -> int | None:
    """Return the scalar element count for one tensor shape."""
    if shape is None:
        return None
    if not shape:
        return 1
    return int(np.prod(shape, dtype=int))


def _estimated_nbytes_for_node(
    shape: tuple[int, ...] | None,
    dtype_text: str | None,
    *,
    element_count: int | None,
) -> int | None:
    """Estimate tensor storage size from shape and dtype when possible."""
    if shape is None or dtype_text is None or element_count is None:
        return None
    try:
        dtype = np.dtype(dtype_text)
    except TypeError:
        return None
    return int(element_count * int(dtype.itemsize))


def _bond_dimension_for_edge(edge: _EdgeData, nodes: dict[int, _NodeData]) -> int | None:
    """Infer the bond dimension carried by one normalized edge."""
    dimensions: set[int] = set()
    for endpoint in edge.endpoints:
        node = nodes.get(endpoint.node_id)
        if node is None or node.shape is None:
            continue
        if endpoint.axis_index < 0 or endpoint.axis_index >= len(node.shape):
            continue
        dimensions.add(int(node.shape[endpoint.axis_index]))
    if not dimensions:
        return None
    if len(dimensions) != 1:
        return None
    return next(iter(dimensions))


def _finalize_graph_diagnostics(graph: _GraphData) -> _GraphData:
    """Fill derived tensor diagnostics shared by snapshots and viewers."""
    finalized_nodes = {
        node_id: _NodeData(
            name=node.name,
            axes_names=node.axes_names,
            degree=node.degree,
            label=node.label,
            is_virtual=node.is_virtual,
            shape=_coerce_shape(node.shape),
            dtype=node.dtype,
            element_count=(
                node.element_count
                if node.element_count is not None
                else _element_count_for_shape(_coerce_shape(node.shape))
            ),
            estimated_nbytes=(
                node.estimated_nbytes
                if node.estimated_nbytes is not None
                else _estimated_nbytes_for_node(
                    _coerce_shape(node.shape),
                    node.dtype,
                    element_count=(
                        node.element_count
                        if node.element_count is not None
                        else _element_count_for_shape(_coerce_shape(node.shape))
                    ),
                )
            ),
        )
        for node_id, node in graph.nodes.items()
    }
    finalized_edges = tuple(
        _EdgeData(
            name=edge.name,
            kind=edge.kind,
            node_ids=edge.node_ids,
            endpoints=edge.endpoints,
            label=edge.label,
            bond_dimension=(
                edge.bond_dimension
                if edge.bond_dimension is not None
                else _bond_dimension_for_edge(edge, finalized_nodes)
            ),
        )
        for edge in graph.edges
    )
    return _GraphData(
        nodes=finalized_nodes,
        edges=finalized_edges,
        contraction_steps=graph.contraction_steps,
        contraction_step_metrics=graph.contraction_step_metrics,
    )


def _resolve_contraction_scheme_by_name(
    graph: _GraphData,
    groups: tuple[tuple[str, ...], ...],
) -> tuple[frozenset[int], ...]:
    """Map per-step tensor names to node ids (non-virtual nodes only)."""
    name_to_id: dict[str, int] = {}
    for nid, nd in graph.nodes.items():
        if nd.is_virtual:
            continue
        n = nd.name
        if not n:
            continue
        if n in name_to_id:
            raise ValueError(
                f"contraction_scheme_by_name: duplicate non-virtual tensor name {n!r} "
                f"(nodes {name_to_id[n]} and {nid})."
            )
        name_to_id[n] = nid
    out: list[frozenset[int]] = []
    for step in groups:
        ids: set[int] = set()
        for name in step:
            if name not in name_to_id:
                raise ValueError(
                    f"contraction_scheme_by_name: unknown tensor name {name!r} among visible nodes."
                )
            ids.add(name_to_id[name])
        out.append(frozenset(ids))
    return tuple(out)
