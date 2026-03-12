"""Shared graph model for normalized tensor network visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

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


def _make_node(
    name: str,
    axes_names: tuple[str, ...],
    *,
    label: str | None = None,
    is_virtual: bool = False,
) -> _NodeData:
    return _NodeData(
        name=name,
        axes_names=axes_names,
        degree=len(axes_names),
        label=label,
        is_virtual=is_virtual,
    )


def _make_dangling_edge(
    endpoint: _EdgeEndpoint,
    *,
    name: str | None,
    label: str | None | object = _DEFAULT_LABEL,
) -> _EdgeData:
    resolved_label = endpoint.axis_name or name
    edge_label = (
        resolved_label if label is _DEFAULT_LABEL else cast(str | None, label)
    )
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
    edge_label = (
        resolved_label if label is _DEFAULT_LABEL else cast(str | None, label)
    )
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
