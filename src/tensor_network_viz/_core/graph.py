"""Shared graph model for normalized tensor network visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
