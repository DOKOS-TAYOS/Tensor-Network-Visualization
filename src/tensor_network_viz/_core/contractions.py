"""Shared helpers for contraction edges used by layout and drawing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .axis_directions import _AXIS_OFFSET_SIGN
from .graph import (
    ContractionNodeIds,
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _require_contraction_endpoints,
    _require_contraction_node_ids,
    _sorted_contraction_node_ids,
)


@dataclass(frozen=True)
class _ContractionRecord:
    edge: _EdgeData
    node_ids: ContractionNodeIds
    endpoints: tuple[_EdgeEndpoint, _EdgeEndpoint]
    key: ContractionNodeIds


@dataclass(frozen=True)
class _ContractionGroups:
    groups: dict[ContractionNodeIds, tuple[_ContractionRecord, ...]]
    offsets: dict[int, tuple[int, int]]


def _offset_sign_from_axis_name(axis_name: str | None) -> int:
    if not axis_name:
        return 0
    return _AXIS_OFFSET_SIGN.get(axis_name.lower().strip(), 0)


def _iter_contractions(graph: _GraphData) -> tuple[_ContractionRecord, ...]:
    records: list[_ContractionRecord] = []
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        node_ids = _require_contraction_node_ids(edge)
        records.append(
            _ContractionRecord(
                edge=edge,
                node_ids=node_ids,
                endpoints=_require_contraction_endpoints(edge),
                key=_sorted_contraction_node_ids(*node_ids),
            )
        )
    return tuple(records)


def _contraction_weights(graph: _GraphData) -> dict[ContractionNodeIds, int]:
    return dict(Counter(record.key for record in _iter_contractions(graph)))


def _group_contractions(graph: _GraphData) -> _ContractionGroups:
    grouped_records: dict[ContractionNodeIds, list[_ContractionRecord]] = {}
    for record in _iter_contractions(graph):
        grouped_records.setdefault(record.key, []).append(record)

    groups: dict[ContractionNodeIds, tuple[_ContractionRecord, ...]] = {}
    offsets: dict[int, tuple[int, int]] = {}
    for key, records in grouped_records.items():
        ordered_records = tuple(
            sorted(
                records,
                key=lambda record: _offset_sign_from_axis_name(record.endpoints[0].axis_name),
            )
        )
        groups[key] = ordered_records
        group_size = len(ordered_records)
        for offset_index, record in enumerate(ordered_records):
            offsets[id(record.edge)] = (offset_index, group_size)

    return _ContractionGroups(groups=groups, offsets=offsets)
