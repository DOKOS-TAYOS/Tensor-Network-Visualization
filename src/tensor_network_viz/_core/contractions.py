"""Shared helpers for contraction edges used in layout and drawing."""

from __future__ import annotations

import weakref
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


@dataclass(frozen=True)
class _ContractionDerived:
    """Cached contraction lists and groupings for one :class:`_GraphData` instance."""

    records: tuple[_ContractionRecord, ...]
    groups: _ContractionGroups


_contraction_derived_by_id: dict[int, _ContractionDerived] = {}


def _offset_sign_from_axis_name(axis_name: str | None) -> int:
    if not axis_name:
        return 0
    return _AXIS_OFFSET_SIGN.get(axis_name.lower().strip(), 0)


def _build_contraction_records(graph: _GraphData) -> tuple[_ContractionRecord, ...]:
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


def _group_contractions_from_records(
    records: tuple[_ContractionRecord, ...],
) -> _ContractionGroups:
    grouped_records: dict[ContractionNodeIds, list[_ContractionRecord]] = {}
    for record in records:
        grouped_records.setdefault(record.key, []).append(record)

    groups: dict[ContractionNodeIds, tuple[_ContractionRecord, ...]] = {}
    offsets: dict[int, tuple[int, int]] = {}
    for key, group_records in grouped_records.items():
        ordered_records = tuple(
            sorted(
                group_records,
                key=lambda record: _offset_sign_from_axis_name(record.endpoints[0].axis_name),
            )
        )
        groups[key] = ordered_records
        group_size = len(ordered_records)
        for offset_index, record in enumerate(ordered_records):
            offsets[id(record.edge)] = (offset_index, group_size)

    return _ContractionGroups(groups=groups, offsets=offsets)


def _contraction_derived(graph: _GraphData) -> _ContractionDerived:
    key = id(graph)
    cached = _contraction_derived_by_id.get(key)
    if cached is not None:
        return cached
    records = _build_contraction_records(graph)
    groups = _group_contractions_from_records(records)
    derived = _ContractionDerived(records=records, groups=groups)
    _contraction_derived_by_id[key] = derived

    def _evict() -> None:
        _contraction_derived_by_id.pop(key, None)

    weakref.finalize(graph, _evict)
    return derived


def _iter_contractions(graph: _GraphData) -> tuple[_ContractionRecord, ...]:
    return _contraction_derived(graph).records


def _group_contractions(graph: _GraphData) -> _ContractionGroups:
    return _contraction_derived(graph).groups
