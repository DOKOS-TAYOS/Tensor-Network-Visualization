"""Shared geometry helpers for layout-space bond and stub reasoning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..contractions import _ContractionGroups, _group_contractions, _iter_contractions
from ..curves import _quadratic_curve
from ..graph import _GraphData
from .parameters import (
    _LAYOUT_BOND_CURVE_NEAR_PAIR_REF,
    _LAYOUT_BOND_CURVE_OFFSET_FACTOR,
    _LAYOUT_BOND_CURVE_SAMPLES,
    _SEGMENT_BOND_CLEAR_2D,
    _STUB_LAYOUT_R0,
    _STUB_LAYOUT_R1,
)
from .types import NodePositions


@dataclass(frozen=True)
class _BondSegment2D:
    node_ids: tuple[int, int]
    start: np.ndarray
    end: np.ndarray
    bbox: tuple[float, float, float, float]


def _segment_point_min_distance_sq_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a2 = np.asarray(a, dtype=float).reshape(-1)[:2]
    b2 = np.asarray(b, dtype=float).reshape(-1)[:2]
    c2 = np.asarray(c, dtype=float).reshape(-1)[:2]
    ab = b2 - a2
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.sum((c2 - a2) ** 2))
    t = float(np.dot(c2 - a2, ab)) / denom
    t = max(0.0, min(1.0, t))
    closest = a2 + t * ab
    return float(np.sum((c2 - closest) ** 2))


def _segment_point_min_distance_sq_2d_many(
    a: np.ndarray,
    b: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    a2 = np.asarray(a, dtype=float).reshape(1, 2)
    b2 = np.asarray(b, dtype=float).reshape(1, 2)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    ab = b2 - a2
    denom = float(np.dot(ab[0], ab[0]))
    if denom < 1e-18:
        delta = pts - a2
        return np.einsum("ij,ij->i", delta, delta)
    t = ((pts - a2) @ ab[0]) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a2 + t[:, np.newaxis] * ab
    delta = pts - closest
    return np.einsum("ij,ij->i", delta, delta)


def _segment_point_min_distance_sq_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a3 = np.asarray(a, dtype=float).reshape(-1)[:3]
    b3 = np.asarray(b, dtype=float).reshape(-1)[:3]
    c3 = np.asarray(c, dtype=float).reshape(-1)[:3]
    ab = b3 - a3
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.sum((c3 - a3) ** 2))
    t = float(np.dot(c3 - a3, ab)) / denom
    t = max(0.0, min(1.0, t))
    closest = a3 + t * ab
    return float(np.sum((c3 - closest) ** 2))


def _segment_segment_min_distance_2d(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    a0 = np.asarray(start_a, dtype=float).reshape(-1)[:2]
    a1 = np.asarray(end_a, dtype=float).reshape(-1)[:2]
    b0 = np.asarray(start_b, dtype=float).reshape(-1)[:2]
    b1 = np.asarray(end_b, dtype=float).reshape(-1)[:2]
    if _segments_cross_2d(a0, a1, b0, b1):
        return 0.0
    return min(
        math.sqrt(_segment_point_min_distance_sq_2d(a0, a1, b0)),
        math.sqrt(_segment_point_min_distance_sq_2d(a0, a1, b1)),
        math.sqrt(_segment_point_min_distance_sq_2d(b0, b1, a0)),
        math.sqrt(_segment_point_min_distance_sq_2d(b0, b1, a1)),
    )


def _segment_segment_min_distance_sq_3d(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    """Squared minimum distance between two 3D segments."""

    delta_a = np.asarray(end_a, dtype=float).reshape(-1)[:3] - np.asarray(
        start_a,
        dtype=float,
    ).reshape(-1)[:3]
    delta_b = np.asarray(end_b, dtype=float).reshape(-1)[:3] - np.asarray(
        start_b,
        dtype=float,
    ).reshape(-1)[:3]
    offset = np.asarray(start_a, dtype=float).reshape(-1)[:3] - np.asarray(
        start_b,
        dtype=float,
    ).reshape(-1)[:3]
    aa = float(np.dot(delta_a, delta_a))
    ab = float(np.dot(delta_a, delta_b))
    bb = float(np.dot(delta_b, delta_b))
    ao = float(np.dot(delta_a, offset))
    bo = float(np.dot(delta_b, offset))
    denom = aa * bb - ab * ab
    small = 1e-12
    s_num = 0.0
    s_den = denom
    t_num = 0.0
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = bo
        t_den = bb
    else:
        s_num = ab * bo - bb * ao
        t_num = aa * bo - ab * ao
        if s_num < 0.0:
            s_num = 0.0
            t_num = bo
            t_den = bb
        elif s_num > s_den:
            s_num = s_den
            t_num = bo + ab
            t_den = bb

    if t_num < 0.0:
        t_num = 0.0
        if -ao < 0.0:
            s_num = 0.0
        elif -ao > aa:
            s_num = s_den
        else:
            s_num = -ao
            s_den = aa
    elif t_num > t_den:
        t_num = t_den
        if (-ao + ab) < 0.0:
            s_num = 0.0
        elif (-ao + ab) > aa:
            s_num = s_den
        else:
            s_num = -ao + ab
            s_den = aa

    s_param = 0.0 if abs(s_num) < small else s_num / s_den
    t_param = 0.0 if abs(t_num) < small else t_num / t_den
    delta = offset + s_param * delta_a - t_param * delta_b
    return float(np.dot(delta, delta))


def _segments_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _point_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _segment_hits_existing_geometry_2d(
    start: np.ndarray,
    end: np.ndarray,
    other_start: np.ndarray,
    other_end: np.ndarray,
) -> bool:
    return (
        _segment_segment_min_distance_2d(start, end, other_start, other_end)
        < _SEGMENT_BOND_CLEAR_2D
    )


def _segment_bboxes_overlap_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    padding: float = 0.0,
) -> bool:
    min_x = min(float(p0[0]), float(p1[0])) - float(padding)
    max_x = max(float(p0[0]), float(p1[0])) + float(padding)
    min_y = min(float(p0[1]), float(p1[1])) - float(padding)
    max_y = max(float(p0[1]), float(p1[1])) + float(padding)
    seg_min_x, seg_max_x, seg_min_y, seg_max_y = bbox
    seg_min_x -= float(padding)
    seg_max_x += float(padding)
    seg_min_y -= float(padding)
    seg_max_y += float(padding)
    return not (max_x < seg_min_x or seg_max_x < min_x or max_y < seg_min_y or seg_max_y < min_y)


def _normalize_2d(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(-1)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def _dangling_stub_segment_2d(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    *,
    draw_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(origin, dtype=float).reshape(-1)[:2]
    d = _normalize_2d(direction_unit)
    s = max(float(draw_scale), 1e-6)
    return o + d * (_STUB_LAYOUT_R0 * s), o + d * (_STUB_LAYOUT_R1 * s)


def _dangling_stub_segment_3d(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    *,
    draw_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(origin, dtype=float).reshape(-1)[:3]
    d = np.asarray(direction_unit, dtype=float).reshape(-1)[:3]
    n = float(np.linalg.norm(d))
    d = np.array([0.0, 0.0, 1.0], dtype=float) if n < 1e-9 else d / n
    s = max(float(draw_scale), 1e-6)
    return o + d * (_STUB_LAYOUT_R0 * s), o + d * (_STUB_LAYOUT_R1 * s)


def _bond_perpendicular_unoriented_2d(delta: np.ndarray) -> np.ndarray:
    direction = delta / max(float(np.linalg.norm(delta)), 1e-6)
    return np.array([-direction[1], direction[0]], dtype=float)


def _layout_quadratic_bond_polyline_2d(
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    *,
    scale: float,
) -> np.ndarray:
    start2 = np.asarray(start, dtype=float).reshape(-1)[:2]
    end2 = np.asarray(end, dtype=float).reshape(-1)[:2]
    midpoint = (start2 + end2) / 2.0
    delta = end2 - start2
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    perpendicular = _bond_perpendicular_unoriented_2d(delta)
    ref_len = _LAYOUT_BOND_CURVE_NEAR_PAIR_REF * scale
    effective_chord = float(math.hypot(distance, ref_len))
    offset = (
        (offset_index - (edge_count - 1) / 2.0)
        * _LAYOUT_BOND_CURVE_OFFSET_FACTOR
        * scale
        * effective_chord
    )
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start2, control, end2, samples=_LAYOUT_BOND_CURVE_SAMPLES)


def _planar_contraction_bond_segments_2d(
    graph: _GraphData,
    positions: NodePositions,
    *,
    scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    records = _planar_contraction_bond_segment_records_2d(
        graph,
        positions,
        scale=scale,
        contraction_groups=contraction_groups,
    )
    return [
        (record.node_ids[0], record.node_ids[1], record.start, record.end) for record in records
    ]


def _planar_contraction_bond_segment_records_2d(
    graph: _GraphData,
    positions: NodePositions,
    *,
    scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
    node_filter: frozenset[int] | None = None,
) -> tuple[_BondSegment2D, ...]:
    groups = contraction_groups if contraction_groups is not None else _group_contractions(graph)
    out: list[_BondSegment2D] = []
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        if node_filter is not None and (
            left_id not in node_filter or right_id not in node_filter
        ):
            continue
        if left_id not in positions or right_id not in positions:
            continue
        offset_index, edge_count = groups.offsets[id(record.edge)]
        start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:2]
        end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:2]
        poly = _layout_quadratic_bond_polyline_2d(start, end, offset_index, edge_count, scale=scale)
        for index in range(int(poly.shape[0]) - 1):
            seg_start = poly[index].copy()
            seg_end = poly[index + 1].copy()
            out.append(
                _BondSegment2D(
                    node_ids=(left_id, right_id),
                    start=seg_start,
                    end=seg_end,
                    bbox=(
                        min(float(seg_start[0]), float(seg_end[0])),
                        max(float(seg_start[0]), float(seg_end[0])),
                        min(float(seg_start[1]), float(seg_end[1])),
                        max(float(seg_start[1]), float(seg_end[1])),
                    ),
                )
            )
    return tuple(out)


__all__ = [
    "_BondSegment2D",
    "_bond_perpendicular_unoriented_2d",
    "_dangling_stub_segment_2d",
    "_dangling_stub_segment_3d",
    "_layout_quadratic_bond_polyline_2d",
    "_normalize_2d",
    "_point_segment_distance_2d",
    "_planar_contraction_bond_segment_records_2d",
    "_planar_contraction_bond_segments_2d",
    "_segment_hits_existing_geometry_2d",
    "_segment_bboxes_overlap_2d",
    "_segment_point_min_distance_sq_2d",
    "_segment_point_min_distance_sq_2d_many",
    "_segment_point_min_distance_sq_3d",
    "_segment_segment_min_distance_2d",
    "_segment_segment_min_distance_sq_3d",
    "_segments_cross_2d",
]
