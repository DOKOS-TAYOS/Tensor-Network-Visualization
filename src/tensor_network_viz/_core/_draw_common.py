"""Shared scale and style parameters for 2D and 3D drawing."""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from matplotlib.textpath import TextPath
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..config import PlotConfig
from ._label_format import format_tensor_node_label
from .contractions import _ContractionGroups, _group_contractions
from .curves import (
    _ellipse_points,
    _ellipse_points_3d,
    _quadratic_curve,
    _require_self_endpoints,
)
from .graph import (
    _EdgeData,
    _EdgeEndpoint,
    _endpoint_index_caption,
    _GraphData,
    _require_contraction_endpoints,
)
from .layout import (
    AxisDirections,
    NodePositions,
    _orthogonal_unit,
)

# Multiedge separation; keep in sync with `_LAYOUT_BOND_CURVE_OFFSET_FACTOR` in layout.py.
_CURVE_OFFSET_FACTOR: float = 0.15
# Blends with chord length so multiedges keep visible separation when endpoints are close.
_CURVE_NEAR_PAIR_REF: float = 0.28
# Extra radius + offset so index captions sit just outside tensor disks (data units).
_NODE_LABEL_MARGIN_FACTOR: float = 1.22
# Small extra perpendicular gap (× layout ``scale``) for 2D bond index labels after half-linewidth.
_INDEX_LABEL_2D_PERP_EXTRA: float = 0.014
# Tighter pad for stroke-flush captions (contractions / physical stubs); only ``hw + this``.
_INDEX_LABEL_2D_STROKE_PAD: float = 0.0035
# Scales ``~1 em`` in data units to extra perp offset so small/large fonts hug the stroke similarly.
_STROKE_LABEL_EM_PERP_FRAC: float = 0.22
# Long bonds (e.g. MERA) cap em-based perp offset vs ``hw`` (max font blows up em in data units).
_STROKE_LABEL_EM_PERP_MAX_HW_MULT: float = 2.0
# If true curve tangent vs blend diverge (dot product), keep left/right using blended normal.
_STROKE_LABEL_GEOM_NORMAL_DOT_MIN: float = 0.5
# Disk clearance as a fraction of shortest bond (``renderer._SHORTEST_EDGE_RADIUS_FRACTION``).
_EDGE_INDEX_NODE_CLEAR_FRAC: float = 0.3
# Target caption span × bond length (data units); contraction vs physical-open axis.
_EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT: float = 0.45 * (1.0 - 2.0 * _EDGE_INDEX_NODE_CLEAR_FRAC)
_EDGE_INDEX_LABEL_SPAN_FRAC_PHYS: float = 0.7 * (1.0 - _EDGE_INDEX_NODE_CLEAR_FRAC)
# Weight of on-curve tangent when blending with chord (2D curved multiedges).
_CURVE_TANGENT_BLEND_LAMBDA: float = 0.2
# Global scale on all bond / stub index caption font sizes (~shortest-bond span still per edge).
_EDGE_INDEX_LABEL_FONT_GLOBAL_SCALE: float = 0.8
# Open / physical legs: drawn label is 20% larger than internal bond captions (after global scale).
_PHYSICAL_INDEX_LABEL_FONT_SCALE: float = 1.2
# All 3D index + tensor labels scale vs 2D (depth reads smaller; sizing uses data-space bond span).
_LABEL_FONT_3D_SCALE: float = 1.2
_AXIS_TIE_EPS: float = 1e-9
# TextPath width under-estimates padded bbox slightly; calibrate so nominal fraction holds visually.
_EDGE_INDEX_LABEL_WIDTH_CALIB: float = 1.12
# Along-edge reference: edge of label aligns at this arc-length fraction from its endpoint.
_EDGE_INDEX_LABEL_ALONG_FRAC: float = 0.3
# Physical dangling legs (2D): inset from **open tip** — smaller ⇒ closer to the free end.
_PHYS_DANGLING_2D_FRAC_FROM_TIP: float = 0.07
# Draw order: bonds < node disks < edge index labels < tensor names (on top).
_ZORDER_NODE_DISK: int = 3
_ZORDER_EDGE_INDEX_LABEL: int = 5
_ZORDER_TENSOR_NAME: int = 8
_EDGE_INDEX_LABEL_GID: str = "tnv_edge_index"
_TENSOR_LABEL_GID: str = "tnv_tensor"
# TextPath under-estimates the final Text bbox; scale diagonal for "fits inside disk" checks.
_TEXT_RENDER_DIAGONAL_FACTOR: float = 1.52
# Keep tensor names slightly inset from the disk / projected octahedron silhouette.
_TENSOR_LABEL_INSIDE_FILL: float = 0.88
# Line end caps / joins: slight rounding reads softer than Matplotlib's default butt/miter.
_EDGE_LINE_CAP_STYLE: str = "round"
_EDGE_LINE_JOIN_STYLE: str = "round"
_FIGURE_MIN_PX_REF: float = 520.0


def _apply_edge_line_style(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("solid_capstyle", _EDGE_LINE_CAP_STYLE)
    kwargs.setdefault("solid_joinstyle", _EDGE_LINE_JOIN_STYLE)


def _apply_text_no_clip(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("clip_on", False)


def _max_perpendicular_bond_curve_offset(
    graph: _GraphData,
    positions: NodePositions,
    contraction_groups: _ContractionGroups,
    scale: float,
) -> float:
    """Max |offset| used for quadratic bond bulge (same formula as ``_curved_edge_points``)."""
    best = 0.0
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        left_id, right_id = edge.node_ids
        offset_index, edge_count = contraction_groups.offsets[id(edge)]
        delta = np.asarray(positions[right_id], dtype=float) - np.asarray(
            positions[left_id], dtype=float
        )
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        effective_chord = float(math.hypot(distance, _CURVE_NEAR_PAIR_REF * scale))
        raw = (
            (offset_index - (edge_count - 1) / 2.0) * _CURVE_OFFSET_FACTOR * scale * effective_chord
        )
        best = max(best, abs(float(raw)))
    return best


def _self_loop_spatial_extent(p: _DrawScaleParams) -> float:
    """Conservative distance from node center to farthest self-loop ellipse shell (2D/3D)."""
    return float(p.r + p.loop_r + math.hypot(float(p.ellipse_w), float(p.ellipse_h)))


def _view_outset_margin_data_units(
    graph: _GraphData,
    positions: NodePositions,
    p: _DrawScaleParams,
    scale: float,
    contraction_groups: _ContractionGroups,
) -> float:
    """Absolute data-units padding so disks, stubs, curved bonds, loops, and
    labels stay in frame."""
    curve = _max_perpendicular_bond_curve_offset(graph, positions, contraction_groups, scale)
    m = float(p.r + p.stub + curve)
    if any(edge.kind == "self" for edge in graph.edges):
        m = max(m, _self_loop_spatial_extent(p))
    m = max(m, float(p.r) + max(p.label_offset * 0.55, p.r * 0.35))
    return m


def _apply_axis_limits_with_outset(
    ax: Any,
    coords: np.ndarray,
    *,
    view_margin: float,
    dimensions: Literal[2, 3],
) -> None:
    if coords.size == 0:
        return
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    span_raw = max_c - min_c
    breathe = np.maximum(span_raw * 0.02, 1e-9)
    lo = min_c - view_margin - breathe
    hi = max_c + view_margin + breathe
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    if dimensions == 3:
        ax.set_zlim(float(lo[2]), float(hi[2]))
        ax.set_box_aspect((hi - lo).astype(float))
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _stack_visible_tensor_coords(graph: _GraphData, positions: NodePositions) -> np.ndarray:
    visible_ids = [nid for nid, n in graph.nodes.items() if not n.is_virtual]
    if visible_ids:
        return np.stack([np.asarray(positions[nid], dtype=float) for nid in visible_ids])
    if not positions:
        return np.zeros((0, 2), dtype=float)
    return np.stack([np.asarray(v, dtype=float) for v in positions.values()])


def _line_halfwidth_data_2d(ax: Axes, lw_pt: float, xy: np.ndarray) -> float:
    """Half of a bond linewidth in data units at *xy* (robust for ``set_aspect('equal')``)."""
    x, y = float(xy[0]), float(xy[1])
    half_w_px = (float(lw_pt) / 72.0) * float(ax.figure.dpi) * 0.5
    t0 = ax.transData.transform(np.array([[x, y]], dtype=float))[0]
    t1 = ax.transData.transform(np.array([[x + 1.0, y]], dtype=float))[0]
    t2 = ax.transData.transform(np.array([[x, y + 1.0]], dtype=float))[0]
    px_per_du = max(abs(float(t1[0] - t0[0])), abs(float(t2[1] - t0[1])), 1e-12)
    return float(half_w_px / px_per_du)


def _edge_index_font_em_data_2d(ax: Axes, xy: np.ndarray, fontsize_pt: float) -> float:
    """Approximate one em in data units at *xy* (isotropic ``transData`` scale).

    Matches the linewidth helper.
    """
    dpi = float(getattr(ax.figure, "dpi", None) or 100.0)
    em_px = (float(fontsize_pt) / 72.0) * dpi
    x, y = float(xy[0]), float(xy[1])
    t0 = ax.transData.transform(np.array([[x, y]], dtype=float))[0]
    t1 = ax.transData.transform(np.array([[x + 1.0, y]], dtype=float))[0]
    t2 = ax.transData.transform(np.array([[x, y + 1.0]], dtype=float))[0]
    px_per_du = max(abs(float(t1[0] - t0[0])), abs(float(t2[1] - t0[1])), 1e-12)
    return float(em_px / px_per_du)


def _edge_index_label_span_frac(*, is_physical: bool) -> float:
    """Bond-length fraction used to size index caption text (non-physical vs physical axis)."""
    return _EDGE_INDEX_LABEL_SPAN_FRAC_PHYS if is_physical else _EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT


def _blend_bond_tangent_with_chord_2d(
    t_curve_2d: np.ndarray,
    bond_start_2d: np.ndarray,
    bond_end_2d: np.ndarray,
    *,
    blend_lambda: float = _CURVE_TANGENT_BLEND_LAMBDA,
) -> np.ndarray:
    """Unit tangent blended toward chord; curved labels track the bulge slightly less."""
    d = np.asarray(bond_end_2d, dtype=float).reshape(2) - np.asarray(
        bond_start_2d, dtype=float
    ).reshape(2)
    L = float(np.linalg.norm(d))
    if L < 1e-15:
        v = np.asarray(t_curve_2d, dtype=float).reshape(2)
        nv = float(np.linalg.norm(v))
        return v / max(nv, 1e-15)
    t_chord = d / L
    t_c = np.asarray(t_curve_2d, dtype=float).reshape(2)
    w = float(np.clip(blend_lambda, 0.0, 1.0))
    b = (1.0 - w) * t_chord + w * t_c
    nb = float(np.linalg.norm(b))
    return b / max(nb, 1e-15)


def _blend_bond_tangent_with_chord_3d(
    t_curve_3d: np.ndarray,
    bond_start_3d: np.ndarray,
    bond_end_3d: np.ndarray,
    *,
    blend_lambda: float = _CURVE_TANGENT_BLEND_LAMBDA,
) -> np.ndarray:
    """3D bond: same chord blend as 2D, for label alignment in world space."""
    d = np.asarray(bond_end_3d, dtype=float).reshape(3) - np.asarray(
        bond_start_3d, dtype=float
    ).reshape(3)
    L = float(np.linalg.norm(d))
    if L < 1e-15:
        v = np.asarray(t_curve_3d, dtype=float).reshape(3)
        nv = float(np.linalg.norm(v))
        return v / max(nv, 1e-15)
    t_chord = d / L
    t_c = np.asarray(t_curve_3d, dtype=float).reshape(3)
    w = float(np.clip(blend_lambda, 0.0, 1.0))
    b = (1.0 - w) * t_chord + w * t_c
    nb = float(np.linalg.norm(b))
    return b / max(nb, 1e-15)


def _stroke_index_normal_screen_unit_2d(
    tg_scr: np.ndarray,
    ta_scr: np.ndarray,
) -> np.ndarray:
    """Unit normal in display plane; matches 2D stroke label left/right side choice."""
    tg = np.asarray(tg_scr, dtype=float).reshape(2)
    tg = tg / max(float(np.linalg.norm(tg)), 1e-15)
    ta = np.asarray(ta_scr, dtype=float).reshape(2)
    ta = ta / max(float(np.linalg.norm(ta)), 1e-15)
    n_blend = np.array([-float(ta[1]), float(ta[0])], dtype=float)
    if float(np.dot(tg, ta)) >= float(_STROKE_LABEL_GEOM_NORMAL_DOT_MIN):
        n = np.array([-float(tg[1]), float(tg[0])], dtype=float)
        if float(np.dot(n, n_blend)) < 0.0:
            n = -n
    else:
        n = n_blend
    nn = float(np.linalg.norm(n))
    return n / max(nn, 1e-15)


def _nominal_figure_px_per_data_unit_3d(ax: Any) -> float:
    """Approximate display pixels per data unit from figure size and axis spans only.

    No camera projection: the max x/y/z extent is mapped to the smaller figure side in pixels.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    span = max(float(x1 - x0), float(y1 - y0), float(z1 - z0), 1e-9)
    dpi = float(getattr(ax.figure, "dpi", None) or 100.0)
    w_in, h_in = ax.figure.get_size_inches()
    min_px = float(min(w_in, h_in) * dpi)
    return float(min_px / span)


def _stroke_perp_distance_data_units_3d(
    ax: Any,
    p: _DrawScaleParams,
    scale: float,
    fontsize_pt: float,
) -> float:
    """Perpendicular stroke clearance in data units (nominal px/du; same layers as 2D)."""
    px_per_du = _nominal_figure_px_per_data_unit_3d(ax)
    dpi = float(getattr(ax.figure, "dpi", None) or 100.0)
    hw_px = (float(p.lw) / 72.0) * dpi * 0.5
    hw_du = hw_px / max(px_per_du, 1e-15)
    pad_du = float(_INDEX_LABEL_2D_STROKE_PAD) * float(scale)
    em_px = (float(fontsize_pt) / 72.0) * dpi * float(_STROKE_LABEL_EM_PERP_FRAC)
    em_du = em_px / max(px_per_du, 1e-15)
    em_du = min(em_du, float(_STROKE_LABEL_EM_PERP_MAX_HW_MULT) * hw_du)
    return float(hw_du + pad_du + em_du)


def _contraction_edge_index_label_3d_placement(
    *,
    Q: np.ndarray,
    t_geom_3d: np.ndarray,
    t_align_3d: np.ndarray,
    text_ep: Literal["left", "right"],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    fontsize_pt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """3D contraction: offset in data units from tangents only (no camera projection)."""
    Q3 = np.asarray(Q[:3], dtype=float).reshape(3)
    tg_u = np.asarray(t_geom_3d, dtype=float).reshape(3)
    tg_u = tg_u / max(float(np.linalg.norm(tg_u)), 1e-15)
    ta_u = np.asarray(t_align_3d, dtype=float).reshape(3)
    ta_u = ta_u / max(float(np.linalg.norm(ta_u)), 1e-15)
    g_perp = tg_u - float(np.dot(tg_u, ta_u)) * ta_u
    gn = float(np.linalg.norm(g_perp))
    if gn < 1e-12:
        n_world = _perpendicular_3d(ta_u)
        n_world = n_world / max(float(np.linalg.norm(n_world)), 1e-15)
    else:
        e2 = g_perp / gn
        tg_xy = np.array(
            [float(np.dot(tg_u, ta_u)), float(np.dot(tg_u, e2))],
            dtype=float,
        )
        ta_xy = np.array([1.0, 0.0], dtype=float)
        n2 = _stroke_index_normal_screen_unit_2d(tg_xy, ta_xy)
        n_world = n2[0] * ta_u + n2[1] * e2
    side = 1.0 if text_ep == "left" else -1.0
    perp_du = _stroke_perp_distance_data_units_3d(ax, p, scale, fontsize_pt)
    pos = Q3 + side * n_world * perp_du
    align_kw = _edge_index_text_kw_tangent_stroke_align(
        text_ep=text_ep,
        tangent=t_align_3d,
        ax=ax,
        dimensions=3,
        data_xy_3d=True,
    )
    return pos, align_kw


def _edge_index_label_is_vertical_axis_2d(d_out: np.ndarray) -> bool:
    """True when |dy| >= |dx| for outward bond direction (vertical placement mode)."""
    d = np.asarray(d_out, dtype=float).reshape(2)
    return abs(float(d[1])) >= abs(float(d[0]))


def _edge_index_label_axis_tie_vertical_2d(d_out: np.ndarray, rng: np.random.Generator) -> bool:
    """When |dx| and |dy| are tied, choose vertical vs horizontal placement at random."""
    d = np.asarray(d_out, dtype=float).reshape(2)
    adx, ady = abs(float(d[0])), abs(float(d[1]))
    if abs(adx - ady) > _AXIS_TIE_EPS:
        return ady >= adx
    return bool(rng.random() < 0.5)


def _polyline_arc_length_total(curve: np.ndarray) -> float:
    pts = np.asarray(curve, dtype=float)
    if int(pts.shape[0]) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _point_tangent_along_polyline_from_start(
    curve: np.ndarray, dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Point at arc *dist* from ``curve[0]`` and unit tangent in the forward polyline direction."""
    pts = np.asarray(curve, dtype=float)
    n = int(pts.shape[0])
    dim = int(pts.shape[1])
    if n < 2:
        t0 = np.zeros(dim, dtype=float)
        t0[0] = 1.0
        return pts[0].copy(), t0
    seg = np.diff(pts, axis=0)
    sl = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([np.array([0.0], dtype=float), np.cumsum(sl)])
    total = float(cum[-1])
    if total < 1e-15:
        t0 = seg[0] / max(float(sl[0]), 1e-15)
        return pts[0].copy(), t0
    target = float(np.clip(dist, 0.0, total))
    for i in range(len(sl)):
        if float(cum[i + 1]) >= target - 1e-12:
            seg_t = (target - float(cum[i])) / max(float(sl[i]), 1e-15)
            seg_t = float(np.clip(seg_t, 0.0, 1.0))
            point = pts[i] + seg_t * seg[i]
            tangent = seg[i] / max(float(sl[i]), 1e-15)
            return point, tangent
    tangent = seg[-1] / max(float(sl[-1]), 1e-15)
    return pts[-1].copy(), tangent


def _point_tangent_along_polyline_from_end(
    curve: np.ndarray, dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``_point_tangent_along_polyline_from_start`` but *dist* counts from
    ``curve[-1]`` backward."""
    pts = np.asarray(curve, dtype=float)
    total = _polyline_arc_length_total(pts)
    return _point_tangent_along_polyline_from_start(pts, max(0.0, total - float(dist)))


def _tangent_screen_angle_deg(
    ax: Any,
    tangent: np.ndarray,
    dimensions: Literal[2, 3],
    *,
    world_anchor: np.ndarray | None = None,
    data_xy_3d: bool = False,
) -> float:
    """Counter-clockwise degrees for the on-screen direction of *tangent*."""
    t = np.asarray(tangent, dtype=float)
    if dimensions == 2:
        return float(math.degrees(math.atan2(float(t[1]), float(t[0]))))
    if data_xy_3d:
        t3 = np.asarray(tangent, dtype=float).reshape(3)
        return float(math.degrees(math.atan2(float(t3[1]), float(t3[0]))))
    M = ax.get_proj()
    if world_anchor is not None:
        q = np.asarray(world_anchor, dtype=float).reshape(3)
        tu = np.asarray(t, dtype=float).reshape(3)
        nrm = float(np.linalg.norm(tu))
        if nrm < 1e-15:
            return 0.0
        tu = tu / nrm
        x0, y0, _ = proj3d.proj_transform(float(q[0]), float(q[1]), float(q[2]), M)
        x1, y1, _ = proj3d.proj_transform(
            float(q[0] + tu[0]),
            float(q[1] + tu[1]),
            float(q[2] + tu[2]),
            M,
        )
        return float(math.degrees(math.atan2(float(y1 - y0), float(x1 - x0))))
    x0, y0, _ = proj3d.proj_transform(0.0, 0.0, 0.0, M)
    x1, y1, _ = proj3d.proj_transform(float(t[0]), float(t[1]), float(t[2]), M)
    return float(math.degrees(math.atan2(float(y1 - y0), float(x1 - x0))))


def _upright_screen_text_rotation_deg_raw(angle_deg: float) -> tuple[float, bool]:
    """Map rotation to ``[-90, 90]``; return whether ±180° was applied (then swap *ha*)."""
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    flipped_180 = False
    if a > 90.0:
        a -= 180.0
        flipped_180 = True
    elif a < -90.0:
        a += 180.0
        flipped_180 = True
    return float(a), flipped_180


def _edge_index_text_kw_tangent_stroke_align(
    *,
    text_ep: Literal["left", "right"],
    tangent: np.ndarray,
    ax: Any,
    dimensions: Literal[2, 3],
    world_anchor: np.ndarray | None = None,
    data_xy_3d: bool = False,
) -> dict[str, Any]:
    """Anchor on bond outer face: +n side uses bottom, −n uses top (swap if upright flips 180°)."""
    rot_raw = _tangent_screen_angle_deg(
        ax,
        tangent,
        dimensions,
        world_anchor=world_anchor,
        data_xy_3d=data_xy_3d,
    )
    rot, flipped = _upright_screen_text_rotation_deg_raw(rot_raw)
    va = "bottom" if text_ep == "left" else "top"
    if flipped:
        va = "top" if va == "bottom" else "bottom"
    return {
        "ha": "center",
        "va": va,
        "rotation": rot,
        "rotation_mode": "anchor",
    }


def _edge_index_along_bond_text_kw(
    *,
    endpoint: Literal["left", "right"],
    tangent: np.ndarray,
    ax: Any,
    dimensions: Literal[2, 3],
    va: str = "center",
    ha: str | None = None,
) -> dict[str, Any]:
    """``ha`` / ``va`` / ``rotation`` for index text along a bond.

    Optional ``ha`` override after upright flip.
    """
    rot_raw = _tangent_screen_angle_deg(ax, tangent, dimensions)
    rot, flipped = _upright_screen_text_rotation_deg_raw(rot_raw)
    if ha is None:
        ha_end = "left" if endpoint == "left" else "right"
        if flipped:
            ha_end = "right" if ha_end == "left" else "left"
    else:
        ha_end = ha
        if flipped and ha_end in ("left", "right"):
            ha_end = "right" if ha_end == "left" else "left"
    return {
        "ha": ha_end,
        "va": va,
        "rotation": rot,
        "rotation_mode": "anchor",
    }


def _edge_index_rim_arc_from_endpoint(*, r_global: float, half_polyline_length: float) -> float:
    """Arc length along a bond from an endpoint reserved by the node disk.

    *r_global* is the graph-wide tensor radius (``p.r`` ≈ shortest contraction bond × 0.3);
    it does not vary per edge. Clamped so the rim does not lie past the bond midpoint.
    """
    return float(min(float(r_global), float(half_polyline_length) * (1.0 - 1e-9)))


def _contraction_edge_index_label_2d_placement(
    *,
    Q: np.ndarray,
    t_geom_2d: np.ndarray,
    t_align_2d: np.ndarray,
    text_ep: Literal["left", "right"],
    p: _DrawScaleParams,
    ax: Axes,
    scale: float,
    fontsize_pt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """2D contraction bond: perp uses true curve tangent; rotation uses chord blend and em term."""
    Q2 = np.asarray(Q[:2], dtype=float)
    tg = np.asarray(t_geom_2d, dtype=float).reshape(2)
    tg = tg / max(float(np.linalg.norm(tg)), 1e-15)
    ta = np.asarray(t_align_2d, dtype=float).reshape(2)
    ta = ta / max(float(np.linalg.norm(ta)), 1e-15)
    n = _stroke_index_normal_screen_unit_2d(tg, ta)
    side = 1.0 if text_ep == "left" else -1.0
    hw = _line_halfwidth_data_2d(ax, float(p.lw), Q2)
    pad = float(_INDEX_LABEL_2D_STROKE_PAD) * float(scale)
    em_du = _edge_index_font_em_data_2d(ax, Q2, float(fontsize_pt))
    em_add = float(_STROKE_LABEL_EM_PERP_FRAC) * em_du
    em_add = min(em_add, float(_STROKE_LABEL_EM_PERP_MAX_HW_MULT) * hw)
    perp = hw + pad + em_add
    pos = Q2 + side * n * perp
    align_kw = _edge_index_text_kw_tangent_stroke_align(
        text_ep=text_ep, tangent=ta, ax=ax, dimensions=2
    )
    return pos, align_kw


def _bond_index_label_perp_offset(
    caption: str,
    *,
    p: _DrawScaleParams,
    scale: float,
    dimensions: Literal[2, 3],
    ax: Any,
    anchor: np.ndarray,
    world_perp_dir: np.ndarray | None = None,
) -> float:
    """Distance from curve to label along the outward normal (2D: data units at *anchor*).

    3D: when *world_perp_dir* is a unit direction in world space, return world distance so the
    on-screen offset matches the 2D stroke + caption-padding stack (pixels).
    """
    display = format_tensor_node_label(caption)
    n = len(display.strip())
    if dimensions == 2:
        xy2 = np.asarray(anchor[:2], dtype=float)
        hw = _line_halfwidth_data_2d(cast(Axes, ax), float(p.lw), xy2)
        excess = max(0, n - 5)
        span_extra = 0.021 * float(scale) * (float(excess) ** 0.82)
        bump = _INDEX_LABEL_2D_PERP_EXTRA * float(scale)
        return float(hw + span_extra + bump)
    if world_perp_dir is not None:
        px_per_du = _nominal_figure_px_per_data_unit_3d(ax)
        excess = max(0, n - 5)
        span_extra_data = 0.021 * float(scale) * (float(excess) ** 0.82)
        bump_data = float(_INDEX_LABEL_2D_PERP_EXTRA) * float(scale)
        dpi = float(getattr(ax.figure, "dpi", None) or 100.0)
        hw_px = (float(p.lw) / 72.0) * dpi * 0.5
        span_px = span_extra_data * px_per_du
        bump_px = bump_data * px_per_du
        total_px = hw_px + span_px + bump_px
        if px_per_du < 1e-12:
            span_extra = 0.026 * float(scale) * (max(0, n - 1) ** 0.8)
            base = max(float(p.label_offset) * 0.40, float(p.r) * 0.26)
            return float(base + span_extra * 1.05)
        return float(total_px / px_per_du)
    span_extra = 0.026 * float(scale) * (max(0, n - 1) ** 0.8)
    base = max(float(p.label_offset) * 0.40, float(p.r) * 0.26)
    return float(base + span_extra * 1.05)


# 3D nodes: octahedron (8 tris / node). Full UV spheres are too heavy for interactive mplot3d.
_UNIT_NODE_TRIS: np.ndarray = np.asarray(
    [
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    ],
    dtype=float,
)
_OCTAHEDRON_TRI_COUNT: int = int(_UNIT_NODE_TRIS.shape[0])
# 3D bond curves use ``p.lw``; octahedron rims are drawn much finer so edges stay subtle vs bonds.
_OCTAHEDRON_EDGE_LINEWIDTH_FACTOR: float = 0.18
_OCTAHEDRON_EDGE_LINEWIDTH_MIN: float = 0.04


def _graph_edge_degree(graph: _GraphData, node_id: int) -> int:
    """Number of graph edges incident on *node_id* (contractions, dangling stubs, self-loops)."""
    return sum(1 for edge in graph.edges for nid in edge.node_ids if nid == node_id)


def _visible_degree_one_mask(graph: _GraphData, visible_node_ids: list[int]) -> np.ndarray:
    """True when a visible tensor has total graph degree 1."""
    return np.array([_graph_edge_degree(graph, nid) == 1 for nid in visible_node_ids], dtype=bool)


class _PlotAdapter(Protocol):
    """Protocol for dimension-aware plotting (2D vs 3D)."""

    def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None: ...
    def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None: ...
    def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None: ...
    def draw_tensor_nodes(
        self,
        coords: np.ndarray,
        *,
        config: PlotConfig,
        p: _DrawScaleParams,
        degree_one_mask: np.ndarray,
    ) -> None: ...
    def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None: ...


def _make_plotter(
    ax: Any,
    *,
    dimensions: Literal[2, 3],
    hover_edge_targets: list[tuple[np.ndarray, str]] | None = None,
) -> _PlotAdapter:
    """Create a dimension-aware plot adapter."""

    if dimensions == 2:

        class _2DPlotter:
            __slots__ = ("_ax", "_edge_segments", "_hover_edge_targets", "_node_disk_collection")

            def __init__(
                self,
                ax_2d: Axes,
                hover_edges: list[tuple[np.ndarray, str]] | None,
            ) -> None:
                self._ax = ax_2d
                self._edge_segments: list[tuple[int, str, float, np.ndarray]] = []
                self._hover_edge_targets = hover_edges
                self._node_disk_collection: PatchCollection | None = None

            def flush_edge_collections(self) -> None:
                """Batch buffered edges into a few LineCollections (call after all edges drawn)."""
                if not self._edge_segments:
                    return
                groups: dict[tuple[int, str, float], list[np.ndarray]] = defaultdict(list)
                for z, color, lw, seg in self._edge_segments:
                    groups[(z, color, lw)].append(seg)
                ax_ = self._ax
                for (z, color, lw), segs in sorted(groups.items(), key=lambda kv: kv[0][0]):
                    coll = LineCollection(
                        segs,
                        colors=color,
                        linewidths=lw,
                        zorder=z,
                        capstyle=_EDGE_LINE_CAP_STYLE,
                        joinstyle=_EDGE_LINE_JOIN_STYLE,
                    )
                    ax_.add_collection(coll)
                self._edge_segments.clear()

            def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
                _apply_edge_line_style(kwargs)
                z = int(kwargs.get("zorder", 1))
                color = str(kwargs.get("color", "#000000"))
                lw = float(kwargs.get("linewidth", 1.0))
                seg = np.array(
                    [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
                    dtype=float,
                )
                self._edge_segments.append((z, color, lw, seg))

            def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
                _apply_edge_line_style(kwargs)
                z = int(kwargs.get("zorder", 1))
                color = str(kwargs.get("color", "#000000"))
                lw = float(kwargs.get("linewidth", 1.0))
                seg = np.asarray(curve[:, :2], dtype=float, order="C")
                self._edge_segments.append((z, color, lw, seg))

            def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
                _apply_text_no_clip(kwargs)
                self._ax.text(pos[0], pos[1], text, **kwargs)

            def draw_tensor_nodes(
                self,
                coords: np.ndarray,
                *,
                config: PlotConfig,
                p: _DrawScaleParams,
                degree_one_mask: np.ndarray,
            ) -> None:
                n = int(coords.shape[0])
                if n == 0:
                    return
                patches = [
                    Circle((float(coords[i, 0]), float(coords[i, 1])), radius=p.r) for i in range(n)
                ]
                faces = [
                    config.node_color_degree_one if degree_one_mask[i] else config.node_color
                    for i in range(n)
                ]
                c1 = config.node_edge_color_degree_one
                c0 = config.node_edge_color
                edges_ = [c1 if degree_one_mask[i] else c0 for i in range(n)]
                coll = PatchCollection(
                    patches,
                    facecolors=faces,
                    edgecolors=edges_,
                    linewidths=float(p.lw),
                    zorder=_ZORDER_NODE_DISK,
                    match_original=False,
                )
                self._ax.add_collection(coll)
                self._node_disk_collection = coll

            def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None:
                _apply_axis_limits_with_outset(
                    self._ax, coords, view_margin=view_margin, dimensions=2
                )

        return _2DPlotter(ax, hover_edge_targets)

    class _3DPlotter:
        def __init__(self, hover_edges: list[tuple[np.ndarray, str]] | None) -> None:
            self._hover_edge_targets = hover_edges

        def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)

        def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], **kwargs)

        def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
            _apply_text_no_clip(kwargs)
            ax.text(pos[0], pos[1], pos[2], text, **kwargs)

        def draw_tensor_nodes(
            self,
            coords: np.ndarray,
            *,
            config: PlotConfig,
            p: _DrawScaleParams,
            degree_one_mask: np.ndarray,
        ) -> None:
            n_nod = int(coords.shape[0])
            if n_nod == 0:
                return
            # Unit octahedron vertices lie on axes at distance 1; scale by p.r so circumradius = p.r
            # (same metric as 2D disks; radius tracks shortest bond via renderer fraction).
            scaled = _UNIT_NODE_TRIS * p.r
            c = coords.astype(float, copy=False)
            stacked = scaled[np.newaxis, :, :, :] + c[:, np.newaxis, np.newaxis, :]
            polys = stacked.reshape(-1, 3, 3)
            node_edge_lw = max(
                float(p.lw) * _OCTAHEDRON_EDGE_LINEWIDTH_FACTOR,
                _OCTAHEDRON_EDGE_LINEWIDTH_MIN,
            )
            face_list: list[str] = []
            edge_list: list[str] = []
            for i in range(n_nod):
                fc = config.node_color_degree_one if degree_one_mask[i] else config.node_color
                ec = (
                    config.node_edge_color_degree_one
                    if degree_one_mask[i]
                    else config.node_edge_color
                )
                face_list.extend([fc] * _OCTAHEDRON_TRI_COUNT)
                edge_list.extend([ec] * _OCTAHEDRON_TRI_COUNT)
            coll = Poly3DCollection(
                polys,
                facecolors=face_list,
                edgecolors=edge_list,
                linewidths=node_edge_lw,
            )
            coll.set_sort_zpos(_ZORDER_NODE_DISK)
            ax.add_collection3d(coll)

        def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None:
            _apply_axis_limits_with_outset(ax, coords, view_margin=view_margin, dimensions=3)

    return _3DPlotter(hover_edge_targets)


def _perpendicular_2d(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]], dtype=float)


def _perpendicular_3d(direction: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    perp = np.cross(direction, reference)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
    return perp / np.linalg.norm(perp)


def _bond_perpendicular_unoriented(
    delta: np.ndarray,
    dimensions: Literal[2, 3],
) -> np.ndarray:
    dist = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / dist
    return _perpendicular_3d(direction) if dimensions == 3 else _perpendicular_2d(direction)


def _signed_bond_perpendicular(
    delta: np.ndarray,
    dimensions: Literal[2, 3],
) -> np.ndarray:
    perpendicular = _bond_perpendicular_unoriented(delta, dimensions)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    if (dimensions == 2 and perpendicular[1] < 0) or (dimensions == 3 and perpendicular[2] < 0):
        perpendicular = -perpendicular
    return perpendicular


def _node_label_clearance(p: _DrawScaleParams) -> float:
    """Minimum distance from a tensor center to an index label anchor (data units)."""
    return float(p.r * _NODE_LABEL_MARGIN_FACTOR + p.label_offset * 0.46)


def _edge_index_text_kwargs(
    config: PlotConfig,
    *,
    fontsize: float,
    stub_kind: Literal["bond", "dangling"] = "bond",
    bbox_pad: float = 0.18,
) -> dict[str, Any]:
    """Matplotlib kwargs for index labels (semi-transparent, tinted bbox)."""
    if stub_kind == "dangling":
        rgb = (0.99, 0.92, 0.91)
        edgecolor: tuple[float, float, float, float] = (0.58, 0.36, 0.36, 0.52)
    else:
        rgb = (0.90, 0.93, 0.99)
        edgecolor = (0.36, 0.42, 0.58, 0.52)
    if bbox_pad <= 0.09:
        alpha_fill = 0.52
    elif bbox_pad <= 0.14:
        alpha_fill = 0.64
    else:
        alpha_fill = 0.74
    facecolor: tuple[float, float, float, float] = (rgb[0], rgb[1], rgb[2], alpha_fill)
    return {
        "color": config.label_color,
        "fontsize": fontsize,
        "zorder": _ZORDER_EDGE_INDEX_LABEL,
        "gid": _EDGE_INDEX_LABEL_GID,
        "ha": "center",
        "va": "center",
        "bbox": {
            "boxstyle": f"round,pad={bbox_pad}",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 0.45,
        },
    }


def _curve_index_outside_disk(
    curve: np.ndarray,
    anchor: np.ndarray,
    clearance: float,
    *,
    from_start: bool,
) -> int:
    """Sample index along a polyline where the point clears a disk around ``anchor``."""
    n = int(curve.shape[0])
    if n <= 2:
        return min(1, max(0, n - 2))
    if from_start:
        for i in range(n - 1):
            if float(np.linalg.norm(curve[i] - anchor)) >= clearance:
                return min(max(i, 1), n - 2)
        return min(max(1, n // 4), n - 2)
    for i in range(n - 1, 0, -1):
        if float(np.linalg.norm(curve[i] - anchor)) >= clearance:
            return max(min(i, n - 2), 1)
    return max(1, (3 * (n - 1)) // 4)


def _estimate_drawn_label_count(
    graph: _GraphData,
    *,
    show_tensor_labels: bool,
    show_index_labels: bool,
) -> int:
    """Rough count of text labels drawn, for font crowding heuristics."""
    count = 0
    if show_tensor_labels:
        count += sum(1 for node in graph.nodes.values() if not node.is_virtual)
    if not show_index_labels:
        return max(1, count)

    for edge in graph.edges:
        if edge.kind == "dangling":
            if edge.label:
                count += 1
            continue
        if edge.kind in ("self", "contraction"):
            endpoints = (
                _require_self_endpoints(edge)
                if edge.kind == "self"
                else _require_contraction_endpoints(edge)
            )
            count += sum(1 for ep in endpoints if _endpoint_index_caption(ep, edge, graph))
    return max(1, count)


def _contraction_hover_label_text(edge: _EdgeData, graph: _GraphData) -> str:
    ep_l, ep_r = _require_contraction_endpoints(edge)
    parts: list[str] = []
    for ep in (ep_l, ep_r):
        cap = _endpoint_index_caption(ep, edge, graph)
        if cap:
            shown = format_tensor_node_label(cap).strip()
            if shown:
                parts.append(shown)
    return " · ".join(parts)


def _self_loop_hover_label_text(edge: _EdgeData, graph: _GraphData) -> str:
    endpoint_a, endpoint_b = _require_self_endpoints(edge)
    parts: list[str] = []
    for ep in (endpoint_a, endpoint_b):
        cap = _endpoint_index_caption(ep, edge, graph)
        if cap:
            shown = format_tensor_node_label(cap).strip()
            if shown:
                parts.append(shown)
    return " · ".join(parts)


def _dangling_hover_label_text(edge: _EdgeData) -> str:
    if not edge.label:
        return ""
    return format_tensor_node_label(edge.label).strip()


def _sqdist_point_to_segment(
    px: float,
    py: float,
    ax_: float,
    ay_: float,
    bx: float,
    by: float,
) -> float:
    abx, aby = bx - ax_, by - ay_
    apx, apy = px - ax_, py - ay_
    den = abx * abx + aby * aby
    if den <= 1e-18:
        return apx * apx + apy * apy
    t = float(np.clip((apx * abx + apy * aby) / den, 0.0, 1.0))
    qx = ax_ + t * abx
    qy = ay_ + t * aby
    dx = px - qx
    dy = py - qy
    return float(dx * dx + dy * dy)


def _min_sqdist_point_to_polyline_display(
    ax: Axes,
    poly_data: np.ndarray,
    x_disp: float,
    y_disp: float,
) -> float:
    pts = np.asarray(poly_data, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return math.inf
    scr = ax.transData.transform(pts[:, :2])
    best = math.inf
    for i in range(int(scr.shape[0]) - 1):
        x0, y0 = float(scr[i, 0]), float(scr[i, 1])
        x1, y1 = float(scr[i + 1, 0]), float(scr[i + 1, 1])
        d = _sqdist_point_to_segment(x_disp, y_disp, x0, y0, x1, y1)
        best = min(best, d)
    return best


def _min_sqdist_point_to_polyline_display_3d(
    ax: Any,
    poly_world: np.ndarray,
    x_disp: float,
    y_disp: float,
) -> float:
    pts = np.asarray(poly_world, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return math.inf
    M = ax.get_proj()
    n = int(pts.shape[0])
    scr = np.empty((n, 2), dtype=float)
    for i in range(n):
        x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
        xs, ys, _zs = proj3d.proj_transform(x, y, z, M)
        t = np.asarray(ax.transData.transform((xs, ys)), dtype=float).ravel()
        scr[i, 0] = float(t[0])
        scr[i, 1] = float(t[1])
    best = math.inf
    for i in range(n - 1):
        x0, y0 = scr[i, 0], scr[i, 1]
        x1, y1 = scr[i + 1, 0], scr[i + 1, 1]
        d = _sqdist_point_to_segment(x_disp, y_disp, x0, y0, x1, y1)
        best = min(best, d)
    return best


_HOVER_EDGE_PICK_RADIUS_PX: float = 10.0


def _disconnect_tensor_network_hover(fig: Figure) -> None:
    cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
    if cid is not None:
        with suppress(ValueError, KeyError):
            fig.canvas.mpl_disconnect(int(cid))
        fig._tensor_network_viz_hover_cid = None


def _register_2d_hover_labels(
    ax: Axes,
    *,
    node_patch_coll: PatchCollection | None,
    visible_node_ids: list[int],
    tensor_hover: dict[int, tuple[str, float]],
    edge_hover: list[tuple[np.ndarray, str]],
    line_width_px_hint: float,
) -> None:
    """Show tensor / bond labels in a tooltip while the pointer hovers (2D axes)."""
    fig = ax.figure
    _disconnect_tensor_network_hover(fig)

    if not tensor_hover and not edge_hover:
        return

    pick_r = max(_HOVER_EDGE_PICK_RADIUS_PX, float(line_width_px_hint) * 2.0)

    ann = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(12, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#1A202C",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": (0.99, 0.97, 0.92, 0.94),
            "edgecolor": (0.35, 0.35, 0.4, 0.55),
            "linewidth": 0.6,
        },
        visible=False,
        zorder=10_000,
        clip_on=False,
    )
    fig._tensor_network_viz_hover_ann = ann

    def on_move(event: Any) -> None:
        if event.inaxes != ax or event.x is None or event.y is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        x_d, y_d = float(event.x), float(event.y)
        label: str | None = None
        fs_hint = 10.0

        if tensor_hover and node_patch_coll is not None:
            hit, props = node_patch_coll.contains(event)
            if hit:
                inds = props.get("ind")
                if inds is not None and len(inds):
                    k = int(inds[0])
                    if 0 <= k < len(visible_node_ids):
                        nid = visible_node_ids[k]
                        pair = tensor_hover.get(nid)
                        if pair:
                            label, fs_hint = pair[0], float(pair[1])

        if label is None and edge_hover:
            best = math.inf
            best_txt = ""
            for poly, txt in edge_hover:
                if not txt:
                    continue
                d = _min_sqdist_point_to_polyline_display(ax, poly, x_d, y_d)
                if d < best:
                    best = d
                    best_txt = txt
            if best <= pick_r * pick_r:
                label = best_txt
                fs_hint = 9.0

        if not label:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        if event.xdata is None or event.ydata is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        ann.xy = (float(event.xdata), float(event.ydata))
        ann.set_text(label)
        ann.set_fontsize(max(7.5, min(14.0, fs_hint)))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig._tensor_network_viz_hover_cid = fig.canvas.mpl_connect("motion_notify_event", on_move)


def _register_3d_hover_labels(
    ax: Any,
    fig: Figure,
    *,
    positions: NodePositions,
    visible_node_ids: list[int],
    tensor_hover: dict[int, tuple[str, float]],
    edge_hover: list[tuple[np.ndarray, str]],
    line_width_px_hint: float,
    p: _DrawScaleParams,
) -> None:
    """Show tensor / bond labels in a figure-space tooltip while the pointer hovers (3D)."""
    _disconnect_tensor_network_hover(fig)

    if not tensor_hover and not edge_hover:
        return

    pick_r = max(_HOVER_EDGE_PICK_RADIUS_PX, float(line_width_px_hint) * 2.0)

    ann = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xycoords="figure pixels",
        xytext=(12, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#1A202C",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": (0.99, 0.97, 0.92, 0.94),
            "edgecolor": (0.35, 0.35, 0.4, 0.55),
            "linewidth": 0.6,
        },
        visible=False,
        zorder=1_000_000,
        clip_on=False,
    )
    fig._tensor_network_viz_hover_ann = ann

    def on_move(event: Any) -> None:
        if event.inaxes != ax or event.x is None or event.y is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        x_d, y_d = float(event.x), float(event.y)
        label: str | None = None
        fs_hint = 10.0

        if tensor_hover:
            best_d2 = math.inf
            best_pair: tuple[str, float] | None = None
            for nid in visible_node_ids:
                pair = tensor_hover.get(nid)
                if not pair:
                    continue
                c = np.asarray(positions[nid], dtype=float).reshape(-1)
                if c.size < 3:
                    c3 = np.zeros(3, dtype=float)
                    c3[: c.size] = c
                    c = c3
                rpx = _tensor_disk_radius_px(ax, c, p, 3)
                M = ax.get_proj()
                xs, ys, _zs = proj3d.proj_transform(float(c[0]), float(c[1]), float(c[2]), M)
                pt = np.asarray(ax.transData.transform((xs, ys)), dtype=float).ravel()
                dx = float(pt[0]) - x_d
                dy = float(pt[1]) - y_d
                d2 = dx * dx + dy * dy
                if d2 <= rpx * rpx and d2 < best_d2:
                    best_d2 = d2
                    best_pair = pair
            if best_pair is not None:
                label, fs_hint = best_pair[0], float(best_pair[1])

        if label is None and edge_hover:
            best = math.inf
            best_txt = ""
            for poly, txt in edge_hover:
                if not txt:
                    continue
                d = _min_sqdist_point_to_polyline_display_3d(ax, poly, x_d, y_d)
                if d < best:
                    best = d
                    best_txt = txt
            if best <= pick_r * pick_r:
                label = best_txt
                fs_hint = 9.0

        if not label:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        ann.xy = (x_d, y_d)
        ann.set_text(label)
        ann.set_fontsize(max(7.5, min(14.0, fs_hint)))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig._tensor_network_viz_hover_cid = fig.canvas.mpl_connect("motion_notify_event", on_move)


def _figure_size_sqrt_ratio(fig: Figure) -> float:
    dpi = float(getattr(fig, "dpi", None) or 100.0)
    width_in, height_in = fig.get_size_inches()
    min_px = float(min(width_in, height_in) * dpi)
    return float(math.sqrt(min_px / _FIGURE_MIN_PX_REF))


def _textpath_width_pts(text: str, *, fontsize_pt: float) -> float:
    """Horizontal advance of *text* from Matplotlib TextPath at *fontsize_pt* (points)."""
    if not text.strip():
        return 0.0
    fp = FontProperties(size=float(fontsize_pt))
    tp = TextPath((0.0, 0.0), text, prop=fp)
    return float(tp.get_extents().width)


def _bond_endpoints_xyz3(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Promote (x,y) or (x,y,z) endpoints to 3-vectors for mplot3d projection."""
    s = np.asarray(start, dtype=float).reshape(-1)
    e = np.asarray(end, dtype=float).reshape(-1)
    s3 = np.zeros(3, dtype=float)
    e3 = np.zeros(3, dtype=float)
    s3[: min(3, s.size)] = s[: min(3, s.size)]
    e3[: min(3, e.size)] = e[: min(3, e.size)]
    return s3, e3


def _bond_reference_span_px_for_font(
    ax: Any,
    bond_start: np.ndarray,
    bond_end: np.ndarray,
    dimensions: Literal[2, 3],
) -> float:
    """Stable pixel length for sizing index labels vs a *data-space* bond length.

    Before the first full canvas draw, ``Axes.transData`` can be anisotropic (different
    ∂x/∂data vs ∂y/∂data) even with ``equal`` aspect, so chord length in display space
    depends on whether the bond is horizontal or vertical. Using
    ``L_data * (‖∂T/∂x‖ + ‖∂T/∂y‖) / 2`` in 2D avoids lateral legs reading tiny when Matplotlib's
    ``transData`` is temporarily anisotropic before the first full draw (horizontal vs
    vertical ``phys`` on MPS) without pinning all legs to the most inflated scale.

    In 3D, ``bond_px = L_3 * px_per_du`` with ``L_3`` the Euclidean segment length in data
    space and ``px_per_du`` from figure size and axis spans only (see
    ``_nominal_figure_px_per_data_unit_3d``) — no camera projection.
    Pairs ``left``/``right`` share fontsize via ``peer_captions_for_width``.
    """
    s = np.asarray(bond_start, dtype=float)
    e = np.asarray(bond_end, dtype=float)
    if dimensions == 2:
        L = float(np.linalg.norm(e - s))
        if L < 1e-15:
            return 0.0
        mid = 0.5 * (s + e)
        ax2 = cast(Axes, ax)
        x, y = float(mid[0]), float(mid[1])
        T = ax2.transData.transform
        p0 = T(np.array([[x, y]], dtype=float))[0]
        px = T(np.array([[x + 1.0, y]], dtype=float))[0]
        py = T(np.array([[x, y + 1.0]], dtype=float))[0]
        ex = float(np.hypot(float(px[0] - p0[0]), float(px[1] - p0[1])))
        ey = float(np.hypot(float(py[0] - p0[0]), float(py[1] - p0[1])))
        return float(L * 0.5 * (ex + ey))
    s3, e3 = _bond_endpoints_xyz3(bond_start, bond_end)
    L3 = float(np.linalg.norm(e3 - s3))
    if L3 < 1e-15:
        return 0.0
    k = _nominal_figure_px_per_data_unit_3d(ax)
    return float(L3 * k)


def _edge_index_fontsize_for_bond(
    caption: str,
    *,
    bond_start: np.ndarray,
    bond_end: np.ndarray,
    ax: Any,
    dimensions: Literal[2, 3],
    is_physical: bool = False,
    peer_captions_for_width: tuple[str, ...] | None = None,
) -> float:
    """Font size from **this** bond's on-screen length × span fraction.

    Disk radius is not used here. ``p.r`` (shortest-bond / global) affects label *position*
    along edges elsewhere, not this value.

    When ``peer_captions_for_width`` lists both endpoint captions on the same bond, use the
    widest TextPath among them so ``left`` / ``right`` (etc.) share one fontsize.
    """
    show = format_tensor_node_label(caption).strip()
    if not show:
        return 1.0
    bond_px = _bond_reference_span_px_for_font(ax, bond_start, bond_end, dimensions)
    target_px = float(_edge_index_label_span_frac(is_physical=is_physical)) * bond_px
    dpi = float(getattr(ax.figure, "dpi", None) or 100.0)
    if peer_captions_for_width is not None and len(peer_captions_for_width) > 1:
        w_ref = 0.0
        for raw_peer in peer_captions_for_width:
            peer_show = format_tensor_node_label(raw_peer).strip()
            if not peer_show:
                continue
            w_ref = max(
                w_ref,
                _textpath_width_pts(peer_show, fontsize_pt=10.0)
                * float(_EDGE_INDEX_LABEL_WIDTH_CALIB),
            )
        if w_ref < 1e-12:
            w_ref = _textpath_width_pts(show, fontsize_pt=10.0) * float(
                _EDGE_INDEX_LABEL_WIDTH_CALIB
            )
    else:
        w_ref = _textpath_width_pts(show, fontsize_pt=10.0) * float(_EDGE_INDEX_LABEL_WIDTH_CALIB)
    if w_ref < 1e-12 or target_px < 1e-12:
        return 1.0
    fs = 10.0 * target_px * 72.0 / (dpi * w_ref)
    fs *= _EDGE_INDEX_LABEL_FONT_GLOBAL_SCALE
    if is_physical:
        fs *= _PHYSICAL_INDEX_LABEL_FONT_SCALE
    if dimensions == 3:
        fs *= _LABEL_FONT_3D_SCALE
    cap = 22.0 * _LABEL_FONT_3D_SCALE if dimensions == 3 else 22.0
    return float(min(fs, cap))


def _figure_relative_font_scale(fig: Figure, label_count: int) -> float:
    """Scale font sizes from figure dimensions and expected label crowding."""
    size_part = _figure_size_sqrt_ratio(fig)
    crowd = 3.0 + math.sqrt(float(label_count))
    raw = size_part * 7.0 / crowd
    return float(np.clip(raw, 0.26, 1.28))


def _figure_base_size_scale(fig: Figure) -> float:
    """Figure pixel size only (no bond / index label crowding).

    Tensor names use this for their *upper* bound so MERA/PEPS-sized index counts do not
    squash every node title to the same tiny cap; per-node fit still clamps by disk.
    """
    size_part = _figure_size_sqrt_ratio(fig)
    return float(np.clip(size_part, 0.35, 1.28))


def _index_label_bbox_pad(label_slots: int) -> float:
    """Tighter rounded box on dense plots so semi-transparent patches overlap less."""
    if label_slots <= 36:
        return 0.18
    if label_slots <= 90:
        return 0.13
    return 0.085


def _curved_edge_points(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    dimensions: Literal[2, 3],
    scale: float = 1.0,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    perpendicular = _bond_perpendicular_unoriented(delta, dimensions)
    ref_len = _CURVE_NEAR_PAIR_REF * scale
    effective_chord = float(math.hypot(distance, ref_len))
    offset = (
        (offset_index - (edge_count - 1) / 2.0) * _CURVE_OFFSET_FACTOR * scale * effective_chord
    )
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


def _plot_contraction_index_captions(
    *,
    plotter: _PlotAdapter,
    curve: np.ndarray,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    left_id: int,
    right_id: int,
    config: PlotConfig,
    p: _DrawScaleParams,
    dimensions: Literal[2, 3],
    ax: Any,
    scale: float,
) -> None:
    ep_l, ep_r = _require_contraction_endpoints(edge)
    cap_l: str | None = _endpoint_index_caption(ep_l, edge, graph)
    cap_r: str | None = _endpoint_index_caption(ep_r, edge, graph)
    if not cap_l and not cap_r:
        return
    cvm = np.asarray(curve, dtype=float)
    L = _polyline_arc_length_total(cvm)
    L_half = 0.5 * L
    rim_arc = _edge_index_rim_arc_from_endpoint(r_global=float(p.r), half_polyline_length=L_half)
    s_slot = 0.5 * (rim_arc + L_half)
    Q_l, t_l = _point_tangent_along_polyline_from_start(cvm, s_slot)
    Q_r, t_r = _point_tangent_along_polyline_from_end(cvm, s_slot)
    bond_start = np.asarray(positions[left_id], dtype=float)
    bond_end = np.asarray(positions[right_id], dtype=float)
    bond_start_2d = np.asarray(bond_start[:2], dtype=float)
    bond_end_2d = np.asarray(bond_end[:2], dtype=float)
    _cap_pairs: tuple[
        tuple[_EdgeEndpoint, str | None, np.ndarray, np.ndarray, Literal["left", "right"]],
        tuple[_EdgeEndpoint, str | None, np.ndarray, np.ndarray, Literal["left", "right"]],
    ] = (
        (ep_l, cap_l, Q_l, t_l, "left"),
        (ep_r, cap_r, Q_r, t_r, "right"),
    )
    peer_caps: tuple[str, ...] = tuple(
        c for c in (cap_l, cap_r) if c and format_tensor_node_label(c).strip()
    )
    peer_for_width: tuple[str, ...] | None = peer_caps if len(peer_caps) > 1 else None
    for _ep, cap, Q, t_fwd, text_ep in _cap_pairs:
        if not cap:
            continue
        fs = _edge_index_fontsize_for_bond(
            cap,
            bond_start=bond_start,
            bond_end=bond_end,
            ax=ax,
            dimensions=dimensions,
            is_physical=False,
            peer_captions_for_width=peer_for_width,
        )
        tk = _edge_index_text_kwargs(
            config,
            fontsize=fs,
            stub_kind="bond",
            bbox_pad=p.index_bbox_pad,
        )
        if dimensions == 2:
            t_curve_2d = np.asarray(t_fwd[:2], dtype=float)
            t_blend = _blend_bond_tangent_with_chord_2d(
                t_curve_2d,
                bond_start_2d,
                bond_end_2d,
            )
            ax2 = cast(Axes, ax)
            pos2, align_kw = _contraction_edge_index_label_2d_placement(
                Q=Q,
                t_geom_2d=t_curve_2d,
                t_align_2d=t_blend,
                text_ep=text_ep,
                p=p,
                ax=ax2,
                scale=scale,
                fontsize_pt=float(fs),
            )
            tk = {**tk, **align_kw}
            plotter.plot_text(
                pos2,
                format_tensor_node_label(cap),
                **tk,
            )
            continue
        t_blend_3d = _blend_bond_tangent_with_chord_3d(
            np.asarray(t_fwd, dtype=float).reshape(3),
            bond_start,
            bond_end,
        )
        pos3, align_kw = _contraction_edge_index_label_3d_placement(
            Q=np.asarray(Q, dtype=float),
            t_geom_3d=np.asarray(t_fwd, dtype=float).reshape(3),
            t_align_3d=t_blend_3d,
            text_ep=text_ep,
            p=p,
            ax=ax,
            scale=scale,
            fontsize_pt=float(fs),
        )
        tk = {**tk, **align_kw}
        plotter.plot_text(
            pos3,
            format_tensor_node_label(cap),
            **tk,
        )


def _draw_dangling_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
) -> None:
    endpoint = edge.endpoints[0]
    direction = directions[(endpoint.node_id, endpoint.axis_index)]
    center = positions[endpoint.node_id]
    # 2D: Circle radius is in data units, so the stub starts on the rim.
    # 3D: scatter marker size is in points^2; zoom changes apparent size in data
    # units while p.r does not, so rim-based starts look detached. Match bonds
    # (center–center) by anchoring at the node center and extending r + stub.
    if dimensions == 3:
        start = center
        end = center + direction * (p.r + p.stub)
    else:
        start = center + direction * p.r
        end = start + direction * p.stub
    plotter.plot_line(start, end, color=config.dangling_edge_color, linewidth=p.lw, zorder=2)
    if show_index_labels and edge.label:
        hover_ht = getattr(plotter, "_hover_edge_targets", None)
        if config.hover_labels and hover_ht is not None:
            cap = _dangling_hover_label_text(edge)
            if cap:
                if dimensions == 2:
                    stub_xy = np.stack(
                        [
                            np.asarray(start[:2], dtype=float),
                            np.asarray(end[:2], dtype=float),
                        ],
                        axis=0,
                    )
                    hover_ht.append((stub_xy, cap))
                else:
                    s3 = np.zeros(3, dtype=float)
                    e3 = np.zeros(3, dtype=float)
                    sa = np.asarray(start, dtype=float).reshape(-1)
                    ea = np.asarray(end, dtype=float).reshape(-1)
                    s3[: min(3, sa.size)] = sa[: min(3, sa.size)]
                    e3[: min(3, ea.size)] = ea[: min(3, ea.size)]
                    stub_xyz = np.stack([s3, e3], axis=0)
                    hover_ht.append((stub_xyz, cap))
            return
        raw_lbl = edge.label
        fs_d = _edge_index_fontsize_for_bond(
            raw_lbl,
            bond_start=start,
            bond_end=end,
            ax=ax,
            dimensions=dimensions,
            is_physical=True,
        )
        dk = _edge_index_text_kwargs(
            config,
            fontsize=fs_d,
            stub_kind="dangling",
            bbox_pad=p.index_bbox_pad,
        )
        stub_seg = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)], axis=0)
        Ls = _polyline_arc_length_total(stub_seg)
        if dimensions == 2:
            ax2 = cast(Axes, ax)
            start_2 = np.asarray(start[:2], dtype=float)
            end_2 = np.asarray(end[:2], dtype=float)
            # Dangling == open leg: hug the **tip** (see ``_PHYS_DANGLING_2D_FRAC_FROM_TIP``).
            s_from_tip = float(_PHYS_DANGLING_2D_FRAC_FROM_TIP) * Ls
            Qp, t_stub = _point_tangent_along_polyline_from_end(stub_seg, s_from_tip)
            t_stub_2 = np.asarray(t_stub[:2], dtype=float)
            t_blend = _blend_bond_tangent_with_chord_2d(t_stub_2, start_2, end_2)
            label_pos, align_d = _contraction_edge_index_label_2d_placement(
                Q=Qp,
                t_geom_2d=t_stub_2,
                t_align_2d=t_blend,
                text_ep="left",
                p=p,
                ax=ax2,
                scale=scale,
                fontsize_pt=float(fs_d),
            )
            dk = {**dk, **align_d}
        else:
            s_from_tip = float(_PHYS_DANGLING_2D_FRAC_FROM_TIP) * Ls
            Qp, t_stub = _point_tangent_along_polyline_from_end(stub_seg, s_from_tip)
            start_3 = np.asarray(start, dtype=float).reshape(3)
            end_3 = np.asarray(end, dtype=float).reshape(3)
            t_blend_3d = _blend_bond_tangent_with_chord_3d(
                np.asarray(t_stub, dtype=float).reshape(3),
                start_3,
                end_3,
            )
            label_pos, align_d = _contraction_edge_index_label_3d_placement(
                Q=Qp,
                t_geom_3d=np.asarray(t_stub, dtype=float).reshape(3),
                t_align_3d=t_blend_3d,
                text_ep="left",
                p=p,
                ax=ax,
                scale=scale,
                fontsize_pt=float(fs_d),
            )
            dk = {**dk, **align_d}
        plotter.plot_text(
            label_pos,
            format_tensor_node_label(raw_lbl),
            **dk,
        )


def _draw_self_loop_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
) -> None:
    endpoint_a, endpoint_b = _require_self_endpoints(edge)
    direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
    direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
    orientation = direction_a + direction_b
    if np.linalg.norm(orientation) < 1e-6:
        orientation = (
            np.array([1.0, 0.0, 0.0], dtype=float)
            if dimensions == 3
            else np.array([1.0, 0.0], dtype=float)
        )
    orientation = orientation / np.linalg.norm(orientation)
    center_pt = positions[endpoint_a.node_id] + orientation * (p.r + p.loop_r)
    if dimensions == 2:
        normal = _perpendicular_2d(orientation)
        curve = _ellipse_points(
            center_pt,
            orientation,
            normal,
            width=p.ellipse_w,
            height=p.ellipse_h,
        )
        label_offset_dir = normal * p.ellipse_w
    else:
        normal = _orthogonal_unit(orientation)
        binormal = np.cross(orientation, normal)
        binormal = binormal / np.linalg.norm(binormal)
        curve = _ellipse_points_3d(
            center_pt,
            normal,
            binormal,
            width=p.ellipse_w,
            height=p.ellipse_h,
        )
        label_offset_dir = binormal * p.ellipse_w
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=2)
    if show_index_labels:
        hover_ht = getattr(plotter, "_hover_edge_targets", None)
        if config.hover_labels and hover_ht is not None:
            cap = _self_loop_hover_label_text(edge, graph)
            if cap:
                if dimensions == 2:
                    hover_ht.append((np.asarray(curve[:, :2], dtype=float, order="C"), cap))
                else:
                    hover_ht.append((np.asarray(curve[:, :3], dtype=float, order="C"), cap))
            return
        ca = _endpoint_index_caption(endpoint_a, edge, graph)
        cb = _endpoint_index_caption(endpoint_b, edge, graph)
        n = int(curve.shape[0])
        node_c = positions[endpoint_a.node_id]
        cl = _node_label_clearance(p)
        ia = _curve_index_outside_disk(curve, node_c, cl, from_start=True)
        ib = _curve_index_outside_disk(curve, node_c, cl, from_start=False)
        min_sep = max(2, n // 8)
        if abs(ia - ib) < min_sep:
            ib = min(n - 2, ia + min_sep)
        if ib <= ia:
            ia = max(1, ib - min_sep)
        dir_norm = float(np.linalg.norm(label_offset_dir))
        dir_unit = label_offset_dir / max(dir_norm, 1e-9)
        loop_span_a = np.asarray(curve[ia], dtype=float)
        loop_span_b = np.asarray(curve[ib], dtype=float)
        sub_loop = np.asarray(curve[ia : ib + 1], dtype=float)
        L_loop = _polyline_arc_length_total(sub_loop)
        along_loop = float(_EDGE_INDEX_LABEL_ALONG_FRAC) * L_loop
        Q_a, t_a = _point_tangent_along_polyline_from_start(sub_loop, along_loop)
        Q_b, t_b = _point_tangent_along_polyline_from_end(sub_loop, along_loop)
        peer_loop_caps: tuple[str, ...] = tuple(
            c for c in (ca, cb) if c and format_tensor_node_label(c).strip()
        )
        peer_loop_width: tuple[str, ...] | None = (
            peer_loop_caps if len(peer_loop_caps) > 1 else None
        )
        perp3 = dir_unit if dimensions == 3 else None
        if ca:
            off_a_mag = _bond_index_label_perp_offset(
                ca,
                p=p,
                scale=scale,
                dimensions=dimensions,
                ax=ax,
                anchor=Q_a,
                world_perp_dir=perp3,
            )
            off_a = dir_unit * off_a_mag
        else:
            off_a = dir_unit * 0.0
        if cb:
            off_b_mag = _bond_index_label_perp_offset(
                cb,
                p=p,
                scale=scale,
                dimensions=dimensions,
                ax=ax,
                anchor=Q_b,
                world_perp_dir=perp3,
            )
            off_b = -dir_unit * off_b_mag * 0.88
        else:
            off_b = -dir_unit * 0.0
        if ca:
            fs_a = _edge_index_fontsize_for_bond(
                ca,
                bond_start=loop_span_a,
                bond_end=loop_span_b,
                ax=ax,
                dimensions=dimensions,
                is_physical=False,
                peer_captions_for_width=peer_loop_width,
            )
            tk_a = _edge_index_text_kwargs(config, fontsize=fs_a, bbox_pad=p.index_bbox_pad)
            tk_a = {
                **tk_a,
                **_edge_index_along_bond_text_kw(
                    endpoint="left", tangent=t_a, ax=ax, dimensions=dimensions
                ),
            }
            plotter.plot_text(
                np.asarray(Q_a, dtype=float) + off_a,
                format_tensor_node_label(ca),
                **tk_a,
            )
        if cb:
            fs_b = _edge_index_fontsize_for_bond(
                cb,
                bond_start=loop_span_a,
                bond_end=loop_span_b,
                ax=ax,
                dimensions=dimensions,
                is_physical=False,
                peer_captions_for_width=peer_loop_width,
            )
            tk_b = _edge_index_text_kwargs(config, fontsize=fs_b, bbox_pad=p.index_bbox_pad)
            tk_b = {
                **tk_b,
                **_edge_index_along_bond_text_kw(
                    endpoint="right", tangent=t_b, ax=ax, dimensions=dimensions
                ),
            }
            plotter.plot_text(
                np.asarray(Q_b, dtype=float) + off_b,
                format_tensor_node_label(cb),
                **tk_b,
            )


def _draw_contraction_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
) -> None:
    left_id, right_id = edge.node_ids
    offset_index, edge_count = contraction_groups.offsets[id(edge)]
    start_base = positions[left_id]
    end_base = positions[right_id]
    # 2D: bonds run through the disk (center–center); disks are drawn on top (higher zorder).
    # 3D: unchanged (already center–center in layout space).
    curve = _curved_edge_points(
        start=start_base,
        end=end_base,
        offset_index=offset_index,
        edge_count=edge_count,
        dimensions=dimensions,
        scale=scale,
    )
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=1)
    if show_index_labels:
        hover_ht = getattr(plotter, "_hover_edge_targets", None)
        if config.hover_labels and hover_ht is not None:
            cap = _contraction_hover_label_text(edge, graph)
            if cap:
                if dimensions == 2:
                    hover_ht.append((np.asarray(curve[:, :2], dtype=float, order="C"), cap))
                else:
                    hover_ht.append((np.asarray(curve[:, :3], dtype=float, order="C"), cap))
        else:
            _plot_contraction_index_captions(
                plotter=plotter,
                curve=curve,
                edge=edge,
                graph=graph,
                positions=positions,
                left_id=left_id,
                right_id=right_id,
                config=config,
                p=p,
                dimensions=dimensions,
                ax=ax,
                scale=scale,
            )


def _draw_edges(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
) -> None:
    for edge in graph.edges:
        if edge.kind == "dangling":
            _draw_dangling_edge(
                plotter=plotter,
                edge=edge,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=dimensions,
                p=p,
                ax=ax,
                scale=scale,
            )
        elif edge.kind == "self":
            _draw_self_loop_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=dimensions,
                p=p,
                ax=ax,
                scale=scale,
            )
        else:
            _draw_contraction_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                contraction_groups=contraction_groups,
                show_index_labels=show_index_labels,
                config=config,
                scale=scale,
                dimensions=dimensions,
                p=p,
                ax=ax,
            )


def _display_disk_radius_px_2d(ax: Axes, center: np.ndarray, r_data: float) -> float:
    """Pixel radius for horizontal *r_data* at *center* (equal-aspect 2D)."""
    c = np.asarray(center[:2], dtype=float)
    row0 = c.reshape(1, -1)
    row1 = (c + np.array([float(r_data), 0.0], dtype=float)).reshape(1, -1)
    t0 = ax.transData.transform(row0)[0]
    t1 = ax.transData.transform(row1)[0]
    return float(np.hypot(float(t0[0] - t1[0]), float(t0[1] - t1[1])))


def _display_disk_radius_px_3d(ax: Any, center: np.ndarray, r_data: float) -> float:
    """Conservative screen radius for data-space sphere *r_data* (current 3D view)."""
    c = np.asarray(center, dtype=float)
    r = float(r_data)
    M = ax.get_proj()
    xs0, ys0, _zs0 = proj3d.proj_transform(c[0], c[1], c[2], M)
    pt_center = ax.transData.transform((xs0, ys0))
    md = math.inf
    for ex, ey, ez in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        p = c + r * np.array([ex, ey, ez], dtype=float)
        xs, ys, _zs = proj3d.proj_transform(p[0], p[1], p[2], M)
        pt = ax.transData.transform((xs, ys))
        d = float(np.hypot(pt[0] - pt_center[0], pt[1] - pt_center[1]))
        md = min(md, d)
    if not math.isfinite(md):
        return 0.0
    return float(md)


def _tensor_disk_radius_px(
    ax: Any,
    anchor: np.ndarray,
    p: _DrawScaleParams,
    dimensions: Literal[2, 3],
) -> float:
    if dimensions == 2:
        return _display_disk_radius_px_2d(cast(Axes, ax), anchor, p.r)
    return _display_disk_radius_px_3d(ax, anchor, p.r)


@functools.lru_cache(maxsize=1024)
def _textpath_diagonal_points_ref10(text: str) -> float:
    """Diagonal of TextPath at ref fontsize=10pt in points (path units × calibration factor)."""
    fp = FontProperties(size=10.0)
    tp = TextPath((0.0, 0.0), text, prop=fp)
    ex = tp.get_extents()
    return float(math.hypot(float(ex.width), float(ex.height)) * _TEXT_RENDER_DIAGONAL_FACTOR)


def _tensor_label_fontsize_to_fit(
    *,
    text: str,
    cap_pt: float,
    pixel_radius: float,
    fig: Figure,
) -> float:
    """First-pass font size from TextPath; refined in `_refit_tensor_labels_to_disks`."""
    if not text.strip():
        return float(max(3.0, cap_pt))
    ref = 10.0
    diag_pts = _textpath_diagonal_points_ref10(text)
    if diag_pts <= 1e-12:
        return float(max(3.0, cap_pt))
    diag_px_ref = diag_pts * float(fig.dpi) / 72.0
    allow = 2.0 * max(float(pixel_radius), 1e-9) * _TENSOR_LABEL_INSIDE_FILL
    max_fs = ref * allow / diag_px_ref
    lo, hi = 3.0, max(3.0, float(cap_pt))
    return float(max(lo, min(hi, max_fs)))


def _draw_nodes(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    config: PlotConfig,
    p: _DrawScaleParams,
) -> np.ndarray:
    visible_node_ids = [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]
    if visible_node_ids:
        coords = np.stack(
            [np.asarray(positions[node_id], dtype=float) for node_id in visible_node_ids]
        )
        deg1 = _visible_degree_one_mask(graph, visible_node_ids)
        plotter.draw_tensor_nodes(coords, config=config, p=p, degree_one_mask=deg1)
        return coords
    return _stack_visible_tensor_coords(graph, positions)


def _draw_labels(
    *,
    plotter: _PlotAdapter,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    show_tensor_labels: bool,
    config: PlotConfig,
    p: _DrawScaleParams,
    dimensions: Literal[2, 3],
    tensor_hover_by_node: dict[int, tuple[str, float]] | None = None,
) -> None:
    if show_tensor_labels:
        fig = ax.figure
        for node_id, node in graph.nodes.items():
            if node.is_virtual:
                continue
            pos = positions[node_id]
            r_px = _tensor_disk_radius_px(ax, pos, p, dimensions)
            display_name = format_tensor_node_label(node.name)
            fs = _tensor_label_fontsize_to_fit(
                text=display_name,
                cap_pt=p.font_tensor_label_max,
                pixel_radius=r_px,
                fig=fig,
            )
            if dimensions == 3:
                cap_tensor = float(p.font_tensor_label_max) * _LABEL_FONT_3D_SCALE
                fs = min(float(fs) * _LABEL_FONT_3D_SCALE, cap_tensor)
            if tensor_hover_by_node is not None:
                tensor_hover_by_node[node_id] = (display_name, float(fs))
                continue
            plotter.plot_text(
                pos,
                display_name,
                color=config.tensor_label_color,
                ha="center",
                va="center",
                fontsize=fs,
                zorder=_ZORDER_TENSOR_NAME,
                gid=_TENSOR_LABEL_GID,
            )


def _tensor_label_data_anchor(t: Any, *, dimensions: Literal[2, 3]) -> np.ndarray:
    """World coordinates of the tensor name anchor (disk center)."""
    if dimensions == 3 and hasattr(t, "get_position_3d"):
        return np.asarray(t.get_position_3d(), dtype=float)
    x, y = t.get_position()
    if dimensions == 2:
        return np.array([float(x), float(y)], dtype=float)
    z = float(getattr(t, "_z", 0.0))
    return np.array([float(x), float(y), z], dtype=float)


def _refit_tensor_labels_to_disks(
    *,
    ax: Any,
    p: _DrawScaleParams,
    dimensions: Literal[2, 3],
) -> None:
    """Shrink tensor tags using true rendered bboxes so names stay inside disks."""
    fig = ax.figure
    labels = [t for t in ax.texts if t.get_gid() == _TENSOR_LABEL_GID]
    if not labels:
        return
    fs_cap = float(p.font_tensor_label_max) * (_LABEL_FONT_3D_SCALE if dimensions == 3 else 1.0)
    n_ts = len(labels)
    max_passes = 5 if n_ts <= 35 else (3 if n_ts <= 75 else 2)
    for _ in range(max_passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tightened = False
        for t in labels:
            anchor = _tensor_label_data_anchor(t, dimensions=dimensions)
            r_px = _tensor_disk_radius_px(ax, anchor, p, dimensions)
            allow = 2.0 * max(r_px, 1e-9) * _TENSOR_LABEL_INSIDE_FILL
            bb = t.get_window_extent(renderer=renderer)
            diag = float(math.hypot(float(bb.width), float(bb.height)))
            if diag <= allow + 1.5:
                continue
            fs = float(t.get_fontsize())
            new_fs = max(3.0, min(fs_cap, fs * (allow / max(diag, 1e-9)) * 0.97))
            if new_fs < fs - 0.05:
                t.set_fontsize(new_fs)
                tightened = True
        if not tightened:
            break


@dataclass(frozen=True)
class _DrawScaleParams:
    """Resolved scale-dependent parameters for drawing."""

    r: float
    stub: float
    loop_r: float
    lw: float
    font_tensor_label_max: float
    index_bbox_pad: float
    label_offset: float
    ellipse_w: float
    ellipse_h: float


def _draw_scale_params(
    config: PlotConfig,
    scale: float,
    *,
    fig: Figure,
    is_3d: bool,
    font_figure_scale: float = 1.0,
    label_slots: int = 1,
) -> _DrawScaleParams:
    """Compute scale-dependent drawing parameters from config."""
    fs = font_figure_scale
    tensor_fs = _figure_base_size_scale(fig)
    bbox_pad = _index_label_bbox_pad(max(1, label_slots))
    r = (
        config.node_radius if config.node_radius is not None else PlotConfig.DEFAULT_NODE_RADIUS
    ) * scale
    stub = (
        config.stub_length if config.stub_length is not None else PlotConfig.DEFAULT_STUB_LENGTH
    ) * scale
    loop_r = (
        config.self_loop_radius
        if config.self_loop_radius is not None
        else PlotConfig.DEFAULT_SELF_LOOP_RADIUS
    ) * scale
    lw_default = PlotConfig.DEFAULT_LINE_WIDTH_3D if is_3d else PlotConfig.DEFAULT_LINE_WIDTH_2D
    lw_attr = config.line_width_3d if is_3d else config.line_width_2d
    lw = (lw_attr if lw_attr is not None else lw_default) * scale

    # Tensor names: cap from layout + figure *size* only — not `fs` (bond-label crowding).
    font_tensor_label_max = float(max(3.0, min(15.0, 10.0 * scale * tensor_fs)))
    return _DrawScaleParams(
        r=r,
        stub=stub,
        loop_r=loop_r,
        lw=lw,
        font_tensor_label_max=font_tensor_label_max,
        index_bbox_pad=bbox_pad,
        label_offset=0.08 * scale * float(np.clip(0.82 + 0.22 * fs, 0.75, 1.2)),
        ellipse_w=0.16 * scale,
        ellipse_h=0.12 * scale,
    )


_ZOOM_FONT_CLAMP: tuple[float, float] = (0.28, 5.5)


def _on_2d_limits_changed(ax: Axes) -> None:
    state = getattr(ax, "_tensor_network_viz_zoom_fonts", None)
    if state is None:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(float(x1 - x0), float(y1 - y0), 1e-9)
    factor = float(state["ref_span"] / span)
    lo, hi = _ZOOM_FONT_CLAMP
    factor = float(np.clip(factor, lo, hi))
    for text, base_fs in state["sizes"].items():
        if text.figure is None:
            continue
        text.set_fontsize(max(3.0, base_fs * factor))


def _register_2d_zoom_font_scaling(ax: Axes) -> None:
    old_cids: list[Any] = getattr(ax, "_tensor_network_viz_zoom_cids", [])
    for cid in old_cids:
        with suppress(ValueError, KeyError):
            ax.callbacks.disconnect(cid)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ref_span = max(float(x1 - x0), float(y1 - y0), 1e-9)
    sizes = {t: float(t.get_fontsize()) for t in ax.texts}
    ax._tensor_network_viz_zoom_fonts = {
        "ref_span": ref_span,
        "sizes": sizes,
    }

    def _cb(_: object) -> None:
        _on_2d_limits_changed(ax)

    cx = ax.callbacks.connect("xlim_changed", _cb)
    cy = ax.callbacks.connect("ylim_changed", _cb)
    ax._tensor_network_viz_zoom_cids = [cx, cy]


def _draw_graph(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float = 1.0,
) -> None:
    _disconnect_tensor_network_hover(ax.figure)
    ax.cla()
    contraction_groups = _group_contractions(graph)
    label_slots = _estimate_drawn_label_count(
        graph,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
    )
    font_figure_scale = _figure_relative_font_scale(ax.figure, label_slots)
    params = _draw_scale_params(
        config,
        scale,
        fig=ax.figure,
        is_3d=dimensions == 3,
        font_figure_scale=font_figure_scale,
        label_slots=max(1, label_slots),
    )
    hover_edge_list: list[tuple[np.ndarray, str]] | None = None
    tensor_hover_map: dict[int, tuple[str, float]] | None = None
    if config.hover_labels:
        if show_index_labels:
            hover_edge_list = []
        if show_tensor_labels:
            tensor_hover_map = {}
    plotter = _make_plotter(
        ax,
        dimensions=dimensions,
        hover_edge_targets=hover_edge_list,
    )
    pre_coords = _stack_visible_tensor_coords(graph, positions)
    view_margin = _view_outset_margin_data_units(
        graph, positions, params, scale, contraction_groups
    )
    _apply_axis_limits_with_outset(ax, pre_coords, view_margin=view_margin, dimensions=dimensions)

    _draw_edges(
        plotter=plotter,
        graph=graph,
        positions=positions,
        directions=directions,
        contraction_groups=contraction_groups,
        show_index_labels=show_index_labels,
        config=config,
        scale=scale,
        dimensions=dimensions,
        p=params,
        ax=ax,
    )
    if dimensions == 2:
        flush = getattr(plotter, "flush_edge_collections", None)
        if callable(flush):
            flush()
    coords = _draw_nodes(
        plotter=plotter,
        graph=graph,
        positions=positions,
        config=config,
        p=params,
    )
    plotter.style_axes(coords, view_margin=view_margin)
    _draw_labels(
        plotter=plotter,
        ax=ax,
        graph=graph,
        positions=positions,
        show_tensor_labels=show_tensor_labels,
        config=config,
        p=params,
        dimensions=dimensions,
        tensor_hover_by_node=tensor_hover_map,
    )
    if config.refine_tensor_labels:
        _refit_tensor_labels_to_disks(ax=ax, p=params, dimensions=dimensions)
    if dimensions == 2:
        _register_2d_zoom_font_scaling(cast(Axes, ax))
    if config.hover_labels and (show_tensor_labels or show_index_labels):
        vis_ids = [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]
        if dimensions == 2:
            node_coll = getattr(plotter, "_node_disk_collection", None)
            _register_2d_hover_labels(
                cast(Axes, ax),
                node_patch_coll=node_coll if show_tensor_labels else None,
                visible_node_ids=vis_ids,
                tensor_hover=tensor_hover_map or {},
                edge_hover=list(hover_edge_list or ()),
                line_width_px_hint=float(params.lw),
            )
        else:
            _register_3d_hover_labels(
                ax,
                ax.figure,
                positions=positions,
                visible_node_ids=vis_ids,
                tensor_hover=tensor_hover_map or {},
                edge_hover=list(hover_edge_list or ()),
                line_width_px_hint=float(params.lw),
                p=params,
            )
