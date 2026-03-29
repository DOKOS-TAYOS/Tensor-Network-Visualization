from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import proj3d

from .._label_format import format_tensor_node_label
from ..contractions import _ContractionGroups
from ..graph import (
    _GraphData,
)
from ..layout import (
    NodePositions,
)
from .constants import *
from .fonts_and_scale import _DrawScaleParams, _textpath_width_pts
from .vectors import _perpendicular_3d


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


__all__ = [
    "_apply_axis_limits_with_outset",
    "_apply_edge_line_style",
    "_apply_text_no_clip",
    "_blend_bond_tangent_with_chord_2d",
    "_blend_bond_tangent_with_chord_3d",
    "_bond_endpoints_xyz3",
    "_bond_index_label_perp_offset",
    "_bond_reference_span_px_for_font",
    "_contraction_edge_index_label_2d_placement",
    "_contraction_edge_index_label_3d_placement",
    "_edge_index_along_bond_text_kw",
    "_edge_index_font_em_data_2d",
    "_edge_index_fontsize_for_bond",
    "_edge_index_label_axis_tie_vertical_2d",
    "_edge_index_label_is_vertical_axis_2d",
    "_edge_index_label_span_frac",
    "_edge_index_rim_arc_from_endpoint",
    "_edge_index_text_kw_tangent_stroke_align",
    "_line_halfwidth_data_2d",
    "_max_perpendicular_bond_curve_offset",
    "_nominal_figure_px_per_data_unit_3d",
    "_point_tangent_along_polyline_from_end",
    "_point_tangent_along_polyline_from_start",
    "_polyline_arc_length_total",
    "_self_loop_spatial_extent",
    "_stack_visible_tensor_coords",
    "_stroke_index_normal_screen_unit_2d",
    "_stroke_perp_distance_data_units_3d",
    "_tangent_screen_angle_deg",
    "_upright_screen_text_rotation_deg_raw",
    "_view_outset_margin_data_units",
]
