from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes

from ...config import PlotConfig
from .._label_format import format_tensor_node_label
from ..contractions import _ContractionGroups
from ..curves import (
    _ellipse_points,
    _ellipse_points_3d,
    _quadratic_curve,
    _require_self_endpoints,
)
from ..graph import (
    _EdgeData,
    _EdgeEndpoint,
    _endpoint_index_caption,
    _GraphData,
    _require_contraction_endpoints,
)
from ..layout import (
    AxisDirections,
    NodePositions,
    _orthogonal_unit,
)
from .constants import *
from .fonts_and_scale import _DrawScaleParams
from .labels_misc import (
    _contraction_hover_label_text,
    _curve_index_outside_disk,
    _dangling_hover_label_text,
    _edge_index_text_kwargs,
    _node_label_clearance,
    _self_loop_hover_label_text,
)
from .plotter import _PlotAdapter
from .vectors import _bond_perpendicular_unoriented, _perpendicular_2d
from .viewport_geometry import (
    _blend_bond_tangent_with_chord_2d,
    _blend_bond_tangent_with_chord_3d,
    _bond_index_label_perp_offset,
    _contraction_edge_index_label_2d_placement,
    _contraction_edge_index_label_3d_placement,
    _edge_index_along_bond_text_kw,
    _edge_index_fontsize_for_bond,
    _edge_index_rim_arc_from_endpoint,
    _point_tangent_along_polyline_from_end,
    _point_tangent_along_polyline_from_start,
    _polyline_arc_length_total,
)

_ZORDER_FLAT_BOND_LINE: float = 1.0
_ZORDER_FLAT_DANGLE_SELF_LINE: float = 2.0


def _edge_stable_bond_sort_key(edge: _EdgeData) -> tuple[str, tuple[int, ...], int]:
    return (edge.kind, edge.node_ids, id(edge))


def _edge_stable_dangling_sort_key(edge: _EdgeData) -> tuple[tuple[int, ...], int]:
    return (edge.node_ids, id(edge))


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
    zorder_label: float | None = None,
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
            zorder=zorder_label,
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
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    scale: float,
    zorder_line: float | None = None,
    zorder_label: float | None = None,
) -> None:
    endpoint = edge.endpoints[0]
    direction = directions[(endpoint.node_id, endpoint.axis_index)]
    center = positions[endpoint.node_id]
    # 2D tensors: stub starts on the circle rim (radius in data units).
    # 3D and 2D virtual (hyperedge) hubs: anchor at the node center like bonds (center–center).
    use_center_anchor = dimensions == 3 or graph.nodes[endpoint.node_id].is_virtual
    if use_center_anchor:
        start = center
        end = center + direction * (p.r + p.stub)
    else:
        start = center + direction * p.r
        end = start + direction * p.stub
    zl = float(_ZORDER_FLAT_DANGLE_SELF_LINE) if zorder_line is None else float(zorder_line)
    plotter.plot_line(start, end, color=config.dangling_edge_color, linewidth=p.lw, zorder=zl)
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
            zorder=zorder_label,
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
    zorder_line: float | None = None,
    zorder_label: float | None = None,
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
    z_loop = float(_ZORDER_FLAT_DANGLE_SELF_LINE) if zorder_line is None else float(zorder_line)
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=z_loop)
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
            tk_a = _edge_index_text_kwargs(
                config, fontsize=fs_a, bbox_pad=p.index_bbox_pad, zorder=zorder_label
            )
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
            tk_b = _edge_index_text_kwargs(
                config, fontsize=fs_b, bbox_pad=p.index_bbox_pad, zorder=zorder_label
            )
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
    zorder_line: float | None = None,
    zorder_label: float | None = None,
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
    zc = float(_ZORDER_FLAT_BOND_LINE) if zorder_line is None else float(zorder_line)
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=zc)
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
                zorder_label=zorder_label,
            )


def _draw_edges_2d_layered(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    visible_order: list[int],
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    p: _DrawScaleParams,
    ax: Any,
) -> None:
    """Enqueue 2D edges with per-node z-order; caller must ``flush_edge_collections`` once."""
    by_node: dict[int, list[_EdgeData]] = defaultdict(list)
    for edge in graph.edges:
        for node_id in edge.node_ids:
            by_node[node_id].append(edge)
    drawn: set[int] = set()
    for i, nid in enumerate(visible_order):
        zb = _ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_BOND
        zd = _ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DANGLING
        zidx = _ZORDER_LAYER_BASE + i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_EDGE_INDEX
        incident = [e for e in by_node.get(nid, ()) if id(e) not in drawn]
        bonds = sorted(
            [e for e in incident if e.kind != "dangling"],
            key=_edge_stable_bond_sort_key,
        )
        dangles = sorted(
            [e for e in incident if e.kind == "dangling"],
            key=_edge_stable_dangling_sort_key,
        )
        for edge in bonds:
            if edge.kind == "self":
                _draw_self_loop_edge(
                    plotter=plotter,
                    edge=edge,
                    graph=graph,
                    positions=positions,
                    directions=directions,
                    show_index_labels=show_index_labels,
                    config=config,
                    dimensions=2,
                    p=p,
                    ax=ax,
                    scale=scale,
                    zorder_line=zb,
                    zorder_label=zidx,
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
                    dimensions=2,
                    p=p,
                    ax=ax,
                    zorder_line=zb,
                    zorder_label=zidx,
                )
            drawn.add(id(edge))
        for edge in dangles:
            _draw_dangling_edge(
                plotter=plotter,
                edge=edge,
                graph=graph,
                positions=positions,
                directions=directions,
                show_index_labels=show_index_labels,
                config=config,
                dimensions=2,
                p=p,
                ax=ax,
                scale=scale,
                zorder_line=zd,
                zorder_label=zidx,
            )
            drawn.add(id(edge))

    # Dangling legs on virtual (e.g. hyperedge hub) nodes never appear in *visible_order*, so they
    # were skipped above; draw them once using the top tensor layer z-order.
    if visible_order:
        last_i = len(visible_order) - 1
        z_orphan_line = _ZORDER_LAYER_BASE + last_i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_DANGLING
        z_orphan_idx = _ZORDER_LAYER_BASE + last_i * _ZORDER_LAYER_STRIDE + _ZORDER_LAYER_EDGE_INDEX
    else:
        z_orphan_line = _ZORDER_LAYER_BASE + _ZORDER_LAYER_DANGLING
        z_orphan_idx = _ZORDER_LAYER_BASE + _ZORDER_LAYER_EDGE_INDEX
    for edge in graph.edges:
        if edge.kind != "dangling" or id(edge) in drawn:
            continue
        _draw_dangling_edge(
            plotter=plotter,
            edge=edge,
            graph=graph,
            positions=positions,
            directions=directions,
            show_index_labels=show_index_labels,
            config=config,
            dimensions=2,
            p=p,
            ax=ax,
            scale=scale,
            zorder_line=z_orphan_line,
            zorder_label=z_orphan_idx,
        )
        drawn.add(id(edge))


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


__all__ = [
    "_curved_edge_points",
    "_draw_contraction_edge",
    "_draw_dangling_edge",
    "_draw_edges",
    "_draw_edges_2d_layered",
    "_draw_self_loop_edge",
    "_plot_contraction_index_captions",
]
