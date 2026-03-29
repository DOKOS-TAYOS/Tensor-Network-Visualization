from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ...config import PlotConfig
from .._label_format import format_tensor_node_label
from ..curves import (
    _require_self_endpoints,
)
from ..graph import (
    _EdgeData,
    _endpoint_index_caption,
    _GraphData,
    _require_contraction_endpoints,
)
from .constants import *
from .fonts_and_scale import _DrawScaleParams


def _node_label_clearance(p: _DrawScaleParams) -> float:
    """Minimum distance from a tensor center to an index label anchor (data units)."""
    return float(p.r * _NODE_LABEL_MARGIN_FACTOR + p.label_offset * 0.46)


def _edge_index_text_kwargs(
    config: PlotConfig,
    *,
    fontsize: float,
    stub_kind: Literal["bond", "dangling"] = "bond",
    bbox_pad: float = 0.18,
    zorder: float | None = None,
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
    z = float(_ZORDER_EDGE_INDEX_LABEL) if zorder is None else float(zorder)
    return {
        "color": config.label_color,
        "fontsize": fontsize,
        "zorder": z,
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


__all__ = [
    "_contraction_hover_label_text",
    "_curve_index_outside_disk",
    "_dangling_hover_label_text",
    "_edge_index_text_kwargs",
    "_estimate_drawn_label_count",
    "_node_label_clearance",
    "_self_loop_hover_label_text",
]
