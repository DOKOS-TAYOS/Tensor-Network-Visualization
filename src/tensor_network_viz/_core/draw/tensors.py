from __future__ import annotations

import functools
import math
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ...config import PlotConfig
from .._label_format import format_tensor_node_label
from ..graph import _GraphData
from ..layout import NodePositions
from .constants import *
from .disk_metrics import _tensor_disk_radius_px
from .fonts_and_scale import _DrawScaleParams
from .plotter import _PlotAdapter, _visible_degree_one_mask
from .viewport_geometry import _stack_visible_tensor_coords


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


def _visible_node_ids_in_graph_order(graph: _GraphData) -> list[int]:
    """Visible tensors in ``graph.nodes`` dict order (same draw / layout semantics as before)."""
    return [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]


def _draw_nodes(
    *,
    plotter: _PlotAdapter,
    graph: _GraphData,
    positions: NodePositions,
    config: PlotConfig,
    p: _DrawScaleParams,
) -> np.ndarray:
    visible_node_ids = _visible_node_ids_in_graph_order(graph)
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
    visible_draw_order: list[int] | None = None,
    tensor_label_zorder_by_node: dict[int, float] | None = None,
) -> None:
    if show_tensor_labels:
        fig = ax.figure
        ordered_ids: list[int]
        if visible_draw_order is not None:
            ordered_ids = list(visible_draw_order)
        else:
            ordered_ids = _visible_node_ids_in_graph_order(graph)
        for node_id in ordered_ids:
            node = graph.nodes.get(node_id)
            if node is None or node.is_virtual:
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
            if tensor_label_zorder_by_node is None:
                z_lbl = float(_ZORDER_TENSOR_NAME)
            else:
                z_lbl = float(tensor_label_zorder_by_node.get(node_id, _ZORDER_TENSOR_NAME))
            plotter.plot_text(
                pos,
                display_name,
                color=config.tensor_label_color,
                ha="center",
                va="center",
                fontsize=fs,
                zorder=z_lbl,
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


__all__ = [
    "_draw_labels",
    "_draw_nodes",
    "_refit_tensor_labels_to_disks",
    "_tensor_label_data_anchor",
    "_tensor_label_fontsize_to_fit",
    "_textpath_diagonal_points_ref10",
    "_visible_node_ids_in_graph_order",
]
