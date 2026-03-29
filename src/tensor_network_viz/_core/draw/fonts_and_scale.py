from __future__ import annotations

import functools
import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ...config import PlotConfig
from .constants import *


def _figure_size_sqrt_ratio(fig: Figure) -> float:
    dpi = float(getattr(fig, "dpi", None) or 100.0)
    width_in, height_in = fig.get_size_inches()
    min_px = float(min(width_in, height_in) * dpi)
    return float(math.sqrt(min_px / _FIGURE_MIN_PX_REF))


_TEXTPATH_WIDTH_CACHE_MAX: int = 8192


@functools.lru_cache(maxsize=_TEXTPATH_WIDTH_CACHE_MAX)
def _textpath_width_pts_cached(text: str, fontsize_pt_key: float) -> float:
    """Cached TextPath width; *fontsize_pt_key* should be ``round(fontsize_pt, 6)``."""
    if not text.strip():
        return 0.0
    fp = FontProperties(size=float(fontsize_pt_key))
    tp = TextPath((0.0, 0.0), text, prop=fp)
    return float(tp.get_extents().width)


def _textpath_width_pts(text: str, *, fontsize_pt: float) -> float:
    """Horizontal advance of *text* from Matplotlib TextPath at *fontsize_pt* (points)."""
    key = round(float(fontsize_pt), 6)
    return float(_textpath_width_pts_cached(text, key))


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


__all__ = [
    "_DrawScaleParams",
    "_draw_scale_params",
    "_figure_base_size_scale",
    "_figure_relative_font_scale",
    "_figure_size_sqrt_ratio",
    "_index_label_bbox_pad",
    "_on_2d_limits_changed",
    "_register_2d_zoom_font_scaling",
    "_textpath_width_pts",
    "_textpath_width_pts_cached",
]
