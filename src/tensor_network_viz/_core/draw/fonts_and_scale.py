from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextPath

from ..._matplotlib_state import (
    get_zoom_cids,
    get_zoom_font_state,
    set_zoom_cids,
    set_zoom_font_state,
)
from ...config import PlotConfig
from .constants import _FIGURE_MIN_PX_REF


def _figure_size_sqrt_ratio(fig: Figure) -> float:
    dpi = float(getattr(fig, "dpi", None) or 100.0)
    width_in, height_in = fig.get_size_inches()
    min_px = float(min(width_in, height_in) * dpi)
    return float(math.sqrt(min_px / _FIGURE_MIN_PX_REF))


_TEXTPATH_WIDTH_CACHE_MAX: int = 8192
_FAST_TEXT_METRICS_LABEL_THRESHOLD: int = 256


@functools.lru_cache(maxsize=_TEXTPATH_WIDTH_CACHE_MAX)
def _textpath_width_pts_cached(text: str, fontsize_pt_key: float) -> float:
    """Cached TextPath width; *fontsize_pt_key* should be ``round(fontsize_pt, 6)``."""
    if not text.strip():
        return 0.0
    fp = FontProperties(size=float(fontsize_pt_key))
    tp = TextPath((0.0, 0.0), text, prop=fp)
    return float(tp.get_extents().width)


@functools.lru_cache(maxsize=_TEXTPATH_WIDTH_CACHE_MAX)
def _textpath_width_pts_fast_cached(text: str, fontsize_pt_key: float) -> float:
    """Fast TextPath width approximation from raw vertices.

    ``TextPath.get_extents()`` walks Bezier extrema and becomes expensive for many unique labels.
    The vertex span is slightly less exact, but avoids that path for dense interactive views.
    """
    if not text.strip():
        return 0.0
    fp = FontProperties(size=float(fontsize_pt_key))
    tp = TextPath((0.0, 0.0), text, prop=fp)
    vertices = np.asarray(tp.vertices, dtype=float)
    if vertices.size == 0:
        return 0.0
    return float(vertices[:, 0].max() - vertices[:, 0].min())


def _textpath_width_pts_fast(text: str, *, fontsize_pt: float) -> float:
    """Fast horizontal span of *text* at *fontsize_pt* for dense label views."""
    key = round(float(fontsize_pt), 6)
    return float(_textpath_width_pts_fast_cached(text, key))


def _textpath_width_pts(
    text: str,
    *,
    fontsize_pt: float,
    fast_metrics: bool = False,
) -> float:
    """Horizontal advance of *text* from Matplotlib TextPath at *fontsize_pt* (points)."""
    if fast_metrics:
        return _textpath_width_pts_fast(text, fontsize_pt=fontsize_pt)
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
    fast_text_metrics: bool = False


def _draw_scale_params(
    config: PlotConfig,
    scale: float,
    *,
    fig: Figure,
    is_3d: bool,
    font_figure_scale: float = 1.0,
    label_slots: int = 1,
    fast_text_metrics: bool | None = None,
) -> _DrawScaleParams:
    """Compute scale-dependent drawing parameters from config."""
    fs = font_figure_scale
    tensor_fs = _figure_base_size_scale(fig)
    resolved_label_slots = max(1, int(label_slots))
    bbox_pad = _index_label_bbox_pad(resolved_label_slots)
    use_fast_text_metrics = (
        resolved_label_slots >= _FAST_TEXT_METRICS_LABEL_THRESHOLD
        if fast_text_metrics is None
        else bool(fast_text_metrics)
    )
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
        fast_text_metrics=use_fast_text_metrics,
    )


_ZOOM_FONT_CLAMP: tuple[float, float] = (0.28, 5.5)


def _current_2d_span(ax: Axes) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return max(float(x1 - x0), float(y1 - y0), 1e-9)


def _zoom_font_factor(*, ref_span: float, current_span: float) -> float:
    lo, hi = _ZOOM_FONT_CLAMP
    factor = float(ref_span) / max(float(current_span), 1e-9)
    return float(np.clip(factor, lo, hi))


def _text_font_sizes(ax: Axes) -> dict[Text, float]:
    return {text: float(text.get_fontsize()) for text in ax.texts if isinstance(text, Text)}


def _pruned_zoom_font_sizes(
    sizes: dict[Any, float],
    *,
    current_texts: tuple[Text, ...],
) -> dict[Text, float]:
    current_ids = {id(text) for text in current_texts}
    return {
        text: float(base_fontsize)
        for text, base_fontsize in sizes.items()
        if isinstance(text, Text) and id(text) in current_ids and text.figure is not None
    }


def _resolve_zoom_font_state(
    ax: Axes,
    *,
    preserve_ref_span: bool,
) -> tuple[float, dict[Text, float]]:
    current_span = _current_2d_span(ax)
    current_texts = tuple(text for text in ax.texts if isinstance(text, Text))
    current_sizes = _text_font_sizes(ax)
    if not preserve_ref_span:
        return current_span, current_sizes

    existing_state = get_zoom_font_state(ax)
    if existing_state is None:
        return current_span, current_sizes

    ref_span = float(existing_state["ref_span"])
    existing_sizes = _pruned_zoom_font_sizes(
        cast(dict[Any, float], existing_state["sizes"]),
        current_texts=current_texts,
    )
    resolved_sizes: dict[Text, float] = {}
    for text in current_texts:
        base_fontsize = existing_sizes.get(text)
        if base_fontsize is None:
            base_fontsize = float(current_sizes[text])
        resolved_sizes[text] = float(base_fontsize)
    return ref_span, resolved_sizes


def _zoom_callback_registered(ax: Axes, signal_name: str, cid: object) -> bool:
    callbacks = getattr(ax.callbacks, "callbacks", None)
    if not isinstance(callbacks, dict):
        return False
    signal_callbacks = callbacks.get(signal_name)
    return isinstance(signal_callbacks, dict) and cid in signal_callbacks


def _zoom_callbacks_connected(ax: Axes, cids: list[Any]) -> bool:
    if len(cids) != 2:
        return False
    return _zoom_callback_registered(ax, "xlim_changed", cids[0]) and _zoom_callback_registered(
        ax,
        "ylim_changed",
        cids[1],
    )


def _disconnect_zoom_callbacks(ax: Axes, cids: list[Any]) -> None:
    for cid in cids:
        try:
            ax.callbacks.disconnect(cid)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue


def _on_2d_limits_changed(ax: Axes) -> None:
    state = get_zoom_font_state(ax)
    if state is None:
        return
    factor = _zoom_font_factor(
        ref_span=float(state["ref_span"]),
        current_span=_current_2d_span(ax),
    )
    for text, base_fs in state["sizes"].items():
        if text.figure is None:
            continue
        text.set_fontsize(max(3.0, base_fs * factor))


def _register_2d_zoom_font_scaling(
    ax: Axes,
    *,
    preserve_ref_span: bool = False,
) -> None:
    ref_span, sizes = _resolve_zoom_font_state(ax, preserve_ref_span=preserve_ref_span)
    set_zoom_font_state(ax, ref_span=ref_span, sizes=sizes)

    old_cids = get_zoom_cids(ax)
    if _zoom_callbacks_connected(ax, old_cids):
        _on_2d_limits_changed(ax)
        return
    if old_cids:
        _disconnect_zoom_callbacks(ax, old_cids)

    def _cb(_: object) -> None:
        _on_2d_limits_changed(ax)

    cx = ax.callbacks.connect("xlim_changed", _cb)
    cy = ax.callbacks.connect("ylim_changed", _cb)
    set_zoom_cids(ax, [cx, cy])
    _on_2d_limits_changed(ax)


__all__ = [
    "_DrawScaleParams",
    "_draw_scale_params",
    "_figure_base_size_scale",
    "_figure_relative_font_scale",
    "_figure_size_sqrt_ratio",
    "_FAST_TEXT_METRICS_LABEL_THRESHOLD",
    "_index_label_bbox_pad",
    "_on_2d_limits_changed",
    "_register_2d_zoom_font_scaling",
    "_textpath_width_pts",
    "_textpath_width_pts_cached",
    "_textpath_width_pts_fast",
    "_textpath_width_pts_fast_cached",
]
