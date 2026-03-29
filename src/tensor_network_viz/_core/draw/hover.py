from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from ..layout import NodePositions
from .constants import *
from .fonts_and_scale import _DrawScaleParams
from .pick_distance import (
    _min_sqdist_point_to_polyline_display,
    _min_sqdist_point_to_polyline_display_3d,
)
from .tensors import _tensor_disk_radius_px


def _disconnect_tensor_network_hover(fig: Figure) -> None:
    cid = getattr(fig, "_tensor_network_viz_hover_cid", None)
    if cid is not None:
        with suppress(ValueError, KeyError):
            fig.canvas.mpl_disconnect(int(cid))
        fig._tensor_network_viz_hover_cid = None


def _register_2d_hover_labels(
    ax: Axes,
    *,
    node_patch_coll: PatchCollection | Sequence[PatchCollection] | None,
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
            if isinstance(node_patch_coll, PatchCollection):
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
            else:
                for coll_idx, coll in enumerate(node_patch_coll):
                    hit, props = coll.contains(event)
                    if not hit:
                        continue
                    if coll_idx < len(visible_node_ids):
                        nid = visible_node_ids[coll_idx]
                        pair = tensor_hover.get(nid)
                        if pair:
                            label, fs_hint = pair[0], float(pair[1])
                    break

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


__all__ = [
    "_disconnect_tensor_network_hover",
    "_register_2d_hover_labels",
    "_register_3d_hover_labels",
]
