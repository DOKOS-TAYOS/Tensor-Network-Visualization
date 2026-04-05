from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PathCollection
from mpl_toolkits.mplot3d import proj3d

from ..._matplotlib_state import (
    clear_hover_annotation,
    clear_hover_cid,
    get_hover_annotation,
    get_hover_cid,
    set_hover_annotation,
    set_hover_cid,
)
from ..._typing import FigureLike, root_figure
from ..layout import NodePositions
from .constants import _HOVER_EDGE_PICK_RADIUS_PX
from .fonts_and_scale import _DrawScaleParams
from .pick_distance import (
    _min_sqdist_point_to_polyline_display,
    _min_sqdist_point_to_polyline_display_3d,
)
from .tensors import _tensor_disk_radius_px


@dataclass(frozen=True)
class _RenderHoverState:
    ax: Any
    figure: FigureLike
    dimensions: int
    node_patch_coll: (
        PatchCollection | PathCollection | Sequence[PatchCollection | PathCollection] | None
    )
    visible_node_ids: tuple[int, ...]
    tensor_hover: dict[int, tuple[str, float]]
    edge_hover: tuple[tuple[np.ndarray, str], ...]
    line_width_px_hint: float
    positions: NodePositions | None = None
    params: _DrawScaleParams | None = None
    tensor_disk_radius_px_3d: float | None = None


def _display_point_in_projected_aabb(
    ax: Any,
    bounds: tuple[float, float, float, float, float, float],
    x_pix: float,
    y_pix: float,
    margin_px: float,
) -> bool:
    """True if pointer (figure pixels) lies in the screen-space bounding box of a 3D AABB."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )
    M = ax.get_proj()
    xs_disp: list[float] = []
    ys_disp: list[float] = []
    for row in corners:
        xp, yp, _zp = proj3d.proj_transform(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            M,
        )
        pt = np.asarray(ax.transData.transform((xp, yp)), dtype=float).ravel()
        xs_disp.append(float(pt[0]))
        ys_disp.append(float(pt[1]))
    x0, x1 = min(xs_disp), max(xs_disp)
    y0, y1 = min(ys_disp), max(ys_disp)
    m = float(margin_px)
    return (x0 - m <= x_pix <= x1 + m) and (y0 - m <= y_pix <= y1 + m)


def _disconnect_tensor_network_hover(fig: FigureLike) -> None:
    resolved_figure = root_figure(fig)
    cid = get_hover_cid(resolved_figure)
    if cid is not None:
        with suppress(ValueError, KeyError):
            resolved_figure.canvas.mpl_disconnect(int(cid))
        clear_hover_cid(resolved_figure)
    ann = get_hover_annotation(resolved_figure)
    if ann is not None:
        setter = getattr(ann, "set_visible", None)
        if callable(setter):
            with suppress(AttributeError, TypeError, ValueError):
                setter(False)
        remover = getattr(ann, "remove", None)
        if callable(remover):
            with suppress(AttributeError, NotImplementedError, TypeError, ValueError):
                remover()
        clear_hover_annotation(resolved_figure)
        draw_idle = getattr(resolved_figure.canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()


def _register_2d_hover_labels(
    ax: Axes,
    *,
    node_patch_coll: (
        PatchCollection | PathCollection | Sequence[PatchCollection | PathCollection] | None
    ),
    visible_node_ids: list[int],
    tensor_hover: dict[int, tuple[str, float]],
    edge_hover: list[tuple[np.ndarray, str]],
    line_width_px_hint: float,
    scheme_hover_patches: Sequence[tuple[Any, str]] | None = None,
) -> None:
    """Show tensor / bond labels in a tooltip while the pointer hovers (2D axes)."""
    fig = ax.figure
    _disconnect_tensor_network_hover(fig)

    scheme_entries = tuple(scheme_hover_patches or ())
    if not tensor_hover and not edge_hover and not scheme_entries:
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
    set_hover_annotation(fig, ann)

    def on_move(event: Any) -> None:
        if event.inaxes != ax or event.x is None or event.y is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        x_d, y_d = float(event.x), float(event.y)
        label: str | None = None
        fs_hint = 10.0

        if tensor_hover and node_patch_coll is not None:
            if isinstance(node_patch_coll, (PatchCollection, PathCollection)):
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

        if label is None and scheme_entries:
            for patch, txt in scheme_entries:
                vis = getattr(patch, "get_visible", None)
                if callable(vis) and not vis():
                    continue
                contains_fn = getattr(patch, "contains", None)
                if not callable(contains_fn):
                    continue
                try:
                    contains_result = contains_fn(event)
                except (NotImplementedError, TypeError, ValueError):
                    continue
                if (
                    not isinstance(contains_result, tuple)
                    or len(contains_result) != 2
                    or not isinstance(contains_result[0], bool)
                ):
                    continue
                hit = contains_result[0]
                if hit:
                    label = txt
                    fs_hint = 8.0
                    break

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
        ann.set_fontsize(max(7.0, min(14.0, fs_hint)))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    set_hover_cid(fig, fig.canvas.mpl_connect("motion_notify_event", on_move))


def _register_3d_hover_labels(
    ax: Any,
    fig: FigureLike,
    *,
    positions: NodePositions,
    visible_node_ids: list[int],
    tensor_hover: dict[int, tuple[str, float]],
    edge_hover: list[tuple[np.ndarray, str]],
    line_width_px_hint: float,
    p: _DrawScaleParams,
    tensor_disk_radius_px_3d: float | None = None,
    scheme_hover_aabbs: Sequence[tuple[tuple[float, float, float, float, float, float], str, Any]]
    | None = None,
) -> None:
    """Show tensor / bond labels in a figure-space tooltip while the pointer hovers (3D)."""
    resolved_figure = root_figure(fig)
    _disconnect_tensor_network_hover(resolved_figure)

    scheme_3d = tuple(scheme_hover_aabbs or ())
    if not tensor_hover and not edge_hover and not scheme_3d:
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
    set_hover_annotation(resolved_figure, ann)

    def on_move(event: Any) -> None:
        if event.inaxes != ax or event.x is None or event.y is None:
            ann.set_visible(False)
            resolved_figure.canvas.draw_idle()
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
                if tensor_disk_radius_px_3d is not None:
                    rpx = float(tensor_disk_radius_px_3d)
                else:
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

        if label is None and scheme_3d:
            for bounds, txt, artist in scheme_3d:
                if artist is not None:
                    vis = getattr(artist, "get_visible", None)
                    if callable(vis) and not vis():
                        continue
                if _display_point_in_projected_aabb(ax, bounds, x_d, y_d, pick_r):
                    label = txt
                    fs_hint = 8.0
                    break

        if not label:
            ann.set_visible(False)
            resolved_figure.canvas.draw_idle()
            return

        ann.xy = (x_d, y_d)
        ann.set_text(label)
        ann.set_fontsize(max(7.0, min(14.0, fs_hint)))
        ann.set_visible(True)
        resolved_figure.canvas.draw_idle()

    set_hover_cid(
        resolved_figure,
        resolved_figure.canvas.mpl_connect(
            "motion_notify_event",
            on_move,
        ),
    )


def _apply_saved_hover_state(
    state: _RenderHoverState,
    *,
    scheme_patches_2d: Sequence[tuple[Any, str]] | None = None,
    scheme_aabbs_3d: Sequence[tuple[tuple[float, float, float, float, float, float], str, Any]]
    | None = None,
) -> None:
    scheme_2d = tuple(scheme_patches_2d or ())
    scheme_3d = tuple(scheme_aabbs_3d or ())
    want_label_hover = bool(state.tensor_hover) or bool(state.edge_hover)
    want_scheme_hover = bool(scheme_2d) or bool(scheme_3d)
    if not (want_label_hover or want_scheme_hover):
        _disconnect_tensor_network_hover(state.figure)
        return

    if state.dimensions == 2:
        node_collection = state.node_patch_coll if state.tensor_hover else None
        _register_2d_hover_labels(
            state.ax,
            node_patch_coll=node_collection,
            visible_node_ids=list(state.visible_node_ids),
            tensor_hover=dict(state.tensor_hover),
            edge_hover=list(state.edge_hover),
            line_width_px_hint=float(state.line_width_px_hint),
            scheme_hover_patches=scheme_2d,
        )
        return

    assert state.positions is not None
    assert state.params is not None
    _register_3d_hover_labels(
        state.ax,
        state.figure,
        positions=state.positions,
        visible_node_ids=list(state.visible_node_ids),
        tensor_hover=dict(state.tensor_hover),
        edge_hover=list(state.edge_hover),
        line_width_px_hint=float(state.line_width_px_hint),
        p=state.params,
        tensor_disk_radius_px_3d=state.tensor_disk_radius_px_3d,
        scheme_hover_aabbs=scheme_3d,
    )


__all__ = [
    "_apply_saved_hover_state",
    "_disconnect_tensor_network_hover",
    "_RenderHoverState",
    "_register_2d_hover_labels",
    "_register_3d_hover_labels",
]
