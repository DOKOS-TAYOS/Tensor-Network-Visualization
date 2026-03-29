from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import proj3d


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


__all__ = [
    "_min_sqdist_point_to_polyline_display",
    "_min_sqdist_point_to_polyline_display_3d",
    "_sqdist_point_to_segment",
]
