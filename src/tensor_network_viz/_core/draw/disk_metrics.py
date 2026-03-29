from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import proj3d

from .fonts_and_scale import _DrawScaleParams


def _tensor_disk_radius_px_3d_nominal(ax: Any, p: _DrawScaleParams) -> float:
    """Pixels per data unit from axis spans only (no per-point projection); ``r`` in data units."""
    from .viewport_geometry import _nominal_figure_px_per_data_unit_3d

    k = _nominal_figure_px_per_data_unit_3d(ax)
    return float(max(float(p.r) * k, 1e-9))


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


__all__ = [
    "_display_disk_radius_px_2d",
    "_display_disk_radius_px_3d",
    "_tensor_disk_radius_px",
    "_tensor_disk_radius_px_3d_nominal",
]
