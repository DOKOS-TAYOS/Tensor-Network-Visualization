"""Shared assertions for Matplotlib axes after tensor-network 2D plots."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, PathCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection


def line_collection_segment_count(ax: Any) -> int:
    """Number of polylines drawn as LineCollections (2D batched edges)."""
    return sum(len(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection))


def line_collection_segments(ax: Any) -> list[np.ndarray]:
    """All polyline segments from LineCollections, in collection order."""
    out: list[np.ndarray] = []
    for c in ax.collections:
        if isinstance(c, LineCollection):
            out.extend(c.get_segments())
    return out


def patch_collection_circle_count(ax: Any) -> int:
    """Tensor nodes in 2D are a single PatchCollection of circles."""
    n = 0
    for c in ax.collections:
        if isinstance(c, PatchCollection):
            n += len(c.get_paths())
    return n


def path_collection_point_count(ax: Any) -> int:
    """Count marker points stored in 2D PathCollections."""
    n = 0
    for c in ax.collections:
        if isinstance(c, PathCollection):
            offsets = np.asarray(c.get_offsets(), dtype=float)
            if offsets.ndim == 2:
                n += int(offsets.shape[0])
    return n


def point_collection_sizes(ax: Any) -> list[tuple[float, ...]]:
    """Marker areas from 2D PathCollections, preserving collection grouping."""
    out: list[tuple[float, ...]] = []
    for c in ax.collections:
        if isinstance(c, PathCollection):
            sizes = np.asarray(c.get_sizes(), dtype=float).reshape(-1)
            out.append(tuple(float(value) for value in sizes))
    return out


def point_collection_facecolors(ax: Any) -> list[tuple[float, ...]]:
    """Face colors from 2D PathCollections."""
    out: list[tuple[float, ...]] = []
    for c in ax.collections:
        if isinstance(c, PathCollection):
            for row in np.asarray(c.get_facecolors(), dtype=float):
                out.append(tuple(float(value) for value in row))
    return out


def poly3d_node_collection_count(ax: Any) -> int:
    """Count 3D polygon collections, used by octahedron nodes."""
    return sum(1 for c in ax.collections if isinstance(c, Poly3DCollection))


def path3d_collection_point_count(ax: Any) -> int:
    """Count marker points stored in 3D scatter collections."""
    n = 0
    for c in ax.collections:
        if isinstance(c, Path3DCollection):
            xs, _ys, _zs = c._offsets3d
            n += len(xs)
    return n


def path3d_collection_sizes(ax: Any) -> list[tuple[float, ...]]:
    """Marker areas from 3D scatter collections."""
    out: list[tuple[float, ...]] = []
    for c in ax.collections:
        if isinstance(c, Path3DCollection):
            sizes = np.asarray(c.get_sizes(), dtype=float).reshape(-1)
            out.append(tuple(float(value) for value in sizes))
    return out


def path3d_collection_facecolors(ax: Any) -> list[tuple[float, ...]]:
    """Face colors from 3D scatter collections."""
    out: list[tuple[float, ...]] = []
    for c in ax.collections:
        if isinstance(c, Path3DCollection):
            for row in np.asarray(c.get_facecolors(), dtype=float):
                out.append(tuple(float(value) for value in row))
    return out
