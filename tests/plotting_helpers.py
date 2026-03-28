"""Shared assertions for Matplotlib axes after tensor-network 2D plots."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection


def line_collection_segment_count(ax: Any) -> int:
    """Number of polylines drawn as LineCollections (2D batched edges)."""
    return sum(
        len(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection)
    )


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
