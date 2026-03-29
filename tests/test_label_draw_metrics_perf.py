"""Performance/regression checks for label metrics (TextPath cache, 3D disk px, refine).

**Baselines (example dev machine, 2026-03-29):**

- ``_textpath_width_pts`` — example: 200 unique labels × 6 passes (1200 calls) ~8.8s **without**
  LRU (all TextPath builds) vs ~1.5s **with** LRU on the same workload (hits after prime), same
  machine 2026-03-29.
- ``_display_disk_radius_px_3d`` — 300 calls ~0.042s; nominal scale is one multiply per node
  after a single axis read (~0.0002s order for the same 300 logical uses if reused).
- ``refine_tensor_labels`` — turning it off avoids repeated ``canvas.draw()`` in
  ``_refit_tensor_labels_to_disks`` (see ``PlotConfig`` docstring); magnitude depends on figure
  complexity — use ``test_draw_performance`` / local ``time.perf_counter`` for full draws.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

from tensor_network_viz._core.draw.disk_metrics import (
    _display_disk_radius_px_3d,
    _tensor_disk_radius_px_3d_nominal,
)
from tensor_network_viz._core.draw.fonts_and_scale import (
    _DrawScaleParams,
    _textpath_width_pts,
    _textpath_width_pts_cached,
)


def test_textpath_width_cache_speedup_on_repeat() -> None:
    labels = [f"bond{i}" for i in range(120)]
    _textpath_width_pts_cached.cache_clear()

    t0 = time.perf_counter()
    for _ in range(5):
        for s in labels:
            _textpath_width_pts(s, fontsize_pt=10.0)
    cold_block = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(5):
        for s in labels:
            _textpath_width_pts(s, fontsize_pt=10.0)
    warm_block = time.perf_counter() - t1

    info = _textpath_width_pts_cached.cache_info()
    assert info.hits >= 400, "expected mostly cache hits on second block"
    assert warm_block < cold_block * 0.35, (
        f"expected warm << cold (cold={cold_block:.4f}s warm={warm_block:.4f}s)"
    )


def test_3d_nominal_disk_px_cheaper_than_many_projections() -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    p = _DrawScaleParams(
        r=0.08,
        stub=0.1,
        loop_r=0.2,
        lw=1.0,
        font_tensor_label_max=10.0,
        index_bbox_pad=0.1,
        label_offset=0.1,
        ellipse_w=0.1,
        ellipse_h=0.1,
    )
    centers = [np.array([float(i), float(i) * 0.5, float(i) * 0.3]) for i in range(200)]

    t0 = time.perf_counter()
    for c in centers:
        _display_disk_radius_px_3d(ax, c, p.r)
    exact_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    r_nom = _tensor_disk_radius_px_3d_nominal(ax, p)
    nominal_s = time.perf_counter() - t1

    plt.close(fig)

    assert r_nom > 0
    assert nominal_s < exact_s * 0.15, (
        f"nominal reuse << per-node projection (exact={exact_s:.4f}s nominal_loop={nominal_s:.4f}s)"
    )
