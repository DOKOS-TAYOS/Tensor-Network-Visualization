"""Unit tests for 3D edge index label geometry helpers (no image baselines)."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_viz._core._draw_common import (
    _blend_bond_tangent_with_chord_3d,
    _nominal_figure_px_per_data_unit_3d,
    _stroke_index_normal_screen_unit_2d,
)


def test_blend_bond_tangent_with_chord_3d_aligned_with_chord() -> None:
    t = np.array([0.0, 1.0, 0.0], dtype=float)
    a = np.array([0.0, 0.0, 0.0], dtype=float)
    b = np.array([0.0, 2.0, 0.0], dtype=float)
    out = _blend_bond_tangent_with_chord_3d(t, a, b)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_blend_bond_tangent_with_chord_3d_degenerate_chord_returns_curve_dir() -> None:
    t = np.array([1.0, 2.0, -1.0], dtype=float)
    a = np.array([3.0, 3.0, 3.0], dtype=float)
    b = a.copy()
    out = _blend_bond_tangent_with_chord_3d(t, a, b)
    t_u = t / np.linalg.norm(t)
    np.testing.assert_allclose(out, t_u, atol=1e-9)


def test_stroke_index_normal_screen_unit_2d_perpendicular_to_align_tangent() -> None:
    tg = np.array([1.0, 0.0], dtype=float)
    ta = np.array([1.0, 0.0], dtype=float)
    n = _stroke_index_normal_screen_unit_2d(tg, ta)
    assert float(np.dot(n, ta)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.linalg.norm(n)) == pytest.approx(1.0, abs=1e-9)


def test_nominal_figure_px_per_data_unit_3d_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 2.0)
    k = _nominal_figure_px_per_data_unit_3d(ax)
    assert np.isfinite(k)
    assert k > 0.0
    fig.canvas.draw()
    # span = 2 -> min_px/2
    dpi = float(fig.dpi)
    min_px = min(4.0, 3.0) * dpi
    assert k == pytest.approx(min_px / 2.0, rel=1e-5)
