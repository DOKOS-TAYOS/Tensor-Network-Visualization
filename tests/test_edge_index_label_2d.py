"""Unit tests for 2D edge index label geometry helpers (no image baselines)."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_viz._core._draw_common import (
    _EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT,
    _EDGE_INDEX_LABEL_SPAN_FRAC_PHYS,
    _EDGE_INDEX_NODE_CLEAR_FRAC,
    _blend_bond_tangent_with_chord_2d,
    _edge_index_label_axis_tie_vertical_2d,
    _edge_index_label_is_vertical_axis_2d,
    _edge_index_label_span_frac,
    _edge_index_rim_arc_from_endpoint,
)


def test_node_clear_frac_matches_shortest_edge_model() -> None:
    assert _EDGE_INDEX_NODE_CLEAR_FRAC == 0.3


def test_rim_arc_uses_global_radius_until_midpoint() -> None:
    got = _edge_index_rim_arc_from_endpoint(r_global=0.3, half_polyline_length=1.0)
    assert got == pytest.approx(0.3)
    got2 = _edge_index_rim_arc_from_endpoint(r_global=0.9, half_polyline_length=0.4)
    assert got2 == pytest.approx(0.4 * (1.0 - 1e-9))


def test_span_fracs_match_spec() -> None:
    assert pytest.approx(0.18) == _EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT
    assert pytest.approx(0.49) == _EDGE_INDEX_LABEL_SPAN_FRAC_PHYS


def test_edge_index_label_span_frac_dispatch() -> None:
    assert _edge_index_label_span_frac(is_physical=False) == _EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT
    assert _edge_index_label_span_frac(is_physical=True) == _EDGE_INDEX_LABEL_SPAN_FRAC_PHYS


def test_vertical_axis_classification() -> None:
    assert not _edge_index_label_is_vertical_axis_2d(np.array([1.0, 0.0]))
    assert not _edge_index_label_is_vertical_axis_2d(np.array([0.7, -0.3]))
    assert _edge_index_label_is_vertical_axis_2d(np.array([0.0, 1.0]))
    assert _edge_index_label_is_vertical_axis_2d(np.array([-0.2, 0.8]))
    assert _edge_index_label_is_vertical_axis_2d(np.array([1.0, 1.0]))


def test_blend_straight_matches_chord() -> None:
    t_c = np.array([1.0, 0.0], dtype=float)
    a = np.array([0.0, 0.0], dtype=float)
    b = np.array([2.0, 0.0], dtype=float)
    out = _blend_bond_tangent_with_chord_2d(t_c, a, b)
    assert float(out[0]) == pytest.approx(1.0)
    assert float(out[1]) == pytest.approx(0.0)


def test_axis_tie_random_vertical_vs_horizontal_reproducible() -> None:
    d = np.array([1.0, 1.0], dtype=float)
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)
    a = _edge_index_label_axis_tie_vertical_2d(d, rng_a)
    b = _edge_index_label_axis_tie_vertical_2d(d, rng_b)
    assert a == b


def test_axis_tie_not_used_when_components_differ() -> None:
    rng = np.random.default_rng(999)
    assert not _edge_index_label_axis_tie_vertical_2d(np.array([1.0, 0.05], dtype=float), rng)
    assert _edge_index_label_axis_tie_vertical_2d(np.array([0.05, 1.0], dtype=float), rng)


def test_blend_curved_partially_follows_curve() -> None:
    """Chord is +x; curve tangent +y — blended direction tilts toward +y."""
    t_c = np.array([0.0, 1.0], dtype=float)
    a = np.array([0.0, 0.0], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)
    out = _blend_bond_tangent_with_chord_2d(t_c, a, b)
    assert float(out[0]) > 0.0 and float(out[1]) > 0.0
    assert float(np.linalg.norm(out)) == pytest.approx(1.0)
