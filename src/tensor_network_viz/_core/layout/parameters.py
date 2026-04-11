"""Layout tuning constants."""

from __future__ import annotations

_LAYOUT_TARGET_NORM: float = 1.6
_FORCE_LAYOUT_K: float = 1.6
_FORCE_LAYOUT_COOLING_FACTOR: float = 0.985
_FREE_DIR_OVERLAP_THRESHOLD: float = 0.7
_FREE_DIR_SAMPLES_2D: int = 72
# Dangling-stub segment in layout units (rim → tip), for 2D crossing checks vs `_draw_*` scale.
_STUB_LAYOUT_R0: float = 0.06
# Keep (R1 − R0) * draw_scale ≈ ``PlotConfig.DEFAULT_STUB_LENGTH * draw_scale`` (rim → tip).
_STUB_LAYOUT_R1: float = 0.22
_STUB_TIP_NODE_CLEAR: float = 0.26
_STUB_TIP_TIP_CLEAR: float = 0.26
_SEGMENT_BOND_CLEAR_2D: float = 0.15
_STUB_BOND_CLEAR_3D: float = 0.2
_STUB_ORIGIN_PAIR_CLEAR: float = 0.12
_STUB_PARALLEL_DOT: float = 0.92
_ATTACHMENT_RADII_2D: tuple[float, ...] = (1.0, 1.35)
_ATTACHMENT_NODE_CLEAR_2D: float = 0.26
_ATTACHMENT_TARGET_CLEAR_2D: float = 0.3
_ATTACHMENT_PARALLEL_DOT_2D: float = 0.92
# Keep in sync with `_draw_common._CURVE_OFFSET_FACTOR` (stub–bond clash vs drawn bonds).
_LAYOUT_BOND_CURVE_OFFSET_FACTOR: float = 0.10
_LAYOUT_BOND_CURVE_NEAR_PAIR_REF: float = 0.28
_LAYOUT_BOND_CURVE_SAMPLES: int = 24
_COMPONENT_GAP: float = 1.4
# Minimum spacing (layout units) between virtual hyperedge hubs that share the same tensor neighbors
# (otherwise they collapse to the same barycenter).
_VIRTUAL_HUB_MIN_SEPARATION: float = 0.38
# Perpendicular displacement (layout units) so a virtual hub leaves the tensor–tensor chord when
# that pair also has a direct contraction (e.g. matmul axis alongside a batch hyperedge).
_VIRTUAL_HUB_CHORD_CLEARANCE: float = 0.24
_LAYER_SPACING: float = 0.55
_LAYER_SEQUENCE: tuple[int, ...] = (0, 1, -1, 2, -2, 3, -3)

__all__ = [
    "_COMPONENT_GAP",
    "_ATTACHMENT_NODE_CLEAR_2D",
    "_ATTACHMENT_PARALLEL_DOT_2D",
    "_ATTACHMENT_RADII_2D",
    "_ATTACHMENT_TARGET_CLEAR_2D",
    "_FORCE_LAYOUT_COOLING_FACTOR",
    "_FORCE_LAYOUT_K",
    "_FREE_DIR_OVERLAP_THRESHOLD",
    "_FREE_DIR_SAMPLES_2D",
    "_LAYER_SEQUENCE",
    "_VIRTUAL_HUB_CHORD_CLEARANCE",
    "_VIRTUAL_HUB_MIN_SEPARATION",
    "_LAYER_SPACING",
    "_LAYOUT_BOND_CURVE_NEAR_PAIR_REF",
    "_LAYOUT_BOND_CURVE_OFFSET_FACTOR",
    "_LAYOUT_BOND_CURVE_SAMPLES",
    "_LAYOUT_TARGET_NORM",
    "_SEGMENT_BOND_CLEAR_2D",
    "_STUB_LAYOUT_R0",
    "_STUB_LAYOUT_R1",
    "_STUB_BOND_CLEAR_3D",
    "_STUB_ORIGIN_PAIR_CLEAR",
    "_STUB_PARALLEL_DOT",
    "_STUB_TIP_NODE_CLEAR",
    "_STUB_TIP_TIP_CLEAR",
]
