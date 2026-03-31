"""Drawing tuning constants (shared 2D/3D)."""

from __future__ import annotations

import numpy as np

# Multiedge separation; keep in sync with
# `_LAYOUT_BOND_CURVE_OFFSET_FACTOR` in `layout/parameters.py`.
_CURVE_OFFSET_FACTOR: float = 0.15
# Blends with chord length so multiedges keep visible separation when endpoints are close.
_CURVE_NEAR_PAIR_REF: float = 0.28
# Extra radius + offset so index captions sit just outside tensor disks (data units).
_NODE_LABEL_MARGIN_FACTOR: float = 1.22
# Small extra perpendicular gap (× layout ``scale``) for 2D bond index labels after half-linewidth.
_INDEX_LABEL_2D_PERP_EXTRA: float = 0.014
# Tighter pad for stroke-flush captions (contractions / physical stubs); only ``hw + this``.
_INDEX_LABEL_2D_STROKE_PAD: float = 0.0035
# Scales ``~1 em`` in data units to extra perp offset so small/large fonts hug the stroke similarly.
_STROKE_LABEL_EM_PERP_FRAC: float = 0.22
# Long bonds (e.g. MERA) cap em-based perp offset vs ``hw`` (max font blows up em in data units).
_STROKE_LABEL_EM_PERP_MAX_HW_MULT: float = 2.0
# If true curve tangent vs blend diverge (dot product), keep left/right using blended normal.
_STROKE_LABEL_GEOM_NORMAL_DOT_MIN: float = 0.5
# Disk clearance as a fraction of shortest bond (``renderer._SHORTEST_EDGE_RADIUS_FRACTION``).
_EDGE_INDEX_NODE_CLEAR_FRAC: float = 0.3
# Target caption span × bond length (data units); contraction vs physical-open axis.
_EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT: float = 0.45 * (1.0 - 2.0 * _EDGE_INDEX_NODE_CLEAR_FRAC)
_EDGE_INDEX_LABEL_SPAN_FRAC_PHYS: float = 0.7 * (1.0 - _EDGE_INDEX_NODE_CLEAR_FRAC)
# Weight of on-curve tangent when blending with chord (2D curved multiedges).
_CURVE_TANGENT_BLEND_LAMBDA: float = 0.2
# Global scale on all bond / stub index caption font sizes (~shortest-bond span still per edge).
_EDGE_INDEX_LABEL_FONT_GLOBAL_SCALE: float = 0.8
# Open / physical legs: drawn label is 20% larger than internal bond captions (after global scale).
_PHYSICAL_INDEX_LABEL_FONT_SCALE: float = 1.2
# All 3D index + tensor labels scale vs 2D (depth reads smaller; sizing uses data-space bond span).
_LABEL_FONT_3D_SCALE: float = 1.2
_AXIS_TIE_EPS: float = 1e-9
# TextPath width under-estimates padded bbox slightly; calibrate so nominal fraction holds visually.
_EDGE_INDEX_LABEL_WIDTH_CALIB: float = 1.12
# Along-edge reference: edge of label aligns at this arc-length fraction from its endpoint.
_EDGE_INDEX_LABEL_ALONG_FRAC: float = 0.3
# Physical dangling legs (2D): inset from **open tip** — smaller ⇒ closer to the free end.
_PHYS_DANGLING_2D_FRAC_FROM_TIP: float = 0.07
# Draw order: contraction scheme < bonds < node disks < edge index labels < tensor names (on top).
_ZORDER_CONTRACTION_SCHEME: float = 0.5
# Legacy flat 3D / pre-layered 2D stacking.
_ZORDER_NODE_DISK: int = 3
_ZORDER_EDGE_INDEX_LABEL: int = 5
_ZORDER_TENSOR_NAME: int = 8
# 2D layered draw: per visible node i (in graph order), z = base + i * stride + offset.
# Stride >= 4 keeps bond/label/disk of node i below the next node's bonds.
_ZORDER_LAYER_BASE: float = 10.0
_ZORDER_LAYER_STRIDE: float = 4.0
_ZORDER_LAYER_BOND: float = 0.0
_ZORDER_LAYER_DANGLING: float = 1.0
_ZORDER_LAYER_DISK: float = 2.0
_ZORDER_LAYER_EDGE_INDEX: float = 2.5
_ZORDER_LAYER_TENSOR_NAME: float = 3.25
_EDGE_INDEX_LABEL_GID: str = "tnv_edge_index"
_TENSOR_LABEL_GID: str = "tnv_tensor"
# TextPath under-estimates the final Text bbox; scale diagonal for "fits inside disk" checks.
_TEXT_RENDER_DIAGONAL_FACTOR: float = 1.52
# Keep tensor names slightly inset from the disk / projected octahedron silhouette.
_TENSOR_LABEL_INSIDE_FILL: float = 0.88
# Line end caps / joins: slight rounding reads softer than Matplotlib's default butt/miter.
_EDGE_LINE_CAP_STYLE: str = "round"
_EDGE_LINE_JOIN_STYLE: str = "round"
_FIGURE_MIN_PX_REF: float = 520.0

# 3D nodes: octahedron (8 tris / node). Full UV spheres are too heavy for interactive mplot3d.
_UNIT_NODE_TRIS: np.ndarray = np.asarray(
    [
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    ],
    dtype=float,
)
_OCTAHEDRON_TRI_COUNT: int = int(_UNIT_NODE_TRIS.shape[0])
# 3D bond curves use ``p.lw``; octahedron rims are drawn much finer so edges stay subtle vs bonds.
_OCTAHEDRON_EDGE_LINEWIDTH_FACTOR: float = 0.18
_OCTAHEDRON_EDGE_LINEWIDTH_MIN: float = 0.04

_HOVER_EDGE_PICK_RADIUS_PX: float = 10.0

_ZOOM_FONT_CLAMP: tuple[float, float] = (0.28, 5.5)


__all__ = [
    "_AXIS_TIE_EPS",
    "_CURVE_NEAR_PAIR_REF",
    "_CURVE_OFFSET_FACTOR",
    "_CURVE_TANGENT_BLEND_LAMBDA",
    "_EDGE_INDEX_LABEL_ALONG_FRAC",
    "_EDGE_INDEX_LABEL_FONT_GLOBAL_SCALE",
    "_EDGE_INDEX_LABEL_GID",
    "_EDGE_INDEX_LABEL_SPAN_FRAC_CONTRACT",
    "_EDGE_INDEX_LABEL_SPAN_FRAC_PHYS",
    "_EDGE_INDEX_LABEL_WIDTH_CALIB",
    "_EDGE_INDEX_NODE_CLEAR_FRAC",
    "_EDGE_LINE_CAP_STYLE",
    "_EDGE_LINE_JOIN_STYLE",
    "_FIGURE_MIN_PX_REF",
    "_HOVER_EDGE_PICK_RADIUS_PX",
    "_INDEX_LABEL_2D_PERP_EXTRA",
    "_INDEX_LABEL_2D_STROKE_PAD",
    "_LABEL_FONT_3D_SCALE",
    "_NODE_LABEL_MARGIN_FACTOR",
    "_OCTAHEDRON_EDGE_LINEWIDTH_FACTOR",
    "_OCTAHEDRON_EDGE_LINEWIDTH_MIN",
    "_OCTAHEDRON_TRI_COUNT",
    "_PHYS_DANGLING_2D_FRAC_FROM_TIP",
    "_PHYSICAL_INDEX_LABEL_FONT_SCALE",
    "_STROKE_LABEL_EM_PERP_FRAC",
    "_STROKE_LABEL_EM_PERP_MAX_HW_MULT",
    "_STROKE_LABEL_GEOM_NORMAL_DOT_MIN",
    "_TENSOR_LABEL_GID",
    "_TENSOR_LABEL_INSIDE_FILL",
    "_TEXT_RENDER_DIAGONAL_FACTOR",
    "_UNIT_NODE_TRIS",
    "_ZOOM_FONT_CLAMP",
    "_ZORDER_CONTRACTION_SCHEME",
    "_ZORDER_EDGE_INDEX_LABEL",
    "_ZORDER_LAYER_BASE",
    "_ZORDER_LAYER_BOND",
    "_ZORDER_LAYER_DANGLING",
    "_ZORDER_LAYER_DISK",
    "_ZORDER_LAYER_EDGE_INDEX",
    "_ZORDER_LAYER_STRIDE",
    "_ZORDER_LAYER_TENSOR_NAME",
    "_ZORDER_NODE_DISK",
    "_ZORDER_TENSOR_NAME",
]
