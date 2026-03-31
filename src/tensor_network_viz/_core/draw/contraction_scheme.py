"""Colored hull outlines for ordered contraction steps (optional translucent fill)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from ...config import PlotConfig
from ..graph import _GraphData, _resolve_contraction_scheme_by_name
from ..layout import NodePositions
from .constants import _ZORDER_CONTRACTION_SCHEME
from .fonts_and_scale import _DrawScaleParams

_DEFAULT_SCHEME_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_CONTRACTION_SCHEME_GID: str = "tnv_contraction_scheme"
_CONTRACTION_SCHEME_LABEL_GID: str = "tnv_contraction_scheme_label"
# FancyBboxPatch ``round`` style pad (fraction of box size; not a point radius).
_CONTRACTION_SCHEME_ROUND_PAD: float = 0.022
# Extra padding per step so nested / growing regions read clearly without numeric labels.
_CONTRACTION_SCHEME_STEP_PAD_FRAC: float = 0.11


def _effective_contraction_steps(
    graph: _GraphData,
    config: PlotConfig,
) -> tuple[frozenset[int], ...] | None:
    if config.contraction_scheme_by_name is not None:
        return _resolve_contraction_scheme_by_name(graph, config.contraction_scheme_by_name)
    return graph.contraction_steps


def _scheme_color_rgba(
    step_index: int,
    *,
    config: PlotConfig,
    alpha: float,
) -> tuple[float, float, float, float]:
    palette = config.contraction_scheme_colors or _DEFAULT_SCHEME_COLORS
    base = palette[step_index % len(palette)]
    r, g, b, _ = to_rgba(base)
    return (r, g, b, float(alpha))


def _padding_data_units(p: _DrawScaleParams) -> float:
    return float(p.r) * 1.28


def _strict_subset_padding_bonus(
    step_index: int,
    steps: tuple[frozenset[int], ...],
    p: _DrawScaleParams,
) -> float:
    """Extra pad when this step's tensor set strictly contains an earlier step's (nested box)."""
    si = steps[step_index]
    for j in range(step_index):
        sj = steps[j]
        if sj < si:
            return float(p.r) * 0.14
    return 0.0


def _step_index_padding_bonus(step_index: int, p: _DrawScaleParams) -> float:
    """Monotone pad growth with contraction order (larger effective tensors → roomier hull)."""
    return float(step_index) * float(p.r) * _CONTRACTION_SCHEME_STEP_PAD_FRAC


def _coords_for_step(
    graph: _GraphData,
    positions: NodePositions,
    node_ids: frozenset[int],
    dimensions: Literal[2, 3],
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for nid in node_ids:
        if nid not in graph.nodes or nid not in positions:
            continue
        coord = np.asarray(positions[nid], dtype=float)
        if coord.size < dimensions:
            pad = np.zeros(dimensions, dtype=float)
            pad[: coord.size] = coord.flat[: coord.size]
            coord = pad
        rows.append(coord[:dimensions])
    if not rows:
        return None
    return np.stack(rows, axis=0)


def _iter_scheme_draw_order(
    steps: tuple[frozenset[int], ...],
) -> list[tuple[int, frozenset[int]]]:
    """(original_step_index, node_ids) with last contraction first (drawn underneath)."""
    items: list[tuple[int, frozenset[int]]] = [
        (i, node_ids) for i, node_ids in enumerate(steps) if node_ids
    ]
    items.reverse()
    return items


def _draw_contraction_scheme(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    steps: tuple[frozenset[int], ...],
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float,
    p: _DrawScaleParams,
) -> None:
    base_pad = _padding_data_units(p)
    fill_a = float(np.clip(config.contraction_scheme_alpha, 0.0, 1.0))
    edge_a_raw = config.contraction_scheme_edge_alpha
    if edge_a_raw is not None:
        edge_a = float(np.clip(edge_a_raw, 0.0, 1.0))
    elif fill_a < 1e-3:
        edge_a = 0.88
    else:
        edge_a = float(np.clip(fill_a + 0.35, 0.0, 1.0))
    lw_attr = config.contraction_scheme_linewidth
    lw = (
        float(lw_attr) * scale
        if lw_attr is not None
        else PlotConfig.DEFAULT_CONTRACTION_SCHEME_LINEWIDTH * scale
    )

    draw_order = _iter_scheme_draw_order(steps)
    for draw_rank, (i, node_ids) in enumerate(draw_order):
        pts = _coords_for_step(graph, positions, node_ids, dimensions)
        if pts is None or pts.shape[0] == 0:
            continue
        pad = (
            base_pad
            + _strict_subset_padding_bonus(i, steps, p)
            + _step_index_padding_bonus(i, p)
        )
        fill_rgba = _scheme_color_rgba(i, config=config, alpha=fill_a)
        edge_rgba = _scheme_color_rgba(i, config=config, alpha=edge_a)
        z_patch = _ZORDER_CONTRACTION_SCHEME + 0.001 * float(draw_rank)

        if dimensions == 2:
            xmin = float(np.min(pts[:, 0]) - pad)
            xmax = float(np.max(pts[:, 0]) + pad)
            ymin = float(np.min(pts[:, 1]) - pad)
            ymax = float(np.max(pts[:, 1]) + pad)
            w = xmax - xmin
            h = ymax - ymin
            boxstyle = f"round,pad={_CONTRACTION_SCHEME_ROUND_PAD}"
            fancy = FancyBboxPatch(
                (xmin, ymin),
                w,
                h,
                boxstyle=boxstyle,
                facecolor=fill_rgba,
                edgecolor=edge_rgba,
                linewidth=lw,
                zorder=z_patch,
                gid=_CONTRACTION_SCHEME_GID,
            )
            ax.add_patch(fancy)
        else:
            xmin = float(np.min(pts[:, 0]) - pad)
            xmax = float(np.max(pts[:, 0]) + pad)
            ymin = float(np.min(pts[:, 1]) - pad)
            ymax = float(np.max(pts[:, 1]) + pad)
            zmin = float(np.min(pts[:, 2]) - pad)
            zmax = float(np.max(pts[:, 2]) + pad)
            _draw_scheme_box_3d(
                ax,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                color=edge_rgba[:3],
                alpha=float(edge_a),
                linewidth=lw,
                zorder_base=z_patch,
            )


def _draw_scheme_box_3d(
    ax: Any,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    color: tuple[float, float, float],
    alpha: float,
    linewidth: float,
    zorder_base: float,
) -> None:
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
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for a, b in edge_pairs:
        p0, p1 = corners[a], corners[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder_base,
            gid=_CONTRACTION_SCHEME_GID,
        )


__all__ = [
    "_CONTRACTION_SCHEME_GID",
    "_CONTRACTION_SCHEME_LABEL_GID",
    "_draw_contraction_scheme",
    "_effective_contraction_steps",
]
