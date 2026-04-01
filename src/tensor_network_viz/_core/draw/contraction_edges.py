from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from ...config import PlotConfig
from ..contractions import _ContractionGroups
from ..curves import _quadratic_curve
from ..graph import _EdgeData, _GraphData
from ..layout import NodePositions
from .constants import _CURVE_NEAR_PAIR_REF, _CURVE_OFFSET_FACTOR
from .edge_labels import _plot_contraction_index_captions
from .fonts_and_scale import _DrawScaleParams
from .labels_misc import _contraction_hover_label_text
from .plotter import _PlotAdapter
from .vectors import _bond_perpendicular_unoriented

_ZORDER_FLAT_BOND_LINE: float = 1.0


def _curved_edge_points(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    dimensions: Literal[2, 3],
    scale: float = 1.0,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    perpendicular = _bond_perpendicular_unoriented(delta, dimensions)
    ref_len = _CURVE_NEAR_PAIR_REF * scale
    effective_chord = float(math.hypot(distance, ref_len))
    offset = (
        (offset_index - (edge_count - 1) / 2.0) * _CURVE_OFFSET_FACTOR * scale * effective_chord
    )
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


def _draw_contraction_edge(
    *,
    plotter: _PlotAdapter,
    edge: _EdgeData,
    graph: _GraphData,
    positions: NodePositions,
    contraction_groups: _ContractionGroups,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float,
    dimensions: Literal[2, 3],
    p: _DrawScaleParams,
    ax: Any,
    zorder_line: float | None = None,
    zorder_label: float | None = None,
) -> None:
    left_id, right_id = edge.node_ids
    offset_index, edge_count = contraction_groups.offsets[id(edge)]
    curve = _curved_edge_points(
        start=positions[left_id],
        end=positions[right_id],
        offset_index=offset_index,
        edge_count=edge_count,
        dimensions=dimensions,
        scale=scale,
    )
    zorder_curve = float(_ZORDER_FLAT_BOND_LINE) if zorder_line is None else float(zorder_line)
    plotter.plot_curve(curve, color=config.bond_edge_color, linewidth=p.lw, zorder=zorder_curve)
    if not show_index_labels:
        return

    hover_targets = getattr(plotter, "_hover_edge_targets", None)
    if config.hover_labels and hover_targets is not None:
        caption = _contraction_hover_label_text(edge, graph)
        if caption:
            if dimensions == 2:
                hover_targets.append((np.asarray(curve[:, :2], dtype=float, order="C"), caption))
            else:
                hover_targets.append((np.asarray(curve[:, :3], dtype=float, order="C"), caption))
        return

    _plot_contraction_index_captions(
        plotter=plotter,
        curve=curve,
        edge=edge,
        graph=graph,
        positions=positions,
        left_id=left_id,
        right_id=right_id,
        config=config,
        p=p,
        dimensions=dimensions,
        ax=ax,
        scale=scale,
        zorder_label=zorder_label,
    )


__all__ = [
    "_curved_edge_points",
    "_draw_contraction_edge",
]
