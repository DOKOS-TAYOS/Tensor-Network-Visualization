"""2D drawing for tensor networks."""

from __future__ import annotations

from matplotlib.axes import Axes

from ..config import PlotConfig
from ._draw_common import (
    _draw_edges,
    _draw_labels,
    _draw_nodes,
    _draw_scale_params,
    _make_2d_plotter,
)
from .curves import _group_contractions
from .graph import _GraphData
from .layout import AxisDirections, NodePositions


def _draw_2d(
    *,
    ax: Axes,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float = 1.0,
) -> None:
    ax.cla()
    pair_groups = _group_contractions(graph)
    p = _draw_scale_params(config, scale, is_3d=False)
    plotter = _make_2d_plotter(ax)

    _draw_edges(
        plotter=plotter,
        graph=graph,
        positions=positions,
        directions=directions,
        pair_groups=pair_groups,
        show_index_labels=show_index_labels,
        config=config,
        scale=scale,
        dimensions=2,
        p=p,
    )
    coords = _draw_nodes(
        plotter=plotter,
        graph=graph,
        positions=positions,
        config=config,
        p=p,
    )
    _draw_labels(
        plotter=plotter,
        graph=graph,
        positions=positions,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        config=config,
        p=p,
    )
    plotter.style_axes(coords)
