"""3D drawing for tensor networks."""

from __future__ import annotations

from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from ._draw_common import (
    _draw_graph,
)
from .graph import _GraphData
from .layout import AxisDirections, NodePositions


def _draw_3d(
    *,
    ax: Axes3D,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    scale: float = 1.0,
) -> None:
    _draw_graph(
        ax=ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        config=config,
        dimensions=3,
        scale=scale,
    )
