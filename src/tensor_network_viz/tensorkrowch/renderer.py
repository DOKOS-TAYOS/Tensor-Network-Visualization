"""Public entry points for TensorKrowch plotting."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._core.renderer import _plot_graph_2d, _plot_graph_3d
from ..config import PlotConfig
from .graph import _build_graph


def plot_tensorkrowch_network_2d(
    network: Any,
    *,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    """Plot a TensorKrowch tensor network in 2D."""
    graph = _build_graph(network)
    return _plot_graph_2d(
        graph,
        ax=ax,
        config=config,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        seed=seed,
        renderer_name="plot_tensorkrowch_network_2d",
    )


def plot_tensorkrowch_network_3d(
    network: Any,
    *,
    ax: Axes | Axes3D | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes3D]:
    """Plot a TensorKrowch tensor network in 3D."""
    graph = _build_graph(network)
    return _plot_graph_3d(
        graph,
        ax=ax,
        config=config,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        seed=seed,
        renderer_name="plot_tensorkrowch_network_3d",
    )
