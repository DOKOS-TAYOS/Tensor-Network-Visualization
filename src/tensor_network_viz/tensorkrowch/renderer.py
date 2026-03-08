"""Main entry points for TensorKrowch tensor network plotting."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from .draw_2d import _draw_2d
from .draw_3d import _draw_3d
from .graph import _build_graph
from .layout import _compute_axis_directions, _compute_layout


def _resolve_flag(value: bool | None, default: bool) -> bool:
    if value is None:
        return default
    return value


def _compute_scale(n_nodes: int) -> float:
    """Scale factor for visual elements: larger for few nodes, smaller for many."""
    if n_nodes <= 1:
        return 1.2
    return max(0.5, min(1.6, 2.2 - 0.07 * n_nodes))


def _prepare_axes_2d(
    ax: Axes | None,
    *,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (14, 10))
        return fig, ax

    if getattr(ax, "name", "") == "3d":
        raise ValueError("plot_tensorkrowch_network_2d requires a 2D Matplotlib axis.")
    return ax.figure, ax


def _prepare_axes_3d(
    ax: Axes | Axes3D | None,
    *,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes3D]:
    if ax is None:
        fig = plt.figure(figsize=figsize or (14, 10))
        created_ax = fig.add_subplot(111, projection="3d")
        return fig, cast(Axes3D, created_ax)

    if getattr(ax, "name", "") != "3d":
        raise ValueError("plot_tensorkrowch_network_3d requires a 3D Matplotlib axis.")
    return ax.figure, cast(Axes3D, ax)


def plot_tensorkrowch_network_2d(
    network: Any,
    *,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    """Plot a TensorKrowch tensor network in 2D.

    Args:
        network: TensorKrowch TensorNetwork with nodes and edges.
        ax: Matplotlib 2D axes; if None, creates a new figure.
        config: Styling options; uses defaults if None.
        show_tensor_labels: Override config; None uses config value.
        show_index_labels: Override config; None uses config value.
        seed: Random seed for layout when using force-directed positioning.

    Returns:
        Tuple of (Figure, Axes) for further customization.
    """
    style = config or PlotConfig()
    graph = _build_graph(network)
    fig, ax = _prepare_axes_2d(ax=ax, figsize=style.figsize)
    positions = _compute_layout(graph, dimensions=2, seed=seed)
    directions = _compute_axis_directions(graph, positions, dimensions=2)
    scale = _compute_scale(len(graph.nodes))
    _draw_2d(
        ax=ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=_resolve_flag(show_tensor_labels, style.show_tensor_labels),
        show_index_labels=_resolve_flag(show_index_labels, style.show_index_labels),
        config=style,
        scale=scale,
    )
    return fig, ax


def plot_tensorkrowch_network_3d(
    network: Any,
    *,
    ax: Axes | Axes3D | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes3D]:
    """Plot a TensorKrowch tensor network in 3D.

    Args:
        network: TensorKrowch TensorNetwork with nodes and edges.
        ax: Matplotlib 3D axes; if None, creates a new figure with 3D projection.
        config: Styling options; uses defaults if None.
        show_tensor_labels: Override config; None uses config value.
        show_index_labels: Override config; None uses config value.
        seed: Random seed for layout when using force-directed positioning.

    Returns:
        Tuple of (Figure, Axes3D) for further customization.
    """
    style = config or PlotConfig()
    graph = _build_graph(network)
    fig, ax = _prepare_axes_3d(ax=ax, figsize=style.figsize)
    positions = _compute_layout(graph, dimensions=3, seed=seed)
    directions = _compute_axis_directions(graph, positions, dimensions=3)
    scale = _compute_scale(len(graph.nodes))
    _draw_3d(
        ax=ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=_resolve_flag(show_tensor_labels, style.show_tensor_labels),
        show_index_labels=_resolve_flag(show_index_labels, style.show_index_labels),
        config=style,
        scale=scale,
    )
    return fig, ax
