"""Shared renderer for normalized tensor network graphs."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from .draw_2d import _draw_2d
from .draw_3d import _draw_3d
from .graph import _GraphData
from .layout import _compute_axis_directions, _compute_layout

NodePositions = dict[int, np.ndarray]


def _apply_custom_positions(
    graph: _GraphData,
    custom_positions: dict[int, tuple[float, ...]],
    dimensions: int,
) -> NodePositions:
    """Apply custom positions, using layout for missing nodes, then center and scale."""
    node_ids = list(graph.nodes)
    positions_arr = np.zeros((len(node_ids), dimensions), dtype=float)
    missing: list[int] = []
    for i, nid in enumerate(node_ids):
        if nid in custom_positions:
            pos = np.array(custom_positions[nid], dtype=float)
            n = min(len(pos), dimensions)
            positions_arr[i, :n] = pos[:n]
        else:
            missing.append(nid)
    if missing:
        fallback = _compute_layout(graph, dimensions=dimensions, seed=0)
        for i, nid in enumerate(node_ids):
            if nid in missing:
                positions_arr[i] = fallback[nid]
    positions_arr -= positions_arr.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(positions_arr, axis=1).max()
    if max_norm > 1e-6:
        positions_arr /= max_norm / 1.6
    return {nid: positions_arr[i].copy() for i, nid in enumerate(node_ids)}


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
    renderer_name: str,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (14, 10))
        return fig, ax

    if getattr(ax, "name", "") == "3d":
        raise ValueError(f"{renderer_name} requires a 2D Matplotlib axis.")
    return ax.figure, ax


def _prepare_axes_3d(
    ax: Axes | Axes3D | None,
    *,
    figsize: tuple[float, float] | None,
    renderer_name: str,
) -> tuple[Figure, Axes3D]:
    if ax is None:
        fig = plt.figure(figsize=figsize or (14, 10))
        created_ax = fig.add_subplot(111, projection="3d")
        return fig, cast(Axes3D, created_ax)

    if getattr(ax, "name", "") != "3d":
        raise ValueError(f"{renderer_name} requires a 3D Matplotlib axis.")
    return ax.figure, cast(Axes3D, ax)


def _plot_graph_2d(
    graph: _GraphData,
    *,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
    renderer_name: str,
) -> tuple[Figure, Axes]:
    style = config or PlotConfig()
    fig, ax = _prepare_axes_2d(ax=ax, figsize=style.figsize, renderer_name=renderer_name)
    if style.positions is not None:
        positions = _apply_custom_positions(graph, style.positions, dimensions=2)
    else:
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
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return fig, ax


def _plot_graph_3d(
    graph: _GraphData,
    *,
    ax: Axes | Axes3D | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
    renderer_name: str,
) -> tuple[Figure, Axes3D]:
    style = config or PlotConfig()
    fig, ax = _prepare_axes_3d(ax=ax, figsize=style.figsize, renderer_name=renderer_name)
    if style.positions is not None:
        positions = _apply_custom_positions(graph, style.positions, dimensions=3)
    else:
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
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return fig, ax
