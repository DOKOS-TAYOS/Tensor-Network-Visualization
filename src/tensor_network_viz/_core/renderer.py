"""Shared renderer for normalized tensor network graphs."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from .draw_2d import _draw_2d
from .draw_3d import _draw_3d
from .graph import _GraphData
from .layout import (
    NodePositions,
    _compute_axis_directions,
    _compute_layout,
    _normalize_positions,
)


def _apply_custom_positions(
    graph: _GraphData,
    custom_positions: dict[int, tuple[float, ...]],
    dimensions: int,
    *,
    iterations: int,
    validate: bool = False,
) -> NodePositions:
    """Apply custom positions, using layout for missing nodes, then center and scale."""
    node_ids = list(graph.nodes)
    node_id_set = set(graph.nodes)

    if validate:
        for key, pos in custom_positions.items():
            if key not in node_id_set:
                warnings.warn(
                    f"Custom positions key {key} does not match any node id; ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            elif len(pos) < dimensions:
                warnings.warn(
                    f"Custom position for node {key} has {len(pos)} coords but view "
                    f"requires {dimensions}; missing coords will be zero-filled.",
                    UserWarning,
                    stacklevel=2,
                )
    positions_arr = np.zeros((len(node_ids), dimensions), dtype=float)
    missing: set[int] = set()
    for i, nid in enumerate(node_ids):
        if nid in custom_positions:
            pos = np.array(custom_positions[nid], dtype=float)
            n = min(len(pos), dimensions)
            positions_arr[i, :n] = pos[:n]
        else:
            missing.add(nid)
    if missing:
        fallback = _compute_layout(graph, dimensions=dimensions, seed=0, iterations=iterations)
        for i, nid in enumerate(node_ids):
            if nid in missing:
                positions_arr[i] = fallback[nid]
    positions = {nid: positions_arr[i].copy() for i, nid in enumerate(node_ids)}
    return _normalize_positions(positions, node_ids)


def _resolve_flag(value: bool | None, default: bool) -> bool:
    if value is None:
        return default
    return value


def _count_visible_nodes(graph: _GraphData) -> int:
    visible_nodes = sum(1 for node in graph.nodes.values() if not node.is_virtual)
    return visible_nodes or len(graph.nodes)


_SCALE_SINGLE_NODE: float = 1.2
_SCALE_MIN: float = 0.5
_SCALE_MAX: float = 1.6
_SCALE_BASE: float = 2.2
_SCALE_PER_NODE: float = 0.07


def _compute_scale(n_nodes: int) -> float:
    """Scale factor for visual elements: larger for few nodes, smaller for many."""
    if n_nodes <= 1:
        return _SCALE_SINGLE_NODE
    return max(_SCALE_MIN, min(_SCALE_MAX, _SCALE_BASE - _SCALE_PER_NODE * n_nodes))


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
    iterations = (
        style.layout_iterations
        if style.layout_iterations is not None
        else PlotConfig.DEFAULT_LAYOUT_ITERATIONS
    )
    if style.positions is not None:
        positions = _apply_custom_positions(
            graph,
            style.positions,
            dimensions=2,
            iterations=iterations,
            validate=style.validate_positions,
        )
    else:
        positions = _compute_layout(graph, dimensions=2, seed=seed, iterations=iterations)
    directions = _compute_axis_directions(graph, positions, dimensions=2)
    scale = _compute_scale(_count_visible_nodes(graph))
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
    iterations = (
        style.layout_iterations
        if style.layout_iterations is not None
        else PlotConfig.DEFAULT_LAYOUT_ITERATIONS
    )
    if style.positions is not None:
        positions = _apply_custom_positions(
            graph,
            style.positions,
            dimensions=3,
            iterations=iterations,
            validate=style.validate_positions,
        )
    else:
        positions = _compute_layout(graph, dimensions=3, seed=seed, iterations=iterations)
    directions = _compute_axis_directions(graph, positions, dimensions=3)
    scale = _compute_scale(_count_visible_nodes(graph))
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


def _make_plot_functions(
    build_graph_fn: Callable[[Any], _GraphData],
    renderer_2d_name: str,
    renderer_3d_name: str,
    doc_2d: str,
    doc_3d: str,
) -> tuple[
    Callable[..., tuple[Figure, Axes]],
    Callable[..., tuple[Figure, Axes3D]],
]:
    """Create plot_2d and plot_3d functions for a backend."""

    def plot_2d(
        network: Any,
        *,
        ax: Axes | None = None,
        config: PlotConfig | None = None,
        show_tensor_labels: bool | None = None,
        show_index_labels: bool | None = None,
        seed: int = 0,
    ) -> tuple[Figure, Axes]:
        graph = build_graph_fn(network)
        return _plot_graph_2d(
            graph,
            ax=ax,
            config=config,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
            seed=seed,
            renderer_name=renderer_2d_name,
        )

    def plot_3d(
        network: Any,
        *,
        ax: Axes | Axes3D | None = None,
        config: PlotConfig | None = None,
        show_tensor_labels: bool | None = None,
        show_index_labels: bool | None = None,
        seed: int = 0,
    ) -> tuple[Figure, Axes3D]:
        graph = build_graph_fn(network)
        return _plot_graph_3d(
            graph,
            ax=ax,
            config=config,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
            seed=seed,
            renderer_name=renderer_3d_name,
        )

    plot_2d.__doc__ = doc_2d
    plot_3d.__doc__ = doc_3d
    return plot_2d, plot_3d
