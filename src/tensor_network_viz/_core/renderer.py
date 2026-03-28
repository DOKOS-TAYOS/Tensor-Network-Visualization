"""Shared renderer for normalized tensor network graphs."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from ._draw_common import _draw_graph
from .graph import _GraphData
from .layout import (
    NodePositions,
    _compute_axis_directions,
    _compute_layout,
    _normalize_positions,
)

RenderedAxes: TypeAlias = Axes | Axes3D
_Dimensions = Literal[2, 3]


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
        for key, position in custom_positions.items():
            if key not in node_id_set:
                warnings.warn(
                    f"Custom positions key {key} does not match any node id; ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            elif len(position) < dimensions:
                warnings.warn(
                    f"Custom position for node {key} has {len(position)} coords but view "
                    f"requires {dimensions}; missing coords will be zero-filled.",
                    UserWarning,
                    stacklevel=2,
                )

    positions_array = np.zeros((len(node_ids), dimensions), dtype=float)
    missing_node_ids: set[int] = set()
    for index, node_id in enumerate(node_ids):
        if node_id not in custom_positions:
            missing_node_ids.add(node_id)
            continue
        position = np.array(custom_positions[node_id], dtype=float)
        copy_dimensions = min(len(position), dimensions)
        positions_array[index, :copy_dimensions] = position[:copy_dimensions]

    if missing_node_ids:
        fallback_positions = _compute_layout(
            graph,
            dimensions=dimensions,
            seed=0,
            iterations=iterations,
        )
        for index, node_id in enumerate(node_ids):
            if node_id in missing_node_ids:
                positions_array[index] = fallback_positions[node_id]

    positions = {node_id: positions_array[index].copy() for index, node_id in enumerate(node_ids)}
    return _normalize_positions(positions, node_ids)


def _resolve_flag(value: bool | None, default: bool) -> bool:
    return default if value is None else value


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


_EXTENT_REF_SPAN: float = 2.6
_EXTENT_REF_NN: float = 0.38
_EXTENT_FACTOR_CLAMP: tuple[float, float] = (0.5, 1.3)
_DRAW_SCALE_CLAMP: tuple[float, float] = (0.35, 1.85)


def _visible_tensor_coords(positions: NodePositions, graph: _GraphData) -> np.ndarray:
    """Stack positions of non-virtual (tensor) nodes for layout metrics."""
    visible_ids = [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]
    if not visible_ids:
        values = list(positions.values())
        if not values:
            return np.zeros((0, 1), dtype=float)
        return np.stack(values)
    return np.stack([positions[node_id] for node_id in visible_ids])


def _median_nearest_neighbor_distance(coords: np.ndarray) -> float:
    """Median nearest-neighbor distance; 1.0 when there is at most one point."""
    count = int(coords.shape[0])
    if count < 2:
        return 1.0
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    return float(np.median(nearest))


def _extent_scale_factor(coords: np.ndarray) -> float:
    """Scale multiplier from bbox span and crowding (smaller when spread out or dense)."""
    if coords.size == 0:
        return 1.0
    span_axes = np.ptp(coords, axis=0)
    span_axes = span_axes[np.isfinite(span_axes)]
    span = float(np.max(span_axes)) if span_axes.size else 0.0
    span = max(span, 1e-6)
    f_span: float = _EXTENT_REF_SPAN / span
    d_nn: float = _median_nearest_neighbor_distance(coords)
    f_nn: float = d_nn / _EXTENT_REF_NN
    combined: float = 0.5 * f_span + 0.5 * f_nn
    lo, hi = _EXTENT_FACTOR_CLAMP
    return float(np.clip(combined, lo, hi))


def _resolve_draw_scale(graph: _GraphData, positions: NodePositions) -> float:
    """Combine node-count scale with layout extent / nearest-neighbor crowding."""
    count_scale = _compute_scale(_count_visible_nodes(graph))
    extent_scale = _extent_scale_factor(_visible_tensor_coords(positions, graph))
    product = float(count_scale * extent_scale)
    lo, hi = _DRAW_SCALE_CLAMP
    return float(np.clip(product, lo, hi))


def _prepare_axes(
    ax: RenderedAxes | None,
    *,
    figsize: tuple[float, float] | None,
    renderer_name: str,
    dimensions: _Dimensions,
) -> tuple[Figure | SubFigure, RenderedAxes]:
    if ax is None:
        if dimensions == 2:
            fig, created_ax = plt.subplots(figsize=figsize or (14, 10))
            return fig, created_ax
        fig = plt.figure(figsize=figsize or (14, 10))
        return fig, cast(Axes3D, fig.add_subplot(111, projection="3d"))

    axis_name = getattr(ax, "name", "")
    if dimensions == 2 and axis_name == "3d":
        raise ValueError(f"{renderer_name} requires a 2D Matplotlib axis.")
    if dimensions == 3 and axis_name != "3d":
        raise ValueError(f"{renderer_name} requires a 3D Matplotlib axis.")
    return ax.figure, ax


def _effective_layout_iterations(config: PlotConfig, *, n_nodes: int) -> int:
    """Force-layout iteration count: explicit config wins; else scale down for large n."""
    if config.layout_iterations is not None:
        return config.layout_iterations
    n = max(n_nodes, 1)
    return int(
        min(
            float(PlotConfig.DEFAULT_LAYOUT_ITERATIONS),
            max(45.0, 14.0 * math.sqrt(float(n))),
        )
    )


def _resolve_positions(
    graph: _GraphData,
    config: PlotConfig,
    *,
    dimensions: _Dimensions,
    seed: int,
) -> NodePositions:
    iterations = _effective_layout_iterations(config, n_nodes=len(graph.nodes))
    if config.positions is None:
        return _compute_layout(graph, dimensions=dimensions, seed=seed, iterations=iterations)
    return _apply_custom_positions(
        graph,
        config.positions,
        dimensions=dimensions,
        iterations=iterations,
        validate=config.validate_positions,
    )


def _plot_graph(
    graph: _GraphData,
    *,
    dimensions: _Dimensions,
    ax: RenderedAxes | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
    renderer_name: str,
) -> tuple[Figure | SubFigure, RenderedAxes]:
    style = config or PlotConfig()
    fig, resolved_ax = _prepare_axes(
        ax=ax,
        figsize=style.figsize,
        renderer_name=renderer_name,
        dimensions=dimensions,
    )
    positions = _resolve_positions(graph, style, dimensions=dimensions, seed=seed)
    if dimensions == 2:
        scale = _resolve_draw_scale(graph, positions)
        directions = _compute_axis_directions(
            graph, positions, dimensions=dimensions, draw_scale=scale
        )
    else:
        scale = _compute_scale(_count_visible_nodes(graph))
        directions = _compute_axis_directions(graph, positions, dimensions=dimensions)
    _draw_graph(
        ax=resolved_ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=_resolve_flag(show_tensor_labels, style.show_tensor_labels),
        show_index_labels=_resolve_flag(show_index_labels, style.show_index_labels),
        config=style,
        dimensions=dimensions,
        scale=scale,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return fig, resolved_ax


def _make_plot_functions(
    build_graph_fn: Callable[[Any], _GraphData],
    renderer_2d_name: str,
    renderer_3d_name: str,
    doc_2d: str,
    doc_3d: str,
) -> tuple[
    Callable[..., tuple[Figure | SubFigure, Axes]],
    Callable[..., tuple[Figure | SubFigure, Axes3D]],
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
    ) -> tuple[Figure | SubFigure, Axes]:
        graph = build_graph_fn(network)
        fig, resolved_ax = _plot_graph(
            graph,
            dimensions=2,
            ax=ax,
            config=config,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
            seed=seed,
            renderer_name=renderer_2d_name,
        )
        return fig, cast(Axes, resolved_ax)

    def plot_3d(
        network: Any,
        *,
        ax: Axes | Axes3D | None = None,
        config: PlotConfig | None = None,
        show_tensor_labels: bool | None = None,
        show_index_labels: bool | None = None,
        seed: int = 0,
    ) -> tuple[Figure | SubFigure, Axes3D]:
        graph = build_graph_fn(network)
        fig, resolved_ax = _plot_graph(
            graph,
            dimensions=3,
            ax=ax,
            config=config,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
            seed=seed,
            renderer_name=renderer_3d_name,
        )
        return fig, cast(Axes3D, resolved_ax)

    plot_2d.__doc__ = doc_2d
    plot_3d.__doc__ = doc_3d
    return plot_2d, plot_3d
