"""Shared renderer for normalized tensor network graphs."""

from __future__ import annotations

import math
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._logging import package_logger
from .._matplotlib_state import get_reserved_bottom
from .._typing import PositionMapping, root_figure
from ..config import PlotConfig
from .contractions import _ContractionGroups, _group_contractions, _iter_contractions
from .draw.constants import _CURVE_NEAR_PAIR_REF, _CURVE_OFFSET_FACTOR
from .draw.graph_pipeline import _draw_graph
from .focus import filter_graph_for_focus
from .graph import _GraphData
from .graph_cache import _get_or_build_graph
from .layout import (
    AxisDirections,
    NodePositions,
    _analyze_layout_components_cached,
    _compute_axis_directions,
    _compute_layout,
    _compute_layout_from_components,
    _normalize_positions,
)
from .layout_structure import _LayoutComponent

RenderedAxes: TypeAlias = Axes | Axes3D
_Dimensions = Literal[2, 3]
_LayoutCacheKey: TypeAlias = tuple[int, int, int]
# Main axes: use almost the full figure width (interactive widgets sit in the bottom margin).
_FIGURE_ADJUST_LEFT: float = 0.006
_FIGURE_ADJUST_RIGHT: float = 0.994

_layout_positions_by_id: dict[
    int,
    dict[_LayoutCacheKey, tuple[NodePositions, tuple[_LayoutComponent, ...]]],
] = {}
_resolved_geometry_by_id: dict[int, dict[_LayoutCacheKey, _ResolvedGeometry]] = {}


@dataclass(frozen=True)
class _ResolvedGeometry:
    positions: NodePositions
    layout_components: tuple[_LayoutComponent, ...]
    contraction_groups: _ContractionGroups
    scale: float
    bond_curve_pad: float
    directions: AxisDirections


def _layout_positions_bucket(
    graph: _GraphData,
) -> dict[_LayoutCacheKey, tuple[NodePositions, tuple[_LayoutComponent, ...]]]:
    key = id(graph)
    cached = _layout_positions_by_id.get(key)
    if cached is not None:
        return cached

    bucket: dict[_LayoutCacheKey, tuple[NodePositions, tuple[_LayoutComponent, ...]]] = {}
    _layout_positions_by_id[key] = bucket

    def _evict() -> None:
        _layout_positions_by_id.pop(key, None)

    weakref.finalize(graph, _evict)
    return bucket


def _resolved_geometry_bucket(graph: _GraphData) -> dict[_LayoutCacheKey, _ResolvedGeometry]:
    key = id(graph)
    cached = _resolved_geometry_by_id.get(key)
    if cached is not None:
        return cached

    bucket: dict[_LayoutCacheKey, _ResolvedGeometry] = {}
    _resolved_geometry_by_id[key] = bucket

    def _evict() -> None:
        _resolved_geometry_by_id.pop(key, None)

    weakref.finalize(graph, _evict)
    return bucket


def _apply_custom_positions(
    graph: _GraphData,
    custom_positions: PositionMapping,
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

# Node disk radius ≈ this fraction of the shortest **contraction** edge length (center to center),
# when geometry is available. Resolved scale is (fraction * d_min) / DEFAULT_NODE_RADIUS so that
# ``r = scale * node_radius`` matches ``fraction * d_min`` for the default ``node_radius``.
_SHORTEST_EDGE_RADIUS_FRACTION: float = 0.3

_EXTENT_REF_SPAN: float = 2.6
_EXTENT_REF_NN: float = 0.38
_EXTENT_FACTOR_CLAMP: tuple[float, float] = (0.5, 1.3)
_DRAW_SCALE_CLAMP: tuple[float, float] = (0.35, 1.85)


def _compute_scale(n_nodes: int) -> float:
    """Heuristic scale factor when graph geometry does not define a shortest bond length."""
    if n_nodes <= 1:
        return _SCALE_SINGLE_NODE
    return max(_SCALE_MIN, min(_SCALE_MAX, _SCALE_BASE - _SCALE_PER_NODE * n_nodes))


def _min_contraction_edge_length(graph: _GraphData, positions: NodePositions) -> float | None:
    """Shortest center–center distance among non-degenerate contraction edges, or None."""
    best: float | None = None
    for record in _iter_contractions(graph):
        a_id, b_id = record.node_ids
        if a_id == b_id:
            continue
        if a_id not in positions or b_id not in positions:
            continue
        delta = np.asarray(positions[a_id], dtype=float) - np.asarray(positions[b_id], dtype=float)
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-12 or not math.isfinite(dist):
            continue
        best = dist if best is None else min(best, dist)
    return best


def _geometric_draw_scale(d_min: float) -> float:
    """Scale so ``DEFAULT_NODE_RADIUS * scale == fraction * d_min`` (exact for default radius)."""
    return float((_SHORTEST_EDGE_RADIUS_FRACTION * d_min) / PlotConfig.DEFAULT_NODE_RADIUS)


def _heuristic_draw_scale(graph: _GraphData, positions: NodePositions) -> float:
    """Previous node-count × extent heuristic (2D/3D fallback when no contraction bonds exist)."""
    count_scale = _compute_scale(_count_visible_nodes(graph))
    extent_scale = _extent_scale_factor(_visible_tensor_coords(positions, graph))
    product = float(count_scale * extent_scale)
    lo, hi = _DRAW_SCALE_CLAMP
    return float(np.clip(product, lo, hi))


def _visible_tensor_coords(positions: NodePositions, graph: _GraphData) -> np.ndarray:
    """Stack positions of non-virtual (tensor) nodes for layout metrics."""
    visible_ids = [node_id for node_id, node in graph.nodes.items() if not node.is_virtual]
    if not visible_ids:
        values = list(positions.values())
        if not values:
            return np.zeros((0, 1), dtype=float)
        return np.stack(values)
    return np.stack([positions[node_id] for node_id in visible_ids])


_NN_MEDIAN_MAX_POINTS: int = 256


def _median_nearest_neighbor_distance(coords: np.ndarray) -> float:
    """Median nearest-neighbor distance; 1.0 when there is at most one point.

    For more than ``_NN_MEDIAN_MAX_POINTS`` points, uses a fixed-seed subsample so cost stays
    bounded while preserving the extent heuristic behavior.
    """
    count = int(coords.shape[0])
    if count < 2:
        return 1.0
    if count > _NN_MEDIAN_MAX_POINTS:
        rng = np.random.default_rng(0)
        sel = rng.choice(count, size=_NN_MEDIAN_MAX_POINTS, replace=False)
        coords = coords[sel]
        count = _NN_MEDIAN_MAX_POINTS
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
    """Pick draw scale so node radius tracks the shortest contraction edge (see fraction constant).

    Uses center–center distances of contraction edges (two distinct endpoints) only.
    Falls back to the legacy node-count × extent heuristic when that set is empty or degenerate.
    """
    d_min = _min_contraction_edge_length(graph, positions)
    if d_min is not None:
        return _geometric_draw_scale(d_min)
    package_logger.debug(
        "Falling back to heuristic draw scale because no valid contraction-edge length was found."
    )
    return _heuristic_draw_scale(graph, positions)


def _resolve_draw_scale_and_bond_curve_pad(
    graph: _GraphData,
    positions: NodePositions,
    contraction_groups: _ContractionGroups,
) -> tuple[float, float]:
    """Single structural pass over contractions for scale, then one pass for bond bulge padding."""
    records = _iter_contractions(graph)
    d_min: float | None = None
    for record in records:
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        delta = np.asarray(positions[right_id], dtype=float) - np.asarray(
            positions[left_id], dtype=float
        )
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-12 or not math.isfinite(dist):
            continue
        d_min = dist if d_min is None else min(d_min, dist)

    scale = (
        _geometric_draw_scale(d_min)
        if d_min is not None
        else _heuristic_draw_scale(graph, positions)
    )
    if d_min is None:
        package_logger.debug(
            "Falling back to heuristic draw scale and curve padding for renderer graph."
        )

    best_curve = 0.0
    for record in records:
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        offset_index, edge_count = contraction_groups.offsets[id(record.edge)]
        delta = np.asarray(positions[right_id], dtype=float) - np.asarray(
            positions[left_id], dtype=float
        )
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        effective_chord = float(math.hypot(distance, _CURVE_NEAR_PAIR_REF * scale))
        raw = (
            (offset_index - (edge_count - 1) / 2.0) * _CURVE_OFFSET_FACTOR * scale * effective_chord
        )
        best_curve = max(best_curve, abs(float(raw)))
    return scale, best_curve


def _prepare_axes(
    ax: RenderedAxes | None,
    *,
    figsize: tuple[float, float] | None,
    renderer_name: str,
    dimensions: _Dimensions,
) -> tuple[Figure, RenderedAxes]:
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
    return root_figure(ax.figure), ax


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
) -> tuple[NodePositions, tuple[_LayoutComponent, ...]]:
    """Resolve positions and layout components (one structural analysis per plot)."""
    iterations = _effective_layout_iterations(config, n_nodes=len(graph.nodes))
    components = _analyze_layout_components_cached(graph)
    node_ids = list(graph.nodes)
    if len(node_ids) == 1:
        return (
            {node_ids[0]: np.zeros(dimensions, dtype=float)},
            components,
        )
    if config.positions is None:
        cache_key: _LayoutCacheKey = (int(dimensions), int(seed), int(iterations))
        cache_bucket = _layout_positions_bucket(graph)
        cached = cache_bucket.get(cache_key)
        if cached is not None:
            return cached
        resolved = (
            _compute_layout_from_components(
                graph,
                components,
                int(dimensions),
                seed,
                iterations=iterations,
            ),
            components,
        )
        cache_bucket[cache_key] = resolved
        return resolved
    return (
        _apply_custom_positions(
            graph,
            config.positions,
            dimensions=dimensions,
            iterations=iterations,
            validate=config.validate_positions,
        ),
        components,
    )


def _resolve_geometry(
    graph: _GraphData,
    config: PlotConfig,
    *,
    dimensions: _Dimensions,
    seed: int,
) -> _ResolvedGeometry:
    positions, layout_components = _resolve_positions(
        graph,
        config,
        dimensions=dimensions,
        seed=seed,
    )
    if config.positions is not None:
        contraction_groups = _group_contractions(graph)
        scale, bond_curve_pad = _resolve_draw_scale_and_bond_curve_pad(
            graph,
            positions,
            contraction_groups,
        )
        directions = _compute_axis_directions(
            graph,
            positions,
            dimensions=dimensions,
            draw_scale=scale,
            contraction_groups=contraction_groups,
            layout_components=layout_components,
        )
        return _ResolvedGeometry(
            positions=positions,
            layout_components=layout_components,
            contraction_groups=contraction_groups,
            scale=scale,
            bond_curve_pad=bond_curve_pad,
            directions=directions,
        )

    iterations = _effective_layout_iterations(config, n_nodes=len(graph.nodes))
    cache_key: _LayoutCacheKey = (int(dimensions), int(seed), int(iterations))
    cache_bucket = _resolved_geometry_bucket(graph)
    cached = cache_bucket.get(cache_key)
    if cached is not None:
        return cached

    contraction_groups = _group_contractions(graph)
    scale, bond_curve_pad = _resolve_draw_scale_and_bond_curve_pad(
        graph,
        positions,
        contraction_groups,
    )
    directions = _compute_axis_directions(
        graph,
        positions,
        dimensions=dimensions,
        draw_scale=scale,
        contraction_groups=contraction_groups,
        layout_components=layout_components,
    )
    resolved = _ResolvedGeometry(
        positions=positions,
        layout_components=layout_components,
        contraction_groups=contraction_groups,
        scale=scale,
        bond_curve_pad=bond_curve_pad,
        directions=directions,
    )
    cache_bucket[cache_key] = resolved
    return resolved


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
    build_contraction_controls: bool = True,
    contraction_controls_build_ui: bool = True,
    register_contraction_controls_on_figure: bool = True,
    build_scene_state: bool = True,
) -> tuple[Figure, RenderedAxes]:
    style = config or PlotConfig()
    created_figure = ax is None
    fig, resolved_ax = _prepare_axes(
        ax=ax,
        figsize=style.figsize,
        renderer_name=renderer_name,
        dimensions=dimensions,
    )
    geometry = _resolve_geometry(
        graph,
        style,
        dimensions=dimensions,
        seed=seed,
    )
    draw_graph = filter_graph_for_focus(graph, style.focus)
    draw_positions = {
        node_id: geometry.positions[node_id]
        for node_id in draw_graph.nodes
        if node_id in geometry.positions
    }
    draw_directions = {
        (node_id, axis_index): direction
        for (node_id, axis_index), direction in geometry.directions.items()
        if node_id in draw_graph.nodes
    }
    draw_contraction_groups = (
        geometry.contraction_groups if draw_graph is graph else _group_contractions(draw_graph)
    )
    _draw_graph(
        ax=resolved_ax,
        graph=draw_graph,
        positions=draw_positions,
        directions=draw_directions,
        show_tensor_labels=_resolve_flag(show_tensor_labels, style.show_tensor_labels),
        show_index_labels=_resolve_flag(show_index_labels, style.show_index_labels),
        config=style,
        dimensions=dimensions,
        scale=geometry.scale,
        contraction_groups=draw_contraction_groups,
        bond_curve_pad=geometry.bond_curve_pad,
        build_contraction_controls=build_contraction_controls,
        contraction_controls_build_ui=contraction_controls_build_ui,
        register_contraction_controls_on_figure=register_contraction_controls_on_figure,
        build_scene_state=build_scene_state,
    )
    if created_figure:
        reserved_bottom = get_reserved_bottom(fig)
        fig.subplots_adjust(
            left=_FIGURE_ADJUST_LEFT,
            right=_FIGURE_ADJUST_RIGHT,
            bottom=reserved_bottom,
            top=0.98,
        )
    return fig, resolved_ax


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
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
        _build_scene_state: bool = True,
    ) -> tuple[Figure, Axes]:
        graph = _get_or_build_graph(network, build_graph_fn)
        fig, resolved_ax = _plot_graph(
            graph,
            dimensions=2,
            ax=ax,
            config=config,
            seed=seed,
            renderer_name=renderer_2d_name,
            build_contraction_controls=_build_contraction_controls,
            contraction_controls_build_ui=_contraction_controls_build_ui,
            register_contraction_controls_on_figure=_register_contraction_controls_on_figure,
            build_scene_state=_build_scene_state,
        )
        return fig, cast(Axes, resolved_ax)

    def plot_3d(
        network: Any,
        *,
        ax: Axes | Axes3D | None = None,
        config: PlotConfig | None = None,
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
        _build_scene_state: bool = True,
    ) -> tuple[Figure, Axes3D]:
        graph = _get_or_build_graph(network, build_graph_fn)
        fig, resolved_ax = _plot_graph(
            graph,
            dimensions=3,
            ax=ax,
            config=config,
            seed=seed,
            renderer_name=renderer_3d_name,
            build_contraction_controls=_build_contraction_controls,
            contraction_controls_build_ui=_contraction_controls_build_ui,
            register_contraction_controls_on_figure=_register_contraction_controls_on_figure,
            build_scene_state=_build_scene_state,
        )
        return fig, cast(Axes3D, resolved_ax)

    plot_2d.__doc__ = doc_2d
    plot_3d.__doc__ = doc_3d
    return plot_2d, plot_3d
