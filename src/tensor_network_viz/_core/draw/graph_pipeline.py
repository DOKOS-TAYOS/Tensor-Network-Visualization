from __future__ import annotations

from typing import Any, Literal

from ...config import PlotConfig
from ...contraction_viewer import (
    _ContractionControls,
    _ContractionSchemeBundle,
    attach_playback_to_tensor_network_figure,
)
from ...einsum_module.contraction_cost import format_contraction_step_tooltip
from ..contractions import _ContractionGroups
from ..graph import _GraphData
from ..layout import AxisDirections, NodePositions
from .contraction_scheme import (
    _contraction_step_metrics_for_draw,
    _draw_contraction_scheme,
    _effective_contraction_steps,
)
from .disk_metrics import _tensor_disk_radius_px_3d_nominal
from .render_prep import (
    _apply_render_hover_state,
    _build_interactive_scene_state,
    _draw_edges_nodes_and_labels,
    _prepare_render_context,
    _register_render_hover,
    _RenderPrepContext,
)


def _has_contraction_scheme_source(
    graph: _GraphData,
    config: PlotConfig,
) -> bool:
    if config.contraction_scheme_by_name is not None:
        return len(config.contraction_scheme_by_name) > 0
    steps = graph.contraction_steps
    return steps is not None and len(steps) > 0


def _scheme_bounds_2d(
    artists_by_step: list[Any | None],
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for artist in artists_by_step:
        if artist is None:
            continue
        get_x = getattr(artist, "get_x", None)
        get_y = getattr(artist, "get_y", None)
        get_width = getattr(artist, "get_width", None)
        get_height = getattr(artist, "get_height", None)
        if not all(callable(fn) for fn in (get_x, get_y, get_width, get_height)):
            continue
        x0 = float(get_x())
        y0 = float(get_y())
        x1 = x0 + float(get_width())
        y1 = y0 + float(get_height())
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def _scheme_bounds_3d(
    scheme_aabb: list[tuple[float, float, float, float, float, float] | None],
) -> tuple[float, float, float, float, float, float] | None:
    boxes = [box for box in scheme_aabb if box is not None]
    if not boxes:
        return None
    xmin = min(box[0] for box in boxes)
    xmax = max(box[1] for box in boxes)
    ymin = min(box[2] for box in boxes)
    ymax = max(box[3] for box in boxes)
    zmin = min(box[4] for box in boxes)
    zmax = max(box[5] for box in boxes)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def _expand_axes_for_scheme_bounds(
    *,
    ax: Any,
    dimensions: Literal[2, 3],
    bounds_2d: tuple[float, float, float, float] | None,
    bounds_3d: tuple[float, float, float, float, float, float] | None,
) -> None:
    if dimensions == 2 and bounds_2d is not None:
        xmin, xmax, ymin, ymax = bounds_2d
        cur_x0, cur_x1 = ax.get_xlim()
        cur_y0, cur_y1 = ax.get_ylim()
        ax.set_xlim(min(float(cur_x0), xmin), max(float(cur_x1), xmax))
        ax.set_ylim(min(float(cur_y0), ymin), max(float(cur_y1), ymax))
        return
    if dimensions == 3 and bounds_3d is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds_3d
        cur_x0, cur_x1 = ax.get_xlim3d()
        cur_y0, cur_y1 = ax.get_ylim3d()
        cur_z0, cur_z1 = ax.get_zlim3d()
        ax.set_xlim3d(min(float(cur_x0), xmin), max(float(cur_x1), xmax))
        ax.set_ylim3d(min(float(cur_y0), ymin), max(float(cur_y1), ymax))
        ax.set_zlim3d(min(float(cur_z0), zmin), max(float(cur_z1), zmax))


def _build_contraction_scheme_bundle(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float,
    context: _RenderPrepContext,
    strict: bool,
) -> _ContractionSchemeBundle:
    scheme_steps_eff = _effective_contraction_steps(graph, config)
    if not scheme_steps_eff:
        raise ValueError("contraction scheme requires a non-empty contraction step sequence.")

    per_step_artists, scheme_aabb = _draw_contraction_scheme(
        ax=ax,
        graph=graph,
        positions=positions,
        steps=scheme_steps_eff,
        config=config,
        dimensions=dimensions,
        scale=scale,
        p=context.params,
    )
    has_drawable_artists = any(artist is not None for artist in per_step_artists)
    if strict and not has_drawable_artists:
        raise ValueError("contraction scheme requires at least one drawable contraction step.")

    metrics_row = _contraction_step_metrics_for_draw(graph, scheme_steps_eff)
    tooltips = tuple(
        (
            format_contraction_step_tooltip(metrics_row[index])
            if (
                metrics_row is not None
                and index < len(metrics_row)
                and metrics_row[index] is not None
            )
            else None
        )
        for index in range(len(per_step_artists))
    )
    viewer = attach_playback_to_tensor_network_figure(
        artists_by_step=per_step_artists,
        fig=ax.figure,
        ax=ax,
        config=config,
        build_ui=False,
    )
    bounds_2d = _scheme_bounds_2d(per_step_artists) if dimensions == 2 else None
    bounds_3d = _scheme_bounds_3d(scheme_aabb) if dimensions == 3 else None
    _expand_axes_for_scheme_bounds(
        ax=ax,
        dimensions=dimensions,
        bounds_2d=bounds_2d,
        bounds_3d=bounds_3d,
    )
    return _ContractionSchemeBundle(
        availability="computed",
        steps=scheme_steps_eff,
        artists_by_step=per_step_artists,
        scheme_aabb=scheme_aabb,
        metrics_row=tuple(metrics_row) if metrics_row is not None else None,
        tooltips=tooltips,
        viewer=viewer,
        bounds_2d=bounds_2d,
        bounds_3d=bounds_3d,
    )


def _draw_graph(
    *,
    ax: Any,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
    config: PlotConfig,
    dimensions: Literal[2, 3],
    scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
    bond_curve_pad: float | None = None,
    build_contraction_controls: bool = True,
    contraction_controls_build_ui: bool = True,
    register_contraction_controls_on_figure: bool = True,
) -> None:
    context = _prepare_render_context(
        ax=ax,
        graph=graph,
        positions=positions,
        config=config,
        dimensions=dimensions,
        scale=scale,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        contraction_groups=contraction_groups,
        bond_curve_pad=bond_curve_pad,
    )

    if config.contraction_playback and not config.show_contraction_scheme:
        raise ValueError(
            "contraction_playback=True requires show_contraction_scheme=True so contraction "
            "steps can be drawn and stepped."
        )

    tensor_disk_radius_px_3d: float | None = None
    if dimensions == 3 and config.approximate_3d_tensor_disk_px:
        tensor_disk_radius_px_3d = _tensor_disk_radius_px_3d_nominal(ax, context.params)

    _draw_edges_nodes_and_labels(
        ax=ax,
        context=context,
        directions=directions,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )

    hover_state = _register_render_hover(
        ax=ax,
        context=context,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        scheme_patches_2d=[],
        scheme_aabbs_3d=[],
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )
    controls: _ContractionControls | None = None
    if _has_contraction_scheme_source(graph, config) and build_contraction_controls:
        controls = _ContractionControls(
            fig=ax.figure,
            ax=ax,
            config=config,
            build_controls=contraction_controls_build_ui,
            register_on_figure=register_contraction_controls_on_figure,
            bundle_builder=lambda strict: _build_contraction_scheme_bundle(
                ax=ax,
                graph=graph,
                positions=positions,
                config=config,
                dimensions=dimensions,
                scale=scale,
                context=context,
                strict=strict,
            ),
            refresh_hover=lambda scheme_patches_2d, scheme_aabbs_3d: _apply_render_hover_state(
                hover_state,
                scheme_patches_2d=scheme_patches_2d,
                scheme_aabbs_3d=scheme_aabbs_3d,
            ),
        )
    scene = _build_interactive_scene_state(
        ax=ax,
        context=context,
        directions=directions,
        scale=scale,
        hover_state=hover_state,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )
    scene.contraction_controls = controls
    ax._tensor_network_viz_scene = scene  # type: ignore[attr-defined]
    if controls is not None:
        ax._tensor_network_viz_contraction_controls = controls  # type: ignore[attr-defined]
    if not _has_contraction_scheme_source(graph, config):
        return


__all__ = ["_draw_graph"]
