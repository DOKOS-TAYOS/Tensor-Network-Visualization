from __future__ import annotations

from typing import Any, Literal

from ...config import PlotConfig
from ...contraction_viewer import attach_playback_to_tensor_network_figure
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
    _draw_edges_nodes_and_labels,
    _prepare_render_context,
    _register_render_hover,
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

    per_step_artists: list[Any | None] | None = None
    scheme_steps_eff: tuple[frozenset[int], ...] | None = None
    scheme_aabb: list[tuple[float, float, float, float, float, float] | None] | None = None
    if config.show_contraction_scheme:
        scheme_steps_eff = _effective_contraction_steps(graph, config)
        if scheme_steps_eff:
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
            if config.contraction_playback:
                if not any(artist is not None for artist in per_step_artists):
                    raise ValueError(
                        "contraction_playback requires at least one drawable "
                        "contraction scheme step."
                    )
                attach_playback_to_tensor_network_figure(
                    artists_by_step=per_step_artists,
                    fig=ax.figure,
                    ax=ax,
                    config=config,
                )
        elif config.contraction_playback:
            raise ValueError(
                "contraction_playback requires a non-empty contraction step sequence on the graph."
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

    metrics_row = (
        _contraction_step_metrics_for_draw(graph, scheme_steps_eff) if scheme_steps_eff else None
    )
    scheme_patches_2d: list[tuple[Any, str]] = []
    scheme_aabbs_3d: list[tuple[tuple[float, float, float, float, float, float], str, Any]] = []
    if (
        config.contraction_scheme_cost_hover
        and metrics_row is not None
        and per_step_artists is not None
        and scheme_aabb is not None
    ):
        for index, artist in enumerate(per_step_artists):
            if index >= len(metrics_row):
                break
            metric = metrics_row[index]
            if metric is None or artist is None:
                continue
            tooltip = format_contraction_step_tooltip(metric)
            if dimensions == 2:
                scheme_patches_2d.append((artist, tooltip))
                continue
            box = scheme_aabb[index] if index < len(scheme_aabb) else None
            if box is not None:
                scheme_aabbs_3d.append((box, tooltip, artist))

    _register_render_hover(
        ax=ax,
        context=context,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        scheme_patches_2d=scheme_patches_2d,
        scheme_aabbs_3d=scheme_aabbs_3d,
        tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
    )


__all__ = ["_draw_graph"]
