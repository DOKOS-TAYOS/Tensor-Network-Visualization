from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ...config import PlotConfig
from ..contractions import _ContractionGroups
from ..graph import _GraphData
from ..layout import AxisDirections, NodePositions
from .disk_metrics import _tensor_disk_radius_px_3d_nominal
from .render_prep import (
    _draw_edges_nodes_and_labels,
    _prepare_render_context,
    _register_render_hover,
    _RenderPrepContext,
)

if TYPE_CHECKING:
    from ..._interaction.scheme import _ContractionSchemeBundle


def _refresh_contraction_hover(
    *,
    ax: Any,
    hover_state: Any,
) -> None:
    from ..._interactive_scene import _apply_scene_hover_state
    from ..._matplotlib_state import get_scene
    from .render_prep import _apply_render_hover_state
    from .scene_state import _InteractiveSceneState

    scene = get_scene(ax)
    if isinstance(scene, _InteractiveSceneState):
        hover_on = bool(scene.hover_state.tensor_hover or scene.hover_state.edge_hover)
        _apply_scene_hover_state(scene, hover_on=hover_on)
        return
    _apply_render_hover_state(
        hover_state,
        scheme_patches_2d=(),
        scheme_aabbs_3d=(),
    )


def _has_contraction_scheme_source(
    graph: _GraphData,
    config: PlotConfig,
) -> bool:
    if config.contraction_scheme_by_name is not None:
        return len(config.contraction_scheme_by_name) > 0
    steps = graph.contraction_steps
    return steps is not None and len(steps) > 0


def _effective_contraction_steps(
    graph: _GraphData,
    config: PlotConfig,
) -> Any:
    from .contraction_scheme import _effective_contraction_steps as _impl

    return _impl(graph, config)


def _build_contraction_playback_states(
    *,
    graph: _GraphData,
    steps: Any,
    config: PlotConfig,
) -> Any:
    from .contraction_scheme import _build_contraction_playback_states as _impl

    return _impl(graph=graph, steps=steps, config=config)


def _contraction_step_metrics_for_draw(
    graph: _GraphData,
    steps: Any,
) -> Any:
    from .contraction_scheme import _contraction_step_metrics_for_draw as _impl

    return _impl(graph, steps)


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
    include_viewer: bool = True,
) -> _ContractionSchemeBundle:
    from ..._interaction.scheme import _ContractionSchemeBundle
    from ...contraction_viewer import attach_tensor_network_playback_to_figure
    from ...einsum_module.contraction_cost import format_contraction_step_panel_text
    from .contraction_scheme import _ContractionSceneApplier

    scheme_steps_eff = _effective_contraction_steps(graph, config)
    if not scheme_steps_eff:
        raise ValueError("contraction scheme requires a non-empty contraction step sequence.")
    playback_states = _build_contraction_playback_states(
        graph=graph,
        steps=scheme_steps_eff,
        config=config,
    )
    if strict and len(playback_states) <= 1:
        raise ValueError("contraction scheme requires at least one drawable contraction step.")

    metrics_row = _contraction_step_metrics_for_draw(graph, scheme_steps_eff)
    step_details_list: list[str | None] = []
    for index in range(len(scheme_steps_eff)):
        detail: str | None = None
        if metrics_row is not None and index < len(metrics_row):
            m = metrics_row[index]
            if m is not None:
                detail = format_contraction_step_panel_text(m)
        step_details_list.append(detail)
    step_details = tuple(step_details_list)
    scene_applier = _ContractionSceneApplier(states=playback_states)
    viewer = (
        attach_tensor_network_playback_to_figure(
            step_count=len(scheme_steps_eff),
            scene_applier=scene_applier,
            step_details_by_step=step_details,
            fig=ax.figure,
            ax=ax,
            config=config,
            build_ui=False,
        )
        if include_viewer
        else None
    )
    return _ContractionSchemeBundle(
        availability="computed",
        steps=scheme_steps_eff,
        playback_states=playback_states,
        metrics_row=tuple(metrics_row) if metrics_row is not None else None,
        step_details=step_details,
        viewer=viewer,
        scene_applier=scene_applier,
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
    build_scene_state: bool = True,
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
    controls: Any | None = None
    if _has_contraction_scheme_source(graph, config):
        if build_contraction_controls:
            from ..._interaction.scheme import _ContractionControls

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
                refresh_hover=lambda scheme_patches_2d, scheme_aabbs_3d: _refresh_contraction_hover(
                    ax=ax,
                    hover_state=hover_state,
                ),
            )
        elif config.show_contraction_scheme:
            _build_contraction_scheme_bundle(
                ax=ax,
                graph=graph,
                positions=positions,
                config=config,
                dimensions=dimensions,
                scale=scale,
                context=context,
                strict=False,
                include_viewer=False,
            )
    if build_scene_state:
        from ..._matplotlib_state import set_contraction_controls, set_scene
        from .render_prep import _build_interactive_scene_state

        scene = _build_interactive_scene_state(
            ax=ax,
            context=context,
            directions=directions,
            scale=scale,
            hover_state=hover_state,
            tensor_disk_radius_px_3d=tensor_disk_radius_px_3d,
        )
        scene.contraction_controls = controls
        set_scene(ax, scene)
        if controls is not None:
            set_contraction_controls(ax, controls)
            controls.bind_scene(scene)
    else:
        from ..._matplotlib_state import clear_contraction_controls, clear_scene

        clear_scene(ax)
        clear_contraction_controls(ax)
    if not _has_contraction_scheme_source(graph, config):
        return


__all__ = ["_draw_graph"]
