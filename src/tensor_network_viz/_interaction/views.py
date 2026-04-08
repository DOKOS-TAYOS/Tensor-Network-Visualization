from __future__ import annotations

from collections.abc import Callable
from typing import cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._contraction_viewer_ui import _MAIN_FIGURE_BOTTOM_RESERVED, _PLAYBACK_DETAILS_TOP
from .._core.draw.scene_state import _InteractiveSceneState
from .._interactive_scene import (
    _ensure_scene_label_descriptors,
    _node_mode_from_show_nodes,
    _scene_from_axes,
)
from .._matplotlib_state import get_contraction_controls
from .._typing import root_figure
from .._ui_utils import _set_axes_visible, _set_figure_bottom_reserved
from ..config import ViewName
from .state import InteractiveViewCache

RenderedAxes = Axes | Axes3D

# Manual axes positions: 2D extends slightly below *base*, 3D starts higher (base + lift).
_INTERACTIVE_2D_BOTTOM_EXTRA: float = 0.022
_INTERACTIVE_3D_BOTTOM_LIFT: float = 0.084
_BASE_INTERACTIVE_HEIGHT: float = 0.09
_SCHEME_OFF_FIGURE_BOTTOM_PAD: float = 0.02
_MAIN_FIGURE_BOTTOM_SCHEME_OFF: float = (
    _PLAYBACK_DETAILS_TOP - _BASE_INTERACTIVE_HEIGHT + _SCHEME_OFF_FIGURE_BOTTOM_PAD
)


class _InteractiveViewManager:
    def __init__(
        self,
        *,
        render_view: Callable[[ViewName, RenderedAxes | None], tuple[Figure, RenderedAxes]],
        initial_ax: RenderedAxes | None,
        external_ax: bool,
    ) -> None:
        self._render_view = render_view
        self._initial_ax = initial_ax
        self._external_ax = external_ax
        self.view_caches: dict[ViewName, InteractiveViewCache] = {
            "2d": InteractiveViewCache(view="2d"),
            "3d": InteractiveViewCache(view="3d"),
        }

    def current_scene(self, view: ViewName) -> _InteractiveSceneState | None:
        return self.view_caches[view].scene

    def build_initial_view(
        self,
        view: ViewName,
        *,
        show_nodes: bool,
    ) -> tuple[Figure, RenderedAxes, _InteractiveSceneState | None]:
        fig, rendered_ax = self.build_view(view, ax=self._initial_ax)
        scene = self.current_scene(view)
        if scene is not None and self.scene_requires_node_mode_rerender(
            scene, show_nodes=show_nodes
        ):
            scene = self.rerender_cached_view(view)
            rendered_ax = scene.ax
            fig = root_figure(rendered_ax.figure)
        return fig, rendered_ax, scene

    def ensure_view(
        self,
        view: ViewName,
        *,
        figure: Figure | None,
        show_nodes: bool,
    ) -> tuple[Figure, RenderedAxes, _InteractiveSceneState | None]:
        cache = self.view_caches[view]
        if cache.ax is None or cache.scene is None:
            target_ax: RenderedAxes | None = None
            if figure is not None and not self._external_ax:
                if view == "3d":
                    target_ax = cast(RenderedAxes, figure.add_subplot(111, projection="3d"))
                else:
                    target_ax = cast(RenderedAxes, figure.add_subplot(111))
            fig, rendered_ax = self.build_view(view, ax=target_ax)
            scene = self.current_scene(view)
        else:
            rendered_ax = cache.ax
            fig = root_figure(rendered_ax.figure)
            scene = cache.scene
        if scene is not None and self.scene_requires_node_mode_rerender(
            scene, show_nodes=show_nodes
        ):
            scene = self.rerender_cached_view(view)
            rendered_ax = scene.ax
            fig = root_figure(rendered_ax.figure)
        return fig, rendered_ax, scene

    def scene_requires_node_mode_rerender(
        self,
        scene: _InteractiveSceneState,
        *,
        show_nodes: bool,
    ) -> bool:
        desired_mode = _node_mode_from_show_nodes(show_nodes)
        return (
            scene.dimensions == 2
            and scene.active_node_mode != desired_mode
            and any(edge.kind == "dangling" for edge in scene.graph.edges)
        )

    def rerender_cached_view(self, view: ViewName) -> _InteractiveSceneState:
        cache = self.view_caches[view]
        assert cache.ax is not None
        fig, rendered_ax = self._render_view(view, cache.ax)
        scene = _scene_from_axes(rendered_ax)
        assert scene is not None
        cache.ax = rendered_ax
        cache.scene = scene
        scene.contraction_controls = get_contraction_controls(rendered_ax)
        _ensure_scene_label_descriptors(scene)
        return scene

    def build_view(
        self,
        view: ViewName,
        *,
        ax: RenderedAxes | None,
    ) -> tuple[Figure, RenderedAxes]:
        cache = self.view_caches[view]
        if cache.ax is not None and cache.scene is not None:
            return root_figure(cache.ax.figure), cache.ax
        fig, rendered_ax = self._render_view(view, ax)
        scene = _scene_from_axes(rendered_ax)
        cache.ax = rendered_ax
        cache.scene = scene
        if scene is not None:
            scene.contraction_controls = get_contraction_controls(rendered_ax)
            _ensure_scene_label_descriptors(scene)
        return fig, rendered_ax

    def deactivate_non_current_views(self, current_view: ViewName) -> None:
        for view_name, cache in self.view_caches.items():
            if cache.ax is None:
                continue
            is_current = view_name == current_view
            _set_axes_visible(cache.ax, is_current)
            scene = cache.scene
            if scene is None or scene.contraction_controls is None:
                continue
            viewer = scene.contraction_controls._viewer
            if viewer is not None and not is_current:
                viewer.pause()
                viewer.set_playback_widgets_visible(False)

    def sync_layout(
        self,
        *,
        figure: Figure | None,
        scheme_chrome_on: bool,
    ) -> None:
        if figure is None or self._external_ax:
            return
        base = self._interactive_main_axes_bottom(scheme_chrome_on=scheme_chrome_on)
        lows: list[float] = []
        if self.view_caches["2d"].ax is not None:
            lows.append(base - float(_INTERACTIVE_2D_BOTTOM_EXTRA))
        if self.view_caches["3d"].ax is not None:
            lows.append(base + float(_INTERACTIVE_3D_BOTTOM_LIFT))
        _set_figure_bottom_reserved(figure, min(lows) if lows else base)
        self._sync_data_axes_vertical_layout(figure=figure, scheme_chrome_on=scheme_chrome_on)

    def _interactive_main_axes_bottom(self, *, scheme_chrome_on: bool) -> float:
        return float(
            _MAIN_FIGURE_BOTTOM_RESERVED if scheme_chrome_on else _MAIN_FIGURE_BOTTOM_SCHEME_OFF
        )

    def _shared_data_axes_top(self) -> float:
        ax3 = self.view_caches["3d"].ax
        if ax3 is not None:
            position = ax3.get_position()
            return float(position.y0 + position.height)
        ax2 = self.view_caches["2d"].ax
        if ax2 is not None:
            position = ax2.get_position()
            return float(position.y0 + position.height)
        return 0.9

    def _sync_data_axes_vertical_layout(
        self,
        *,
        figure: Figure,
        scheme_chrome_on: bool,
    ) -> None:
        if self._external_ax:
            return
        base = self._interactive_main_axes_bottom(scheme_chrome_on=scheme_chrome_on)
        top = self._shared_data_axes_top()
        ax2 = self.view_caches["2d"].ax
        ax3 = self.view_caches["3d"].ax
        if ax2 is not None:
            bottom_2d = base - float(_INTERACTIVE_2D_BOTTOM_EXTRA)
            position = ax2.get_position()
            height = max(top - bottom_2d, 0.08)
            ax2.set_position((position.x0, bottom_2d, position.width, height))
        if ax3 is not None:
            bottom_3d = base + float(_INTERACTIVE_3D_BOTTOM_LIFT)
            position = ax3.get_position()
            height = max(top - bottom_3d, 0.08)
            ax3.set_position((position.x0, bottom_3d, position.width, height))


__all__ = ["_InteractiveViewManager"]
