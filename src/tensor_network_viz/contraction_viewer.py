"""Matplotlib playback helpers for stepping through contraction schemes.

The main public integration path is ``show_tensor_network(..., config=PlotConfig(
show_contraction_scheme=True))``. That path reuses this module to attach a slider plus
playback buttons to an existing tensor-network figure without redrawing the scene from
scratch. The standalone viewers remain useful for demos and tests that want the same step
logic on arbitrary Matplotlib artists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._contraction_viewer_style import (
    apply_scheme_2d_highlight_current as _apply_scheme_2d_highlight_current,
)
from ._contraction_viewer_style import (
    apply_scheme_2d_highlight_past as _apply_scheme_2d_highlight_past,
)
from ._contraction_viewer_style import (
    box_poly3d_faces as _box_poly3d_faces,
)
from ._contraction_viewer_style import (
    is_tensor_network_scheme_artist as _is_tensor_network_scheme_artist,
)
from ._contraction_viewer_style import (
    is_tensor_network_scheme_fancy_patch as _is_tensor_network_scheme_fancy_patch,
)
from ._contraction_viewer_style import (
    restore_style as _restore_style,
)
from ._contraction_viewer_style import (
    safe_set_alpha as _safe_set_alpha,
)
from ._contraction_viewer_style import (
    safe_set_color as _safe_set_color,
)
from ._contraction_viewer_style import (
    safe_set_linewidth as _safe_set_linewidth,
)
from ._contraction_viewer_style import (
    safe_set_visible as _safe_set_visible,
)
from ._contraction_viewer_style import (
    snapshot_style as _snapshot_style,
)
from ._contraction_viewer_ui import (
    _PLAYBACK_MAIN_BOTTOM,
)
from ._contraction_viewer_ui import (
    create_playback_buttons as _create_playback_buttons,
)
from ._contraction_viewer_ui import (
    create_playback_details_panel as _create_playback_details_panel,
)
from ._contraction_viewer_ui import (
    create_playback_slider as _create_playback_slider,
)
from ._matplotlib_state import set_contraction_viewer
from ._typing import root_figure
from ._ui_utils import _reserve_figure_bottom, _set_axes_visible, _set_widget_active
from .config import PlotConfig

VisualizerMode = Literal["cumulative", "highlight_current", "window"]
_SchemeAvailability = Literal["not_computed", "computed", "unavailable"]


@dataclass
class _ContractionSchemeBundle:
    """Cached contraction-scheme artists, bounds, metrics, and playback helpers."""

    availability: _SchemeAvailability = "not_computed"
    steps: tuple[frozenset[int], ...] | None = None
    artists_by_step: list[Artist | None] | None = None
    scheme_aabb: list[tuple[float, float, float, float, float, float] | None] | None = None
    metrics_row: tuple[Any | None, ...] | None = None
    step_details: tuple[str | None, ...] | None = None
    viewer: _ContractionViewerBase | None = None
    bounds_2d: tuple[float, float, float, float] | None = None
    bounds_3d: tuple[float, float, float, float, float, float] | None = None


class _ContractionViewerBase:
    """Shared stepping logic for contraction playback viewers.

    The base class only mutates already-created Matplotlib artists. Subclasses decide how
    those artists are created and which axes type they require.
    """

    def __init__(
        self,
        artists: Sequence[Artist | None],
        *,
        fig: Figure,
        ax_main: Axes | Axes3D,
        step_details_by_step: Sequence[str | None] | None = None,
        config: PlotConfig | None = None,
        enable_playback: bool | None = None,
        mode: VisualizerMode = "highlight_current",
        past_color: str | tuple[float, ...] = "0.55",
        current_color: str | tuple[float, ...] = "tab:red",
        future_visible: bool = False,
        past_alpha: float = 0.45,
        current_alpha: float = 1.0,
        future_alpha: float = 0.25,
        past_linewidth: float = 0.6,
        current_linewidth: float = 2.0,
        future_linewidth: float = 0.6,
        window_size: int = 3,
        interval_ms: int = 300,
        loop: bool = False,
        initial_step: int | None = None,
        scheme_2d_highlight_fill_alpha: float = 0.07,
    ) -> None:
        self._artists: list[Artist | None] = list(artists)
        self.figure = fig
        self._ax_main = ax_main
        self.config = config
        self._enable_playback = enable_playback if enable_playback is not None else False
        self.mode = mode
        self.past_color = past_color
        self.current_color = current_color
        self.future_visible = future_visible
        self.past_alpha = past_alpha
        self.current_alpha = current_alpha
        self.future_alpha = future_alpha
        self.past_linewidth = past_linewidth
        self.current_linewidth = current_linewidth
        self.future_linewidth = future_linewidth
        self.window_size = max(1, int(window_size))
        self.interval_ms = int(interval_ms)
        self.loop = loop
        self.scheme_2d_highlight_fill_alpha = float(
            np.clip(scheme_2d_highlight_fill_alpha, 0.0, 1.0)
        )
        self._snapshots: list[dict[str, Any] | None] = [
            _snapshot_style(a) if a is not None else None for a in self._artists
        ]
        details = list(step_details_by_step or ())
        if len(details) < len(self._artists):
            details.extend([None] * (len(self._artists) - len(details)))
        self._step_details_by_step: tuple[str | None, ...] = tuple(details[: len(self._artists)])
        self._details_enabled: bool = False

        init_step = initial_step if initial_step is not None else len(self._artists)
        self.current_step: int = int(np.clip(init_step, 0, len(self._artists)))
        self._initial_step = initial_step

        self.slider: Slider | None = None
        self._btn_play: Button | None = None
        self._btn_pause: Button | None = None
        self._btn_reset: Button | None = None
        self._timer: Any = None
        self._is_playing: bool = False
        self._slider_callback_guard: bool = False
        self._ui_built: bool = False
        self._playback_widgets_visible: bool = False
        self._cid_close: int | None = None
        self._cost_panel_ax: Axes | None = None
        self._cost_text_artist: Text | None = None
        self._step_changed_callbacks: list[Callable[[int], None]] = []

    @property
    def num_steps(self) -> int:
        return len(self._artists)

    def _coerce_step(self, k: int | float) -> int:
        return int(np.clip(round(float(k)), 0, self.num_steps))

    def _restore_base_style(self, i: int, artist: Artist) -> None:
        snap = self._snapshots[i]
        if snap:
            _restore_style(artist, snap)

    def _apply_future_style(self, artist: Artist) -> None:
        if self.future_visible:
            _safe_set_visible(artist, True)
            _safe_set_alpha(artist, self.future_alpha)
            _safe_set_color(artist, self.past_color)
            _safe_set_linewidth(artist, self.future_linewidth)
        else:
            _safe_set_visible(artist, False)

    def _apply_past_style(self, artist: Artist) -> None:
        _safe_set_visible(artist, True)
        _safe_set_alpha(artist, self.past_alpha)
        _safe_set_color(artist, self.past_color)
        _safe_set_linewidth(artist, self.past_linewidth)

    def _apply_current_style(self, artist: Artist) -> None:
        _safe_set_visible(artist, True)
        _safe_set_alpha(artist, self.current_alpha)
        _safe_set_color(artist, self.current_color)
        _safe_set_linewidth(artist, self.current_linewidth)

    def _apply_step_visuals(self, k: int) -> None:
        """Update artists for step ``k`` without recreating geometry."""
        mode = self.mode
        wsz = self.window_size
        for i, art in enumerate(self._artists):
            if art is None:
                continue
            if mode == "cumulative":
                if i < k:
                    _safe_set_visible(art, True)
                    self._restore_base_style(i, art)
                else:
                    self._apply_future_style(art)
            elif mode == "highlight_current":
                if _is_tensor_network_scheme_fancy_patch(art):
                    if i < k - 1:
                        _apply_scheme_2d_highlight_past(art)
                    elif i == k - 1 and k > 0:
                        _apply_scheme_2d_highlight_current(
                            art,
                            accent=self.current_color,
                            fill_alpha=self.scheme_2d_highlight_fill_alpha,
                            edge_alpha=self.current_alpha,
                            linewidth=self.current_linewidth,
                        )
                    else:
                        self._apply_future_style(art)
                elif _is_tensor_network_scheme_artist(art):
                    if i < k - 1:
                        _safe_set_visible(art, False)
                    elif i == k - 1 and k > 0:
                        self._apply_current_style(art)
                    else:
                        self._apply_future_style(art)
                elif i < k - 1:
                    self._apply_past_style(art)
                elif i == k - 1 and k > 0:
                    self._apply_current_style(art)
                else:
                    self._apply_future_style(art)
            elif mode == "window":
                lo = k - wsz
                if lo <= i < k:
                    _safe_set_visible(art, True)
                    self._restore_base_style(i, art)
                    if i == k - 1 and k > 0:
                        _safe_set_alpha(art, self.current_alpha)
                        _safe_set_color(art, self.current_color)
                        _safe_set_linewidth(art, self.current_linewidth)
                else:
                    self._apply_future_style(art)
            else:
                self._restore_base_style(i, art)

    def set_step(self, k: int | float) -> None:
        """Clamp *k* to ``[0, num_steps]``, update artists and optional slider."""
        k_clamped = self._coerce_step(k)
        self.current_step = k_clamped
        self._apply_step_visuals(k_clamped)

        if self.slider is not None and not self._slider_callback_guard:
            try:
                if float(self.slider.val) != float(k_clamped):
                    self._slider_callback_guard = True
                    self.slider.set_val(float(k_clamped))
            finally:
                self._slider_callback_guard = False

        self._refresh_step_details_panel()
        for callback in tuple(self._step_changed_callbacks):
            callback(k_clamped)
        self.figure.canvas.draw_idle()

    def add_step_changed_callback(
        self,
        callback: Callable[[int], None],
        *,
        call_immediately: bool = False,
    ) -> None:
        if callback not in self._step_changed_callbacks:
            self._step_changed_callbacks.append(callback)
        if call_immediately:
            callback(self.current_step)

    def remove_step_changed_callback(self, callback: Callable[[int], None]) -> None:
        if callback in self._step_changed_callbacks:
            self._step_changed_callbacks.remove(callback)

    def set_step_details_enabled(self, enabled: bool) -> None:
        self._details_enabled = bool(enabled)
        self._refresh_step_details_panel()

    def _build_step_details_panel(self) -> None:
        if self._cost_panel_ax is not None:
            return
        ax_details, text = _create_playback_details_panel(self.figure)
        self._cost_panel_ax = ax_details
        self._cost_text_artist = text
        _set_axes_visible(ax_details, False)
        text.set_visible(False)

    def _current_step_details_text(self) -> str | None:
        if (
            not self._details_enabled
            or not self._playback_widgets_visible
            or self.current_step <= 0
        ):
            return None
        step_index = self.current_step - 1
        if step_index >= len(self._step_details_by_step):
            return None
        return self._step_details_by_step[step_index]

    def _refresh_step_details_panel(self) -> None:
        if self._cost_panel_ax is None or self._cost_text_artist is None:
            return
        detail_text = self._current_step_details_text()
        visible = bool(detail_text)
        self._cost_text_artist.set_text(detail_text or "")
        self._cost_text_artist.set_visible(visible)
        _set_axes_visible(self._cost_panel_ax, visible)

    def build_ui(self, *, initialize_step: bool = True) -> None:
        """Create slider and buttons on 2D axes; no-op if ``enable_playback`` is False."""
        if not self._enable_playback or self._ui_built:
            return

        n = self.num_steps
        _reserve_figure_bottom(self.figure, _PLAYBACK_MAIN_BOTTOM)
        self._build_step_details_panel()
        slider = _create_playback_slider(
            self.figure,
            num_steps=n,
            initial_step=int(self._initial_step if self._initial_step is not None else n),
        )
        self.slider = slider

        btn_play, btn_pause, btn_reset = _create_playback_buttons(self.figure)
        self._btn_play = btn_play
        self._btn_pause = btn_pause
        self._btn_reset = btn_reset

        def _on_slider_change(_: float) -> None:
            if self._slider_callback_guard or self.slider is None:
                return
            self.set_step(float(self.slider.val))

        slider.on_changed(_on_slider_change)
        btn_play.on_clicked(lambda _e: self.play())
        btn_pause.on_clicked(lambda _e: self.pause())
        btn_reset.on_clicked(lambda _e: self.reset())

        def _on_close(_: Any) -> None:
            self.pause()
            if self._cid_close is not None:
                self.figure.canvas.mpl_disconnect(self._cid_close)
                self._cid_close = None

        self._cid_close = self.figure.canvas.mpl_connect("close_event", _on_close)

        self._timer = self.figure.canvas.new_timer(interval=self.interval_ms)
        self._timer.add_callback(self._tick_playback)

        self._ui_built = True
        self.set_playback_widgets_visible(True)
        if initialize_step:
            init = self._initial_step if self._initial_step is not None else n
            self.set_step(init)

    def _tick_playback(self) -> None:
        if not self._is_playing:
            return
        n = self.num_steps
        nxt = self.current_step + 1
        if nxt > n:
            if self.loop:
                nxt = 0
            else:
                self.pause()
                return
        self.set_step(nxt)

    def play(self) -> None:
        if not self._enable_playback or self._timer is None:
            return
        if self._is_playing:
            return
        self._is_playing = True
        self._timer.start()

    def pause(self) -> None:
        self._is_playing = False
        if self._timer is not None:
            self._timer.stop()

    def reset(self) -> None:
        self.pause()
        self.set_step(0)

    def set_playback_widgets_visible(self, visible: bool) -> None:
        if not self._ui_built:
            return
        widget_pairs = [
            (self.slider, self.slider.ax) if self.slider is not None else None,
            (self._btn_play, self._btn_play.ax) if self._btn_play is not None else None,
            (self._btn_pause, self._btn_pause.ax) if self._btn_pause is not None else None,
            (self._btn_reset, self._btn_reset.ax) if self._btn_reset is not None else None,
        ]
        for pair in widget_pairs:
            if pair is None:
                continue
            widget, widget_ax = pair
            if not visible:
                mouse_grabber = getattr(self.figure.canvas, "mouse_grabber", None)
                if mouse_grabber is widget_ax:
                    with suppress(AttributeError, RuntimeError, TypeError, ValueError):
                        self.figure.canvas.release_mouse(widget_ax)
                slider = self.slider
                if slider is not None and slider is widget:
                    slider.drag_active = False
            _set_widget_active(widget, visible)
            _set_axes_visible(widget_ax, visible)
        self._playback_widgets_visible = visible
        self._refresh_step_details_panel()
        self.figure.canvas.draw_idle()

    def show_static_scheme(self) -> None:
        for i, art in enumerate(self._artists):
            if art is None:
                continue
            _safe_set_visible(art, True)
            self._restore_base_style(i, art)
        self.figure.canvas.draw_idle()

    def hide_scheme_artists(self) -> None:
        for art in self._artists:
            if art is None:
                continue
            _safe_set_visible(art, False)
        self.figure.canvas.draw_idle()

    def show_dynamic_scheme(self) -> None:
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        """Standalone demos; in-library use relies on ``show_tensor_network`` / ``_show_figure``."""
        from .viewer import _show_figure

        self.build_ui()
        _show_figure(self.figure)


class ContractionViewer2D(_ContractionViewerBase):
    """Playback viewer for 2D Matplotlib artists such as patches and rectangles."""

    def __init__(
        self,
        artists: Sequence[Artist | None],
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        config: PlotConfig | None = None,
        enable_playback: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if fig is None and ax is not None:
            fig = root_figure(ax.figure)
        if fig is None:
            fig, ax_new = plt.subplots()
            ax = cast(Axes, ax_new)
        assert ax is not None
        if getattr(ax, "name", None) == "3d":
            raise TypeError("ContractionViewer2D requires a 2D Axes")
        super().__init__(
            artists,
            fig=fig,
            ax_main=ax,
            config=config,
            enable_playback=enable_playback,
            **kwargs,
        )

    @classmethod
    def from_rectangles(
        cls,
        rectangles: Sequence[tuple[float, float, float, float]],
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        facecolor: str | tuple[float, ...] = "powderblue",
        edgecolor: str | tuple[float, ...] = "navy",
        **kwargs: Any,
    ) -> ContractionViewer2D:
        """Create a 2D viewer from ``(x, y, width, height)`` rectangle tuples."""
        if fig is None and ax is not None:
            fig = root_figure(ax.figure)
        if fig is None:
            fig, ax_new = plt.subplots()
            ax = cast(Axes, ax_new)
        assert ax is not None
        artists: list[Rectangle] = []
        for x, y, w, h in rectangles:
            r = Rectangle(
                (x, y),
                w,
                h,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
            )
            ax.add_patch(r)
            artists.append(r)
        return cls(artists, fig=fig, ax=ax, **kwargs)


class ContractionViewer3D(_ContractionViewerBase):
    """Playback viewer for 3D Matplotlib collections."""

    def __init__(
        self,
        artists: Sequence[Artist | None],
        *,
        fig: Figure | None = None,
        ax: Axes3D | None = None,
        config: PlotConfig | None = None,
        enable_playback: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if fig is None and ax is not None:
            fig = root_figure(ax.figure)
        if fig is None:
            fig = plt.figure()
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        assert ax is not None
        if not hasattr(ax, "zaxis"):
            raise TypeError("ContractionViewer3D requires Axes3D")
        super().__init__(
            artists,
            fig=fig,
            ax_main=ax,
            config=config,
            enable_playback=enable_playback,
            **kwargs,
        )

    @classmethod
    def from_boxes(
        cls,
        boxes: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        fig: Figure | None = None,
        ax: Axes3D | None = None,
        facecolor: str | tuple[float, ...] = (0.2, 0.5, 0.9, 0.15),
        edgecolor: str | tuple[float, ...] = "0.2",
        **kwargs: Any,
    ) -> ContractionViewer3D:
        """Create a 3D viewer from axis-aligned ``(min_corner, max_corner)`` boxes."""
        if fig is None and ax is not None:
            fig = root_figure(ax.figure)
        if fig is None:
            fig = plt.figure()
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        assert ax is not None
        artists: list[Poly3DCollection] = []
        for mn, mx in boxes:
            mn = np.asarray(mn, dtype=float).ravel()[:3]
            mx = np.asarray(mx, dtype=float).ravel()[:3]
            verts = _box_poly3d_faces(
                float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2])
            )
            poly = Poly3DCollection(
                verts,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=0.8,
            )
            ax.add_collection3d(poly)
            artists.append(poly)
        return cls(artists, fig=fig, ax=ax, **kwargs)


class _SceneStepApplier(Protocol):
    """Protocol for scene objects that can redraw themselves for one playback step."""

    def apply_step(self, step: int) -> None: ...
    def set_enabled(self, enabled: bool) -> None: ...


class _TensorNetworkContractionViewer(_ContractionViewerBase):
    """Viewer adapter that drives tensor-network scene playback without standalone artists."""

    def __init__(
        self,
        *,
        step_count: int,
        scene_applier: _SceneStepApplier,
        fig: Figure,
        ax: Axes | Axes3D,
        step_details_by_step: Sequence[str | None] | None,
        config: PlotConfig,
    ) -> None:
        super().__init__(
            [None] * int(step_count),
            fig=fig,
            ax_main=ax,
            step_details_by_step=step_details_by_step,
            config=config,
            enable_playback=True,
        )
        self._scene_applier = scene_applier
        self.add_step_changed_callback(self._scene_applier.apply_step, call_immediately=False)

    def hide_scheme_artists(self) -> None:
        self._scene_applier.set_enabled(False)
        self.figure.canvas.draw_idle()

    def show_dynamic_scheme(self) -> None:
        self._scene_applier.set_enabled(True)
        self._scene_applier.apply_step(self.current_step)
        self.figure.canvas.draw_idle()


def attach_playback_to_tensor_network_figure(
    *,
    artists_by_step: Sequence[Artist | None],
    step_details_by_step: Sequence[str | None] | None = None,
    fig: Figure,
    ax: Axes | Axes3D,
    config: PlotConfig,
    build_ui: bool = True,
) -> _ContractionViewerBase:
    """Attach playback UI to the figure produced by ``show_tensor_network``."""
    if getattr(ax, "name", None) == "3d":
        v: _ContractionViewerBase = ContractionViewer3D(
            artists_by_step,
            fig=fig,
            ax=cast(Axes3D, ax),
            step_details_by_step=step_details_by_step,
            config=config,
            enable_playback=True,
        )
    else:
        v = ContractionViewer2D(
            artists_by_step,
            fig=fig,
            ax=cast(Axes, ax),
            step_details_by_step=step_details_by_step,
            config=config,
            enable_playback=True,
        )
    if build_ui:
        v.build_ui()
    set_contraction_viewer(fig, v)
    return v


def attach_tensor_network_playback_to_figure(
    *,
    step_count: int,
    scene_applier: _SceneStepApplier,
    step_details_by_step: Sequence[str | None] | None,
    fig: Figure,
    ax: Axes | Axes3D,
    config: PlotConfig,
    build_ui: bool = True,
) -> _ContractionViewerBase:
    viewer = _TensorNetworkContractionViewer(
        step_count=step_count,
        scene_applier=scene_applier,
        fig=fig,
        ax=ax,
        step_details_by_step=step_details_by_step,
        config=config,
    )
    if build_ui:
        viewer.build_ui()
    set_contraction_viewer(fig, viewer)
    return viewer


__all__ = [
    "ContractionViewer2D",
    "ContractionViewer3D",
]
