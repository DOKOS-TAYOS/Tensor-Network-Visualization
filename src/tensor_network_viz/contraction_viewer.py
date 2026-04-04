"""Matplotlib-only interactive stepping through ordered contraction visuals.

Primary integration: ``show_tensor_network(..., config=PlotConfig(
show_contraction_scheme=True, contraction_playback=True))`` adds a slider and
Play / Pause / Reset on the same figure (2D widget axes only).

For a richer visual policy, subclass ``ContractionViewer2D`` / ``ContractionViewer3D``
and override ``_apply_step_visuals``. The draw pipeline attaches highlights; this module
only adds matplotlib widgets and mutates existing artists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._typing import root_figure
from ._ui_utils import _reserve_figure_bottom, _set_axes_visible, _set_widget_active
from .config import PlotConfig

VisualizerMode = Literal["cumulative", "highlight_current", "window"]
_SchemeAvailability = Literal["not_computed", "computed", "unavailable"]

# Must stay aligned with ``_CONTRACTION_SCHEME_GID`` in ``_core.draw.contraction_scheme``.
_TNV_CONTRACTION_SCHEME_PATCH_GID: Final[str] = "tnv_contraction_scheme"

_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
_PLAYBACK_MAIN_BOTTOM: float = 0.40
_PLAYBACK_DETAILS_BOUNDS: tuple[float, float, float, float] = (0.25, 0.116, 0.68, 0.12)
_PLAYBACK_SLIDER_BOUNDS: tuple[float, float, float, float] = (0.25, 0.067, 0.33, 0.024)
_PLAYBACK_BUTTON_START_X: float = 0.73
_PLAYBACK_BUTTON_Y: float = 0.058
_PLAYBACK_BUTTON_WIDTH: float = 0.055
_PLAYBACK_BUTTON_HEIGHT: float = 0.038
_PLAYBACK_BUTTON_GAP: float = 0.012
_PLAYBACK_RESET_WIDTH: float = 0.065
_CONTROLS_MAIN_BOTTOM: float = _PLAYBACK_MAIN_BOTTOM
_CONTROLS_CHECKBOX_BOUNDS: tuple[float, float, float, float] = (0.02, 0.045, 0.13, 0.10)
_SCHEME_LABELS: tuple[str, str, str] = ("Scheme", "Playback", "Costs")
_CONTROL_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_CONTROL_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_CONTROL_CHECK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}


@dataclass
class _ContractionSchemeBundle:
    availability: _SchemeAvailability = "not_computed"
    steps: tuple[frozenset[int], ...] | None = None
    artists_by_step: list[Artist | None] | None = None
    scheme_aabb: list[tuple[float, float, float, float, float, float] | None] | None = None
    metrics_row: tuple[Any | None, ...] | None = None
    step_details: tuple[str | None, ...] | None = None
    viewer: _ContractionViewerBase | None = None
    bounds_2d: tuple[float, float, float, float] | None = None
    bounds_3d: tuple[float, float, float, float, float, float] | None = None


def _is_tensor_network_scheme_artist(artist: Artist) -> bool:
    """True for contraction-scheme artists tagged by the draw pipeline."""
    getter = getattr(artist, "get_gid", None)
    if not callable(getter):
        return False
    with suppress(TypeError, ValueError):
        return getter() == _TNV_CONTRACTION_SCHEME_PATCH_GID
    return False


def _is_tensor_network_scheme_fancy_patch(artist: Artist) -> bool:
    """True for 2D contraction-scheme hulls (playback restyle only; static draw unchanged)."""
    return isinstance(artist, FancyBboxPatch) and _is_tensor_network_scheme_artist(artist)


def _apply_scheme_2d_highlight_past(artist: Artist) -> None:
    """Transparent fill and edge (like wire-only 3D: no colored hull)."""
    _safe_set_visible(artist, True)
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc):
        with suppress(TypeError, ValueError):
            setter_fc(_TRANSPARENT)
    setter_ec = getattr(artist, "set_edgecolor", None)
    if callable(setter_ec):
        with suppress(TypeError, ValueError):
            setter_ec(_TRANSPARENT)
    _safe_set_linewidth(artist, 0.0)
    _safe_clear_patch_alpha(artist)


def _apply_scheme_2d_highlight_current(
    artist: Artist,
    *,
    accent: Any,
    fill_alpha: float,
    edge_alpha: float,
    linewidth: float,
) -> None:
    """Very faint tint inside; strong accent on border (aligns with 3D edge emphasis)."""
    r, g, b, _ = to_rgba(accent)
    face = (float(r), float(g), float(b), float(np.clip(fill_alpha, 0.0, 1.0)))
    edge = to_rgba(accent, alpha=float(np.clip(edge_alpha, 0.0, 1.0)))
    _safe_set_visible(artist, True)
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc):
        with suppress(TypeError, ValueError):
            setter_fc(face)
    setter_ec = getattr(artist, "set_edgecolor", None)
    if callable(setter_ec):
        with suppress(TypeError, ValueError):
            setter_ec(edge)
    _safe_set_linewidth(artist, float(linewidth))
    _safe_clear_patch_alpha(artist)


def _safe_set_visible(artist: Artist, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        with suppress(AttributeError, TypeError, ValueError):
            setter(visible)


def _safe_set_alpha(artist: Artist, alpha: float | None) -> None:
    setter = getattr(artist, "set_alpha", None)
    if callable(setter) and alpha is not None:
        with suppress(AttributeError, TypeError, ValueError):
            setter(alpha)


def _safe_clear_patch_alpha(artist: Artist) -> None:
    """So facecolor/edgecolor RGBA alphas are not multiplied by a stale artist alpha."""
    setter = getattr(artist, "set_alpha", None)
    if callable(setter):
        with suppress(AttributeError, TypeError, ValueError):
            setter(None)


def _safe_set_color(artist: Artist, color: Any) -> None:
    for name in ("set_edgecolor", "set_color", "set_facecolor"):
        setter = getattr(artist, name, None)
        if callable(setter):
            try:
                setter(color)
                return
            except (AttributeError, TypeError, ValueError):
                continue


def _safe_set_linewidth(artist: Artist, lw: float) -> None:
    setter = getattr(artist, "set_linewidth", None)
    if callable(setter):
        try:
            setter(lw)
            return
        except (AttributeError, TypeError, ValueError):
            pass
    setter2 = getattr(artist, "set_linewidths", None)
    if callable(setter2):
        with suppress(AttributeError, TypeError, ValueError):
            setter2(lw)


def _snapshot_style(artist: Artist) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for attr, key in (
        ("get_edgecolor", "edgecolor"),
        ("get_facecolor", "facecolor"),
        ("get_color", "color"),
        ("get_linewidth", "linewidth"),
        ("get_linewidths", "linewidths"),
        ("get_alpha", "alpha"),
    ):
        fn = getattr(artist, attr, None)
        if callable(fn):
            with suppress(AttributeError, TypeError, ValueError):
                snap[key] = fn()
    return snap


def _restore_style(artist: Artist, snap: dict[str, Any]) -> None:
    ec = snap.get("edgecolor")
    if ec is not None:
        _safe_set_color(artist, ec)
    fc = snap.get("facecolor")
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc) and fc is not None:
        with suppress(AttributeError, TypeError, ValueError):
            setter_fc(fc)
    col = snap.get("color")
    if col is not None and not hasattr(artist, "set_edgecolor"):
        _safe_set_color(artist, col)
    lw = snap.get("linewidth")
    if lw is not None:
        try:
            _safe_set_linewidth(artist, float(np.ravel(lw)[0]))
        except (TypeError, ValueError, IndexError):
            _safe_set_linewidth(artist, float(lw))  # type: ignore[arg-type]
    else:
        lws = snap.get("linewidths")
        if lws is not None:
            with suppress(TypeError, ValueError, IndexError):
                _safe_set_linewidth(artist, float(np.ravel(lws)[0]))
    al = snap.get("alpha")
    if al is not None:
        try:
            a0 = float(np.ravel(al)[0])
        except (TypeError, ValueError, IndexError):
            try:
                a0 = float(al)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                a0 = None
        if a0 is not None:
            _safe_set_alpha(artist, a0)


def _box_poly3d_faces(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> list:
    return [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]


class _ContractionViewerBase:
    """Shared stepping logic; UI is built only when ``enable_playback`` is True."""

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
        self._enable_playback = (
            enable_playback
            if enable_playback is not None
            else (bool(config.contraction_playback) if config is not None else False)
        )
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

    @property
    def num_steps(self) -> int:
        return len(self._artists)

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

    def set_step(self, k: int) -> None:
        """Clamp *k* to ``[0, num_steps]``, update artists and optional slider."""
        n = self.num_steps
        k_clamped = int(np.clip(k, 0, n))
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
        self.figure.canvas.draw_idle()

    def set_step_details_enabled(self, enabled: bool) -> None:
        self._details_enabled = bool(enabled)
        self._refresh_step_details_panel()

    def _build_step_details_panel(self) -> None:
        if self._cost_panel_ax is not None:
            return
        ax_details = self.figure.add_axes(_PLAYBACK_DETAILS_BOUNDS)
        ax_details.set_xticks([])
        ax_details.set_yticks([])
        ax_details.patch.set_alpha(0.0)
        for spine in ax_details.spines.values():
            spine.set_visible(False)
        text = ax_details.text(
            0.0,
            1.0,
            "",
            transform=ax_details.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            wrap=True,
        )
        self._cost_panel_ax = ax_details
        self._cost_text_artist = text
        _set_axes_visible(ax_details, False)
        text.set_visible(False)

    def _current_step_details_text(self) -> str | None:
        if not self._details_enabled or not self._playback_widgets_visible or self.current_step <= 0:
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
        ax_slider = self.figure.add_axes(_PLAYBACK_SLIDER_BOUNDS)
        slider = Slider(
            ax_slider,
            "Step",
            0,
            float(max(0, n)),
            valinit=float(self._initial_step if self._initial_step is not None else n),
            valstep=1,
        )
        self.slider = slider

        bx = _PLAYBACK_BUTTON_START_X
        by = _PLAYBACK_BUTTON_Y
        bw = _PLAYBACK_BUTTON_WIDTH
        bh = _PLAYBACK_BUTTON_HEIGHT
        gap = _PLAYBACK_BUTTON_GAP
        ax_play = self.figure.add_axes((bx, by, bw, bh))
        ax_pause = self.figure.add_axes((bx + bw + gap, by, bw, bh))
        ax_reset = self.figure.add_axes((bx + 2.0 * (bw + gap), by, _PLAYBACK_RESET_WIDTH, bh))
        btn_play = Button(ax_play, "Play")
        btn_pause = Button(ax_pause, "Pause")
        btn_reset = Button(ax_reset, "Reset")
        self._btn_play = btn_play
        self._btn_pause = btn_pause
        self._btn_reset = btn_reset

        def _on_slider_change(_: float) -> None:
            if self._slider_callback_guard or self.slider is None:
                return
            self.set_step(int(self.slider.val))

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

    def show(self) -> None:
        """Standalone demos; in-library use relies on ``show_tensor_network`` / ``_show_figure``."""
        from .viewer import _show_figure

        self.build_ui()
        _show_figure(self.figure)


class _ContractionControls:
    """Per-figure controller for lazy contraction scheme, playback, and cost-hover toggles."""

    def __init__(
        self,
        *,
        fig: Figure,
        ax: Axes | Axes3D,
        config: PlotConfig,
        build_controls: bool,
        register_on_figure: bool,
        bundle_builder: Callable[[bool], _ContractionSchemeBundle],
        refresh_hover: Callable[
            [
                Sequence[tuple[Artist, str]],
                Sequence[tuple[tuple[float, float, float, float, float, float], str, Artist]],
            ],
            None,
        ],
    ) -> None:
        self.figure = fig
        self.ax = ax
        self.config = config
        self._build_controls_ui = bool(build_controls)
        self._register_on_figure = bool(register_on_figure)
        self._bundle_builder = bundle_builder
        self._refresh_hover_callback = refresh_hover
        self._bundle = _ContractionSchemeBundle()
        self._viewer: _ContractionViewerBase | None = None
        self.scheme_on: bool = bool(config.show_contraction_scheme)
        self.playback_on: bool = bool(config.contraction_playback)
        self.cost_hover_on: bool = bool(config.contraction_scheme_cost_hover)
        if self.cost_hover_on:
            self.scheme_on = True
            self.playback_on = True
        elif self.playback_on:
            self.scheme_on = True
        self._controls_ax: Axes | None = None
        self._checkbuttons: CheckButtons | None = None
        self._checkbuttons_cid: int | None = None
        self._callback_guard: bool = False

        if self._build_controls_ui:
            _reserve_figure_bottom(fig, _CONTROLS_MAIN_BOTTOM)
            self._build_controls()
        if self._register_on_figure:
            fig._tensor_network_viz_contraction_controls = self  # type: ignore[attr-defined]
        ax._tensor_network_viz_contraction_controls = self  # type: ignore[attr-defined]
        if self.scheme_on:
            self._ensure_bundle(strict=self.playback_on, swallow_errors=False)
        self._apply_visual_state()
        self._refresh_hover()

    def _build_controls(self) -> None:
        controls_ax = self.figure.add_axes(_CONTROLS_CHECKBOX_BOUNDS)
        self._controls_ax = controls_ax
        self._checkbuttons = CheckButtons(
            controls_ax,
            list(_SCHEME_LABELS),
            [self.scheme_on, self.playback_on, self.cost_hover_on],
            label_props=_CONTROL_LABEL_PROPS,
            frame_props=_CONTROL_FRAME_PROPS,
            check_props=_CONTROL_CHECK_PROPS,
        )
        self._checkbuttons_cid = self._checkbuttons.on_clicked(self._on_toggle)

    def _set_checkbox_state(self, index: int, value: bool) -> None:
        if self._checkbuttons is None:
            return
        current = bool(self._checkbuttons.get_status()[index])
        if current == value:
            return
        self._callback_guard = True
        try:
            self._checkbuttons.set_active(index, state=value)
        finally:
            self._callback_guard = False

    def _sync_checkbuttons(self) -> None:
        self._set_checkbox_state(0, self.scheme_on)
        self._set_checkbox_state(1, self.playback_on)
        self._set_checkbox_state(2, self.cost_hover_on)

    def _on_toggle(self, label: str | None) -> None:
        if self._callback_guard or self._checkbuttons is None:
            return
        ui_scheme, ui_playback, ui_cost = [bool(v) for v in self._checkbuttons.get_status()]
        self.set_states(
            scheme_on=ui_scheme,
            playback_on=ui_playback,
            cost_hover_on=ui_cost,
            source_label=label,
        )

    def set_states(
        self,
        *,
        scheme_on: bool,
        playback_on: bool,
        cost_hover_on: bool,
        source_label: str | None = None,
    ) -> None:
        prev_scheme = self.scheme_on
        prev_playback = self.playback_on
        prev_cost = self.cost_hover_on

        new_scheme = bool(scheme_on)
        new_playback = bool(playback_on)
        new_cost = bool(cost_hover_on)
        if new_cost:
            new_scheme = True
            new_playback = True
        elif source_label is None:
            if new_playback:
                new_scheme = True
        elif source_label == "Playback" and new_playback:
            new_scheme = True

        strict = bool(new_scheme or new_playback or new_cost)
        if strict:
            bundle = self._ensure_bundle(strict=True, swallow_errors=True)
            if bundle is None:
                self.scheme_on = prev_scheme
                self.playback_on = prev_playback
                self.cost_hover_on = prev_cost
                self._sync_checkbuttons()
                self._apply_visual_state()
                self._refresh_hover()
                return
            self._viewer = bundle.viewer

        self.scheme_on = new_scheme
        self.playback_on = new_playback
        self.cost_hover_on = new_cost
        self._sync_checkbuttons()
        self._apply_visual_state()
        self._refresh_hover()

    def _ensure_bundle(
        self,
        *,
        strict: bool,
        swallow_errors: bool = True,
    ) -> _ContractionSchemeBundle | None:
        if self._bundle.availability == "unavailable":
            return None
        if self._bundle.availability == "computed":
            return self._bundle

        try:
            bundle = self._bundle_builder(strict)
        except ValueError:
            self._bundle.availability = "unavailable"
            if swallow_errors:
                return None
            raise

        self._bundle = bundle
        self._viewer = bundle.viewer
        if self._viewer is not None:
            self._viewer.build_ui(initialize_step=False)
            self._viewer.set_step_details_enabled(self.cost_hover_on)
            self._viewer.set_playback_widgets_visible(False)
            self.figure._tensor_network_viz_contraction_viewer = self._viewer  # type: ignore[attr-defined]
        return bundle

    def _scheme_entries_2d(self) -> tuple[tuple[Artist, str], ...]:
        return ()

    def _scheme_entries_3d(
        self,
    ) -> tuple[tuple[tuple[float, float, float, float, float, float], str, Artist], ...]:
        return ()

    def _apply_visual_state(self) -> None:
        if self._viewer is None:
            if not self.scheme_on:
                self.figure.canvas.draw_idle()
            return
        self._viewer.set_step_details_enabled(self.cost_hover_on)

        if not self.scheme_on:
            self._viewer.pause()
            self._viewer.set_playback_widgets_visible(False)
            self._viewer.hide_scheme_artists()
            return

        if self.playback_on:
            self._viewer.set_playback_widgets_visible(True)
            self._viewer.set_step(self._viewer.current_step)
            return

        self._viewer.pause()
        self._viewer.set_playback_widgets_visible(False)
        self._viewer.show_static_scheme()

    def _refresh_hover(self) -> None:
        self._refresh_hover_callback((), ())


class ContractionViewer2D(_ContractionViewerBase):
    """Step through 2D patches (e.g. ``FancyBboxPatch``, ``Rectangle``)."""

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
    """Step through 3D line or poly collections."""

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
    fig._tensor_network_viz_contraction_viewer = v  # type: ignore[attr-defined]
    return v


__all__ = [
    "ContractionViewer2D",
    "ContractionViewer3D",
]
