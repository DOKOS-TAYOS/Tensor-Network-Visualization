"""Matplotlib-only interactive stepping through ordered contraction visuals.

Primary integration: ``show_tensor_network(..., config=PlotConfig(
show_contraction_scheme=True, contraction_playback=True))`` adds a slider and
Play / Pause / Reset on the same figure (2D widget axes only).

For a richer visual policy, subclass ``ContractionViewer2D`` / ``ContractionViewer3D``
and override ``_apply_step_visuals``. The draw pipeline attaches highlights; this module
only adds matplotlib widgets and mutates existing artists.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Final, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._typing import root_figure
from .config import PlotConfig
from .viewer import _show_figure

VisualizerMode = Literal["cumulative", "highlight_current", "window"]

# Must stay aligned with ``_CONTRACTION_SCHEME_GID`` in ``_core.draw.contraction_scheme``.
_TNV_CONTRACTION_SCHEME_PATCH_GID: Final[str] = "tnv_contraction_scheme"

_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
_PLAYBACK_MAIN_BOTTOM: float = 0.24
_PLAYBACK_SLIDER_BOUNDS: tuple[float, float, float, float] = (0.16, 0.065, 0.46, 0.028)
_PLAYBACK_BUTTON_START_X: float = 0.66
_PLAYBACK_BUTTON_Y: float = 0.058
_PLAYBACK_BUTTON_WIDTH: float = 0.058
_PLAYBACK_BUTTON_HEIGHT: float = 0.04
_PLAYBACK_BUTTON_GAP: float = 0.012
_PLAYBACK_RESET_WIDTH: float = 0.068


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

        self.current_step: int = 0
        self._initial_step = initial_step

        self.slider: Slider | None = None
        self._btn_play: Button | None = None
        self._btn_pause: Button | None = None
        self._btn_reset: Button | None = None
        self._timer: Any = None
        self._is_playing: bool = False
        self._slider_callback_guard: bool = False
        self._ui_built: bool = False
        self._cid_close: int | None = None

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

        self.figure.canvas.draw_idle()

    def build_ui(self) -> None:
        """Create slider and buttons on 2D axes; no-op if ``enable_playback`` is False."""
        if not self._enable_playback or self._ui_built:
            return

        n = self.num_steps
        self.figure.subplots_adjust(bottom=_PLAYBACK_MAIN_BOTTOM)
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
        ax_reset = self.figure.add_axes(
            (bx + 2.0 * (bw + gap), by, _PLAYBACK_RESET_WIDTH, bh)
        )
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

    def show(self) -> None:
        """Standalone demos; in-library use relies on ``show_tensor_network`` / ``_show_figure``."""
        self.build_ui()
        _show_figure(self.figure)


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
    fig: Figure,
    ax: Axes | Axes3D,
    config: PlotConfig,
) -> _ContractionViewerBase:
    """Attach playback UI to the figure produced by ``show_tensor_network``."""
    if getattr(ax, "name", None) == "3d":
        v: _ContractionViewerBase = ContractionViewer3D(
            artists_by_step,
            fig=fig,
            ax=cast(Axes3D, ax),
            config=config,
            enable_playback=True,
        )
    else:
        v = ContractionViewer2D(
            artists_by_step,
            fig=fig,
            ax=cast(Axes, ax),
            config=config,
            enable_playback=True,
        )
    v.build_ui()
    fig._tensor_network_viz_contraction_viewer = v  # type: ignore[attr-defined]
    return v


__all__ = [
    "ContractionViewer2D",
    "ContractionViewer3D",
]


if __name__ == "__main__":
    # Demo 2D standalone; library: show_tensor_network + PlotConfig(contraction_playback=True).
    rng = np.random.default_rng(0)
    rects = []
    for _ in range(5):
        rects.append(
            (
                float(rng.random()),
                float(rng.random()),
                0.12 + float(rng.random()) * 0.15,
                0.1 + float(rng.random()) * 0.12,
            )
        )
    v2 = ContractionViewer2D.from_rectangles(
        rects,
        mode="highlight_current",
        current_color="tab:red",
        past_color="0.65",
        interval_ms=350,
        enable_playback=True,
    )
    v2._ax_main.set_xlim(0, 1.3)
    v2._ax_main.set_ylim(0, 1.3)
    v2._ax_main.set_aspect("equal")
    v2.show()

    # Demo 3D (opens after closing the 2D window when using an interactive backend).
    boxes_demo = [
        (np.array([0, 0, 0]), np.array([1, 0.6, 0.5])),
        (np.array([1.1, 0.1, 0.1]), np.array([1.8, 0.9, 0.7])),
        (np.array([0.2, 0.8, 0.2]), np.array([0.9, 1.4, 0.9])),
    ]
    v3 = ContractionViewer3D.from_boxes(
        boxes_demo,
        mode="highlight_current",
        current_color="tab:red",
        past_color="0.65",
        interval_ms=400,
        enable_playback=True,
    )
    cast(Axes3D, v3._ax_main).set_xlim(0, 2)
    cast(Axes3D, v3._ax_main).set_ylim(0, 2)
    cast(Axes3D, v3._ax_main).set_zlim(0, 1.2)
    v3.show()
