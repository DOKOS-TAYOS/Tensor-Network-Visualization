from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .._matplotlib_state import set_contraction_controls, set_contraction_viewer
from .._ui_utils import _reserve_figure_bottom
from ..config import PlotConfig
from ..contraction_viewer import (
    _CONTROL_CHECK_PROPS,
    _CONTROL_FRAME_PROPS,
    _CONTROL_LABEL_PROPS,
    _CONTROLS_CHECKBOX_BOUNDS,
    _CONTROLS_MAIN_BOTTOM,
    _PLAYBACK_TRAY_FRAME,
)


@dataclass
class _ContractionSchemeBundle:
    availability: str = "not_computed"
    steps: tuple[frozenset[int], ...] | None = None
    artists_by_step: list[Artist | None] | None = None
    scheme_aabb: list[tuple[float, float, float, float, float, float] | None] | None = None
    metrics_row: tuple[Any | None, ...] | None = None
    step_details: tuple[str | None, ...] | None = None
    viewer: Any = None
    bounds_2d: tuple[float, float, float, float] | None = None
    bounds_3d: tuple[float, float, float, float, float, float] | None = None


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
        self._viewer: Any = None
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
            set_contraction_controls(fig, self)
        set_contraction_controls(ax, self)
        if self.scheme_on:
            self._ensure_bundle(strict=self.playback_on, swallow_errors=False)
        self._apply_visual_state()
        self._refresh_hover()

    def _build_controls(self) -> None:
        controls_ax = self.figure.add_axes(_CONTROLS_CHECKBOX_BOUNDS)
        controls_ax.set_xticks([])
        controls_ax.set_yticks([])
        controls_ax.set_navigate(False)
        controls_ax.patch.set_facecolor((0.97, 0.97, 0.99))
        controls_ax.patch.set_alpha(0.88)
        controls_ax.patch.set_edgecolor(_PLAYBACK_TRAY_FRAME)
        controls_ax.patch.set_linewidth(0.6)
        for spine in controls_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color(_PLAYBACK_TRAY_FRAME)
        self._controls_ax = controls_ax
        self._checkbuttons = CheckButtons(
            controls_ax,
            ["Scheme", "Playback", "Costs"],
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
            set_contraction_viewer(self.figure, self._viewer)
        return bundle

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


__all__ = [
    "_ContractionControls",
    "_ContractionSchemeBundle",
]
