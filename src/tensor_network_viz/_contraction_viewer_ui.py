from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider

from ._widgets import _SafeSlider

_PLAYBACK_DETAILS_BOUNDS: tuple[float, float, float, float] = (0.23, 0.116, 0.7, 0.12)
# Top of the cost / step-details axis; interactive chrome (checkboxes, 2d/3d) aligns to this y.
_PLAYBACK_DETAILS_TOP: float = _PLAYBACK_DETAILS_BOUNDS[1] + _PLAYBACK_DETAILS_BOUNDS[3]
# Aligned with the top of the cost / scheme chrome (no extra gap above the widgets).
_MAIN_FIGURE_BOTTOM_RESERVED: float = _PLAYBACK_DETAILS_TOP
_PLAYBACK_MAIN_BOTTOM: float = _MAIN_FIGURE_BOTTOM_RESERVED
_CONTROLS_MAIN_BOTTOM: float = _MAIN_FIGURE_BOTTOM_RESERVED
_PLAYBACK_SLIDER_HEIGHT: float = 0.058
_PLAYBACK_SLIDER_BOUNDS: tuple[float, float, float, float] = (
    0.33,
    0.062,
    0.345,
    _PLAYBACK_SLIDER_HEIGHT,
)
_PLAYBACK_SLIDER_HANDLE_STYLE: dict[str, Any] = {
    "facecolor": "#2563eb",
    "edgecolor": "#1d4ed8",
    "size": 11,
}
_PLAYBACK_BUTTON_START_X: float = 0.73
_PLAYBACK_BUTTON_Y: float = 0.058
_PLAYBACK_BUTTON_WIDTH: float = 0.055
_PLAYBACK_BUTTON_HEIGHT: float = 0.038
_PLAYBACK_BUTTON_GAP: float = 0.012
_PLAYBACK_RESET_WIDTH: float = 0.065
_CONTROLS_CHECKBOX_TOP: float = _PLAYBACK_DETAILS_TOP
_CONTROLS_CHECKBOX_HEIGHT: float = 0.10
_CONTROLS_CHECKBOX_BOUNDS: tuple[float, float, float, float] = (
    0.02,
    _CONTROLS_CHECKBOX_TOP - _CONTROLS_CHECKBOX_HEIGHT,
    0.13,
    _CONTROLS_CHECKBOX_HEIGHT,
)
_PLAYBACK_TRAY_FACE: tuple[float, float, float] = (0.96, 0.96, 0.98)
_PLAYBACK_TRAY_FRAME: tuple[float, float, float] = (0.78, 0.78, 0.82)
_SCHEME_LABELS: tuple[str, str] = ("Scheme", "Costs")
_CONTROL_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_CONTROL_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_CONTROL_CHECK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}


def create_playback_details_panel(fig: Figure) -> tuple[Axes, Text]:
    ax_details = fig.add_axes(_PLAYBACK_DETAILS_BOUNDS)
    ax_details.set_xticks([])
    ax_details.set_yticks([])
    ax_details.set_navigate(False)
    ax_details.patch.set_facecolor(_PLAYBACK_TRAY_FACE)
    ax_details.patch.set_alpha(0.92)
    ax_details.patch.set_edgecolor(_PLAYBACK_TRAY_FRAME)
    ax_details.patch.set_linewidth(0.6)
    for spine in ax_details.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(_PLAYBACK_TRAY_FRAME)
    text = ax_details.text(
        0.0,
        1.0,
        "",
        transform=ax_details.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        wrap=False,
    )
    return ax_details, text


def create_playback_slider(
    fig: Figure,
    *,
    num_steps: int,
    initial_step: int,
) -> Slider:
    ax_slider = fig.add_axes(_PLAYBACK_SLIDER_BOUNDS)
    return _SafeSlider(
        ax_slider,
        "Step",
        0,
        float(max(0, num_steps)),
        valinit=float(initial_step),
        valstep=1,
        handle_style=_PLAYBACK_SLIDER_HANDLE_STYLE,
    )


def create_playback_buttons(fig: Figure) -> tuple[Button, Button, Button]:
    bx = _PLAYBACK_BUTTON_START_X
    by = _PLAYBACK_BUTTON_Y
    bw = _PLAYBACK_BUTTON_WIDTH
    bh = _PLAYBACK_BUTTON_HEIGHT
    gap = _PLAYBACK_BUTTON_GAP
    ax_play = fig.add_axes((bx, by, bw, bh))
    ax_pause = fig.add_axes((bx + bw + gap, by, bw, bh))
    ax_reset = fig.add_axes((bx + 2.0 * (bw + gap), by, _PLAYBACK_RESET_WIDTH, bh))
    return (
        Button(ax_play, "Play"),
        Button(ax_pause, "Pause"),
        Button(ax_reset, "Reset"),
    )


__all__ = [
    "_CONTROL_CHECK_PROPS",
    "_CONTROL_FRAME_PROPS",
    "_CONTROL_LABEL_PROPS",
    "_CONTROLS_CHECKBOX_BOUNDS",
    "_CONTROLS_MAIN_BOTTOM",
    "_MAIN_FIGURE_BOTTOM_RESERVED",
    "_PLAYBACK_DETAILS_TOP",
    "_PLAYBACK_MAIN_BOTTOM",
    "_PLAYBACK_TRAY_FRAME",
    "_SCHEME_LABELS",
    "create_playback_buttons",
    "create_playback_details_panel",
    "create_playback_slider",
]
