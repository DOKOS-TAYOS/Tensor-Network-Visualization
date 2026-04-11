from __future__ import annotations

from contextlib import suppress
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.widgets import Button

from ._matplotlib_state import get_reserved_bottom, set_reserved_bottom

_CONTROL_TRAY_FACE: tuple[float, float, float] = (0.97, 0.97, 0.99)
_CONTROL_TRAY_FRAME: tuple[float, float, float] = (0.78, 0.78, 0.82)
_CONTROL_BUTTON_FACE: str = "#F8FAFC"
_CONTROL_BUTTON_HOVER_FACE: str = "#EEF6FF"
_CONTROL_BUTTON_EDGE: str = "#CBD5E1"
_CONTROL_BUTTON_TEXT: str = "#0F172A"
_CONTROL_BUTTON_ACTIVE_FACE: str = "#DBEAFE"
_CONTROL_BUTTON_ACTIVE_EDGE: str = "#2563EB"
_CONTROL_BUTTON_MUTED_FACE: str = "#E5E7EB"
_CONTROL_BUTTON_MUTED_EDGE: str = "#CBD5E1"
_CONTROL_BUTTON_MUTED_TEXT: str = "#64748B"
_CONTROL_SLIDER_TRACK_COLOR: str = "#D7EAF2"
_CONTROL_SLIDER_HANDLE_SIZE: float = 11.0
_CONTROL_SLIDER_TRACK_WHITE_MIX: float = 0.82


def _reserve_figure_bottom(fig: Figure, bottom: float) -> None:
    current = get_reserved_bottom(fig)
    target = max(current, float(bottom))
    set_reserved_bottom(fig, target)
    fig.subplots_adjust(bottom=target)


def _set_figure_bottom_reserved(fig: Figure, bottom: float) -> None:
    """Store and apply *bottom*; unlike `_reserve_figure_bottom`, can shrink the reserved strip."""
    b = float(bottom)
    set_reserved_bottom(fig, b)
    p = fig.subplotpars
    fig.subplots_adjust(left=p.left, right=p.right, top=p.top, bottom=b)


def _set_axes_visible(ax: Any, visible: bool) -> None:
    ax.set_visible(visible)
    ax.patch.set_visible(visible)
    for child in ax.get_children():
        setter = getattr(child, "set_visible", None)
        if callable(setter):
            with suppress(AttributeError, TypeError, ValueError):
                setter(bool(visible))


def _set_widget_active(widget: Any, active: bool) -> None:
    setter = getattr(widget, "set_active", None)
    if callable(setter):
        with suppress(AttributeError, TypeError, ValueError):
            setter(bool(active))


def _style_control_tray_axes(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_navigate(False)
    ax.patch.set_facecolor(_CONTROL_TRAY_FACE)
    ax.patch.set_alpha(0.88)
    ax.patch.set_edgecolor(_CONTROL_TRAY_FRAME)
    ax.patch.set_linewidth(0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(_CONTROL_TRAY_FRAME)


def _slider_track_color(
    base_color: str,
    *,
    white_mix: float = _CONTROL_SLIDER_TRACK_WHITE_MIX,
) -> tuple[float, float, float, float]:
    r, g, b, _ = to_rgba(base_color)
    mix = float(white_mix)
    return (
        mix + (1.0 - mix) * float(r),
        mix + (1.0 - mix) * float(g),
        mix + (1.0 - mix) * float(b),
        1.0,
    )


def _control_slider_handle_style(
    *,
    active_color: str,
    edge_color: str | None = None,
    size: float = _CONTROL_SLIDER_HANDLE_SIZE,
) -> dict[str, Any]:
    return {
        "facecolor": active_color,
        "edgecolor": active_color if edge_color is None else edge_color,
        "size": float(size),
    }


def _style_control_button(
    button: Button,
    *,
    active: bool = False,
    muted: bool = False,
    font_size: float | None = None,
) -> None:
    if muted:
        face = _CONTROL_BUTTON_MUTED_FACE
        edge = _CONTROL_BUTTON_MUTED_EDGE
        text = _CONTROL_BUTTON_MUTED_TEXT
        hover = _CONTROL_BUTTON_MUTED_FACE
    elif active:
        face = _CONTROL_BUTTON_ACTIVE_FACE
        edge = _CONTROL_BUTTON_ACTIVE_EDGE
        text = _CONTROL_BUTTON_TEXT
        hover = _CONTROL_BUTTON_HOVER_FACE
    else:
        face = _CONTROL_BUTTON_FACE
        edge = _CONTROL_BUTTON_EDGE
        text = _CONTROL_BUTTON_TEXT
        hover = _CONTROL_BUTTON_HOVER_FACE

    button.color = face
    button.hovercolor = hover
    button.ax.patch.set_facecolor(face)
    button.ax.patch.set_alpha(1.0)
    button.ax.patch.set_edgecolor(edge)
    button.ax.patch.set_linewidth(0.8 if active else 0.6)
    for spine in button.ax.spines.values():
        spine.set_visible(True)
        spine.set_color(edge)
        spine.set_linewidth(0.8 if active else 0.6)
    button.label.set_color(text)
    button.label.set_fontweight("semibold" if active else "normal")
    if font_size is not None:
        button.label.set_fontsize(float(font_size))


__all__ = [
    "_CONTROL_BUTTON_ACTIVE_FACE",
    "_CONTROL_BUTTON_FACE",
    "_CONTROL_BUTTON_MUTED_FACE",
    "_CONTROL_SLIDER_TRACK_COLOR",
    "_CONTROL_TRAY_FACE",
    "_CONTROL_TRAY_FRAME",
    "_control_slider_handle_style",
    "_reserve_figure_bottom",
    "_set_figure_bottom_reserved",
    "_set_axes_visible",
    "_set_widget_active",
    "_slider_track_color",
    "_style_control_button",
    "_style_control_tray_axes",
]
