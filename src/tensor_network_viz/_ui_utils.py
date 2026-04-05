from __future__ import annotations

from contextlib import suppress
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._matplotlib_state import get_reserved_bottom, set_reserved_bottom

_CONTROL_TRAY_FACE: tuple[float, float, float] = (0.97, 0.97, 0.99)
_CONTROL_TRAY_FRAME: tuple[float, float, float] = (0.78, 0.78, 0.82)


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


__all__ = [
    "_CONTROL_TRAY_FACE",
    "_CONTROL_TRAY_FRAME",
    "_reserve_figure_bottom",
    "_set_figure_bottom_reserved",
    "_set_axes_visible",
    "_set_widget_active",
    "_style_control_tray_axes",
]
