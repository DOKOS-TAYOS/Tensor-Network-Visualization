from __future__ import annotations

from contextlib import suppress
from typing import Any

from matplotlib.figure import Figure


def _reserve_figure_bottom(fig: Figure, bottom: float) -> None:
    current = float(getattr(fig, "_tensor_network_viz_reserved_bottom", 0.02))
    target = max(current, float(bottom))
    fig._tensor_network_viz_reserved_bottom = target  # type: ignore[attr-defined]
    fig.subplots_adjust(bottom=target)


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


__all__ = ["_reserve_figure_bottom", "_set_axes_visible", "_set_widget_active"]
