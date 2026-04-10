from __future__ import annotations

from contextlib import suppress
from typing import Any

from matplotlib.widgets import Button, Slider


def _release_stale_mouse_grabber(widget: Any, event: Any) -> None:
    canvas = getattr(event, "canvas", None)
    mouse_grabber = None if canvas is None else getattr(canvas, "mouse_grabber", None)
    if mouse_grabber is None or mouse_grabber is widget.ax:
        return
    release_mouse = None if canvas is None else getattr(canvas, "release_mouse", None)
    with suppress(AttributeError, RuntimeError, TypeError, ValueError):
        if callable(release_mouse):
            release_mouse(mouse_grabber)


class _SafeButton(Button):
    def _click(self, event: Any) -> None:
        if self.eventson and not self.ignore(event) and self.ax.contains(event)[0]:
            _release_stale_mouse_grabber(self, event)
        super()._click(event)


class _SafeSlider(Slider):
    def _update(self, event: Any) -> None:
        if (
            getattr(event, "name", None) == "button_press_event"
            and getattr(event, "button", None) == 1
            and self.ax.contains(event)[0]
        ):
            _release_stale_mouse_grabber(self, event)
        super()._update(event)


__all__ = ["_SafeButton", "_SafeSlider"]
