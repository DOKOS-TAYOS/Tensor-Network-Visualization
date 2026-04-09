from __future__ import annotations

from contextlib import suppress
from typing import Any

from matplotlib.widgets import Slider


class _SafeSlider(Slider):
    def _update(self, event: Any) -> None:
        if (
            getattr(event, "name", None) == "button_press_event"
            and getattr(event, "button", None) == 1
            and self.ax.contains(event)[0]
        ):
            canvas = getattr(event, "canvas", None)
            mouse_grabber = None if canvas is None else getattr(canvas, "mouse_grabber", None)
            if mouse_grabber is not None and mouse_grabber is not self.ax:
                with suppress(AttributeError, RuntimeError, TypeError, ValueError):
                    canvas.release_mouse(mouse_grabber)
        super()._update(event)


__all__ = ["_SafeSlider"]
