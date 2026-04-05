from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import NonGuiException
from matplotlib.figure import Figure

from .._logging import package_logger
from .._tensor_elements_controller import _TensorElementsFigureController
from .._tensor_elements_data import _PlaybackStepRecord
from .._tensor_elements_support import _TensorRecord
from ..config import EngineName
from ..tensor_elements import _show_tensor_records
from ..tensor_elements_config import TensorElementsConfig


class _StepPlaybackViewer(Protocol):
    current_step: int

    def add_step_changed_callback(
        self,
        callback: Callable[[int], None],
        *,
        call_immediately: bool,
    ) -> None: ...

    def remove_step_changed_callback(self, callback: Callable[[int], None]) -> None: ...


def _reveal_auxiliary_figure(figure: Figure) -> None:
    manager = getattr(figure.canvas, "manager", None)
    manager_show = getattr(manager, "show", None)
    if callable(manager_show):
        with suppress(AttributeError, NonGuiException, RuntimeError, TypeError, ValueError):
            manager_show()
    else:
        figure_show = getattr(figure, "show", None)
        if callable(figure_show):
            with suppress(AttributeError, NonGuiException, RuntimeError, TypeError, ValueError):
                figure_show()
    draw_idle = getattr(figure.canvas, "draw_idle", None)
    if callable(draw_idle):
        with suppress(AttributeError, RuntimeError, TypeError, ValueError):
            draw_idle()
    flush_events = getattr(figure.canvas, "flush_events", None)
    if callable(flush_events):
        with suppress(AttributeError, RuntimeError, TypeError, ValueError):
            flush_events()


class _LinkedTensorInspectorController:
    def __init__(
        self,
        *,
        step_records: tuple[_PlaybackStepRecord, ...],
        placeholder_engine: EngineName,
        on_closed: Callable[[], None],
    ) -> None:
        self._step_records = step_records
        self._placeholder_engine: EngineName = placeholder_engine
        self._on_closed = on_closed
        self._config = TensorElementsConfig()
        self._enabled: bool = False
        self._viewer: _StepPlaybackViewer | None = None
        self._figure: Figure | None = None
        self._elements_controller: _TensorElementsFigureController | None = None
        self._saved_mode: str | None = None
        self._closing_programmatically: bool = False
        self._close_cid: int | None = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def bind_viewer(self, viewer: _StepPlaybackViewer) -> None:
        if self._viewer is viewer:
            if self._enabled and self._viewer is not None:
                self._viewer.add_step_changed_callback(
                    self._sync_to_step,
                    call_immediately=True,
                )
            return
        if self._viewer is not None:
            self._viewer.remove_step_changed_callback(self._sync_to_step)
        self._viewer = viewer
        package_logger.debug("Bound linked tensor inspector to playback viewer.")
        if self._viewer is not None:
            self._viewer.add_step_changed_callback(
                self._sync_to_step,
                call_immediately=self._enabled,
            )

    def set_enabled(self, enabled: bool, *, reveal: bool = False) -> None:
        target = bool(enabled)
        if target == self._enabled:
            if target and reveal and self._figure is not None:
                _reveal_auxiliary_figure(self._figure)
            if target and self._viewer is not None:
                self._sync_to_step(int(self._viewer.current_step))
            return
        self._enabled = target
        package_logger.debug("Linked tensor inspector enabled=%s reveal=%s.", self._enabled, reveal)
        if not target:
            self._close_figure()
            return
        self._ensure_figure()
        if reveal and self._figure is not None:
            _reveal_auxiliary_figure(self._figure)
        if self._viewer is not None:
            self._sync_to_step(int(self._viewer.current_step))
        else:
            self._render_placeholder("No contraction selected yet.")

    def close_from_owner(self) -> None:
        self._enabled = False
        if self._viewer is not None:
            self._viewer.remove_step_changed_callback(self._sync_to_step)
            self._viewer = None
        self._close_figure()

    def _placeholder_record(self) -> _TensorRecord:
        return _TensorRecord(
            array=np.zeros((1, 1), dtype=float),
            axis_names=(),
            engine=self._placeholder_engine,
            name="Tensor inspector",
        )

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        initial_step = int(self._viewer.current_step) if self._viewer is not None else 0
        record = self._record_for_step(initial_step)
        if record is None:
            record = self._placeholder_record()
        figure, _ax, controller = _show_tensor_records(
            [record],
            config=self._config,
            ax=None,
            show_controls=True,
            show=False,
        )
        self._figure = figure
        self._elements_controller = controller
        package_logger.debug("Created linked tensor inspector figure for step=%s.", initial_step)
        if self._saved_mode is not None:
            with suppress(ValueError):
                controller.set_mode(self._saved_mode, redraw=False)
        self._close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)

    def _record_for_step(self, step: int) -> _TensorRecord | None:
        if step <= 0:
            return None
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            return None
        return self._step_records[index].record

    def _render_placeholder(self, text: str) -> None:
        if self._elements_controller is None:
            return
        self._elements_controller.render_placeholder(text)

    def _sync_to_step(self, step: int) -> None:
        if not self._enabled:
            return
        self._ensure_figure()
        if self._elements_controller is None:
            return
        package_logger.debug("Syncing linked tensor inspector to playback step=%s.", step)
        if step <= 0:
            self._render_placeholder("No contraction selected yet.")
            return
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        step_record = self._step_records[index]
        if step_record.record is None:
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        self._elements_controller.set_single_record(step_record.record)
        self._saved_mode = str(self._elements_controller.selected_mode)

    def _close_figure(self) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        figure = self._figure
        if figure is None:
            self._elements_controller = None
            self._close_cid = None
            return
        if self._close_cid is not None:
            figure.canvas.mpl_disconnect(self._close_cid)
        self._closing_programmatically = True
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        package_logger.debug("Closing linked tensor inspector figure.")
        plt.close(figure)
        self._closing_programmatically = False

    def _on_figure_closed(self, _event: object) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        if self._closing_programmatically:
            return
        self._enabled = False
        package_logger.debug("Linked tensor inspector figure was closed by the user.")
        self._on_closed()


__all__ = ["_LinkedTensorInspectorController", "_StepPlaybackViewer", "_reveal_auxiliary_figure"]
