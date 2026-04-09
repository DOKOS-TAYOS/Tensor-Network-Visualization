from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import NonGuiException
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import Button, RadioButtons

from .._logging import package_logger
from .._tensor_comparison_support import (
    _build_comparison_record,
    _comparison_placeholder_text,
    _comparison_text_payload,
)
from .._tensor_elements_controller import (
    _TensorElementsControlsLayout,
    _TensorElementsFigureController,
)
from .._tensor_elements_data import _PlaybackStepRecord
from .._tensor_elements_support import _TensorRecord
from .._ui_utils import _style_control_tray_axes
from ..config import EngineName
from ..tensor_comparison_config import TensorComparisonConfig, TensorComparisonMode
from ..tensor_elements import _show_tensor_records
from ..tensor_elements_config import TensorElementsConfig

_BaseInspectorCompareMode = Literal["current", "reference"]
_DerivedInspectorCompareMode = Literal[
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
]
_InspectorCompareMode = Literal[
    "current",
    "reference",
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
]
_COMPARE_TOGGLE_BOUNDS: tuple[float, float, float, float] = (0.02, 0.19, 0.11, 0.045)
_COMPARE_SELECTOR_BOUNDS: tuple[float, float, float, float] = (0.02, 0.028, 0.18, 0.16)
_CAPTURE_REFERENCE_BOUNDS: tuple[float, float, float, float] = (0.135, 0.19, 0.035, 0.045)
_CLEAR_REFERENCE_BOUNDS: tuple[float, float, float, float] = (0.175, 0.19, 0.035, 0.045)
_COMPARE_LABEL_PROPS: dict[str, list[float]] = {"fontsize": [9.0]}
_COMPARE_RADIO_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 0.9}
_COMPACT_BUTTON_FONT_SIZE: float = 8.0
_SYMBOL_BUTTON_FONT_FAMILY: str = "STIXGeneral"
_INSPECTOR_FIGSIZE: tuple[float, float] = (8.8, 6.4)
_BUTTON_TOOLTIP_GAP_FIG: float = 0.006
_BUTTON_TOOLTIP_TEXT_KWARGS: dict[str, object] = {
    "ha": "left",
    "va": "bottom",
    "fontsize": 8.5,
    "bbox": {
        "boxstyle": "round,pad=0.25",
        "facecolor": "#F8FAFC",
        "edgecolor": "#CBD5E1",
    },
}
_COMPARE_MODE_OPTIONS: tuple[_InspectorCompareMode, ...] = (
    "current",
    "reference",
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
)
_BASE_COMPARE_MODE_OPTIONS: tuple[_BaseInspectorCompareMode, _BaseInspectorCompareMode] = (
    "current",
    "reference",
)
_DERIVED_COMPARE_MODE_OPTIONS: tuple[
    _DerivedInspectorCompareMode,
    _DerivedInspectorCompareMode,
    _DerivedInspectorCompareMode,
    _DerivedInspectorCompareMode,
    _DerivedInspectorCompareMode,
    _DerivedInspectorCompareMode,
] = (
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
)
_INSPECTOR_TENSOR_ELEMENTS_LAYOUT: _TensorElementsControlsLayout = _TensorElementsControlsLayout(
    controls_bottom=0.31,
    group_selector_bounds=(0.225, 0.04, 0.13, 0.145),
    mode_selector_bounds=(0.36, 0.028, 0.17, 0.16),
    tensor_slider_bounds=(0.59, 0.008, 0.35, 0.05),
    tensor_slider_label_x=-0.08,
    analysis_axis_bounds=(0.535, 0.072, 0.205, 0.16),
    analysis_check_bounds=(0.535, 0.07, 0.215, 0.17),
    analysis_method_bounds=(0.77, 0.11, 0.1, 0.12),
    analysis_slider_bounds=(0.825, 0.098, 0.135, 0.045),
)


def _alternate_base_compare_mode(mode: _BaseInspectorCompareMode) -> _BaseInspectorCompareMode:
    return "reference" if mode == "current" else "current"


def _compare_toggle_label(mode: _BaseInspectorCompareMode) -> str:
    return "Reference" if mode == "current" else "Current"


def _style_compact_button(
    button: Button,
    *,
    fontsize: float = _COMPACT_BUTTON_FONT_SIZE,
    fontfamily: str | None = None,
) -> None:
    button.label.set_fontsize(fontsize)
    if fontfamily is not None:
        button.label.set_fontfamily(fontfamily)


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
        step_records: tuple[_PlaybackStepRecord, ...] | None,
        node_records_by_name: dict[str, _TensorRecord] | None,
        placeholder_engine: EngineName,
        on_closed: Callable[[], None],
    ) -> None:
        self._step_records = tuple(step_records or ())
        self._node_records_by_name = (
            {} if node_records_by_name is None else dict(node_records_by_name)
        )
        self._placeholder_engine: EngineName = placeholder_engine
        self._on_closed = on_closed
        self._config = TensorElementsConfig(figsize=_INSPECTOR_FIGSIZE)
        self._enabled: bool = False
        self._viewer: _StepPlaybackViewer | None = None
        self._figure: Figure | None = None
        self._elements_controller: _TensorElementsFigureController | None = None
        self._saved_mode: str | None = None
        self._selected_node_name: str | None = None
        self._active_record: _TensorRecord | None = None
        self._active_source_kind: str = "placeholder"
        self._compare_mode: _InspectorCompareMode = "current"
        self._base_compare_mode: _BaseInspectorCompareMode = "current"
        self._derived_compare_mode: _DerivedInspectorCompareMode = "abs_diff"
        self._captured_reference_record: _TensorRecord | None = None
        self._compare_toggle_button: Button | None = None
        self._compare_radio: RadioButtons | None = None
        self._capture_reference_button: Button | None = None
        self._clear_reference_button: Button | None = None
        self._button_hover_text: Text | None = None
        self._button_hover_cid: int | None = None
        self._compare_callback_guard: bool = False
        self._closing_programmatically: bool = False
        self._close_cid: int | None = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def has_node_records(self) -> bool:
        return bool(self._node_records_by_name)

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

    def select_node(self, node_name: str, *, reveal: bool = False) -> bool:
        """Pin the inspector to one visible node until cleared."""
        if node_name not in self._node_records_by_name:
            return False
        self._selected_node_name = str(node_name)
        package_logger.debug("Linked tensor inspector selected node=%s.", node_name)
        if self._enabled:
            self._sync_active_source()
            if reveal and self._figure is not None:
                _reveal_auxiliary_figure(self._figure)
        return True

    def clear_selected_node(self, *, reveal: bool = False) -> None:
        """Clear the manual node selection and fall back to playback or placeholder state."""
        if self._selected_node_name is None:
            return
        package_logger.debug(
            "Linked tensor inspector cleared selected node=%s.",
            self._selected_node_name,
        )
        self._selected_node_name = None
        if self._enabled:
            self._sync_active_source()
            if reveal and self._figure is not None:
                _reveal_auxiliary_figure(self._figure)

    def set_compare_mode(self, mode: _InspectorCompareMode | str) -> None:
        """Switch the inspector comparison mode programmatically."""
        resolved_mode = cast(_InspectorCompareMode, str(mode))
        if resolved_mode not in _COMPARE_MODE_OPTIONS:
            raise ValueError(f"Unsupported inspector comparison mode: {mode!r}.")
        self._compare_mode = resolved_mode
        if resolved_mode in _BASE_COMPARE_MODE_OPTIONS:
            self._base_compare_mode = cast(_BaseInspectorCompareMode, resolved_mode)
        else:
            self._derived_compare_mode = cast(_DerivedInspectorCompareMode, resolved_mode)
        self._sync_compare_controls()
        if self._enabled:
            self._sync_active_source()

    def capture_reference(self) -> bool:
        """Freeze the currently visible tensor as an in-session reference."""
        if self._active_record is None:
            return False
        self._captured_reference_record = _TensorRecord(
            array=np.asarray(self._active_record.array).copy(),
            axis_names=tuple(self._active_record.axis_names),
            engine=self._active_record.engine,
            name=f"{self._active_record.name} (captured ref)",
        )
        package_logger.debug(
            "Captured tensor inspector reference from record=%s.",
            self._active_record.name,
        )
        if self._enabled:
            self._sync_active_source()
        return True

    def clear_captured_reference(self) -> None:
        """Drop the captured in-session reference, if any."""
        if self._captured_reference_record is None:
            return
        self._captured_reference_record = None
        package_logger.debug("Cleared tensor inspector captured reference.")
        if self._enabled:
            self._sync_active_source()

    def set_enabled(self, enabled: bool, *, reveal: bool = False) -> None:
        target = bool(enabled)
        if target == self._enabled:
            if target and reveal and self._figure is not None:
                _reveal_auxiliary_figure(self._figure)
            if target:
                self._sync_active_source()
            return
        self._enabled = target
        package_logger.debug("Linked tensor inspector enabled=%s reveal=%s.", self._enabled, reveal)
        if not target:
            self._close_figure()
            return
        self._ensure_figure()
        if reveal and self._figure is not None:
            _reveal_auxiliary_figure(self._figure)
        self._sync_active_source()

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
        record = self._selected_node_record()
        if record is None:
            initial_step = int(self._viewer.current_step) if self._viewer is not None else 0
            record = self._record_for_step(initial_step)
        if record is None:
            record = self._placeholder_record()
        figure, _ax, controller = _show_tensor_records(
            [record],
            config=self._config,
            controls_layout=_INSPECTOR_TENSOR_ELEMENTS_LAYOUT,
            ax=None,
            show_controls=True,
            show=False,
        )
        self._figure = figure
        self._elements_controller = controller
        package_logger.debug("Created linked tensor inspector figure.")
        if self._saved_mode is not None:
            with suppress(ValueError):
                controller.set_mode(self._saved_mode, redraw=False)
        self._build_comparison_controls()
        self._close_cid = figure.canvas.mpl_connect("close_event", self._on_figure_closed)

    def _build_comparison_controls(self) -> None:
        if self._figure is None:
            return
        toggle_ax = self._figure.add_axes(_COMPARE_TOGGLE_BOUNDS)
        _style_control_tray_axes(toggle_ax)
        self._compare_toggle_button = Button(
            toggle_ax,
            _compare_toggle_label(self._base_compare_mode),
        )
        _style_compact_button(self._compare_toggle_button)
        self._compare_toggle_button.on_clicked(self._on_compare_toggle_clicked)

        compare_ax = self._figure.add_axes(_COMPARE_SELECTOR_BOUNDS)
        _style_control_tray_axes(compare_ax)
        self._compare_radio = RadioButtons(
            compare_ax,
            tuple(mode.replace("_", " ") for mode in _DERIVED_COMPARE_MODE_OPTIONS),
            active=_DERIVED_COMPARE_MODE_OPTIONS.index(self._derived_compare_mode),
            label_props=_COMPARE_LABEL_PROPS,
            radio_props=_COMPARE_RADIO_PROPS,
        )
        self._compare_radio.on_clicked(self._on_compare_mode_clicked)
        self._sync_compare_controls()

        capture_ax = self._figure.add_axes(_CAPTURE_REFERENCE_BOUNDS)
        _style_control_tray_axes(capture_ax)
        self._capture_reference_button = Button(capture_ax, "⌖")
        _style_compact_button(
            self._capture_reference_button,
            fontsize=9.0,
            fontfamily=_SYMBOL_BUTTON_FONT_FAMILY,
        )
        self._capture_reference_button.on_clicked(self._on_capture_reference_clicked)

        clear_ax = self._figure.add_axes(_CLEAR_REFERENCE_BOUNDS)
        _style_control_tray_axes(clear_ax)
        self._clear_reference_button = Button(clear_ax, "x")
        _style_compact_button(self._clear_reference_button, fontsize=9.0)
        self._clear_reference_button.on_clicked(self._on_clear_reference_clicked)
        self._install_button_hover_tooltips()

    def _on_compare_mode_clicked(self, label: str | None) -> None:
        if self._compare_callback_guard or label is None:
            return
        self.set_compare_mode(str(label).replace(" ", "_"))

    def _on_compare_toggle_clicked(self, _event: object) -> None:
        self.set_compare_mode(_alternate_base_compare_mode(self._base_compare_mode))

    def _on_capture_reference_clicked(self, _event: object) -> None:
        self.capture_reference()

    def _on_clear_reference_clicked(self, _event: object) -> None:
        self.clear_captured_reference()

    def _install_button_hover_tooltips(self) -> None:
        if self._figure is None or self._button_hover_text is not None:
            return
        self._button_hover_text = self._figure.text(
            0.0,
            0.0,
            "",
            visible=False,
            transform=self._figure.transFigure,
            **_BUTTON_TOOLTIP_TEXT_KWARGS,
        )
        self._button_hover_cid = self._figure.canvas.mpl_connect(
            "motion_notify_event",
            self._on_button_hover_motion,
        )

    def _button_tooltip_target_for_event(self, event: object) -> tuple[Button, str] | None:
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if x is None or y is None:
            return None
        for button, label in (
            (self._capture_reference_button, "Capture reference"),
            (self._clear_reference_button, "Clear reference"),
        ):
            if button is not None and button.ax.bbox.contains(float(x), float(y)):
                return button, label
        return None

    def _on_button_hover_motion(self, event: object) -> None:
        if self._figure is None or self._button_hover_text is None:
            return
        tooltip_target = self._button_tooltip_target_for_event(event)
        if tooltip_target is None:
            if self._button_hover_text.get_visible():
                self._button_hover_text.set_visible(False)
                self._figure.canvas.draw_idle()
            return
        button, label = tooltip_target
        bounds = button.ax.get_position().bounds
        x_fig = min(max(float(bounds[0]), 0.01), 0.99)
        y_fig = min(max(float(bounds[1] + bounds[3] + _BUTTON_TOOLTIP_GAP_FIG), 0.01), 0.99)
        self._button_hover_text.set_position((x_fig, y_fig))
        self._button_hover_text.set_text(label)
        if not self._button_hover_text.get_visible():
            self._button_hover_text.set_visible(True)
        self._figure.canvas.draw_idle()

    def _sync_compare_controls(self) -> None:
        if self._compare_toggle_button is not None:
            label = _compare_toggle_label(self._base_compare_mode)
            if self._compare_toggle_button.label.get_text() != label:
                self._compare_toggle_button.label.set_text(label)
        if self._compare_radio is None:
            return
        active_label = self._derived_compare_mode.replace("_", " ")
        if getattr(self._compare_radio, "value_selected", None) == active_label:
            return
        try:
            self._compare_callback_guard = True
            self._compare_radio.set_active(
                _DERIVED_COMPARE_MODE_OPTIONS.index(self._derived_compare_mode)
            )
        finally:
            self._compare_callback_guard = False

    def _current_step(self) -> int:
        return int(self._viewer.current_step) if self._viewer is not None else 0

    def _record_for_step(self, step: int) -> _TensorRecord | None:
        if step <= 0:
            return None
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            return None
        return self._step_records[index].record

    def _selected_node_record(self) -> _TensorRecord | None:
        if self._selected_node_name is None:
            return None
        return self._node_records_by_name.get(self._selected_node_name)

    def _default_placeholder_text(self) -> str:
        if self._viewer is not None and self._step_records:
            return "No contraction selected yet."
        if self._node_records_by_name:
            return "Click a tensor node to inspect its values."
        return "No inspectable tensor selected yet."

    def _render_placeholder(self, text: str) -> None:
        if self._elements_controller is None:
            return
        self._elements_controller.render_placeholder(text)

    def _reference_record_for_active_source(
        self,
        current_record: _TensorRecord,
        *,
        source_kind: str,
    ) -> _TensorRecord | None:
        if self._captured_reference_record is not None:
            return self._captured_reference_record
        if source_kind != "playback_result":
            return None
        previous_record = self._record_for_step(self._current_step() - 1)
        if previous_record is None:
            return None
        if previous_record.array.shape != current_record.array.shape:
            return None
        return previous_record

    def _render_active_record(
        self,
        record: _TensorRecord,
        *,
        source_kind: str,
    ) -> None:
        if self._elements_controller is None:
            return
        self._active_record = record
        self._active_source_kind = source_kind
        reference_record = self._reference_record_for_active_source(record, source_kind=source_kind)
        if self._compare_mode == "current" or reference_record is None:
            self._elements_controller.set_single_record(record)
            self._saved_mode = str(self._elements_controller.selected_mode)
            return
        if self._compare_mode == "reference":
            self._elements_controller.set_single_record(reference_record)
            self._saved_mode = str(self._elements_controller.selected_mode)
            return
        if record.array.shape != reference_record.array.shape:
            self._render_placeholder(_comparison_placeholder_text(record, reference_record))
            return
        comparison_config = TensorComparisonConfig(
            mode=cast(TensorComparisonMode, self._compare_mode)
        )
        if self._compare_mode == "topk_changes":
            self._elements_controller.render_placeholder(
                _comparison_text_payload(
                    record,
                    reference_record,
                    comparison_config=comparison_config,
                ).text,
                title=f"{record.name} - top-k changes",
            )
            return
        self._elements_controller.set_single_record(
            _build_comparison_record(
                record,
                reference_record,
                comparison_config=comparison_config,
            )
        )
        self._saved_mode = str(self._elements_controller.selected_mode)

    def _sync_active_source(self) -> None:
        if not self._enabled:
            return
        self._ensure_figure()
        if self._elements_controller is None:
            return

        selected_record = self._selected_node_record()
        if selected_record is not None:
            package_logger.debug(
                "Syncing linked tensor inspector to selected node=%s.",
                self._selected_node_name,
            )
            self._render_active_record(selected_record, source_kind="selected_node")
            return
        if self._selected_node_name is not None:
            self._active_record = None
            self._active_source_kind = "placeholder"
            self._render_placeholder(
                f"Tensor for node {self._selected_node_name!r} is not available."
            )
            return

        step = self._current_step()
        package_logger.debug("Syncing linked tensor inspector to playback step=%s.", step)
        if step <= 0:
            self._active_record = None
            self._active_source_kind = "placeholder"
            self._render_placeholder(self._default_placeholder_text())
            return
        index = step - 1
        if index < 0 or index >= len(self._step_records):
            self._active_record = None
            self._active_source_kind = "placeholder"
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        step_record = self._step_records[index]
        if step_record.record is None:
            self._active_record = None
            self._active_source_kind = "placeholder"
            self._render_placeholder(f"Tensor for step {step} is not available.")
            return
        self._render_active_record(step_record.record, source_kind="playback_result")

    def _sync_to_step(self, step: int) -> None:
        _ = step
        self._sync_active_source()

    def _close_figure(self) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        figure = self._figure
        if figure is None:
            self._elements_controller = None
            self._close_cid = None
            self._button_hover_cid = None
            self._button_hover_text = None
            return
        if self._close_cid is not None:
            figure.canvas.mpl_disconnect(self._close_cid)
        if self._button_hover_cid is not None:
            figure.canvas.mpl_disconnect(self._button_hover_cid)
        if self._button_hover_text is not None:
            self._button_hover_text.remove()
        self._closing_programmatically = True
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        self._button_hover_cid = None
        self._button_hover_text = None
        self._compare_toggle_button = None
        self._compare_radio = None
        self._capture_reference_button = None
        self._clear_reference_button = None
        package_logger.debug("Closing linked tensor inspector figure.")
        plt.close(figure)
        self._closing_programmatically = False

    def _on_figure_closed(self, _event: object) -> None:
        if self._elements_controller is not None:
            self._saved_mode = str(self._elements_controller.selected_mode)
        figure = self._figure
        if figure is not None and self._button_hover_cid is not None:
            figure.canvas.mpl_disconnect(self._button_hover_cid)
        if self._button_hover_text is not None:
            self._button_hover_text.remove()
        self._figure = None
        self._elements_controller = None
        self._close_cid = None
        self._button_hover_cid = None
        self._button_hover_text = None
        self._compare_toggle_button = None
        self._compare_radio = None
        self._capture_reference_button = None
        self._clear_reference_button = None
        if self._closing_programmatically:
            return
        self._enabled = False
        package_logger.debug("Linked tensor inspector figure was closed by the user.")
        self._on_closed()


__all__ = ["_LinkedTensorInspectorController", "_StepPlaybackViewer", "_reveal_auxiliary_figure"]
