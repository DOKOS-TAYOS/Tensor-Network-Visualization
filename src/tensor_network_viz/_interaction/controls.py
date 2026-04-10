from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons

from .._ui_utils import _set_axes_visible, _style_control_tray_axes
from .._widgets import _SafeButton
from ..config import ViewName
from .state import InteractiveFeatureState

# Menu column: fixed bottom = tallest stack (inspector + scheme). Without playback, checkboxes
# stay as low as when the bottom row exists.
_INTERACTIVE_MENU_COLUMN_HEIGHT: float = 0.172
_INTERACTIVE_MENU_COLUMN_BOTTOM: float = 0.236 - _INTERACTIVE_MENU_COLUMN_HEIGHT
_INTERACTIVE_CHECKBOX_AXES_BOUNDS: tuple[float, float, float, float] = (
    0.02,
    _INTERACTIVE_MENU_COLUMN_BOTTOM,
    0.21,
    _INTERACTIVE_MENU_COLUMN_HEIGHT,
)
_COMPACT_ROW_GAP: float = 0.008
_COMPACT_BUTTON_GAP: float = 0.006
_COMPACT_ROW_HEIGHT: float = 0.04
_VIEW_TOGGLE_WIDTH: float = 0.05
_FOCUS_MODE_BUTTON_WIDTH: float = 0.08
_FOCUS_RADIUS_BUTTON_WIDTH: float = 0.032
_FOCUS_CLEAR_BUTTON_WIDTH: float = 0.032
_VIEW_TOGGLE_BOUNDS: tuple[float, float, float, float] = (
    _INTERACTIVE_CHECKBOX_AXES_BOUNDS[0],
    _INTERACTIVE_CHECKBOX_AXES_BOUNDS[1] + _INTERACTIVE_CHECKBOX_AXES_BOUNDS[3] + _COMPACT_ROW_GAP,
    _VIEW_TOGGLE_WIDTH,
    _COMPACT_ROW_HEIGHT,
)
_BASE_TOGGLE_LABELS: tuple[str, str, str, str] = (
    "Hover",
    "Nodes",
    "Tensor labels",
    "Edge labels",
)
_SCHEME_TOGGLE_LABELS: tuple[str, str] = ("Scheme", "Costs")
_TENSOR_INSPECTOR_LABEL: str = "Tensor inspector"
_DIAGNOSTICS_LABEL: str = "Dimensions"
_INTERACTIVE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}
_COMPACT_BUTTON_FONT_SIZE: float = 8.5
_FOCUS_MODE_OPTIONS: tuple[str, str] = ("neighborhood", "path")
_FOCUS_RADIUS_OPTIONS: tuple[str, str] = ("1", "2")


@dataclass(frozen=True)
class _InteractiveControlsLayout:
    include_view_selector: bool
    include_scheme_toggles: bool
    include_tensor_inspector: bool
    include_diagnostics: bool = False
    include_focus_controls: bool = False


def _interactive_checkbox_bounds(
    *,
    include_scheme_toggles: bool,
    include_tensor_inspector: bool,
) -> tuple[float, float, float, float]:
    _ = include_scheme_toggles, include_tensor_inspector
    return _INTERACTIVE_CHECKBOX_AXES_BOUNDS


def _alternate_view(view: ViewName) -> ViewName:
    return "3d" if view == "2d" else "2d"


def _view_toggle_label(view: ViewName) -> str:
    return _alternate_view(view).upper()


def _style_compact_button(button: Button) -> None:
    button.label.set_fontsize(_COMPACT_BUTTON_FONT_SIZE)


def _compact_button_bounds(*, left: float, width: float) -> tuple[float, float, float, float]:
    return (left, _VIEW_TOGGLE_BOUNDS[1], width, _VIEW_TOGGLE_BOUNDS[3])


def _alternate_focus_mode(mode: str) -> str:
    if mode == "off":
        return "neighborhood"
    if mode == "neighborhood":
        return "path"
    return "off"


def _focus_mode_button_label(mode: str) -> str:
    if mode == "off":
        return "Off"
    if mode == "neighborhood":
        return "Neighbor"
    return "Path"


def _alternate_focus_radius(radius: int) -> int:
    if int(radius) == 1:
        return 2
    return 1


def _focus_radius_button_label(radius: int) -> str:
    return str(int(radius))


class _InteractiveControlsPanel:
    def __init__(
        self,
        *,
        fig: Figure,
        layout: _InteractiveControlsLayout,
        initial_view: ViewName,
        initial_state: InteractiveFeatureState,
        on_view_selected: Callable[[ViewName], None],
        on_state_changed: Callable[[InteractiveFeatureState], None],
        initial_focus_mode: str = "neighborhood",
        initial_focus_radius: int = 1,
        on_focus_mode_selected: Callable[[str], None] | None = None,
        on_focus_radius_selected: Callable[[int], None] | None = None,
        on_focus_cleared: Callable[[], None] | None = None,
    ) -> None:
        self._figure = fig
        self._layout = layout
        self._on_view_selected = on_view_selected
        self._on_state_changed = on_state_changed
        self._on_focus_mode_selected = on_focus_mode_selected
        self._on_focus_radius_selected = on_focus_radius_selected
        self._on_focus_cleared = on_focus_cleared
        self._last_state = initial_state
        self._last_view: ViewName = initial_view
        self._last_focus_mode = initial_focus_mode
        self._last_focus_radius: int = int(initial_focus_radius)
        self._callback_guard: bool = False
        self.check_ax: Axes
        self.checkbuttons: CheckButtons
        self.view_toggle_ax: Axes | None = None
        self.view_toggle_button: Button | None = None
        self.focus_mode_ax: Axes | None = None
        self.focus_mode_button: Button | None = None
        self.focus_radius_ax: Axes | None = None
        self.focus_radius_button: Button | None = None
        self.focus_clear_ax: Axes | None = None
        self.focus_clear_button: Button | None = None
        self._build()

    def _labels(self) -> list[str]:
        labels = list(_BASE_TOGGLE_LABELS)
        if self._layout.include_scheme_toggles:
            labels.extend(_SCHEME_TOGGLE_LABELS)
        if self._layout.include_tensor_inspector:
            labels.append(_TENSOR_INSPECTOR_LABEL)
        if self._layout.include_diagnostics:
            labels.append(_DIAGNOSTICS_LABEL)
        return labels

    def _statuses(self, state: InteractiveFeatureState) -> list[bool]:
        statuses = [
            state.hover,
            state.nodes,
            state.tensor_labels,
            state.edge_labels,
        ]
        if self._layout.include_scheme_toggles:
            statuses.extend([state.scheme, state.cost_hover])
        if self._layout.include_tensor_inspector:
            statuses.append(state.tensor_inspector)
        if self._layout.include_diagnostics:
            statuses.append(state.diagnostics)
        return statuses

    def _build(self) -> None:
        cb_bounds = _interactive_checkbox_bounds(
            include_scheme_toggles=self._layout.include_scheme_toggles,
            include_tensor_inspector=self._layout.include_tensor_inspector,
        )
        check_ax = self._figure.add_axes(cb_bounds)
        _style_control_tray_axes(check_ax)
        self.check_ax = check_ax
        self.checkbuttons = CheckButtons(
            check_ax,
            self._labels(),
            self._statuses(self._last_state),
            label_props=_INTERACTIVE_LABEL_PROPS,
            frame_props=_INTERACTIVE_CHECK_FRAME_PROPS,
            check_props=_INTERACTIVE_CHECK_MARK_PROPS,
        )
        self.checkbuttons.on_clicked(self._on_toggle_clicked)
        compact_row_left = float(_VIEW_TOGGLE_BOUNDS[0])
        if self._layout.include_view_selector:
            view_toggle_ax = self._figure.add_axes(_VIEW_TOGGLE_BOUNDS)
            _style_control_tray_axes(view_toggle_ax)
            self.view_toggle_ax = view_toggle_ax
            self.view_toggle_button = _SafeButton(
                view_toggle_ax,
                _view_toggle_label(self._last_view),
            )
            _style_compact_button(self.view_toggle_button)
            self.view_toggle_button.on_clicked(self._on_view_toggle_clicked)
            compact_row_left = float(
                _VIEW_TOGGLE_BOUNDS[0] + _VIEW_TOGGLE_BOUNDS[2] + _COMPACT_BUTTON_GAP
            )
        if not self._layout.include_focus_controls:
            return
        self.focus_mode_ax = self._figure.add_axes(
            _compact_button_bounds(left=compact_row_left, width=_FOCUS_MODE_BUTTON_WIDTH)
        )
        _style_control_tray_axes(self.focus_mode_ax)
        self.focus_mode_button = _SafeButton(
            self.focus_mode_ax,
            _focus_mode_button_label(self._last_focus_mode),
        )
        _style_compact_button(self.focus_mode_button)
        self.focus_mode_button.on_clicked(self._on_focus_mode_clicked)

        compact_row_left += _FOCUS_MODE_BUTTON_WIDTH + _COMPACT_BUTTON_GAP
        self.focus_radius_ax = self._figure.add_axes(
            _compact_button_bounds(left=compact_row_left, width=_FOCUS_RADIUS_BUTTON_WIDTH)
        )
        _style_control_tray_axes(self.focus_radius_ax)
        self.focus_radius_button = _SafeButton(
            self.focus_radius_ax,
            _focus_radius_button_label(self._last_focus_radius),
        )
        _style_compact_button(self.focus_radius_button)
        self.focus_radius_button.on_clicked(self._on_focus_radius_clicked)

        compact_row_left += _FOCUS_RADIUS_BUTTON_WIDTH + _COMPACT_BUTTON_GAP
        self.focus_clear_ax = self._figure.add_axes(
            _compact_button_bounds(left=compact_row_left, width=_FOCUS_CLEAR_BUTTON_WIDTH)
        )
        _style_control_tray_axes(self.focus_clear_ax)
        self.focus_clear_button = _SafeButton(self.focus_clear_ax, "x")
        _style_compact_button(self.focus_clear_button)
        self.focus_clear_button.on_clicked(self._on_focus_clear_clicked)
        self._sync_focus_button_labels()

    def sync(
        self,
        *,
        state: InteractiveFeatureState,
        view: ViewName,
        focus_mode: str | None = None,
        focus_radius: int | None = None,
    ) -> None:
        self._last_state = state
        self._last_view = view
        if focus_mode is not None:
            self._last_focus_mode = focus_mode
        if focus_radius is not None:
            self._last_focus_radius = int(focus_radius)
        self._callback_guard = True
        try:
            self._sync_view_toggle_label()
            self._sync_focus_button_labels()
            current = [bool(value) for value in self.checkbuttons.get_status()]
            for index, value in enumerate(self._statuses(state)):
                if index < len(current) and current[index] != value:
                    self.checkbuttons.set_active(index, state=value)
        finally:
            self._callback_guard = False

    def _sync_view_toggle_label(self) -> None:
        if self.view_toggle_button is None:
            return
        label = _view_toggle_label(self._last_view)
        if self.view_toggle_button.label.get_text() != label:
            self.view_toggle_button.label.set_text(label)

    def _sync_focus_button_labels(self) -> None:
        if self.focus_mode_button is not None:
            mode_label = _focus_mode_button_label(self._last_focus_mode)
            if self.focus_mode_button.label.get_text() != mode_label:
                self.focus_mode_button.label.set_text(mode_label)
        if self.focus_radius_button is not None:
            radius_label = _focus_radius_button_label(self._last_focus_radius)
            if self.focus_radius_button.label.get_text() != radius_label:
                self.focus_radius_button.label.set_text(radius_label)
        if self.focus_clear_button is not None and self.focus_clear_button.label.get_text() != "x":
            self.focus_clear_button.label.set_text("x")
        radius_visible = self._last_focus_mode == "neighborhood"
        clear_visible = self._last_focus_mode != "off"
        if self.focus_radius_ax is not None:
            _set_axes_visible(self.focus_radius_ax, radius_visible)
        if self.focus_clear_ax is not None:
            _set_axes_visible(self.focus_clear_ax, clear_visible)

    def _state_from_status(self, status: list[bool]) -> InteractiveFeatureState:
        scheme = self._last_state.scheme
        playback = self._last_state.playback
        cost_hover = self._last_state.cost_hover
        tensor_inspector = self._last_state.tensor_inspector
        diagnostics = self._last_state.diagnostics
        if self._layout.include_scheme_toggles:
            scheme_index = len(_BASE_TOGGLE_LABELS)
            scheme = status[scheme_index]
            cost_hover = status[scheme_index + 1]
            next_index = scheme_index + 2
            if self._layout.include_tensor_inspector:
                tensor_inspector = status[next_index]
                next_index += 1
            if self._layout.include_diagnostics:
                diagnostics = status[next_index]
            playback = bool(scheme or cost_hover)
        else:
            next_index = len(_BASE_TOGGLE_LABELS)
            if self._layout.include_tensor_inspector:
                tensor_inspector = status[next_index]
                next_index += 1
            if self._layout.include_diagnostics:
                diagnostics = status[next_index]
        return InteractiveFeatureState(
            hover=status[0],
            nodes=status[1],
            tensor_labels=status[2],
            edge_labels=status[3],
            scheme=scheme,
            playback=playback,
            cost_hover=cost_hover,
            tensor_inspector=tensor_inspector,
            diagnostics=diagnostics,
        )

    def _on_view_toggle_clicked(self, _event: object) -> None:
        if self._callback_guard:
            return
        self._last_view = _alternate_view(self._last_view)
        self._sync_view_toggle_label()
        self._on_view_selected(self._last_view)

    def _on_toggle_clicked(self, _label: str | None) -> None:
        if self._callback_guard:
            return
        status = [bool(value) for value in self.checkbuttons.get_status()]
        requested_state = self._state_from_status(status)
        self._last_state = requested_state
        self._on_state_changed(requested_state)

    def _on_focus_mode_clicked(self, _event: object) -> None:
        if self._callback_guard or self._on_focus_mode_selected is None:
            return
        self._last_focus_mode = _alternate_focus_mode(self._last_focus_mode)
        self._sync_focus_button_labels()
        self._on_focus_mode_selected(self._last_focus_mode)

    def _on_focus_radius_clicked(self, _event: object) -> None:
        if self._callback_guard or self._on_focus_radius_selected is None:
            return
        self._last_focus_radius = _alternate_focus_radius(self._last_focus_radius)
        self._sync_focus_button_labels()
        self._on_focus_radius_selected(self._last_focus_radius)

    def _on_focus_clear_clicked(self, _event: object) -> None:
        if self._callback_guard or self._on_focus_cleared is None:
            return
        self._on_focus_cleared()


__all__ = [
    "_InteractiveControlsLayout",
    "_InteractiveControlsPanel",
]
