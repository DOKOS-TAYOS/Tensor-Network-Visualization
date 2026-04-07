from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, RadioButtons

from .._ui_utils import _style_control_tray_axes
from ..config import ViewName
from .state import InteractiveFeatureState

# Menu column: fixed bottom = tallest stack (inspector + scheme). Without playback, checkboxes/radio
# stay as low as when the bottom row exists. Top aligned with cost-details top.
_VIEW_SELECTOR_LEFT: float = 0.213
_VIEW_SELECTOR_WIDTH: float = 0.053
_VIEW_SELECTOR_HEIGHT: float = 0.063
_INTERACTIVE_MENU_COLUMN_HEIGHT: float = 0.172
_INTERACTIVE_MENU_COLUMN_BOTTOM: float = 0.236 - _INTERACTIVE_MENU_COLUMN_HEIGHT
_INTERACTIVE_CHECKBOX_AXES_BOUNDS: tuple[float, float, float, float] = (
    0.02,
    _INTERACTIVE_MENU_COLUMN_BOTTOM,
    0.19,
    _INTERACTIVE_MENU_COLUMN_HEIGHT,
)
_BASE_TOGGLE_LABELS: tuple[str, str, str, str] = (
    "Hover",
    "Nodes",
    "Tensor labels",
    "Edge labels",
)
_SCHEME_TOGGLE_LABELS: tuple[str, str] = ("Scheme", "Costs")
_TENSOR_INSPECTOR_LABEL: str = "Tensor inspector"
_INTERACTIVE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.5]}
_INTERACTIVE_CHECK_FRAME_PROPS: dict[str, float] = {"s": 44.0, "linewidth": 0.9}
_INTERACTIVE_CHECK_MARK_PROPS: dict[str, float] = {"s": 34.0, "linewidth": 1.0}
_INTERACTIVE_RADIO_PROPS: dict[str, float] = {"s": 38.0, "linewidth": 0.9}


@dataclass(frozen=True)
class _InteractiveControlsLayout:
    include_view_selector: bool
    include_scheme_toggles: bool
    include_tensor_inspector: bool


def _interactive_checkbox_bounds(
    *,
    include_scheme_toggles: bool,
    include_tensor_inspector: bool,
) -> tuple[float, float, float, float]:
    _ = include_scheme_toggles, include_tensor_inspector
    return _INTERACTIVE_CHECKBOX_AXES_BOUNDS


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
    ) -> None:
        self._figure = fig
        self._layout = layout
        self._on_view_selected = on_view_selected
        self._on_state_changed = on_state_changed
        self._last_state = initial_state
        self._last_view = initial_view
        self._callback_guard: bool = False
        self.check_ax: Axes
        self.checkbuttons: CheckButtons
        self.radio_ax: Axes | None = None
        self.radio: RadioButtons | None = None
        self._build()

    def _labels(self) -> list[str]:
        labels = list(_BASE_TOGGLE_LABELS)
        if self._layout.include_scheme_toggles:
            labels.extend(_SCHEME_TOGGLE_LABELS)
        if self._layout.include_tensor_inspector:
            labels.append(_TENSOR_INSPECTOR_LABEL)
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
        return statuses

    def _build(self) -> None:
        cb_bounds = _interactive_checkbox_bounds(
            include_scheme_toggles=self._layout.include_scheme_toggles,
            include_tensor_inspector=self._layout.include_tensor_inspector,
        )
        cb_bottom = float(cb_bounds[1])
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
        if not self._layout.include_view_selector:
            return
        radio_bounds: tuple[float, float, float, float] = (
            _VIEW_SELECTOR_LEFT,
            cb_bottom,
            _VIEW_SELECTOR_WIDTH,
            _VIEW_SELECTOR_HEIGHT,
        )
        radio_ax = self._figure.add_axes(radio_bounds)
        _style_control_tray_axes(radio_ax)
        self.radio_ax = radio_ax
        self.radio = RadioButtons(
            radio_ax,
            ("2d", "3d"),
            active=0 if self._last_view == "2d" else 1,
            label_props=_INTERACTIVE_LABEL_PROPS,
            radio_props=_INTERACTIVE_RADIO_PROPS,
        )
        self.radio.on_clicked(self._on_view_clicked)

    def sync(self, *, state: InteractiveFeatureState, view: ViewName) -> None:
        self._last_state = state
        self._last_view = view
        self._callback_guard = True
        try:
            if self.radio is not None and self.radio.value_selected != view:
                self.radio.set_active(0 if view == "2d" else 1)
            current = [bool(value) for value in self.checkbuttons.get_status()]
            for index, value in enumerate(self._statuses(state)):
                if index < len(current) and current[index] != value:
                    self.checkbuttons.set_active(index, state=value)
        finally:
            self._callback_guard = False

    def _state_from_status(self, status: list[bool]) -> InteractiveFeatureState:
        scheme = self._last_state.scheme
        playback = self._last_state.playback
        cost_hover = self._last_state.cost_hover
        tensor_inspector = self._last_state.tensor_inspector
        if self._layout.include_scheme_toggles:
            scheme_index = len(_BASE_TOGGLE_LABELS)
            scheme = status[scheme_index]
            cost_hover = status[scheme_index + 1]
            if self._layout.include_tensor_inspector:
                tensor_inspector = status[scheme_index + 2]
            playback = bool(scheme or cost_hover or tensor_inspector)
        return InteractiveFeatureState(
            hover=status[0],
            nodes=status[1],
            tensor_labels=status[2],
            edge_labels=status[3],
            scheme=scheme,
            playback=playback,
            cost_hover=cost_hover,
            tensor_inspector=tensor_inspector,
        )

    def _on_view_clicked(self, label: str | None) -> None:
        if self._callback_guard or label is None:
            return
        self._last_view = cast(ViewName, label)
        self._on_view_selected(self._last_view)

    def _on_toggle_clicked(self, _label: str | None) -> None:
        if self._callback_guard:
            return
        status = [bool(value) for value in self.checkbuttons.get_status()]
        requested_state = self._state_from_status(status)
        self._last_state = requested_state
        self._on_state_changed(requested_state)


__all__ = [
    "_InteractiveControlsLayout",
    "_InteractiveControlsPanel",
]
