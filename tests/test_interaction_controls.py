from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensor_network_viz._interaction.controls import (
    _InteractiveControlsLayout,
    _InteractiveControlsPanel,
)
from tensor_network_viz._interaction.state import InteractiveFeatureState


def test_interactive_controls_panel_sync_updates_widgets_without_emitting_callbacks() -> None:
    fig = plt.figure()
    state_events: list[InteractiveFeatureState] = []
    view_events: list[str] = []
    initial_state = InteractiveFeatureState(
        hover=True,
        nodes=True,
        tensor_labels=False,
        edge_labels=False,
        scheme=False,
        playback=False,
        cost_hover=False,
        tensor_inspector=False,
    )
    try:
        panel = _InteractiveControlsPanel(
            fig=fig,
            layout=_InteractiveControlsLayout(
                include_view_selector=True,
                include_scheme_toggles=True,
                include_tensor_inspector=True,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=view_events.append,
            on_state_changed=state_events.append,
        )

        panel.sync(
            state=InteractiveFeatureState(
                hover=False,
                nodes=False,
                tensor_labels=True,
                edge_labels=True,
                scheme=True,
                playback=True,
                cost_hover=True,
                tensor_inspector=True,
            ),
            view="3d",
        )

        assert state_events == []
        assert view_events == []
        assert panel.radio is not None
        assert panel.radio.value_selected == "3d"
        assert tuple(bool(value) for value in panel.checkbuttons.get_status()) == (
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        )
    finally:
        plt.close(fig)


def test_interactive_controls_panel_emits_raw_requested_state_from_widget_changes() -> None:
    fig = plt.figure()
    state_events: list[InteractiveFeatureState] = []
    view_events: list[str] = []
    initial_state = InteractiveFeatureState(
        hover=True,
        nodes=True,
        tensor_labels=False,
        edge_labels=False,
        scheme=False,
        playback=False,
        cost_hover=False,
        tensor_inspector=False,
    )
    try:
        panel = _InteractiveControlsPanel(
            fig=fig,
            layout=_InteractiveControlsLayout(
                include_view_selector=True,
                include_scheme_toggles=True,
                include_tensor_inspector=True,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=view_events.append,
            on_state_changed=state_events.append,
        )

        panel.checkbuttons.set_active(5)
        panel.checkbuttons.set_active(6)
        assert state_events[-1] == InteractiveFeatureState(
            hover=True,
            nodes=True,
            tensor_labels=False,
            edge_labels=False,
            scheme=False,
            playback=True,
            cost_hover=True,
            tensor_inspector=True,
        )

        assert panel.radio is not None
        panel.radio.set_active(1)
        assert view_events == ["3d"]
    finally:
        plt.close(fig)
