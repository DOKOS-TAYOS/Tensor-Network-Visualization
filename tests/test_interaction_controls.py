from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_hex

from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    TensorNetworkDiagnosticsConfig,
    TensorNetworkFocus,
    einsum,
    show_tensor_network,
)
from tensor_network_viz._interaction.controls import (
    _InteractiveControlsLayout,
    _InteractiveControlsPanel,
)
from tensor_network_viz._interaction.state import InteractiveFeatureState
from tensor_network_viz._matplotlib_state import get_scene


def _einsum_trace() -> EinsumTrace:
    trace = EinsumTrace()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    mid = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)

    trace.bind("A", left)
    trace.bind("B", mid)
    trace.bind("C", right)
    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, r0, r1]  # type: ignore[attr-defined]
    return trace


def _dispatch_button_event_at_data(
    ax: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
) -> MouseEvent:
    ax.figure.canvas.draw()
    x_display, y_display = ax.transData.transform((x, y))
    event = MouseEvent(
        "button_press_event",
        ax.figure.canvas,
        int(round(x_display)),
        int(round(y_display)),
        button=MouseButton.LEFT,
    )
    ax.figure.canvas.callbacks.process("button_press_event", event)
    return event


def _click_button(button: object) -> None:
    button_widget = button
    fig = button_widget.ax.figure  # type: ignore[attr-defined]
    fig.canvas.draw()
    bbox = button_widget.ax.get_window_extent(fig.canvas.get_renderer())  # type: ignore[attr-defined]
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", press)
    fig.canvas.callbacks.process("button_release_event", release)


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
                include_diagnostics=True,
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
        assert panel.view_toggle_button is not None
        assert panel.view_toggle_button.label.get_text() == "2D"
        assert tuple(bool(value) for value in panel.checkbuttons.get_status()) == (
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
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
                include_diagnostics=True,
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
            diagnostics=False,
        )

        assert panel.view_toggle_button is not None
        _click_button(panel.view_toggle_button)
        assert view_events == ["3d"]
        assert panel.view_toggle_button.label.get_text() == "2D"
    finally:
        plt.close(fig)


def test_interactive_controls_panel_keeps_playback_off_when_only_tensor_inspector_is_enabled() -> (
    None
):
    fig = plt.figure()
    state_events: list[InteractiveFeatureState] = []
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
                include_diagnostics=False,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=lambda _view: None,
            on_state_changed=state_events.append,
        )

        panel.checkbuttons.set_active(6)

        assert state_events[-1] == InteractiveFeatureState(
            hover=True,
            nodes=True,
            tensor_labels=False,
            edge_labels=False,
            scheme=False,
            playback=False,
            cost_hover=False,
            tensor_inspector=True,
            diagnostics=False,
        )
    finally:
        plt.close(fig)


def test_show_tensor_network_diagnostics_overlay_enriches_scene_payloads() -> None:
    fig, ax = show_tensor_network(
        _einsum_trace(),
        engine="einsum",
        config=PlotConfig(
            diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=True),
        ),
        show=False,
    )
    controls = fig._tensor_network_viz_interactive_controls  # type: ignore[attr-defined]
    scene = get_scene(ax)

    assert controls.diagnostics_on is True
    assert scene is not None
    assert scene.diagnostic_artists
    diagnostic_texts = [artist.get_text() for artist in scene.diagnostic_artists]
    assert any(text.startswith("(") for text in diagnostic_texts)
    assert any(text.isdigit() for text in diagnostic_texts)
    assert all("shape=" not in text.lower() for text in diagnostic_texts)
    assert all("chi=" not in text.lower() for text in diagnostic_texts)
    assert all(
        to_hex(artist.get_color()).lower() == "#000000" for artist in scene.diagnostic_artists
    )
    diagnostic_bbox_patches = [artist.get_bbox_patch() for artist in scene.diagnostic_artists]
    assert all(patch is not None for patch in diagnostic_bbox_patches)
    assert all(
        patch.get_facecolor()[0] > 0.9
        and patch.get_facecolor()[1] > 0.9
        and patch.get_facecolor()[2] > 0.9
        and patch.get_facecolor()[3] > 0.7
        for patch in diagnostic_bbox_patches
    )
    assert scene.tensor_hover_payload is not None
    assert not any(
        "dtype:" in payload[0].lower() for payload in scene.tensor_hover_payload.values()
    )
    assert any("memory:" in payload[0].lower() for payload in scene.tensor_hover_payload.values())
    assert scene.edge_hover_payload is not None
    assert any("bond dimension" in payload[1].lower() for payload in scene.edge_hover_payload)


def test_show_tensor_network_focus_config_filters_scene_without_relayout() -> None:
    trace = _einsum_trace()

    full_fig, full_ax = show_tensor_network(trace, engine="einsum", show=False)
    focused_fig, focused_ax = show_tensor_network(
        trace,
        engine="einsum",
        config=PlotConfig(
            focus=TensorNetworkFocus(kind="path", endpoints=("A", "C")),
        ),
        show=False,
    )
    full_scene = get_scene(full_ax)
    focused_scene = get_scene(focused_ax)

    assert full_scene is not None
    assert focused_scene is not None
    assert {
        focused_scene.graph.nodes[node_id].name for node_id in focused_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }
    for node_id in focused_scene.visible_node_ids:
        assert np.allclose(focused_scene.positions[node_id], full_scene.positions[node_id])

    plt.close(full_fig)
    plt.close(focused_fig)


def test_interactive_controls_panel_sync_updates_focus_widgets_without_callbacks() -> None:
    fig = plt.figure()
    focus_mode_events: list[str] = []
    focus_radius_events: list[int] = []
    focus_clear_events: list[str] = []
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
                include_scheme_toggles=False,
                include_tensor_inspector=False,
                include_diagnostics=False,
                include_focus_controls=True,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=lambda _view: None,
            on_state_changed=lambda _state: None,
            initial_focus_mode="path",
            initial_focus_radius=2,
            on_focus_mode_selected=focus_mode_events.append,
            on_focus_radius_selected=focus_radius_events.append,
            on_focus_cleared=lambda: focus_clear_events.append("clear"),
        )

        panel.sync(
            state=initial_state,
            view="2d",
            focus_mode="off",
            focus_radius=1,
        )

        assert focus_mode_events == []
        assert focus_radius_events == []
        assert focus_clear_events == []
        assert panel.focus_mode_button is not None
        assert panel.focus_mode_button.label.get_text() == "Off"
        assert panel.focus_radius_button is not None
        assert panel.focus_radius_button.label.get_text() == "1"
        assert panel.focus_radius_button.ax.get_visible() is False
        assert panel.focus_clear_button is not None
        assert panel.focus_clear_button.label.get_text() == "x"
        assert panel.focus_clear_button.ax.get_visible() is False
    finally:
        plt.close(fig)


def test_interactive_controls_panel_focus_buttons_emit_callbacks_and_flip_labels() -> None:
    fig = plt.figure()
    focus_mode_events: list[str] = []
    focus_radius_events: list[int] = []
    focus_clear_events: list[str] = []
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
                include_scheme_toggles=False,
                include_tensor_inspector=False,
                include_diagnostics=False,
                include_focus_controls=True,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=lambda _view: None,
            on_state_changed=lambda _state: None,
            initial_focus_mode="off",
            initial_focus_radius=1,
            on_focus_mode_selected=focus_mode_events.append,
            on_focus_radius_selected=focus_radius_events.append,
            on_focus_cleared=lambda: focus_clear_events.append("clear"),
        )

        assert panel.focus_mode_button is not None
        assert panel.focus_radius_button is not None
        assert panel.focus_clear_button is not None

        assert panel.focus_mode_button.label.get_text() == "Off"
        assert panel.focus_radius_button.label.get_text() == "1"
        assert panel.focus_radius_button.ax.get_visible() is False
        assert panel.focus_clear_button.label.get_text() == "x"
        assert panel.focus_clear_button.ax.get_visible() is False

        _click_button(panel.focus_mode_button)
        assert focus_mode_events == ["neighborhood"]
        assert panel.focus_mode_button.label.get_text() == "Neighbor"
        assert panel.focus_radius_button.ax.get_visible() is True
        assert panel.focus_radius_button.label.get_text() == "1"
        assert panel.focus_clear_button.ax.get_visible() is True

        _click_button(panel.focus_radius_button)
        assert focus_radius_events == [2]
        assert panel.focus_radius_button.label.get_text() == "2"

        _click_button(panel.focus_mode_button)
        assert focus_mode_events == ["neighborhood", "path"]
        assert panel.focus_mode_button.label.get_text() == "Path"
        assert panel.focus_radius_button.ax.get_visible() is False
        assert panel.focus_clear_button.ax.get_visible() is True

        _click_button(panel.focus_clear_button)
        assert focus_clear_events == ["clear"]
        assert panel.focus_clear_button.label.get_text() == "x"

        _click_button(panel.focus_mode_button)
        assert focus_mode_events == ["neighborhood", "path", "off"]
        assert panel.focus_mode_button.label.get_text() == "Off"
        assert panel.focus_radius_button.ax.get_visible() is False
        assert panel.focus_clear_button.ax.get_visible() is False
    finally:
        plt.close(fig)


def test_interactive_focus_button_releases_stale_mouse_grabber_before_click() -> None:
    fig = plt.figure()
    focus_mode_events: list[str] = []
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
                include_scheme_toggles=False,
                include_tensor_inspector=False,
                include_diagnostics=False,
                include_focus_controls=True,
            ),
            initial_view="2d",
            initial_state=initial_state,
            on_view_selected=lambda _view: None,
            on_state_changed=lambda _state: None,
            initial_focus_mode="off",
            initial_focus_radius=1,
            on_focus_mode_selected=focus_mode_events.append,
        )
        stale_ax = fig.add_axes((0.9, 0.9, 0.05, 0.05))
        stale_ax.set_visible(False)

        assert panel.focus_mode_button is not None
        fig.canvas.grab_mouse(stale_ax)
        assert fig.canvas.mouse_grabber is stale_ax

        _click_button(panel.focus_mode_button)

        assert focus_mode_events == ["neighborhood"]
        assert fig.canvas.mouse_grabber is None
    finally:
        plt.close(fig)


def test_show_tensor_network_focus_controls_drive_click_selection_and_clear() -> None:
    fig, ax = show_tensor_network(_einsum_trace(), engine="einsum", show=False)
    controls = fig._tensor_network_viz_interactive_controls  # type: ignore[attr-defined]
    panel = controls._controls_panel
    scene = controls.current_scene

    assert panel is not None
    assert panel.focus_mode_button is not None
    assert panel.focus_mode_button.label.get_text() == "Off"
    assert panel.focus_radius_button is not None
    assert panel.focus_radius_button.label.get_text() == "1"
    assert panel.focus_radius_button.ax.get_visible() is False
    assert panel.focus_clear_button is not None
    assert panel.focus_clear_button.label.get_text() == "x"
    assert panel.focus_clear_button.ax.get_visible() is False
    assert {scene.graph.nodes[node_id].name for node_id in scene.visible_node_ids} == {
        "A",
        "B",
        "C",
    }

    node_id_a = next(node_id for node_id, node in scene.graph.nodes.items() if node.name == "A")
    position_a = np.asarray(scene.positions[node_id_a], dtype=float)
    _dispatch_button_event_at_data(ax, x=float(position_a[0]), y=float(position_a[1]))
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }

    _click_button(panel.focus_mode_button)
    assert controls.focus_mode == "neighborhood"
    assert panel.focus_mode_button.label.get_text() == "Neighbor"
    assert panel.focus_radius_button.ax.get_visible() is True
    assert panel.focus_clear_button.ax.get_visible() is True

    _dispatch_button_event_at_data(ax, x=float(position_a[0]), y=float(position_a[1]))
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
    }

    _click_button(panel.focus_radius_button)
    assert controls.focus_radius == 2
    assert panel.focus_radius_button.label.get_text() == "2"
    assert panel.focus_radius_button.ax.get_visible() is True

    _click_button(panel.focus_clear_button)
    assert controls.focus_mode == "neighborhood"
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }

    _click_button(panel.focus_mode_button)
    assert controls.focus_mode == "path"
    assert panel.focus_mode_button.label.get_text() == "Path"
    assert panel.focus_radius_button.ax.get_visible() is False
    assert panel.focus_clear_button.ax.get_visible() is True
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }
    _dispatch_button_event_at_data(ax, x=float(position_a[0]), y=float(position_a[1]))
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }

    node_id_b = next(
        node_id for node_id, node in controls.current_scene.graph.nodes.items() if node.name == "B"
    )
    position_b = np.asarray(controls.current_scene.positions[node_id_b], dtype=float)
    _dispatch_button_event_at_data(ax, x=float(position_b[0]), y=float(position_b[1]))
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
    }

    _click_button(panel.focus_mode_button)
    assert controls.focus_mode == "off"
    assert panel.focus_mode_button.label.get_text() == "Off"
    assert panel.focus_radius_button.ax.get_visible() is False
    assert panel.focus_clear_button.ax.get_visible() is False
    assert {
        controls.current_scene.graph.nodes[node_id].name
        for node_id in controls.current_scene.visible_node_ids
    } == {
        "A",
        "B",
        "C",
    }

    plt.close(fig)
