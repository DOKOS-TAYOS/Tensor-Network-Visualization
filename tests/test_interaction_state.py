from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text

from tensor_network_viz._interaction.bridge import (
    clear_contraction_controls,
    clear_hover_annotation,
    clear_hover_cid,
    clear_scene,
    get_artist_node_id,
    get_contraction_controls,
    get_hover_annotation,
    get_hover_cid,
    get_reserved_bottom,
    get_scene,
    get_zoom_cids,
    get_zoom_font_state,
    set_active_axes,
    set_artist_node_id,
    set_contraction_controls,
    set_contraction_viewer,
    set_hover_annotation,
    set_hover_cid,
    set_interactive_controls,
    set_reserved_bottom,
    set_scene,
    set_tensor_elements_controls,
    set_zoom_cids,
    set_zoom_font_state,
)
from tensor_network_viz._interaction.state import (
    InteractiveFeatureAvailability,
    InteractiveFeatureState,
    normalize_feature_state,
)


def test_bridge_round_trips_matplotlib_attrs() -> None:
    fig, ax = plt.subplots()
    artist = Text(0.0, 0.0, "node")
    try:
        assert get_reserved_bottom(fig) == 0.02
        set_reserved_bottom(fig, 0.3)
        assert get_reserved_bottom(fig) == 0.3

        scene = object()
        controls = object()
        viewer = object()
        interactive_controls = object()
        tensor_elements_controls = object()
        hover_annotation = object()

        assert get_scene(ax) is None
        set_scene(ax, scene)
        assert get_scene(ax) is scene
        clear_scene(ax)
        assert get_scene(ax) is None

        assert get_contraction_controls(fig) is None
        set_contraction_controls(fig, controls)
        assert get_contraction_controls(fig) is controls
        clear_contraction_controls(fig)
        assert get_contraction_controls(fig) is None

        set_contraction_viewer(fig, viewer)
        assert fig._tensor_network_viz_contraction_viewer is viewer

        set_interactive_controls(fig, interactive_controls)
        assert fig._tensor_network_viz_interactive_controls is interactive_controls

        set_active_axes(fig, ax)
        assert fig._tensor_network_viz_active_axes is ax

        set_tensor_elements_controls(fig, tensor_elements_controls)
        assert fig._tensor_network_viz_tensor_elements_controls is (tensor_elements_controls)

        assert get_hover_cid(fig) is None
        set_hover_cid(fig, 17)
        assert get_hover_cid(fig) == 17
        clear_hover_cid(fig)
        assert get_hover_cid(fig) is None

        assert get_hover_annotation(fig) is None
        set_hover_annotation(fig, hover_annotation)
        assert get_hover_annotation(fig) is hover_annotation
        clear_hover_annotation(fig)
        assert get_hover_annotation(fig) is None

        assert get_zoom_font_state(ax) is None
        set_zoom_font_state(ax, ref_span=2.5, sizes={artist: 9.0})
        assert get_zoom_font_state(ax) == {"ref_span": 2.5, "sizes": {artist: 9.0}}

        assert get_zoom_cids(ax) == []
        set_zoom_cids(ax, [11, 13])
        assert get_zoom_cids(ax) == [11, 13]

        assert get_artist_node_id(artist) is None
        set_artist_node_id(artist, 23)
        assert get_artist_node_id(artist) == 23
    finally:
        plt.close(fig)


def test_normalize_feature_state_enforces_dependency_rules() -> None:
    availability = InteractiveFeatureAvailability(
        scheme=True,
        playback=True,
        cost_hover=True,
        tensor_inspector=True,
    )
    requested = InteractiveFeatureState(
        hover=True,
        tensor_labels=False,
        edge_labels=False,
        scheme=False,
        playback=False,
        cost_hover=True,
        tensor_inspector=False,
    )

    resolved = normalize_feature_state(requested, availability)

    assert resolved.hover is True
    assert resolved.tensor_labels is False
    assert resolved.edge_labels is False
    assert resolved.scheme is True
    assert resolved.playback is True
    assert resolved.cost_hover is True
    assert resolved.tensor_inspector is False


def test_normalize_feature_state_turns_off_unavailable_features() -> None:
    availability = InteractiveFeatureAvailability(
        scheme=True,
        playback=True,
        cost_hover=True,
        tensor_inspector=False,
    )
    requested = InteractiveFeatureState(
        hover=False,
        tensor_labels=True,
        edge_labels=True,
        scheme=False,
        playback=False,
        cost_hover=False,
        tensor_inspector=True,
    )

    resolved = normalize_feature_state(requested, availability)

    assert resolved.hover is False
    assert resolved.tensor_labels is True
    assert resolved.edge_labels is True
    assert resolved.scheme is False
    assert resolved.playback is False
    assert resolved.cost_hover is False
    assert resolved.tensor_inspector is False
