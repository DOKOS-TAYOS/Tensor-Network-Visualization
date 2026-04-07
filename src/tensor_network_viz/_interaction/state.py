from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig, ViewName

if TYPE_CHECKING:
    from .._core.draw.scene_state import _InteractiveSceneState
else:
    _InteractiveSceneState = Any


@dataclass(frozen=True)
class InteractiveFeatureAvailability:
    scheme: bool
    playback: bool
    cost_hover: bool
    tensor_inspector: bool


@dataclass(frozen=True)
class InteractiveFeatureState:
    hover: bool
    nodes: bool
    tensor_labels: bool
    edge_labels: bool
    scheme: bool
    playback: bool
    cost_hover: bool
    tensor_inspector: bool


RenderedAxes = Axes | Axes3D


@dataclass
class InteractiveViewCache:
    view: ViewName
    ax: RenderedAxes | None = None
    scene: _InteractiveSceneState | None = None


def feature_availability_from_scene(
    scene: _InteractiveSceneState | None,
    *,
    tensor_inspector_available: bool,
) -> InteractiveFeatureAvailability:
    controls = getattr(scene, "contraction_controls", None) if scene is not None else None
    bundle = getattr(controls, "_bundle", None)
    bundle_availability = getattr(bundle, "availability", "computed")
    scheme_available = bool(controls is not None and bundle_availability != "unavailable")
    playback_available = bool(scheme_available)
    cost_hover_available = bool(playback_available)
    return InteractiveFeatureAvailability(
        scheme=scheme_available,
        playback=playback_available,
        cost_hover=cost_hover_available,
        tensor_inspector=bool(tensor_inspector_available and playback_available),
    )


def feature_state_from_config(
    config: PlotConfig,
    *,
    tensor_inspector_available: bool,
) -> InteractiveFeatureState:
    requested = InteractiveFeatureState(
        hover=bool(config.hover_labels),
        nodes=bool(config.show_nodes),
        tensor_labels=bool(config.show_tensor_labels),
        edge_labels=bool(config.show_index_labels),
        scheme=bool(config.show_contraction_scheme),
        playback=bool(config.show_contraction_scheme),
        cost_hover=bool(config.contraction_scheme_cost_hover),
        tensor_inspector=bool(config.contraction_tensor_inspector),
    )
    return normalize_feature_state(
        requested,
        InteractiveFeatureAvailability(
            scheme=True,
            playback=True,
            cost_hover=True,
            tensor_inspector=bool(tensor_inspector_available),
        ),
    )


def normalize_feature_state(
    requested: InteractiveFeatureState,
    availability: InteractiveFeatureAvailability,
) -> InteractiveFeatureState:
    scheme = bool(requested.scheme and availability.scheme)
    playback = bool(requested.playback and availability.playback)
    cost_hover = bool(requested.cost_hover and availability.cost_hover)
    tensor_inspector = bool(requested.tensor_inspector and availability.tensor_inspector)

    if cost_hover or tensor_inspector:
        if availability.playback:
            playback = True
        if availability.scheme:
            scheme = True
    elif playback and availability.scheme:
        scheme = True

    if not availability.scheme:
        scheme = False
    if not availability.playback:
        playback = False
    if not availability.cost_hover:
        cost_hover = False
    if not availability.tensor_inspector:
        tensor_inspector = False

    return InteractiveFeatureState(
        hover=bool(requested.hover),
        nodes=bool(requested.nodes),
        tensor_labels=bool(requested.tensor_labels),
        edge_labels=bool(requested.edge_labels),
        scheme=scheme,
        playback=playback,
        cost_hover=cost_hover,
        tensor_inspector=tensor_inspector,
    )


__all__ = [
    "InteractiveFeatureAvailability",
    "InteractiveFeatureState",
    "InteractiveViewCache",
    "feature_availability_from_scene",
    "feature_state_from_config",
    "normalize_feature_state",
]
