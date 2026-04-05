from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import ViewName

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
    "normalize_feature_state",
]
