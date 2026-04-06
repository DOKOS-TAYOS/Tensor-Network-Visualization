from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from matplotlib.artist import Artist

from ...config import PlotConfig
from ..contractions import _ContractionGroups
from ..graph import _EdgeData, _GraphData
from ..layout import AxisDirections, NodePositions
from .fonts_and_scale import _DrawScaleParams
from .hover import _RenderHoverState
from .label_descriptors import _AnyLabelDescriptor, _TextLabelDescriptor
from .plotter import NodeRenderMode, _NodeArtistBundle, _PlotAdapter


@dataclass(frozen=True)
class _RenderedEdgeGeometry:
    edge: _EdgeData
    polyline: np.ndarray


@dataclass
class _InteractiveSceneState:
    ax: Any
    graph: _GraphData
    positions: NodePositions
    directions: AxisDirections
    config: PlotConfig
    dimensions: Literal[2, 3]
    scale: float
    params: _DrawScaleParams
    contraction_groups: _ContractionGroups
    plotter: _PlotAdapter
    visible_node_ids: tuple[int, ...]
    node_patch_coll: Any
    edge_geometry: tuple[_RenderedEdgeGeometry, ...]
    hover_state: _RenderHoverState
    tensor_disk_radius_px_3d: float | None
    edge_artists: list[Artist] = field(default_factory=list)
    scheme_artists: list[Artist] = field(default_factory=list)
    node_artist_bundles: dict[NodeRenderMode, _NodeArtistBundle] = field(default_factory=dict)
    active_node_mode: NodeRenderMode = "normal"
    tensor_label_artists: list[Artist] = field(default_factory=list)
    edge_label_artists: list[Artist] = field(default_factory=list)
    tensor_label_descriptors: tuple[_AnyLabelDescriptor, ...] | None = None
    edge_label_descriptors: tuple[_AnyLabelDescriptor, ...] | None = None
    tensor_hover_payload: dict[int, tuple[str, float]] | None = None
    edge_hover_payload: tuple[tuple[np.ndarray, str], ...] | None = None
    contraction_controls: Any = None


__all__ = [
    "_InteractiveSceneState",
    "_RenderedEdgeGeometry",
    "_AnyLabelDescriptor",
    "_TextLabelDescriptor",
]
