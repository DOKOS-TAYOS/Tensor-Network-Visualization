from __future__ import annotations

from typing import Any, Literal, Protocol, cast

import numpy as np

from ...config import PlotConfig
from .fonts_and_scale import _DrawScaleParams
from .plotter_2d import _make_2d_plotter
from .plotter_3d import _make_3d_plotter
from .plotter_common import (
    NodeRenderMode,
    _edge_outline_effects,
    _graph_edge_degree,
    _node_edge_degrees,
    _NodeArtistBundle,
    _visible_degree_one_mask,
)


class _PlotAdapter(Protocol):
    """Protocol for dimension-aware plotting (2D vs 3D)."""

    def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None: ...
    def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None: ...
    def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None: ...
    def draw_tensor_nodes(
        self,
        coords: np.ndarray,
        *,
        config: PlotConfig,
        p: _DrawScaleParams,
        degree_one_mask: np.ndarray,
        mode: NodeRenderMode,
    ) -> None: ...
    def draw_virtual_hub_markers(
        self,
        coords: np.ndarray,
        *,
        config: PlotConfig,
        zorder: float,
    ) -> None: ...
    def get_node_artist_bundle(self) -> _NodeArtistBundle | None: ...
    def get_edge_artists(self) -> tuple[Any, ...]: ...
    def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None: ...


def _make_plotter(
    ax: Any,
    *,
    dimensions: Literal[2, 3],
    hover_edge_targets: list[tuple[np.ndarray, str]] | None = None,
) -> _PlotAdapter:
    """Create a dimension-aware plot adapter."""
    if dimensions == 2:
        return cast(_PlotAdapter, _make_2d_plotter(ax, hover_edge_targets=hover_edge_targets))
    return cast(_PlotAdapter, _make_3d_plotter(ax, hover_edge_targets=hover_edge_targets))


__all__ = [
    "_PlotAdapter",
    "_NodeArtistBundle",
    "_edge_outline_effects",
    "_graph_edge_degree",
    "_make_plotter",
    "_node_edge_degrees",
    "_visible_degree_one_mask",
]
