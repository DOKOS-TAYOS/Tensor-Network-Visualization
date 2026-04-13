from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from matplotlib import patheffects

from ..graph import _GraphData

_EDGE_OUTLINE_COLOR: str = "black"
_EDGE_OUTLINE_LINEWIDTH_DELTA: float = 0.35
_COMPACT_NODE_MARKER_AREA_2D_PT2: float = 16.0
_COMPACT_NODE_MARKER_AREA_3D_PT2: float = 22.0
_COMPACT_NODE_MARKER_LINEWIDTH_PT: float = 0.7
_VIRTUAL_HUB_MARKER_AREA_2D_PT2: float = 24.0
_VIRTUAL_HUB_MARKER_AREA_3D_PT2: float = 30.0
_VIRTUAL_HUB_MARKER_LINEWIDTH_PT: float = 0.75

NodeRenderMode: TypeAlias = Literal["normal", "compact"]


@dataclass(frozen=True)
class _NodeArtistBundle:
    mode: NodeRenderMode
    artists: tuple[Any, ...]
    hover_target: Any | None


def _edge_outline_effects(linewidth: float) -> list[patheffects.AbstractPathEffect]:
    outline_width = max(
        float(linewidth) + _EDGE_OUTLINE_LINEWIDTH_DELTA,
        float(linewidth) * 1.14,
    )
    return [patheffects.withStroke(linewidth=outline_width, foreground=_EDGE_OUTLINE_COLOR)]


def _node_edge_degrees(graph: _GraphData) -> dict[int, int]:
    """Incident edge count per node (contractions, dangling, self-loops).

    One pass over ``graph.edges``.
    """
    counts: dict[int, int] = {}
    for edge in graph.edges:
        for nid in edge.node_ids:
            counts[int(nid)] = counts.get(int(nid), 0) + 1
    return counts


def _graph_edge_degree(graph: _GraphData, node_id: int) -> int:
    """Number of graph edges incident on *node_id* (contractions, dangling stubs, self-loops)."""
    target = int(node_id)
    c = 0
    for edge in graph.edges:
        for nid in edge.node_ids:
            if int(nid) == target:
                c += 1
    return c


def _visible_degree_one_mask(
    graph: _GraphData,
    visible_node_ids: list[int],
    *,
    node_degrees: dict[int, int] | None = None,
) -> np.ndarray:
    """True when a visible tensor has total graph degree 1."""
    d = node_degrees if node_degrees is not None else _node_edge_degrees(graph)
    return np.array([d.get(int(nid), 0) == 1 for nid in visible_node_ids], dtype=bool)


__all__ = [
    "NodeRenderMode",
    "_COMPACT_NODE_MARKER_AREA_2D_PT2",
    "_COMPACT_NODE_MARKER_AREA_3D_PT2",
    "_COMPACT_NODE_MARKER_LINEWIDTH_PT",
    "_NodeArtistBundle",
    "_VIRTUAL_HUB_MARKER_AREA_2D_PT2",
    "_VIRTUAL_HUB_MARKER_AREA_3D_PT2",
    "_VIRTUAL_HUB_MARKER_LINEWIDTH_PT",
    "_edge_outline_effects",
    "_graph_edge_degree",
    "_node_edge_degrees",
    "_visible_degree_one_mask",
]
