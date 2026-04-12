from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
from matplotlib import patheffects
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection, PathCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...config import PlotConfig
from ..graph import (
    _GraphData,
)
from .constants import (
    _EDGE_LINE_CAP_STYLE,
    _EDGE_LINE_JOIN_STYLE,
    _OCTAHEDRON_EDGE_LINEWIDTH_FACTOR,
    _OCTAHEDRON_EDGE_LINEWIDTH_MIN,
    _OCTAHEDRON_TRI_COUNT,
    _UNIT_NODE_TRIS,
    _ZORDER_NODE_DISK,
)
from .fonts_and_scale import _DrawScaleParams
from .viewport_geometry import (
    _apply_axis_limits_with_outset,
    _apply_edge_line_style,
    _apply_text_no_clip,
)

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

        class _2DPlotter:
            __slots__ = (
                "_ax",
                "_edge_segments",
                "_hover_edge_targets",
                "_edge_artists",
                "_node_disk_collection",
                "_node_disk_collections",
                "_node_artist_bundle",
            )

            def __init__(
                self,
                ax_2d: Axes,
                hover_edges: list[tuple[np.ndarray, str]] | None,
            ) -> None:
                self._ax = ax_2d
                self._edge_segments: list[tuple[float, str, float, np.ndarray]] = []
                self._hover_edge_targets = hover_edges
                self._edge_artists: list[Artist] = []
                self._node_disk_collection: PatchCollection | PathCollection | None = None
                self._node_disk_collections: list[PatchCollection | PathCollection] = []
                self._node_artist_bundle: _NodeArtistBundle | None = None

            def clear_node_disk_collections(self) -> None:
                self._node_disk_collections.clear()
                self._node_disk_collection = None
                self._node_artist_bundle = None

            def get_node_artist_bundle(self) -> _NodeArtistBundle | None:
                return self._node_artist_bundle

            def get_edge_artists(self) -> tuple[Artist, ...]:
                return tuple(self._edge_artists)

            def flush_edge_collections(self) -> None:
                """Batch buffered edges into a few LineCollections (call after all edges drawn)."""
                if not self._edge_segments:
                    return
                groups: dict[tuple[float, str, float], list[np.ndarray]] = defaultdict(list)
                for z, color, lw, seg in self._edge_segments:
                    groups[(z, color, lw)].append(seg)
                ax_ = self._ax
                for (z, color, lw), segs in sorted(groups.items(), key=lambda kv: kv[0][0]):
                    coll = LineCollection(
                        segs,
                        colors=color,
                        linewidths=lw,
                        zorder=z,
                        capstyle=_EDGE_LINE_CAP_STYLE,
                        joinstyle=_EDGE_LINE_JOIN_STYLE,
                    )
                    coll.set_path_effects(_edge_outline_effects(float(lw)))
                    ax_.add_collection(coll, autolim=False)
                    self._edge_artists.append(coll)
                self._edge_segments.clear()

            def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
                _apply_edge_line_style(kwargs)
                z = float(kwargs.get("zorder", 1))
                color = str(kwargs.get("color", "#000000"))
                lw = float(kwargs.get("linewidth", 1.0))
                seg = np.array(
                    [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
                    dtype=float,
                )
                self._edge_segments.append((z, color, lw, seg))

            def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
                _apply_edge_line_style(kwargs)
                z = float(kwargs.get("zorder", 1))
                color = str(kwargs.get("color", "#000000"))
                lw = float(kwargs.get("linewidth", 1.0))
                seg = np.asarray(curve[:, :2], dtype=float, order="C")
                self._edge_segments.append((z, color, lw, seg))

            def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
                _apply_text_no_clip(kwargs)
                self._ax.text(pos[0], pos[1], text, **kwargs)

            def draw_tensor_node(
                self,
                coord: np.ndarray,
                *,
                config: PlotConfig,
                p: _DrawScaleParams,
                degree_one: bool,
                mode: NodeRenderMode,
                zorder: float,
            ) -> None:
                fc = config.node_color_degree_one if degree_one else config.node_color
                ec = config.node_edge_color_degree_one if degree_one else config.node_edge_color
                if mode == "compact":
                    coll = self._ax.scatter(
                        [float(coord[0])],
                        [float(coord[1])],
                        s=_COMPACT_NODE_MARKER_AREA_2D_PT2,
                        c=[fc],
                        edgecolors=[ec],
                        linewidths=_COMPACT_NODE_MARKER_LINEWIDTH_PT,
                        zorder=zorder,
                    )
                else:
                    patch = Circle((float(coord[0]), float(coord[1])), radius=p.r)
                    coll = PatchCollection(
                        [patch],
                        facecolors=[fc],
                        edgecolors=[ec],
                        linewidths=float(p.lw),
                        zorder=zorder,
                        match_original=False,
                    )
                    self._ax.add_collection(coll, autolim=False)
                self._node_disk_collections.append(coll)
                self._node_artist_bundle = _NodeArtistBundle(
                    mode=mode,
                    artists=tuple(self._node_disk_collections),
                    hover_target=tuple(self._node_disk_collections),
                )

            def draw_tensor_nodes(
                self,
                coords: np.ndarray,
                *,
                config: PlotConfig,
                p: _DrawScaleParams,
                degree_one_mask: np.ndarray,
                mode: NodeRenderMode,
            ) -> None:
                n = int(coords.shape[0])
                if n == 0:
                    return
                faces = [
                    config.node_color_degree_one if degree_one_mask[i] else config.node_color
                    for i in range(n)
                ]
                c1 = config.node_edge_color_degree_one
                c0 = config.node_edge_color
                edges_ = [c1 if degree_one_mask[i] else c0 for i in range(n)]
                if mode == "compact":
                    coll = self._ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        s=_COMPACT_NODE_MARKER_AREA_2D_PT2,
                        c=faces,
                        edgecolors=edges_,
                        linewidths=_COMPACT_NODE_MARKER_LINEWIDTH_PT,
                        zorder=_ZORDER_NODE_DISK,
                    )
                else:
                    patches = [
                        Circle((float(coords[i, 0]), float(coords[i, 1])), radius=p.r)
                        for i in range(n)
                    ]
                    coll = PatchCollection(
                        patches,
                        facecolors=faces,
                        edgecolors=edges_,
                        linewidths=float(p.lw),
                        zorder=_ZORDER_NODE_DISK,
                        match_original=False,
                    )
                    self._ax.add_collection(coll, autolim=False)
                self._node_disk_collection = coll
                self._node_disk_collections = [coll]
                self._node_artist_bundle = _NodeArtistBundle(
                    mode=mode,
                    artists=(coll,),
                    hover_target=coll,
                )

            def draw_virtual_hub_markers(
                self,
                coords: np.ndarray,
                *,
                config: PlotConfig,
                zorder: float,
            ) -> None:
                if int(coords.shape[0]) == 0:
                    return
                self._ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    marker="^",
                    s=_VIRTUAL_HUB_MARKER_AREA_2D_PT2,
                    c=[config.dangling_edge_color],
                    edgecolors=[config.node_edge_color_degree_one],
                    linewidths=_VIRTUAL_HUB_MARKER_LINEWIDTH_PT,
                    zorder=zorder,
                )

            def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None:
                _apply_axis_limits_with_outset(
                    self._ax, coords, view_margin=view_margin, dimensions=2
                )

        return _2DPlotter(ax, hover_edge_targets)

    class _3DPlotter:
        def __init__(self, hover_edges: list[tuple[np.ndarray, str]] | None) -> None:
            if hasattr(ax, "computed_zorder"):
                ax.computed_zorder = False
            self._hover_edge_targets = hover_edges
            self._edge_artists: list[Artist] = []
            self._node_artist_bundle: _NodeArtistBundle | None = None

        def get_node_artist_bundle(self) -> _NodeArtistBundle | None:
            return self._node_artist_bundle

        def get_edge_artists(self) -> tuple[Artist, ...]:
            return tuple(self._edge_artists)

        def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            artists = ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)
            linewidth = float(kwargs.get("linewidth", 1.0))
            for artist in artists:
                artist.set_path_effects(_edge_outline_effects(linewidth))
                self._edge_artists.append(artist)

        def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            artists = ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], **kwargs)
            linewidth = float(kwargs.get("linewidth", 1.0))
            for artist in artists:
                artist.set_path_effects(_edge_outline_effects(linewidth))
                self._edge_artists.append(artist)

        def plot_text(self, pos: np.ndarray, text: str, **kwargs: Any) -> None:
            _apply_text_no_clip(kwargs)
            ax.text(pos[0], pos[1], pos[2], text, **kwargs)

        def draw_tensor_nodes(
            self,
            coords: np.ndarray,
            *,
            config: PlotConfig,
            p: _DrawScaleParams,
            degree_one_mask: np.ndarray,
            mode: NodeRenderMode,
        ) -> None:
            n_nod = int(coords.shape[0])
            if n_nod == 0:
                return
            face_list: list[str] = []
            edge_list: list[str] = []
            for i in range(n_nod):
                fc = config.node_color_degree_one if degree_one_mask[i] else config.node_color
                ec = (
                    config.node_edge_color_degree_one
                    if degree_one_mask[i]
                    else config.node_edge_color
                )
                face_list.append(fc)
                edge_list.append(ec)
            if mode == "compact":
                coll = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    s=_COMPACT_NODE_MARKER_AREA_3D_PT2,
                    c=face_list,
                    edgecolors=edge_list,
                    linewidths=_COMPACT_NODE_MARKER_LINEWIDTH_PT,
                    depthshade=False,
                    zorder=float(_ZORDER_NODE_DISK),
                )
                coll.set_zorder(float(_ZORDER_NODE_DISK))
                if hasattr(coll, "set_sort_zpos"):
                    coll.set_sort_zpos(_ZORDER_NODE_DISK)
                self._node_artist_bundle = _NodeArtistBundle(
                    mode=mode,
                    artists=(coll,),
                    hover_target=None,
                )
                return

            # Unit octahedron vertices lie on axes at distance 1; scale by p.r so circumradius = p.r
            # (same metric as 2D disks; radius tracks shortest bond via renderer fraction).
            scaled = _UNIT_NODE_TRIS * p.r
            c = coords.astype(float, copy=False)
            stacked = scaled[np.newaxis, :, :, :] + c[:, np.newaxis, np.newaxis, :]
            polys = stacked.reshape(-1, 3, 3)
            node_edge_lw = max(
                float(p.lw) * _OCTAHEDRON_EDGE_LINEWIDTH_FACTOR,
                _OCTAHEDRON_EDGE_LINEWIDTH_MIN,
            )
            tri_faces: list[str] = []
            tri_edges: list[str] = []
            for fc, ec in zip(face_list, edge_list, strict=False):
                tri_faces.extend([fc] * _OCTAHEDRON_TRI_COUNT)
                tri_edges.extend([ec] * _OCTAHEDRON_TRI_COUNT)
            coll = Poly3DCollection(
                polys,
                facecolors=tri_faces,
                edgecolors=tri_edges,
                linewidths=node_edge_lw,
            )
            coll.set_zorder(float(_ZORDER_NODE_DISK))
            coll.set_sort_zpos(_ZORDER_NODE_DISK)
            ax.add_collection3d(coll)
            self._node_artist_bundle = _NodeArtistBundle(
                mode=mode,
                artists=(coll,),
                hover_target=None,
            )

        def draw_virtual_hub_markers(
            self,
            coords: np.ndarray,
            *,
            config: PlotConfig,
            zorder: float,
        ) -> None:
            if int(coords.shape[0]) == 0:
                return
            coll = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                marker="^",
                s=_VIRTUAL_HUB_MARKER_AREA_3D_PT2,
                c=[config.dangling_edge_color],
                edgecolors=[config.node_edge_color_degree_one],
                linewidths=_VIRTUAL_HUB_MARKER_LINEWIDTH_PT,
                depthshade=False,
                zorder=zorder,
            )
            coll.set_zorder(zorder)
            if hasattr(coll, "set_sort_zpos"):
                coll.set_sort_zpos(zorder)

        def style_axes(self, coords: np.ndarray, *, view_margin: float) -> None:
            _apply_axis_limits_with_outset(ax, coords, view_margin=view_margin, dimensions=3)

    return _3DPlotter(hover_edge_targets)


__all__ = [
    "_PlotAdapter",
    "_NodeArtistBundle",
    "_graph_edge_degree",
    "_make_plotter",
    "_node_edge_degrees",
    "_visible_degree_one_mask",
]
