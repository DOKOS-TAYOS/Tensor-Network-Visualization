from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection, PathCollection
from matplotlib.patches import Circle

from ...config import PlotConfig
from .constants import _EDGE_LINE_CAP_STYLE, _EDGE_LINE_JOIN_STYLE, _ZORDER_NODE_DISK
from .fonts_and_scale import _DrawScaleParams
from .plotter_common import (
    _COMPACT_NODE_MARKER_AREA_2D_PT2,
    _COMPACT_NODE_MARKER_LINEWIDTH_PT,
    _VIRTUAL_HUB_MARKER_AREA_2D_PT2,
    _VIRTUAL_HUB_MARKER_LINEWIDTH_PT,
    NodeRenderMode,
    _edge_outline_effects,
    _NodeArtistBundle,
)
from .viewport_geometry import (
    _apply_axis_limits_with_outset,
    _apply_edge_line_style,
    _apply_text_no_clip,
)


def _make_2d_plotter(
    ax: Axes,
    *,
    hover_edge_targets: list[tuple[np.ndarray, str]] | None = None,
) -> object:
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
                    Circle((float(coords[i, 0]), float(coords[i, 1])), radius=p.r) for i in range(n)
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
            _apply_axis_limits_with_outset(self._ax, coords, view_margin=view_margin, dimensions=2)

    return _2DPlotter(ax, hover_edge_targets)
