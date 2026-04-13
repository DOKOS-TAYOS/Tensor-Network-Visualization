from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ...config import PlotConfig
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
from .plotter_common import (
    _COMPACT_NODE_MARKER_AREA_3D_PT2,
    _COMPACT_NODE_MARKER_LINEWIDTH_PT,
    _VIRTUAL_HUB_MARKER_AREA_3D_PT2,
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


def _make_3d_plotter(
    ax: Any,
    *,
    hover_edge_targets: list[tuple[np.ndarray, str]] | None = None,
) -> object:
    class _3DPlotter:
        def __init__(self, hover_edges: list[tuple[np.ndarray, str]] | None) -> None:
            if hasattr(ax, "computed_zorder"):
                ax.computed_zorder = False
            self._hover_edge_targets = hover_edges
            self._edge_segments: list[tuple[float, str, float, np.ndarray]] = []
            self._edge_artists: list[Artist] = []
            self._node_artist_bundle: _NodeArtistBundle | None = None

        def get_node_artist_bundle(self) -> _NodeArtistBundle | None:
            return self._node_artist_bundle

        def get_edge_artists(self) -> tuple[Artist, ...]:
            return tuple(self._edge_artists)

        def flush_edge_collections(self) -> None:
            if not self._edge_segments:
                return
            groups: dict[tuple[float, str, float], list[np.ndarray]] = defaultdict(list)
            for z, color, lw, seg in self._edge_segments:
                groups[(z, color, lw)].append(seg)
            for (z, color, lw), segs in sorted(groups.items(), key=lambda kv: kv[0][0]):
                coll = Line3DCollection(
                    segs,
                    colors=color,
                    linewidths=lw,
                    zorder=z,
                    capstyle=_EDGE_LINE_CAP_STYLE,
                    joinstyle=_EDGE_LINE_JOIN_STYLE,
                )
                coll.set_path_effects(_edge_outline_effects(float(lw)))
                coll.set_zorder(z)
                if hasattr(coll, "set_sort_zpos"):
                    coll.set_sort_zpos(z)
                ax.add_collection3d(coll)
                self._edge_artists.append(coll)
            self._edge_segments.clear()

        def plot_line(self, start: np.ndarray, end: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            z = float(kwargs.get("zorder", 1))
            color = str(kwargs.get("color", "#000000"))
            lw = float(kwargs.get("linewidth", 1.0))
            seg = np.array(
                [
                    [float(start[0]), float(start[1]), float(start[2])],
                    [float(end[0]), float(end[1]), float(end[2])],
                ],
                dtype=float,
            )
            self._edge_segments.append((z, color, lw, seg))

        def plot_curve(self, curve: np.ndarray, **kwargs: Any) -> None:
            _apply_edge_line_style(kwargs)
            z = float(kwargs.get("zorder", 1))
            color = str(kwargs.get("color", "#000000"))
            lw = float(kwargs.get("linewidth", 1.0))
            seg = np.asarray(curve[:, :3], dtype=float, order="C")
            self._edge_segments.append((z, color, lw, seg))

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
