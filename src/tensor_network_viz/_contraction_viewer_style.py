from __future__ import annotations

from contextlib import suppress
from typing import Any, Final

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

# Must stay aligned with ``_CONTRACTION_SCHEME_GID`` in ``_core.draw.contraction_scheme``.
_TNV_CONTRACTION_SCHEME_PATCH_GID: Final[str] = "tnv_contraction_scheme"
_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def is_tensor_network_scheme_artist(artist: Artist) -> bool:
    """True for contraction-scheme artists tagged by the draw pipeline."""
    getter = getattr(artist, "get_gid", None)
    if not callable(getter):
        return False
    with suppress(TypeError, ValueError):
        return getter() == _TNV_CONTRACTION_SCHEME_PATCH_GID
    return False


def is_tensor_network_scheme_fancy_patch(artist: Artist) -> bool:
    """True for 2D contraction-scheme hulls (playback restyle only; static draw unchanged)."""
    return isinstance(artist, FancyBboxPatch) and is_tensor_network_scheme_artist(artist)


def apply_scheme_2d_highlight_past(artist: Artist) -> None:
    """Transparent fill and edge (like wire-only 3D: no colored hull)."""
    safe_set_visible(artist, True)
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc):
        with suppress(TypeError, ValueError):
            setter_fc(_TRANSPARENT)
    setter_ec = getattr(artist, "set_edgecolor", None)
    if callable(setter_ec):
        with suppress(TypeError, ValueError):
            setter_ec(_TRANSPARENT)
    safe_set_linewidth(artist, 0.0)
    safe_clear_patch_alpha(artist)


def apply_scheme_2d_highlight_current(
    artist: Artist,
    *,
    accent: Any,
    fill_alpha: float,
    edge_alpha: float,
    linewidth: float,
) -> None:
    """Very faint tint inside; strong accent on border (aligns with 3D edge emphasis)."""
    r, g, b, _ = to_rgba(accent)
    face = (float(r), float(g), float(b), float(np.clip(fill_alpha, 0.0, 1.0)))
    edge = to_rgba(accent, alpha=float(np.clip(edge_alpha, 0.0, 1.0)))
    safe_set_visible(artist, True)
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc):
        with suppress(TypeError, ValueError):
            setter_fc(face)
    setter_ec = getattr(artist, "set_edgecolor", None)
    if callable(setter_ec):
        with suppress(TypeError, ValueError):
            setter_ec(edge)
    safe_set_linewidth(artist, float(linewidth))
    safe_clear_patch_alpha(artist)


def safe_set_visible(artist: Artist, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        with suppress(AttributeError, TypeError, ValueError):
            setter(visible)


def safe_set_alpha(artist: Artist, alpha: float | None) -> None:
    setter = getattr(artist, "set_alpha", None)
    if callable(setter) and alpha is not None:
        with suppress(AttributeError, TypeError, ValueError):
            setter(alpha)


def safe_clear_patch_alpha(artist: Artist) -> None:
    """So facecolor/edgecolor RGBA alphas are not multiplied by a stale artist alpha."""
    setter = getattr(artist, "set_alpha", None)
    if callable(setter):
        with suppress(AttributeError, TypeError, ValueError):
            setter(None)


def safe_set_color(artist: Artist, color: Any) -> None:
    for name in ("set_edgecolor", "set_color", "set_facecolor"):
        setter = getattr(artist, name, None)
        if callable(setter):
            try:
                setter(color)
                return
            except (AttributeError, TypeError, ValueError):
                continue


def safe_set_linewidth(artist: Artist, lw: float) -> None:
    setter = getattr(artist, "set_linewidth", None)
    if callable(setter):
        try:
            setter(lw)
            return
        except (AttributeError, TypeError, ValueError):
            pass
    setter2 = getattr(artist, "set_linewidths", None)
    if callable(setter2):
        with suppress(AttributeError, TypeError, ValueError):
            setter2(lw)


def snapshot_style(artist: Artist) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for attr, key in (
        ("get_edgecolor", "edgecolor"),
        ("get_facecolor", "facecolor"),
        ("get_color", "color"),
        ("get_linewidth", "linewidth"),
        ("get_linewidths", "linewidths"),
        ("get_alpha", "alpha"),
    ):
        fn = getattr(artist, attr, None)
        if callable(fn):
            with suppress(AttributeError, TypeError, ValueError):
                snap[key] = fn()
    return snap


def restore_style(artist: Artist, snap: dict[str, Any]) -> None:
    ec = snap.get("edgecolor")
    if ec is not None:
        safe_set_color(artist, ec)
    fc = snap.get("facecolor")
    setter_fc = getattr(artist, "set_facecolor", None)
    if callable(setter_fc) and fc is not None:
        with suppress(AttributeError, TypeError, ValueError):
            setter_fc(fc)
    col = snap.get("color")
    if col is not None and not hasattr(artist, "set_edgecolor"):
        safe_set_color(artist, col)
    lw = snap.get("linewidth")
    if lw is not None:
        try:
            safe_set_linewidth(artist, float(np.ravel(lw)[0]))
        except (TypeError, ValueError, IndexError):
            safe_set_linewidth(artist, float(lw))  # type: ignore[arg-type]
    else:
        lws = snap.get("linewidths")
        if lws is not None:
            with suppress(TypeError, ValueError, IndexError):
                safe_set_linewidth(artist, float(np.ravel(lws)[0]))
    al = snap.get("alpha")
    if al is not None:
        try:
            a0 = float(np.ravel(al)[0])
        except (TypeError, ValueError, IndexError):
            try:
                a0 = float(al)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                a0 = None
        if a0 is not None:
            safe_set_alpha(artist, a0)


def box_poly3d_faces(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> list[list[tuple[float, float, float]]]:
    return [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]


__all__ = [
    "apply_scheme_2d_highlight_current",
    "apply_scheme_2d_highlight_past",
    "box_poly3d_faces",
    "is_tensor_network_scheme_artist",
    "is_tensor_network_scheme_fancy_patch",
    "restore_style",
    "safe_clear_patch_alpha",
    "safe_set_alpha",
    "safe_set_color",
    "safe_set_linewidth",
    "safe_set_visible",
    "snapshot_style",
]
