from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from matplotlib.artist import Artist

if TYPE_CHECKING:
    from ._core.draw.scene_state import _InteractiveSceneState
else:
    _InteractiveSceneState = Any


_ACTIVE_AXES_ATTR = "_tensor_network_viz_active_axes"
_CONTRACTION_CONTROLS_ATTR = "_tensor_network_viz_contraction_controls"
_CONTRACTION_VIEWER_ATTR = "_tensor_network_viz_contraction_viewer"
_HOVER_ANN_ATTR = "_tensor_network_viz_hover_ann"
_HOVER_CID_ATTR = "_tensor_network_viz_hover_cid"
_INTERACTIVE_CONTROLS_ATTR = "_tensor_network_viz_interactive_controls"
_NODE_ID_ATTR = "_tensor_network_viz_node_id"
_RESERVED_BOTTOM_ATTR = "_tensor_network_viz_reserved_bottom"
_SCENE_ATTR = "_tensor_network_viz_scene"
_TENSOR_ELEMENTS_CONTROLS_ATTR = "_tensor_network_viz_tensor_elements_controls"
_ZOOM_CIDS_ATTR = "_tensor_network_viz_zoom_cids"
_ZOOM_FONTS_ATTR = "_tensor_network_viz_zoom_fonts"


def _clear_attr(obj: object, attr_name: str) -> None:
    if hasattr(obj, attr_name):
        delattr(obj, attr_name)


def get_reserved_bottom(fig: object, *, default: float = 0.02) -> float:
    return float(getattr(fig, _RESERVED_BOTTOM_ATTR, default))


def set_reserved_bottom(fig: object, bottom: float) -> None:
    setattr(fig, _RESERVED_BOTTOM_ATTR, float(bottom))


def get_scene(ax: object) -> _InteractiveSceneState | None:
    return cast(_InteractiveSceneState | None, getattr(ax, _SCENE_ATTR, None))


def set_scene(ax: object, scene: _InteractiveSceneState) -> None:
    setattr(ax, _SCENE_ATTR, scene)


def clear_scene(ax: object) -> None:
    _clear_attr(ax, _SCENE_ATTR)


def get_contraction_controls(obj: object) -> object | None:
    return cast(object | None, getattr(obj, _CONTRACTION_CONTROLS_ATTR, None))


def set_contraction_controls(obj: object, controls: object) -> None:
    setattr(obj, _CONTRACTION_CONTROLS_ATTR, controls)


def clear_contraction_controls(obj: object) -> None:
    _clear_attr(obj, _CONTRACTION_CONTROLS_ATTR)


def set_contraction_viewer(fig: object, viewer: object) -> None:
    setattr(fig, _CONTRACTION_VIEWER_ATTR, viewer)


def set_interactive_controls(fig: object, controls: object) -> None:
    setattr(fig, _INTERACTIVE_CONTROLS_ATTR, controls)


def set_active_axes(fig: object, ax: object) -> None:
    setattr(fig, _ACTIVE_AXES_ATTR, ax)


def set_tensor_elements_controls(fig: object, controls: object) -> None:
    setattr(fig, _TENSOR_ELEMENTS_CONTROLS_ATTR, controls)


def get_hover_cid(fig: object) -> int | None:
    raw = getattr(fig, _HOVER_CID_ATTR, None)
    return int(raw) if raw is not None else None


def set_hover_cid(fig: object, cid: int) -> None:
    setattr(fig, _HOVER_CID_ATTR, int(cid))


def clear_hover_cid(fig: object) -> None:
    setattr(fig, _HOVER_CID_ATTR, None)


def get_hover_annotation(fig: object) -> object | None:
    return cast(object | None, getattr(fig, _HOVER_ANN_ATTR, None))


def set_hover_annotation(fig: object, annotation: object) -> None:
    setattr(fig, _HOVER_ANN_ATTR, annotation)


def clear_hover_annotation(fig: object) -> None:
    setattr(fig, _HOVER_ANN_ATTR, None)


def get_zoom_font_state(ax: object) -> dict[str, Any] | None:
    state = getattr(ax, _ZOOM_FONTS_ATTR, None)
    return state if isinstance(state, dict) else None


def set_zoom_font_state(ax: object, *, ref_span: float, sizes: dict[Any, float]) -> None:
    setattr(
        ax,
        _ZOOM_FONTS_ATTR,
        {
            "ref_span": float(ref_span),
            "sizes": sizes,
        },
    )


def get_zoom_cids(ax: object) -> list[Any]:
    raw = getattr(ax, _ZOOM_CIDS_ATTR, None)
    return list(raw) if isinstance(raw, list) else []


def set_zoom_cids(ax: object, cids: list[Any]) -> None:
    setattr(ax, _ZOOM_CIDS_ATTR, list(cids))


def get_artist_node_id(artist: Artist) -> int | None:
    raw = getattr(artist, _NODE_ID_ATTR, None)
    return int(raw) if raw is not None else None


def set_artist_node_id(artist: Artist, node_id: int) -> None:
    setattr(artist, _NODE_ID_ATTR, int(node_id))


__all__ = [
    "clear_contraction_controls",
    "clear_hover_annotation",
    "clear_hover_cid",
    "clear_scene",
    "get_artist_node_id",
    "get_contraction_controls",
    "get_hover_annotation",
    "get_hover_cid",
    "get_reserved_bottom",
    "get_scene",
    "get_zoom_cids",
    "get_zoom_font_state",
    "set_active_axes",
    "set_artist_node_id",
    "set_contraction_controls",
    "set_contraction_viewer",
    "set_hover_annotation",
    "set_hover_cid",
    "set_interactive_controls",
    "set_reserved_bottom",
    "set_scene",
    "set_tensor_elements_controls",
    "set_zoom_cids",
    "set_zoom_font_state",
]
