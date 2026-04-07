from __future__ import annotations

from matplotlib.figure import Figure

from ._interaction.bridge import (
    clear_contraction_controls,
    clear_hover_annotation,
    clear_hover_cid,
    clear_scene,
    get_artist_node_id,
    get_contraction_controls,
    get_hover_annotation,
    get_hover_cid,
    get_reserved_bottom,
    get_scene,
    get_zoom_cids,
    get_zoom_font_state,
    set_active_axes,
    set_artist_node_id,
    set_contraction_controls,
    set_contraction_viewer,
    set_hover_annotation,
    set_hover_cid,
    set_interactive_controls,
    set_reserved_bottom,
    set_scene,
    set_tensor_elements_controls,
    set_zoom_cids,
    set_zoom_font_state,
)


def canvas_supports_live_redraw(figure: Figure) -> bool:
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return False
    if type(canvas).__module__ == "matplotlib.backends.backend_agg":
        return False
    return callable(getattr(canvas, "draw_idle", None))


def request_canvas_redraw(figure: Figure | None) -> None:
    if figure is None:
        return
    if not canvas_supports_live_redraw(figure):
        return
    draw_idle = getattr(figure.canvas, "draw_idle", None)
    if callable(draw_idle):
        draw_idle()


__all__ = [
    "canvas_supports_live_redraw",
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
    "request_canvas_redraw",
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
