"""Shared scale and style parameters for 2D and 3D drawing."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    _DEFAULT_LINE_WIDTH_2D,
    _DEFAULT_LINE_WIDTH_3D,
    _DEFAULT_NODE_RADIUS,
    _DEFAULT_SELF_LOOP_RADIUS,
    _DEFAULT_STUB_LENGTH,
    PlotConfig,
)


@dataclass(frozen=True)
class _DrawScaleParams:
    """Resolved scale-dependent parameters for drawing."""

    r: float
    stub: float
    loop_r: float
    lw: float
    font_dangling: int
    font_bond: int
    font_node: int
    label_offset: float
    ellipse_w: float
    ellipse_h: float
    scatter_s: float


def _draw_scale_params(config: PlotConfig, scale: float, *, is_3d: bool) -> _DrawScaleParams:
    """Compute scale-dependent drawing parameters from config."""
    r = (config.node_radius if config.node_radius is not None else _DEFAULT_NODE_RADIUS) * scale
    stub = (
        config.stub_length if config.stub_length is not None else _DEFAULT_STUB_LENGTH
    ) * scale
    loop_r = (
        config.self_loop_radius
        if config.self_loop_radius is not None
        else _DEFAULT_SELF_LOOP_RADIUS
    ) * scale
    lw_default = _DEFAULT_LINE_WIDTH_3D if is_3d else _DEFAULT_LINE_WIDTH_2D
    lw_attr = config.line_width_3d if is_3d else config.line_width_2d
    lw = (lw_attr if lw_attr is not None else lw_default) * scale
    scatter_s = (120 if is_3d else 900) * (scale**2)

    return _DrawScaleParams(
        r=r,
        stub=stub,
        loop_r=loop_r,
        lw=lw,
        font_dangling=max(7, round(9 * scale)),
        font_bond=max(5, round(5 * scale)),
        font_node=max(8, round(10 * scale)),
        label_offset=0.08 * scale,
        ellipse_w=0.16 * scale,
        ellipse_h=0.12 * scale,
        scatter_s=scatter_s,
    )
