from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from ._engine_specs import EngineName

ViewName: TypeAlias = Literal["2d", "3d"]


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for tensor network plot styling.

    Used by ``show_tensor_network`` and backend ``plot_*_network_*`` helpers.
    Label visibility can also be overridden per call via
    ``show_tensor_network(..., show_tensor_labels=..., show_index_labels=...)``.

    Attributes:
        node_color: Fill color for tensor nodes (hex or named color).
        label_color: Color for axis and edge labels.
        bond_edge_color: Color for contraction edges between tensors.
        dangling_edge_color: Color for dangling index stubs.
        figsize: Figure size as (width, height) in inches; None uses Matplotlib default.
        show_tensor_labels: Whether to display tensor names on nodes.
        show_index_labels: Whether to display axis names on edges.
        node_radius: Base radius of tensor nodes in data units before draw scale; None uses
            default (0.08). Draw scale is chosen so that, for the default radius, disk radius is
            the shortest contraction-edge length times the renderer fraction (see
            ``_SHORTEST_EDGE_RADIUS_FRACTION``); setting this scales that radius
            proportionally.
        stub_length: Length of dangling index stubs; None uses default (0.16).
        self_loop_radius: Radius for self-contraction loops; None uses default (0.2).
        line_width_2d: Line width for 2D plots (node outlines and tensor edges); None uses default.
        line_width_3d: Line width for 3D plots (node outlines and tensor edges); None uses default.
        positions: Custom node positions for grid/PEPS layout; dict mapping node id to
            (x, y) for 2D or (x, y, z) for 3D. None uses automatic layout.
        node_edge_color: Border color for tensor nodes; use dark for contrast on light nodes.
        node_color_degree_one: Fill for non-virtual tensors with graph degree 1 (any edge kinds).
        node_edge_color_degree_one: Border for those same tensors.
        tensor_label_color: Color for tensor names on nodes; use dark for readability.
        layout_iterations: Force-directed layout iterations; None uses default (220).
        validate_positions: If True, warn when custom positions have unknown keys or
            wrong dimension count for the view.
        refine_tensor_labels: If True, run post-draw passes that shrink tensor names so
            they fit the node marker in 2D or 3D (uses extra canvas draws). Set False for
            faster plots when visual polish is less important.
        hover_labels: If True, tensor names and bond index labels are hidden until the pointer
            hovers over a node or edge (2D: hit-testing in axes space; 3D: projected screen
            distance). Use an interactive Matplotlib window.
    """

    DEFAULT_NODE_RADIUS: ClassVar[float] = 0.08
    DEFAULT_STUB_LENGTH: ClassVar[float] = 0.16
    DEFAULT_SELF_LOOP_RADIUS: ClassVar[float] = 0.2
    DEFAULT_LINE_WIDTH_2D: ClassVar[float] = 0.85
    DEFAULT_LINE_WIDTH_3D: ClassVar[float] = 0.75
    DEFAULT_LAYOUT_ITERATIONS: ClassVar[int] = 220

    node_color: str = "#E8E8E8"
    node_edge_color: str = "#2D3748"
    node_color_degree_one: str = "#E8D6D6"
    node_edge_color_degree_one: str = "#4A3436"
    tensor_label_color: str = "#1A202C"
    label_color: str = "#0C1319"
    bond_edge_color: str = "#00008B"
    dangling_edge_color: str = "#8B0000"
    figsize: tuple[float, float] | None = (8, 6)
    show_tensor_labels: bool = True
    show_index_labels: bool = True
    node_radius: float | None = None
    stub_length: float | None = None
    self_loop_radius: float | None = None
    line_width_2d: float | None = None
    line_width_3d: float | None = None
    layout_iterations: int | None = None
    positions: dict[int, tuple[float, ...]] | None = None
    validate_positions: bool = False
    refine_tensor_labels: bool = True
    hover_labels: bool = False


__all__ = ["EngineName", "PlotConfig", "ViewName"]
