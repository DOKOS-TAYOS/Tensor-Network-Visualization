from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

EngineName: TypeAlias = Literal["tensorkrowch", "tensornetwork", "quimb", "tenpy", "einsum"]
ViewName: TypeAlias = Literal["2d", "3d"]


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for tensor network plot styling.

    Input may be a network-like object exposing `nodes` or `leaf_nodes`, or any
    iterable of nodes. Each node must have `edges`, `axes_names` or
    `axis_names`, and `name`. Each edge must have `node1`, `node2`, and `name`.

    Attributes:
        node_color: Fill color for tensor nodes (hex or named color).
        label_color: Color for axis and edge labels.
        bond_edge_color: Color for contraction edges between tensors.
        dangling_edge_color: Color for dangling index stubs.
        figsize: Figure size as (width, height) in inches; None uses Matplotlib default.
        show_tensor_labels: Whether to display tensor names on nodes.
        show_index_labels: Whether to display axis names on edges.
        node_radius: Radius of tensor nodes; None uses default (0.08).
        stub_length: Length of dangling index stubs; None uses default (0.34).
        self_loop_radius: Radius for self-contraction loops; None uses default (0.2).
        line_width_2d: Line width for 2D plots; None uses default (1.8).
        line_width_3d: Line width for 3D plots; None uses default (1.6).
        positions: Custom node positions for grid/PEPS layout; dict mapping node id to
            (x, y) for 2D or (x, y, z) for 3D. None uses automatic layout.
        node_edge_color: Border color for tensor nodes; use dark for contrast on light nodes.
        tensor_label_color: Color for tensor names on nodes; use dark for readability.
        layout_iterations: Force-directed layout iterations; None uses default (220).
    """

    DEFAULT_NODE_RADIUS: ClassVar[float] = 0.08
    DEFAULT_STUB_LENGTH: ClassVar[float] = 0.34
    DEFAULT_SELF_LOOP_RADIUS: ClassVar[float] = 0.2
    DEFAULT_LINE_WIDTH_2D: ClassVar[float] = 1.8
    DEFAULT_LINE_WIDTH_3D: ClassVar[float] = 1.6
    DEFAULT_LAYOUT_ITERATIONS: ClassVar[int] = 220

    node_color: str = "#E8E8E8"
    node_edge_color: str = "#2D3748"
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
