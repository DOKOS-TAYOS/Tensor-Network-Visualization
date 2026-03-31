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
            they fit the node marker in 2D or 3D. Each pass calls ``fig.canvas.draw()`` and
            measures text bounding boxes — often a large share of end-to-end plot time on dense
            figures. Set False for much faster plots when slight overflow of long names is
            acceptable (first-pass TextPath sizing still applies).
        approximate_3d_tensor_disk_px: If True (default), tensor label disk radius in pixels
            uses a single nominal scale from axis spans (cheap). If False, uses per-node
            projection (slower, marginally more accurate under 3D perspective).
        hover_labels: If True, tensor names and bond index labels are hidden until the pointer
            hovers over a node or edge (2D: hit-testing in axes space; 3D: projected screen
            distance). Use an interactive Matplotlib window.
        show_contraction_scheme: If True, draw colored highlights for each contraction step (from
            ``graph.contraction_steps`` or ``contraction_scheme_by_name``).
        contraction_scheme_alpha: Fill alpha for scheme rectangles (2D); 3D uses edge color only.
            Default is 0 (no fill); borders remain visible via ``contraction_scheme_edge_alpha``.
        contraction_scheme_edge_alpha: Stroke alpha for scheme borders; None chooses a visible edge
            (stronger when the fill is fully transparent).
        contraction_scheme_linewidth: Border line width for 2D rounded scheme rectangles
            (data units, scaled like bond lines); None uses a thin default.
        contraction_scheme_colors: Optional cycle of face colors (hex/named); None uses a built-in
            categorical palette.
        contraction_scheme_by_name: Optional override: each inner tuple is one step, tensor names
            matching non-virtual ``node.name`` values. Duplicate names among visible tensors or
            unknown names raise ``ValueError``. When set, this replaces ``graph.contraction_steps``.
        contraction_playback: If True, ``show_tensor_network`` adds a Matplotlib slider and
            Play/Pause/Reset controls on the same figure (2D widget axes only) to step through
            contraction highlights interactively. Requires ``show_contraction_scheme=True`` and a
            non-empty contraction step list.
    """

    DEFAULT_NODE_RADIUS: ClassVar[float] = 0.08
    DEFAULT_STUB_LENGTH: ClassVar[float] = 0.16
    DEFAULT_SELF_LOOP_RADIUS: ClassVar[float] = 0.2
    DEFAULT_LINE_WIDTH_2D: ClassVar[float] = 0.85
    DEFAULT_LINE_WIDTH_3D: ClassVar[float] = 0.75
    DEFAULT_CONTRACTION_SCHEME_LINEWIDTH: ClassVar[float] = 0.12
    DEFAULT_LAYOUT_ITERATIONS: ClassVar[int] = 220

    node_color: str = "#E8EEF5"
    node_edge_color: str = "#1E293B"
    node_color_degree_one: str = "#FEE2E2"
    node_edge_color_degree_one: str = "#7F1D1D"
    tensor_label_color: str = "#0F172A"
    label_color: str = "#334155"
    bond_edge_color: str = "#0369A1"
    dangling_edge_color: str = "#BE123C"
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
    approximate_3d_tensor_disk_px: bool = True
    hover_labels: bool = False
    show_contraction_scheme: bool = False
    contraction_scheme_alpha: float = 0.0
    contraction_scheme_edge_alpha: float | None = None
    contraction_scheme_linewidth: float | None = None
    contraction_scheme_colors: tuple[str, ...] | None = None
    contraction_scheme_by_name: tuple[tuple[str, ...], ...] | None = None
    contraction_playback: bool = False


__all__ = ["EngineName", "PlotConfig", "ViewName"]
