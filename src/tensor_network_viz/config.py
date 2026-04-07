from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from ._engine_specs import EngineName
from ._typing import PositionMapping

ViewName: TypeAlias = Literal["2d", "3d"]
TensorLabelRefinement: TypeAlias = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for ``show_tensor_network``.

    The public constructor is intentionally ordered from high-level behavior first and
    styling/detail later, so it reads naturally when used with keyword arguments.

    Attributes:
        show_nodes: Whether to draw tensor nodes with their full geometry. If ``False``,
            use compact fixed-size markers instead.
        show_tensor_labels: Whether to draw static tensor names on nodes.
        show_index_labels: Whether to draw static index labels on edges.
        hover_labels: Whether to enable hover tooltips for tensor names and edge labels.
            This is independent from the static label toggles.
        show_contraction_scheme: Whether to enable the interactive contraction slider.
        contraction_scheme_cost_hover: Whether to show the per-step contraction-cost panel
            together with the contraction slider.
        contraction_tensor_inspector: Whether to enable the linked tensor-inspector window
            when playback tensors are available.
        tensor_label_refinement: Policy for the post-draw passes that shrink tensor labels
            to fit their node marker. ``"always"`` always refits, ``"never"`` skips it,
            and ``"auto"`` applies it only when it is still cheap.
        approximate_3d_tensor_disk_px: Whether 3D tensor-label fitting should use a faster
            nominal disk radius instead of a per-node projected radius.
        figsize: Figure size in inches. ``None`` leaves Matplotlib's default.
        positions: Optional mapping from node id to custom 2D or 3D coordinates.
        validate_positions: Whether to warn about unknown or mismatched custom positions.
        layout_iterations: Optional override for the force-layout iteration count.
        tensor_label_fontsize: Preferred font size for tensor labels in points. ``None``
            keeps the automatic fit-based behavior. When set, it is still treated as a
            safe preferred size, so labels may shrink if needed to fit.
        edge_label_fontsize: Preferred font size for edge labels in points. ``None`` keeps
            the automatic bond-length heuristic.
        node_radius: Base tensor-node radius in data units before draw scaling.
        stub_length: Length of dangling-index stubs before draw scaling.
        self_loop_radius: Radius used for self-loop edges before draw scaling.
        line_width_2d: 2D line width for node outlines and edges.
        line_width_3d: 3D line width for node outlines and edges.
        node_color: Fill color for tensor nodes.
        node_edge_color: Border color for tensor nodes.
        node_color_degree_one: Fill color for visible degree-one tensors.
        node_edge_color_degree_one: Border color for visible degree-one tensors.
        tensor_label_color: Color for tensor names drawn on nodes.
        label_color: Color for index labels.
        bond_edge_color: Color for contraction edges between tensors.
        dangling_edge_color: Color for dangling-index stubs.
        contraction_scheme_by_name: Optional explicit contraction scheme, expressed as
            tuples of visible tensor names, one tuple per contraction step.
        contraction_scheme_colors: Optional color cycle for contraction groups.
        contraction_scheme_alpha: Reserved for backwards compatibility.
        contraction_scheme_edge_alpha: Reserved for backwards compatibility.
        contraction_scheme_linewidth: Reserved for backwards compatibility.
    """

    DEFAULT_NODE_RADIUS: ClassVar[float] = 0.08
    DEFAULT_STUB_LENGTH: ClassVar[float] = 0.16
    DEFAULT_SELF_LOOP_RADIUS: ClassVar[float] = 0.2
    DEFAULT_LINE_WIDTH_2D: ClassVar[float] = 0.85
    DEFAULT_LINE_WIDTH_3D: ClassVar[float] = 0.75
    DEFAULT_CONTRACTION_SCHEME_LINEWIDTH: ClassVar[float] = 0.12
    DEFAULT_LAYOUT_ITERATIONS: ClassVar[int] = 220

    show_nodes: bool = True
    show_tensor_labels: bool = False
    show_index_labels: bool = False
    hover_labels: bool = True
    show_contraction_scheme: bool = False
    contraction_scheme_cost_hover: bool = False
    contraction_tensor_inspector: bool = False
    tensor_label_refinement: TensorLabelRefinement = "auto"
    approximate_3d_tensor_disk_px: bool = True
    figsize: tuple[float, float] | None = (8, 6)
    positions: PositionMapping | None = None
    validate_positions: bool = False
    layout_iterations: int | None = None
    tensor_label_fontsize: float | None = None
    edge_label_fontsize: float | None = None
    node_radius: float | None = None
    stub_length: float | None = None
    self_loop_radius: float | None = None
    line_width_2d: float | None = None
    line_width_3d: float | None = None
    node_color: str = "#E8EEF5"
    node_edge_color: str = "#1E293B"
    node_color_degree_one: str = "#FEE2E2"
    node_edge_color_degree_one: str = "#7F1D1D"
    tensor_label_color: str = "#0F172A"
    label_color: str = "#334155"
    bond_edge_color: str = "#0369A1"
    dangling_edge_color: str = "#BE123C"
    contraction_scheme_by_name: tuple[tuple[str, ...], ...] | None = None
    contraction_scheme_colors: tuple[str, ...] | None = None
    contraction_scheme_alpha: float = 0.0
    contraction_scheme_edge_alpha: float | None = None
    contraction_scheme_linewidth: float | None = None


__all__ = ["EngineName", "PlotConfig", "TensorLabelRefinement", "ViewName"]
