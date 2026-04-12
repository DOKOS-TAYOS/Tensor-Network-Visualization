"""Public plotting configuration types for tensor-network figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from ._engine_specs import EngineName
from ._typing import PositionMapping

ViewName: TypeAlias = Literal["2d", "3d"]
PlotTheme: TypeAlias = Literal[
    "default",
    "paper",
    "colorblind",
    "dark",
    "midnight",
    "forest",
    "slate",
]
FocusRadius: TypeAlias = Literal[1, 2]
TensorLabelRefinement: TypeAlias = Literal["auto", "always", "never"]

_PLOT_THEME_ORDER: tuple[PlotTheme, ...] = (
    "default",
    "paper",
    "colorblind",
    "dark",
    "midnight",
    "forest",
    "slate",
)
_PLOT_THEME_NAMES: frozenset[str] = frozenset(_PLOT_THEME_ORDER)
_PAPER_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#93C5FD",
    "#FCA5A5",
    "#86EFAC",
    "#FDBA74",
    "#C4B5FD",
    "#67E8F9",
)
_COLORBLIND_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)
_DARK_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#60A5FA",
    "#FBBF24",
    "#34D399",
    "#F472B6",
    "#A78BFA",
    "#F87171",
)
_MIDNIGHT_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#38BDF8",
    "#22D3EE",
    "#818CF8",
    "#C084FC",
    "#F472B6",
    "#F59E0B",
)
_FOREST_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#86EFAC",
    "#FCD34D",
    "#93C5FD",
    "#FCA5A5",
    "#C4B5FD",
    "#67E8F9",
)
_SLATE_CONTRACTION_SCHEME_COLORS: tuple[str, ...] = (
    "#94A3B8",
    "#38BDF8",
    "#22C55E",
    "#F59E0B",
    "#A78BFA",
    "#EF4444",
)
_PLOT_THEME_OVERRIDES: dict[str, dict[str, object]] = {
    "paper": {
        "node_color": "#FFFFFF",
        "node_edge_color": "#111827",
        "node_color_degree_one": "#FFF7ED",
        "node_edge_color_degree_one": "#9A3412",
        "tensor_label_color": "#111827",
        "label_color": "#374151",
        "bond_edge_color": "#1D4ED8",
        "dangling_edge_color": "#B91C1C",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _PAPER_CONTRACTION_SCHEME_COLORS,
    },
    "colorblind": {
        "node_color": "#F7F7F7",
        "node_edge_color": "#000000",
        "node_color_degree_one": "#F0E442",
        "node_edge_color_degree_one": "#000000",
        "tensor_label_color": "#000000",
        "label_color": "#000000",
        "bond_edge_color": "#0072B2",
        "dangling_edge_color": "#D55E00",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _COLORBLIND_CONTRACTION_SCHEME_COLORS,
    },
    "dark": {
        "node_color": "#1F2937",
        "node_edge_color": "#E5E7EB",
        "node_color_degree_one": "#134E4A",
        "node_edge_color_degree_one": "#99F6E4",
        "tensor_label_color": "#F8FAFC",
        "label_color": "#CBD5E1",
        "bond_edge_color": "#60A5FA",
        "dangling_edge_color": "#F59E0B",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _DARK_CONTRACTION_SCHEME_COLORS,
    },
    "midnight": {
        "node_color": "#1E293B",
        "node_edge_color": "#E2E8F0",
        "node_color_degree_one": "#0F766E",
        "node_edge_color_degree_one": "#99F6E4",
        "tensor_label_color": "#F8FAFC",
        "label_color": "#BFDBFE",
        "bond_edge_color": "#38BDF8",
        "dangling_edge_color": "#FB7185",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _MIDNIGHT_CONTRACTION_SCHEME_COLORS,
    },
    "forest": {
        "node_color": "#ECFDF5",
        "node_edge_color": "#14532D",
        "node_color_degree_one": "#FEF3C7",
        "node_edge_color_degree_one": "#A16207",
        "tensor_label_color": "#14532D",
        "label_color": "#365314",
        "bond_edge_color": "#15803D",
        "dangling_edge_color": "#B45309",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _FOREST_CONTRACTION_SCHEME_COLORS,
    },
    "slate": {
        "node_color": "#E2E8F0",
        "node_edge_color": "#334155",
        "node_color_degree_one": "#F8FAFC",
        "node_edge_color_degree_one": "#475569",
        "tensor_label_color": "#0F172A",
        "label_color": "#334155",
        "bond_edge_color": "#0284C7",
        "dangling_edge_color": "#B91C1C",
        "line_width_2d": 1.0,
        "line_width_3d": 0.9,
        "contraction_scheme_colors": _SLATE_CONTRACTION_SCHEME_COLORS,
    },
}
_PLOT_THEME_BACKGROUND_COLORS: dict[str, tuple[str, str]] = {
    "default": ("#FFFFFF", "#FFFFFF"),
    "paper": ("#FFFFFF", "#FFFFFF"),
    "colorblind": ("#FFFFFF", "#FFFFFF"),
    "dark": ("#0B1120", "#111827"),
    "midnight": ("#020617", "#0F172A"),
    "forest": ("#F7FDF9", "#F7FDF9"),
    "slate": ("#F1F5F9", "#F8FAFC"),
}


def _theme_background_colors(theme: str) -> tuple[str, str]:
    """Figure and axes background colors for a visual theme preset."""
    return _PLOT_THEME_BACKGROUND_COLORS.get(theme, ("#FFFFFF", "#FFFFFF"))


@dataclass(frozen=True)
class TensorNetworkDiagnosticsConfig:
    """Configuration for graph diagnostics shown in overlays and hover payloads."""

    show_overlay: bool = False
    include_hover: bool = True


@dataclass(frozen=True)
class TensorNetworkFocus:
    """Configuration for subnetwork focus in normalized snapshots and interactive views."""

    kind: Literal["neighborhood", "path"]
    center: str | None = None
    radius: FocusRadius = 1
    endpoints: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate radius and normalize endpoint names when provided."""
        radius = int(self.radius)
        if radius not in (1, 2):
            raise ValueError("focus.radius must be 1 or 2.")
        normalized_radius: FocusRadius = 1 if radius == 1 else 2
        object.__setattr__(self, "radius", normalized_radius)

        if self.endpoints is None:
            return
        if len(self.endpoints) != 2:
            raise ValueError("focus.endpoints must contain exactly two tensor names.")
        object.__setattr__(
            self,
            "endpoints",
            (str(self.endpoints[0]), str(self.endpoints[1])),
        )


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
        diagnostics: Optional diagnostics settings for shape / bond / memory overlays.
        focus: Optional subnetwork focus settings shared by snapshots and interactive views.
        theme: Visual theme preset. ``"default"`` uses the library defaults, ``"paper"`` uses
            a cleaner high-contrast export style, ``"colorblind"`` uses a colorblind-safe
            palette, ``"dark"`` and ``"midnight"`` use dark canvases, and ``"forest"`` and
            ``"slate"`` provide softer practical light presets. Explicit color and line-width
            overrides still win over theme presets.
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
    DEFAULT_LINE_WIDTH_2D: ClassVar[float] = 0.95
    DEFAULT_LINE_WIDTH_3D: ClassVar[float] = 0.8
    DEFAULT_CONTRACTION_SCHEME_LINEWIDTH: ClassVar[float] = 0.12
    DEFAULT_LAYOUT_ITERATIONS: ClassVar[int] = 220

    show_nodes: bool = True
    show_tensor_labels: bool = False
    show_index_labels: bool = False
    hover_labels: bool = True
    show_contraction_scheme: bool = False
    contraction_scheme_cost_hover: bool = False
    contraction_tensor_inspector: bool = False
    diagnostics: TensorNetworkDiagnosticsConfig | None = None
    focus: TensorNetworkFocus | None = None
    theme: PlotTheme = "default"
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
    node_color: str = "#F5F3FF"
    node_edge_color: str = "#000000"
    node_color_degree_one: str = "#ECFDF5"
    node_edge_color_degree_one: str = "#047857"
    tensor_label_color: str = "#000000"
    label_color: str = "#000000"
    bond_edge_color: str = "#7C3AED"
    dangling_edge_color: str = "#10B981"
    contraction_scheme_by_name: tuple[tuple[str, ...], ...] | None = None
    contraction_scheme_colors: tuple[str, ...] | None = None
    contraction_scheme_alpha: float = 0.0
    contraction_scheme_edge_alpha: float | None = None
    contraction_scheme_linewidth: float | None = None

    def __post_init__(self) -> None:
        """Validate and resolve visual theme presets."""
        theme = str(self.theme)
        if theme not in _PLOT_THEME_NAMES:
            available = ", ".join(f"{name!r}" for name in sorted(_PLOT_THEME_NAMES))
            raise ValueError(f"theme must be one of {available}.")
        object.__setattr__(self, "theme", theme)
        if theme == "default":
            return

        for field_name, themed_value in _PLOT_THEME_OVERRIDES[theme].items():
            base_value = getattr(type(self), field_name)
            if getattr(self, field_name) == base_value:
                object.__setattr__(self, field_name, themed_value)


__all__ = [
    "EngineName",
    "FocusRadius",
    "PlotConfig",
    "PlotTheme",
    "TensorLabelRefinement",
    "TensorNetworkDiagnosticsConfig",
    "TensorNetworkFocus",
    "ViewName",
]
