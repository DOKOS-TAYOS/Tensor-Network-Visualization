"""Tests for the main public API: show_tensor_network, PlotConfig, and error handling."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib
from matplotlib.colors import to_hex

matplotlib.use("Agg")

import pytest

from plotting_helpers import assert_readable_image, assert_rendered_figure
from tensor_network_viz import (
    PlotConfig,
    TensorAnalysisConfig,
    TensorComparisonConfig,
    TensorElementsConfig,
    TensorNetworkDiagnosticsConfig,
    TensorNetworkFocus,
    export_tensor_network_snapshot,
    normalize_tensor_network,
    pair_tensor,
    show_tensor_comparison,
    show_tensor_network,
    translate_tensor_network,
)
from tensor_network_viz._core.renderer import _effective_layout_iterations
from tensor_network_viz.config import EngineName

pytestmark = pytest.mark.filterwarnings("ignore:unit_cell_width.*:UserWarning")


def test_plot_config_has_expected_defaults() -> None:
    config = PlotConfig()
    assert config.figsize == (8, 6)
    assert config.show_nodes is None
    assert config.show_tensor_labels is None
    assert config.show_index_labels is False
    assert config.show_contraction_scheme is False
    assert config.contraction_scheme_cost_hover is False
    assert config.contraction_tensor_inspector is False
    assert config.tensor_inspector_config is None
    assert config.layout_iterations is None
    assert config.positions is None
    assert config.validate_positions is False
    assert config.tensor_label_fontsize is None
    assert config.edge_label_fontsize is None
    assert config.tensor_label_refinement == "auto"
    assert config.approximate_3d_tensor_disk_px is True
    assert config.hover_labels is True
    assert config.diagnostics is None
    assert config.focus is None
    assert config.theme == "default"
    assert config.node_color == "#F5F3FF"
    assert config.node_edge_color == "#000000"
    assert config.node_color_degree_one == "#ECFDF5"
    assert config.node_edge_color_degree_one == "#047857"
    assert config.tensor_label_color == "#000000"
    assert config.label_color == "#000000"
    assert config.bond_edge_color == "#7C3AED"
    assert config.dangling_edge_color == "#10B981"


def test_plot_theme_is_public_type_alias() -> None:
    import tensor_network_viz as tnv

    assert hasattr(tnv, "PlotTheme")
    assert tnv.PlotTheme.__args__ == (
        "default",
        "paper",
        "colorblind",
        "dark",
        "midnight",
        "forest",
        "slate",
    )


def test_translation_target_name_is_public_type_alias() -> None:
    import tensor_network_viz as tnv

    assert hasattr(tnv, "TranslationTargetName")
    assert tnv.TranslationTargetName.__args__ == (
        "tensorkrowch",
        "tensornetwork",
        "quimb",
        "einsum",
    )


def test_tensor_elements_theme_is_public_type_alias() -> None:
    import tensor_network_viz as tnv

    assert hasattr(tnv, "TensorElementsTheme")
    assert tnv.TensorElementsTheme.__args__ == (
        "default",
        "grayscale",
        "contrast",
        "categorical",
        "paper",
        "colorblind",
        "rainbow",
        "spectral",
    )


def test_plot_config_public_signature_orders_modes_before_detail() -> None:
    signature = inspect.signature(PlotConfig)

    assert tuple(signature.parameters) == (
        "show_nodes",
        "show_tensor_labels",
        "show_index_labels",
        "hover_labels",
        "show_contraction_scheme",
        "contraction_scheme_cost_hover",
        "contraction_tensor_inspector",
        "tensor_inspector_config",
        "diagnostics",
        "focus",
        "theme",
        "tensor_label_refinement",
        "approximate_3d_tensor_disk_px",
        "figsize",
        "positions",
        "validate_positions",
        "layout_iterations",
        "tensor_label_fontsize",
        "edge_label_fontsize",
        "node_radius",
        "stub_length",
        "self_loop_radius",
        "line_width_2d",
        "line_width_3d",
        "node_color",
        "node_edge_color",
        "node_color_degree_one",
        "node_edge_color_degree_one",
        "tensor_label_color",
        "label_color",
        "bond_edge_color",
        "dangling_edge_color",
        "contraction_scheme_by_name",
        "contraction_scheme_colors",
        "contraction_scheme_alpha",
        "contraction_scheme_edge_alpha",
        "contraction_scheme_linewidth",
    )


def test_translate_tensor_network_public_signature_is_small_and_explicit() -> None:
    signature = inspect.signature(translate_tensor_network)

    assert tuple(signature.parameters) == (
        "network",
        "engine",
        "target_engine",
        "path",
    )
    assert signature.parameters["engine"].default is None
    assert signature.parameters["path"].default is None


def test_effective_layout_iterations_respects_explicit_setting() -> None:
    cfg = PlotConfig(layout_iterations=400)
    assert _effective_layout_iterations(cfg, n_nodes=10_000) == 400


def test_effective_layout_iterations_auto_scales_below_default_for_small_graphs() -> None:
    cfg = PlotConfig()
    auto = _effective_layout_iterations(cfg, n_nodes=20)
    assert 45 <= auto < PlotConfig.DEFAULT_LAYOUT_ITERATIONS


def test_plot_config_accepts_overrides() -> None:
    config = PlotConfig(
        show_tensor_labels=False,
        diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=True),
        focus=TensorNetworkFocus(kind="neighborhood", center="A", radius=2),
        figsize=(10, 5),
        layout_iterations=100,
        tensor_label_fontsize=13.0,
        edge_label_fontsize=11.0,
        tensor_label_refinement="never",
    )
    assert config.figsize == (10, 5)
    assert config.show_tensor_labels is False
    assert config.layout_iterations == 100
    assert config.tensor_label_fontsize == pytest.approx(13.0)
    assert config.edge_label_fontsize == pytest.approx(11.0)
    assert config.tensor_label_refinement == "never"
    assert config.diagnostics == TensorNetworkDiagnosticsConfig(show_overlay=True)
    assert config.focus == TensorNetworkFocus(kind="neighborhood", center="A", radius=2)


def test_plot_config_rejects_unknown_theme() -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        PlotConfig(theme="presentation")  # type: ignore[arg-type]


def test_plot_config_paper_theme_applies_style_defaults() -> None:
    config = PlotConfig(theme="paper")

    assert config.node_color == "#FFFFFF"
    assert config.node_edge_color == "#111827"
    assert config.node_color_degree_one == "#FFF7ED"
    assert config.node_edge_color_degree_one == "#9A3412"
    assert config.tensor_label_color == "#111827"
    assert config.label_color == "#374151"
    assert config.bond_edge_color == "#1D4ED8"
    assert config.dangling_edge_color == "#B91C1C"
    assert config.line_width_2d == pytest.approx(1.0)
    assert config.line_width_3d == pytest.approx(0.9)
    assert config.contraction_scheme_colors == (
        "#93C5FD",
        "#FCA5A5",
        "#86EFAC",
        "#FDBA74",
        "#C4B5FD",
        "#67E8F9",
    )


def test_plot_config_theme_preserves_manual_overrides() -> None:
    config = PlotConfig(
        theme="paper",
        node_color="#ABCDEF",
        line_width_2d=2.5,
        contraction_scheme_colors=("#123456",),
    )

    assert config.node_color == "#ABCDEF"
    assert config.node_edge_color == "#111827"
    assert config.line_width_2d == pytest.approx(2.5)
    assert config.line_width_3d == pytest.approx(0.9)
    assert config.contraction_scheme_colors == ("#123456",)


def test_plot_config_colorblind_theme_uses_okabe_ito_palette() -> None:
    config = PlotConfig(theme="colorblind")

    assert config.node_color == "#F7F7F7"
    assert config.node_edge_color == "#000000"
    assert config.node_color_degree_one == "#F0E442"
    assert config.node_edge_color_degree_one == "#000000"
    assert config.tensor_label_color == "#000000"
    assert config.label_color == "#000000"
    assert config.bond_edge_color == "#0072B2"
    assert config.dangling_edge_color == "#D55E00"
    assert config.line_width_2d == pytest.approx(1.0)
    assert config.line_width_3d == pytest.approx(0.9)
    assert config.contraction_scheme_colors == (
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    )


@pytest.mark.parametrize(
    ("theme", "expected"),
    [
        (
            "dark",
            {
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
                "contraction_scheme_colors": (
                    "#60A5FA",
                    "#FBBF24",
                    "#34D399",
                    "#F472B6",
                    "#A78BFA",
                    "#F87171",
                ),
            },
        ),
        (
            "midnight",
            {
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
                "contraction_scheme_colors": (
                    "#38BDF8",
                    "#22D3EE",
                    "#818CF8",
                    "#C084FC",
                    "#F472B6",
                    "#F59E0B",
                ),
            },
        ),
        (
            "forest",
            {
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
                "contraction_scheme_colors": (
                    "#86EFAC",
                    "#FCD34D",
                    "#93C5FD",
                    "#FCA5A5",
                    "#C4B5FD",
                    "#67E8F9",
                ),
            },
        ),
        (
            "slate",
            {
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
                "contraction_scheme_colors": (
                    "#94A3B8",
                    "#38BDF8",
                    "#22C55E",
                    "#F59E0B",
                    "#A78BFA",
                    "#EF4444",
                ),
            },
        ),
    ],
)
def test_plot_config_new_practical_themes_apply_style_defaults(
    theme: str,
    expected: dict[str, object],
) -> None:
    config = PlotConfig(theme=theme)  # type: ignore[arg-type]

    assert config.node_color == expected["node_color"]
    assert config.node_edge_color == expected["node_edge_color"]
    assert config.node_color_degree_one == expected["node_color_degree_one"]
    assert config.node_edge_color_degree_one == expected["node_edge_color_degree_one"]
    assert config.tensor_label_color == expected["tensor_label_color"]
    assert config.label_color == expected["label_color"]
    assert config.bond_edge_color == expected["bond_edge_color"]
    assert config.dangling_edge_color == expected["dangling_edge_color"]
    assert config.line_width_2d == pytest.approx(expected["line_width_2d"])
    assert config.line_width_3d == pytest.approx(expected["line_width_3d"])
    assert config.contraction_scheme_colors == expected["contraction_scheme_colors"]


@pytest.mark.parametrize(
    ("theme", "expected_figure_color", "expected_axes_color"),
    [
        ("dark", "#0B1120", "#111827"),
        ("midnight", "#020617", "#0F172A"),
        ("forest", "#F7FDF9", "#F7FDF9"),
        ("slate", "#F1F5F9", "#F8FAFC"),
    ],
)
def test_show_tensor_network_applies_theme_background_colors(
    theme: str,
    expected_figure_color: str,
    expected_axes_color: str,
) -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        view="2d",
        config=PlotConfig(theme=theme),  # type: ignore[arg-type]
        show=False,
        show_controls=False,
    )

    assert to_hex(fig.get_facecolor()).lower() == expected_figure_color.lower()
    assert to_hex(ax.get_facecolor()).lower() == expected_axes_color.lower()


def test_tensor_elements_config_accepts_analysis_overrides() -> None:
    config = TensorElementsConfig(
        mode="slice",
        analysis=TensorAnalysisConfig(
            slice_axis="row",
            slice_index=1,
            reduce_axes=("col",),
            reduce_method="norm",
            profile_axis="row",
            profile_method="mean",
        ),
    )

    assert config.mode == "slice"
    assert config.analysis == TensorAnalysisConfig(
        slice_axis="row",
        slice_index=1,
        reduce_axes=("col",),
        reduce_method="norm",
        profile_axis="row",
        profile_method="mean",
    )


def test_tensor_elements_config_rejects_unknown_theme() -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        TensorElementsConfig(theme="presentation")  # type: ignore[arg-type]


def test_tensor_elements_config_grayscale_theme_applies_style_defaults() -> None:
    config = TensorElementsConfig(theme="grayscale")

    assert config.theme == "grayscale"
    assert config.continuous_cmap == "gray"
    assert config.log_magnitude_cmap == "Greys"
    assert config.phase_cmap == "twilight_shifted"
    assert config.diverging_cmap == "Greys"
    assert config.series_color == "#111111"
    assert config.histogram_color == "#444444"
    assert config.histogram_edge_color == "#111111"
    assert config.zero_marker_color == "#000000"
    assert config.hover_facecolor == "#FFFFFF"
    assert config.hover_edgecolor == "#111111"


@pytest.mark.parametrize(
    ("theme", "expected_continuous_cmap"),
    [
        ("contrast", "CMRmap"),
        ("paper", "inferno"),
        ("colorblind", "cividis"),
    ],
)
def test_tensor_elements_key_presets_use_distinct_continuous_colormaps(
    theme: str,
    expected_continuous_cmap: str,
) -> None:
    config = TensorElementsConfig(theme=theme)  # type: ignore[arg-type]

    assert config.continuous_cmap == expected_continuous_cmap


@pytest.mark.parametrize(
    ("theme", "expected"),
    [
        (
            "contrast",
            {
                "continuous_cmap": "CMRmap",
                "log_magnitude_cmap": "Greys",
                "diverging_cmap": "RdGy",
                "histogram_color": "#64748B",
                "hover_facecolor": "#FFFFFF",
            },
        ),
        (
            "paper",
            {
                "continuous_cmap": "inferno",
                "log_magnitude_cmap": "inferno",
                "diverging_cmap": "coolwarm",
                "histogram_color": "#0369A1",
                "hover_facecolor": "#F8FAFC",
            },
        ),
    ],
)
def test_tensor_elements_contrast_and_paper_swap_key_style_defaults(
    theme: str,
    expected: dict[str, str],
) -> None:
    config = TensorElementsConfig(theme=theme)  # type: ignore[arg-type]

    assert config.continuous_cmap == expected["continuous_cmap"]
    assert config.log_magnitude_cmap == expected["log_magnitude_cmap"]
    assert config.diverging_cmap == expected["diverging_cmap"]
    assert config.histogram_color == expected["histogram_color"]
    assert config.hover_facecolor == expected["hover_facecolor"]


@pytest.mark.parametrize(
    ("theme", "expected"),
    [
        (
            "colorblind",
            {
                "continuous_cmap": "cividis",
                "log_magnitude_cmap": "cividis",
                "diverging_cmap": "coolwarm",
                "series_color": "#0072B2",
                "histogram_color": "#56B4E9",
            },
        ),
        (
            "rainbow",
            {
                "continuous_cmap": "gist_rainbow",
                "log_magnitude_cmap": "gist_rainbow",
                "diverging_cmap": "gist_rainbow",
                "series_color": "#FF00FF",
                "histogram_color": "#00AEEF",
            },
        ),
        (
            "spectral",
            {
                "continuous_cmap": "nipy_spectral",
                "log_magnitude_cmap": "nipy_spectral",
                "diverging_cmap": "Spectral",
                "series_color": "#7C3AED",
                "histogram_color": "#0891B2",
            },
        ),
    ],
)
def test_tensor_elements_config_new_themes_apply_style_defaults(
    theme: str,
    expected: dict[str, str],
) -> None:
    config = TensorElementsConfig(theme=theme)  # type: ignore[arg-type]

    assert config.theme == theme
    assert config.continuous_cmap == expected["continuous_cmap"]
    assert config.log_magnitude_cmap == expected["log_magnitude_cmap"]
    assert config.diverging_cmap == expected["diverging_cmap"]
    assert config.series_color == expected["series_color"]
    assert config.histogram_color == expected["histogram_color"]


def test_tensor_elements_config_theme_preserves_manual_overrides() -> None:
    config = TensorElementsConfig(
        theme="grayscale",
        continuous_cmap="cividis",
        series_color="#ABCDEF",
        hover_facecolor="#FDF6E3",
    )

    assert config.continuous_cmap == "cividis"
    assert config.log_magnitude_cmap == "Greys"
    assert config.series_color == "#ABCDEF"
    assert config.hover_facecolor == "#FDF6E3"
    assert config.hover_edgecolor == "#111111"


def test_show_tensor_network_public_signature_is_config_centric() -> None:
    signature = inspect.signature(show_tensor_network)

    assert tuple(signature.parameters) == (
        "network",
        "engine",
        "view",
        "config",
        "ax",
        "show_controls",
        "show",
    )


def test_show_tensor_comparison_public_signature_is_config_centric() -> None:
    signature = inspect.signature(show_tensor_comparison)

    assert tuple(signature.parameters) == (
        "data",
        "reference",
        "engine",
        "config",
        "comparison_config",
        "ax",
        "show_controls",
        "show",
    )


def test_tensor_comparison_config_has_expected_defaults() -> None:
    config = TensorComparisonConfig()

    assert config.mode == "reference"
    assert config.topk_count == 8


def test_normalize_tensor_network_public_signature_is_structural() -> None:
    signature = inspect.signature(normalize_tensor_network)

    assert tuple(signature.parameters) == (
        "network",
        "engine",
    )


def test_export_tensor_network_snapshot_public_signature_includes_layout_controls() -> None:
    signature = inspect.signature(export_tensor_network_snapshot)

    assert tuple(signature.parameters) == (
        "network",
        "engine",
        "view",
        "config",
        "seed",
    )


def test_show_tensor_network_rejects_invalid_view() -> None:
    """Invalid view must raise ValueError with a clear message."""
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
    right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
    left["b"] ^ right["b"]

    with pytest.raises(ValueError, match="Unsupported tensor network view"):
        show_tensor_network(
            network,
            engine="tensorkrowch",
            view="invalid",  # type: ignore[arg-type]
            show=False,
        )


def test_show_tensor_network_rejects_invalid_engine() -> None:
    """Invalid engine must raise ValueError."""
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    tk.Node(shape=(2,), axes_names=("a",), name="N", network=network)

    with pytest.raises(ValueError, match="Unsupported tensor network engine"):
        show_tensor_network(
            network,
            engine="unknown_engine",  # type: ignore[arg-type]
            view="2d",
            show=False,
        )


def test_show_tensor_network_returns_fig_ax_with_show_false() -> None:
    """show=False must return (fig, ax) without displaying."""
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
    right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
    left["b"] ^ right["b"]

    fig, ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="2d",
        config=PlotConfig(figsize=(6, 4)),
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.name != "3d"
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.current_view == "2d"
    assert controls._view_caches["2d"].ax is ax
    assert len(fig.axes) >= 3


@pytest.mark.parametrize(
    "theme",
    ["paper", "colorblind", "dark", "midnight", "forest", "slate"],
)
@pytest.mark.parametrize("view", ["2d", "3d"])
def test_show_tensor_network_renders_visual_themes(theme: str, view: str) -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        view=view,  # type: ignore[arg-type]
        config=PlotConfig(theme=theme),  # type: ignore[arg-type]
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    if view == "3d":
        assert ax.name == "3d"
    else:
        assert ax.name != "3d"


def test_show_tensor_network_defaults_to_2d_when_view_is_omitted() -> None:
    """Omitting ``view`` should start in 2D by default."""
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
    right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
    left["b"] ^ right["b"]

    fig, ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.name != "3d"
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert controls.current_view == "2d"
    assert controls._view_caches["2d"].ax is ax
    assert len(fig.axes) >= 3


def test_show_tensor_network_headless_save_produces_file() -> None:
    """Figure can be saved to disk without display (headless)."""
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
    right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
    left["b"] ^ right["b"]

    fig, _ = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="2d",
        show=False,
    )

    out_path = Path(".tmp") / "test_public_api_headless.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    image = assert_readable_image(out_path)
    assert image.shape[0] > 0
    assert image.shape[1] > 0


@pytest.mark.parametrize("engine", ["tensorkrowch", "tensornetwork", "quimb", "tenpy", "einsum"])
def test_show_tensor_network_supports_engine_when_available(engine: str) -> None:
    """Each engine, when installed, produces a valid figure via show_tensor_network."""
    _run_engine_smoke(engine)  # type: ignore[arg-type]


def _run_engine_smoke(engine: EngineName) -> None:
    if engine == "tensorkrowch":
        pytest.importorskip("tensorkrowch")
        tk = __import__("tensorkrowch", fromlist=["TensorNetwork", "Node"])
        net = tk.TensorNetwork(name="t")
        left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=net)
        right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=net)
        left["b"] ^ right["b"]
        network = net
    elif engine == "tensornetwork":
        pytest.importorskip("tensornetwork")
        tn = __import__("tensornetwork", fromlist=["Node"])
        left = tn.Node(__import__("numpy").ones((2, 2)), name="L", axis_names=("a", "b"))
        right = tn.Node(__import__("numpy").ones((2, 2)), name="R", axis_names=("b", "c"))
        left["b"] ^ right["b"]
        network = [left, right]
    elif engine == "quimb":
        pytest.importorskip("quimb")
        qtn = __import__("quimb.tensor", fromlist=["Tensor", "TensorNetwork"])
        np = __import__("numpy")
        left = qtn.Tensor(np.ones((2, 2)), inds=("a", "b"), tags={"L"})
        right = qtn.Tensor(np.ones((2, 2)), inds=("b", "c"), tags={"R"})
        network = qtn.TensorNetwork([left, right])
    elif engine == "tenpy":
        pytest.importorskip("tenpy")
        from tenpy.networks.mps import MPS
        from tenpy.networks.site import SpinHalfSite

        sites = [SpinHalfSite() for _ in range(2)]
        network = MPS.from_product_state(sites, ["up", "up"], bc="finite")
    elif engine == "einsum":
        from tensor_network_viz import pair_tensor

        network = [
            pair_tensor("A", "x", "r0", "ab,b->a"),
        ]
    else:
        raise ValueError(engine)

    for view in ("2d", "3d"):
        fig, ax = show_tensor_network(
            network,
            engine=engine,
            view=view,  # type: ignore[arg-type]
            show=False,
        )
        assert_rendered_figure(fig, ax)
        if view == "3d":
            assert ax.name == "3d"
