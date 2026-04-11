"""Tests for the main public API: show_tensor_network, PlotConfig, and error handling."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib

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
)
from tensor_network_viz._core.renderer import _effective_layout_iterations
from tensor_network_viz.config import EngineName

pytestmark = pytest.mark.filterwarnings("ignore:unit_cell_width.*:UserWarning")


def test_plot_config_has_expected_defaults() -> None:
    config = PlotConfig()
    assert config.figsize == (8, 6)
    assert config.show_nodes is True
    assert config.show_tensor_labels is False
    assert config.show_index_labels is False
    assert config.show_contraction_scheme is False
    assert config.contraction_scheme_cost_hover is False
    assert config.contraction_tensor_inspector is False
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
    assert config.node_color == "#F1F5F9"
    assert config.node_edge_color == "#334155"
    assert config.node_color_degree_one == "#FFEDD5"
    assert config.node_edge_color_degree_one == "#C2410C"
    assert config.tensor_label_color == "#111827"
    assert config.bond_edge_color == "#2563EB"
    assert config.dangling_edge_color == "#DC2626"


def test_plot_theme_is_public_type_alias() -> None:
    import tensor_network_viz as tnv

    assert hasattr(tnv, "PlotTheme")
    assert tnv.PlotTheme.__args__ == ("default", "paper", "colorblind")


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


@pytest.mark.parametrize("theme", ["paper", "colorblind"])
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
