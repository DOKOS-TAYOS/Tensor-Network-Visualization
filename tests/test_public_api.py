"""Tests for the main public API: show_tensor_network, PlotConfig, and error handling."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

pytestmark = pytest.mark.filterwarnings("ignore:unit_cell_width.*:UserWarning")

from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.config import EngineName


def test_plot_config_has_expected_defaults() -> None:
    config = PlotConfig()
    assert config.figsize == (8, 6)
    assert config.show_tensor_labels is True
    assert config.show_index_labels is True
    assert config.layout_iterations is None
    assert config.positions is None
    assert config.validate_positions is False


def test_plot_config_accepts_overrides() -> None:
    config = PlotConfig(
        figsize=(10, 5),
        show_tensor_labels=False,
        layout_iterations=100,
    )
    assert config.figsize == (10, 5)
    assert config.show_tensor_labels is False
    assert config.layout_iterations == 100


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

    assert fig is ax.figure
    assert ax.name != "3d"


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

    assert out_path.exists()
    assert out_path.stat().st_size > 0


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
        assert fig is ax.figure
        if view == "3d":
            assert ax.name == "3d"
