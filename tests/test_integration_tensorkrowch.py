import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.tensorkrowch import (
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
)
from tensor_network_viz.tensorkrowch.graph import _build_graph as _build_tensorkrowch_graph

torch = pytest.importorskip("torch")
tk = pytest.importorskip("tensorkrowch")


def test_real_tensorkrowch_network_renders() -> None:
    network = tk.TensorNetwork(name="demo")
    left = tk.Node(shape=(2, 3), axes_names=("input", "bond"), name="left", network=network)
    right = tk.Node(shape=(3, 5), axes_names=("bond", "output"), name="right", network=network)
    left["bond"] ^ right["bond"]

    fig2d, ax2d = plot_tensorkrowch_network_2d(network)
    fig3d, ax3d = plot_tensorkrowch_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"


def test_real_tensorkrowch_contracted_network_exposes_auto_scheme_steps() -> None:
    network = tk.TensorNetwork(name="demo")
    left = tk.Node(
        shape=(2, 3),
        axes_names=("input", "bond"),
        name="left",
        network=network,
        tensor=torch.ones((2, 3)),
    )
    right = tk.Node(
        shape=(3, 5),
        axes_names=("bond", "output"),
        name="right",
        network=network,
        tensor=torch.ones((3, 5)),
    )
    left["bond"] ^ right["bond"]
    _ = left @ right

    graph = _build_tensorkrowch_graph(network)

    assert graph.contraction_steps is not None
    assert len(graph.contraction_steps) == 1
    assert graph.contraction_step_metrics is not None
    assert graph.contraction_step_metrics[0] is not None
    assert graph.contraction_step_metrics[0].multiplicative_cost == 30


def test_real_tensorkrowch_contracted_network_draws_auto_scheme_with_slider() -> None:
    network = tk.TensorNetwork(name="demo")
    left = tk.Node(
        shape=(2, 3),
        axes_names=("input", "bond"),
        name="left",
        network=network,
        tensor=torch.ones((2, 3)),
    )
    right = tk.Node(
        shape=(3, 5),
        axes_names=("bond", "output"),
        name="right",
        network=network,
        tensor=torch.ones((3, 5)),
    )
    left["bond"] ^ right["bond"]
    _ = left @ right

    fig, ax = plot_tensorkrowch_network_2d(
        network,
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_scheme_cost_hover=True,
        ),
    )

    patches = [patch for patch in ax.patches if patch.get_gid() == "tnv_contraction_scheme"]
    controls = getattr(fig, "_tensor_network_viz_contraction_controls", None)

    assert fig is ax.figure
    assert patches == []
    assert controls is not None
    assert controls._viewer is not None
    assert controls._viewer.slider is not None
    assert controls._viewer._cost_panel_ax is not None
    assert controls._viewer._cost_text_artist is not None
    assert "Contraction:" in controls._viewer._cost_text_artist.get_text()


def test_contracted_tensorkrowch_network_exposes_tensor_inspector() -> None:
    network = tk.TensorNetwork(name="demo")
    left = tk.Node(
        shape=(2, 3),
        axes_names=("input", "bond"),
        name="left",
        network=network,
        tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    right = tk.Node(
        shape=(3, 5),
        axes_names=("bond", "output"),
        name="right",
        network=network,
        tensor=torch.arange(15, dtype=torch.float32).reshape(3, 5),
    )
    left["bond"] ^ right["bond"]
    _ = left @ right

    fig, _ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="2d",
        config=PlotConfig(
            show_contraction_scheme=True,
            contraction_scheme_cost_hover=True,
            contraction_tensor_inspector=True,
        ),
        show=False,
    )

    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    inspector = getattr(fig, "_tensor_network_viz_tensor_inspector", None)

    assert controls is not None
    assert controls.tensor_inspector_on is True
    assert controls._checkbuttons is not None
    assert "Tensor inspector" in [label.get_text() for label in controls._checkbuttons.labels]
    assert inspector is not None
    assert inspector._figure is not None
