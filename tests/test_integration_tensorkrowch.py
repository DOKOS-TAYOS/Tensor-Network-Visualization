import matplotlib

matplotlib.use("Agg")

import pytest

from tensorkrowch_engine import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d

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
