import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz.tensornetwork import (
    plot_tensornetwork_network_2d,
    plot_tensornetwork_network_3d,
)

tn = pytest.importorskip("tensornetwork")


def test_real_tensornetwork_nodes_render() -> None:
    left = tn.Node(np.ones((2, 3)), name="left", axis_names=("input", "bond"))
    right = tn.Node(np.ones((3, 5)), name="right", axis_names=("bond", "output"))
    left["bond"] ^ right["bond"]

    fig2d, ax2d = plot_tensornetwork_network_2d([left, right])
    fig3d, ax3d = plot_tensornetwork_network_3d([left, right])

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"
