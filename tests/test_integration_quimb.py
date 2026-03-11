import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d

qtn = pytest.importorskip("quimb.tensor")


def test_real_quimb_tensor_network_renders() -> None:
    left = qtn.Tensor(data=np.ones((2, 3)), inds=("input", "bond"), tags={"left"})
    right = qtn.Tensor(data=np.ones((3, 5)), inds=("bond", "output"), tags={"right"})
    network = qtn.TensorNetwork([left, right])

    fig2d, ax2d = plot_quimb_network_2d(network)
    fig3d, ax3d = plot_quimb_network_3d(network)

    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"
