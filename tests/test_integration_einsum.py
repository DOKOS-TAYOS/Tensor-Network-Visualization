import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz import pair_tensor
from tensor_network_viz.einsum import plot_einsum_network_2d, plot_einsum_network_3d

torch = pytest.importorskip("torch")


def test_pair_tensor_works_with_torch_einsum_and_renders() -> None:
    a_dim = 2
    p_dim = 3
    b_dim = 4

    a0 = torch.ones((p_dim, a_dim))
    x0 = torch.ones((p_dim,))
    a1 = torch.ones((a_dim, p_dim, b_dim))
    x1 = torch.ones((p_dim,))
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
    ]

    r0 = torch.einsum(trace[0], a0, x0)
    r1 = torch.einsum(trace[1], r0, a1)
    r2 = torch.einsum(trace[2], r1, x1)
    fig2d, ax2d = plot_einsum_network_2d(trace)
    fig3d, ax3d = plot_einsum_network_3d(trace)

    assert isinstance(trace[0], str)
    assert tuple(r0.shape) == (a_dim,)
    assert tuple(r1.shape) == (p_dim, b_dim)
    assert tuple(r2.shape) == (b_dim,)
    assert fig2d is ax2d.figure
    assert fig3d is ax3d.figure
    assert ax3d.name == "3d"
