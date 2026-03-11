import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d
from tensor_network_viz.tenpy.graph import _build_graph as _build_tenpy_graph

tenpy = pytest.importorskip("tenpy")

pytestmark = pytest.mark.filterwarnings("ignore:unit_cell_width.*:UserWarning")


def _build_finite_mps(length: int = 3):
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(length)]
    return MPS.from_product_state(sites, ["up"] * length, bc="finite")


def _build_infinite_mps(length: int = 3):
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(length)]
    return MPS.from_product_state(sites, ["up"] * length, bc="infinite")


def _build_finite_mpo(length: int = 3):
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": length, "J": 1.0, "g": 1.0, "bc_MPS": "finite"})
    return model.calc_H_MPO()


def test_build_tenpy_graph_accepts_finite_mps() -> None:
    graph = _build_tenpy_graph(_build_finite_mps())

    assert [node.name for node in graph.nodes.values()] == ["B0", "B1", "B2"]
    assert {edge.kind for edge in graph.edges} >= {"contraction", "dangling"}
    assert {edge.label for edge in graph.edges} >= {"p", "vR<->vL"}


def test_build_tenpy_graph_accepts_finite_mpo() -> None:
    graph = _build_tenpy_graph(_build_finite_mpo())

    assert [node.name for node in graph.nodes.values()] == ["W0", "W1", "W2"]
    assert {edge.kind for edge in graph.edges} >= {"contraction", "dangling"}
    assert {edge.label for edge in graph.edges} >= {"p", "p*", "wR<->wL"}


def test_build_tenpy_graph_rejects_infinite_networks() -> None:
    with pytest.raises(ValueError, match="finite or segment"):
        _build_tenpy_graph(_build_infinite_mps())


def test_plot_tenpy_network_2d_draws_finite_mps() -> None:
    fig, ax = plot_tenpy_network_2d(_build_finite_mps(length=2))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"B0", "B1", "p", "vR<->vL"}
    assert len(ax.lines) >= 1


def test_plot_tenpy_network_3d_returns_3d_axes() -> None:
    fig, ax = plot_tenpy_network_3d(_build_finite_mps(length=2))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) >= 1


def test_plot_tenpy_network_3d_rejects_2d_axis() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="plot_tenpy_network_3d"):
        plot_tenpy_network_3d(_build_finite_mps(length=2), ax=ax)

    plt.close(fig)
