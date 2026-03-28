import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz._core.curves import _require_self_endpoints
from tensor_network_viz._core.graph import _require_contraction_endpoints
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


def _build_infinite_mpo(length: int = 3):
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": length, "J": 1.0, "g": 1.0, "bc_MPS": "infinite"})
    return model.calc_H_MPO()


def test_build_tenpy_graph_accepts_finite_mps() -> None:
    graph = _build_tenpy_graph(_build_finite_mps())

    assert [node.name for node in graph.nodes.values()] == ["B0", "B1", "B2"]
    assert {edge.kind for edge in graph.edges} >= {"contraction", "dangling"}
    assert {edge.label for edge in graph.edges if edge.label} >= {"p"}
    axes = set()
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        lo, hi = _require_contraction_endpoints(edge)
        axes.update({lo.axis_name, hi.axis_name})
    assert {"vR", "vL"} <= axes


def test_build_tenpy_graph_accepts_finite_mpo() -> None:
    graph = _build_tenpy_graph(_build_finite_mpo())

    assert [node.name for node in graph.nodes.values()] == ["W0", "W1", "W2"]
    assert {edge.kind for edge in graph.edges} >= {"contraction", "dangling"}
    assert {edge.label for edge in graph.edges if edge.label} >= {"p", "p*"}
    axes = set()
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        lo, hi = _require_contraction_endpoints(edge)
        axes.update({lo.axis_name, hi.axis_name})
    assert {"wR", "wL"} <= axes


def test_build_tenpy_graph_accepts_infinite_mps_as_closed_unit_cell() -> None:
    graph = _build_tenpy_graph(_build_infinite_mps())

    assert [node.name for node in graph.nodes.values()] == ["B0", "B1", "B2"]
    contraction_pairs = [edge.node_ids for edge in graph.edges if edge.kind == "contraction"]
    assert contraction_pairs == [(0, 1), (1, 2), (2, 0)]


def test_build_tenpy_graph_accepts_infinite_mpo_as_closed_unit_cell() -> None:
    graph = _build_tenpy_graph(_build_infinite_mpo())

    assert [node.name for node in graph.nodes.values()] == ["W0", "W1", "W2"]
    contraction_pairs = [edge.node_ids for edge in graph.edges if edge.kind == "contraction"]
    assert contraction_pairs == [(0, 1), (1, 2), (2, 0)]
    assert {edge.label for edge in graph.edges if edge.label} >= {"p", "p*"}
    axes = set()
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        lo, hi = _require_contraction_endpoints(edge)
        axes.update({lo.axis_name, hi.axis_name})
    assert {"wR", "wL"} <= axes


def test_build_tenpy_graph_builds_single_site_infinite_mps_as_self_edge() -> None:
    graph = _build_tenpy_graph(_build_infinite_mps(length=1))

    self_edges = [edge for edge in graph.edges if edge.kind == "self"]
    assert len(self_edges) == 1
    assert self_edges[0].node_ids == (0,)
    assert self_edges[0].label is None
    ep_a, ep_b = _require_self_endpoints(self_edges[0])
    assert {ep_a.axis_name, ep_b.axis_name} == {"vR", "vL"}


def test_build_tenpy_graph_builds_two_site_infinite_mps_as_parallel_edges() -> None:
    graph = _build_tenpy_graph(_build_infinite_mps(length=2))

    contraction_edges = [edge for edge in graph.edges if edge.kind == "contraction"]
    assert [edge.node_ids for edge in contraction_edges] == [(0, 1), (1, 0)]
    assert all(edge.label is None for edge in contraction_edges)
    assert all(
        {ep.axis_name for ep in _require_contraction_endpoints(edge)} == {"vR", "vL"}
        for edge in contraction_edges
    )


def test_plot_tenpy_network_2d_draws_finite_mps() -> None:
    fig, ax = plot_tenpy_network_2d(_build_finite_mps(length=2))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"B0", "B1", "p", "vR", "vL"}
    from plotting_helpers import line_collection_segment_count

    assert line_collection_segment_count(ax) >= 1


def test_plot_tenpy_network_2d_draws_infinite_mps() -> None:
    fig, ax = plot_tenpy_network_2d(_build_infinite_mps(length=3))

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"B0", "B1", "B2", "p", "vR", "vL"}
    from plotting_helpers import line_collection_segment_count

    assert line_collection_segment_count(ax) == 6


def test_plot_tenpy_network_3d_returns_3d_axes() -> None:
    fig, ax = plot_tenpy_network_3d(_build_finite_mps(length=2))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) >= 1


def test_plot_tenpy_network_3d_draws_infinite_mpo() -> None:
    fig, ax = plot_tenpy_network_3d(_build_infinite_mpo(length=3))

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 9


def test_plot_tenpy_network_3d_rejects_2d_axis() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="plot_tenpy_network_3d"):
        plot_tenpy_network_3d(_build_finite_mps(length=2), ax=ax)

    plt.close(fig)
