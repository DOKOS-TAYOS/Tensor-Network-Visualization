import matplotlib

matplotlib.use("Agg")

from typing import Any

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz._core.curves import _require_self_endpoints
from tensor_network_viz._core.graph import _require_contraction_endpoints
from tensor_network_viz.tenpy import (
    make_tenpy_tensor_network,
    plot_tenpy_network_2d,
    plot_tenpy_network_3d,
)
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


def _build_purification_mps(length: int = 3):
    from tenpy.networks.purification_mps import PurificationMPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(length)]
    return PurificationMPS.from_infiniteT(sites, bc="finite")


def _build_uniform_mps_from_imps(length: int = 3):
    from tenpy.networks.uniform_mps import UniformMPS

    return UniformMPS.from_MPS(_build_infinite_mps(length=length))


class _MomentumLikeForGraph:
    """Minimal stand-in for ``MomentumMPS`` (duck typing) without calling its ``__init__``."""

    def __init__(self, uMPS_GS: Any) -> None:
        self.uMPS_GS = uMPS_GS

    def get_X(self, i: int, copy: bool = False) -> Any:
        return self.uMPS_GS.get_AR(i, copy=copy)


def test_build_tenpy_graph_accepts_purification_mps() -> None:
    graph = _build_tenpy_graph(_build_purification_mps(length=3))

    assert [node.name for node in graph.nodes.values()] == ["B0", "B1", "B2"]
    dangling = {edge.label for edge in graph.edges if edge.kind == "dangling"}
    assert {"p", "q"} <= dangling


def test_build_tenpy_graph_accepts_uniform_mps() -> None:
    graph = _build_tenpy_graph(_build_uniform_mps_from_imps(length=3))

    assert [node.name for node in graph.nodes.values()] == ["B0", "B1", "B2"]
    contraction_pairs = [edge.node_ids for edge in graph.edges if edge.kind == "contraction"]
    assert contraction_pairs == [(0, 1), (1, 2), (2, 0)]


def test_build_tenpy_graph_accepts_momentum_like_chain() -> None:
    u = _build_uniform_mps_from_imps(length=3)
    graph = _build_tenpy_graph(_MomentumLikeForGraph(u))

    assert [node.name for node in graph.nodes.values()] == ["X0", "X1", "X2"]
    contraction_pairs = [edge.node_ids for edge in graph.edges if edge.kind == "contraction"]
    assert contraction_pairs == [(0, 1), (1, 2), (2, 0)]


def test_build_tenpy_graph_real_momentum_mps_if_constructible() -> None:
    from tenpy.networks.momentum_mps import MomentumMPS

    u = _build_uniform_mps_from_imps(length=2)
    try:
        Xs = [u.get_AR(i, copy=True) * 0 for i in range(u.L)]
        mom = MomentumMPS(Xs, u, 0.0)
    except AttributeError as exc:
        if "find_common_type" in str(exc):
            pytest.skip("MomentumMPS incompatible with installed NumPy (find_common_type removed).")
        raise
    graph = _build_tenpy_graph(mom)
    assert [node.name for node in graph.nodes.values()] == ["X0", "X1"]
    assert {edge.kind for edge in graph.edges} >= {"contraction", "dangling"}


def test_build_tenpy_graph_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Unsupported TeNPy input"):
        _build_tenpy_graph(object())


def test_explicit_tn_matches_finite_mps_topology() -> None:
    mps = _build_finite_mps(length=3)
    ref = _build_tenpy_graph(mps)
    tn = make_tenpy_tensor_network(
        nodes=[(f"B{i}", mps.get_B(i)) for i in range(3)],
        bonds=[
            (("B0", "vR"), ("B1", "vL")),
            (("B1", "vR"), ("B2", "vL")),
        ],
    )
    got = _build_tenpy_graph(tn)
    assert len(got.nodes) == len(ref.nodes)
    assert len(got.edges) == len(ref.edges)
    assert {e.kind for e in got.edges} == {e.kind for e in ref.edges}


def test_make_tenpy_tensor_network_rejects_duplicate_leg_use() -> None:
    import numpy as np
    from tenpy.linalg import np_conserved as npc

    leg = npc.LegCharge.from_trivial(2)
    t0 = npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=["j", "d0"])
    t1 = npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=["j", "d1"])
    with pytest.raises(ValueError, match="more than one bond"):
        make_tenpy_tensor_network(
            nodes=[("T0", t0), ("T1", t1)],
            bonds=[
                (("T0", "j"), ("T1", "j")),
                (("T0", "j"), ("T1", "j")),
            ],
        )


def test_build_explicit_tn_hyperedge_hub() -> None:
    import numpy as np
    from tenpy.linalg import np_conserved as npc

    leg = npc.LegCharge.from_trivial(2)
    t0 = npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=["j", "d0"])
    t1 = npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=["j", "d1"])
    t2 = npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=["j", "d2"])
    tn = make_tenpy_tensor_network(
        nodes=[("T0", t0), ("T1", t1), ("T2", t2)],
        bonds=[(("T0", "j"), ("T1", "j"), ("T2", "j"))],
    )
    graph = _build_tenpy_graph(tn)
    virtual = [n for n in graph.nodes.values() if n.is_virtual]
    assert len(virtual) == 1
    assert sum(1 for e in graph.edges if e.kind == "contraction") == 3
    assert sum(1 for e in graph.edges if e.kind == "dangling") == 3


def test_tensor_network_viz_exports_tenpy_explicit() -> None:
    from tensor_network_viz import TenPyTensorNetwork
    from tensor_network_viz import make_tenpy_tensor_network as m

    assert TenPyTensorNetwork.__name__ == "TenPyTensorNetwork"
    assert callable(m)
