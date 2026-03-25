from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Generator

import matplotlib.pyplot as plt
import pytest

from tensor_network_viz import pair_tensor
from tensor_network_viz.einsum_module import (
    plot_einsum_network_2d,
    plot_einsum_network_3d,
)
from tensor_network_viz.einsum_module.graph import _build_graph


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, None, None]:
    yield
    plt.close("all")


def test_pair_tensor_behaves_like_str_and_preserves_metadata() -> None:
    pair = pair_tensor(
        "A0",
        "x0",
        "r0",
        "pa,p->a",
        metadata={"kind": "site"},
    )

    assert isinstance(pair, str)
    assert str(pair) == "pa,p->a"
    assert pair.left_name == "A0"
    assert pair.right_name == "x0"
    assert pair.result_name == "r0"
    assert pair.equation == "pa,p->a"
    assert pair.metadata == {"kind": "site"}


def test_build_einsum_graph_reconstructs_chain_without_intermediate_nodes() -> None:
    graph = _build_graph(
        [
            pair_tensor("A0", "x0", "r0", "pa,p->a"),
            pair_tensor("r0", "A1", "r1", "a,apb->pb"),
            pair_tensor("r1", "x1", "r2", "pb,p->b"),
        ]
    )

    node_names = [node.name for node in graph.nodes.values()]
    contraction_pairs = {
        frozenset(graph.nodes[endpoint.node_id].name for endpoint in edge.endpoints)
        for edge in graph.edges
        if edge.kind == "contraction"
    }
    dangling = [edge for edge in graph.edges if edge.kind == "dangling"]

    assert node_names == ["A0", "x0", "A1", "x1"]
    assert contraction_pairs == {
        frozenset({"A0", "x0"}),
        frozenset({"A0", "A1"}),
        frozenset({"A1", "x1"}),
    }
    assert len(dangling) == 1
    assert dangling[0].label == "b"
    assert graph.nodes[dangling[0].endpoints[0].node_id].name == "A1"


def test_build_einsum_graph_supports_disconnected_components() -> None:
    graph = _build_graph(
        [
            pair_tensor("A", "x", "r0", "ab,b->a"),
            pair_tensor("B", "y", "r1", "cd,d->c"),
        ]
    )

    assert [node.name for node in graph.nodes.values()] == ["A", "x", "B", "y"]
    assert [edge.kind for edge in graph.edges].count("contraction") == 2
    assert [edge.label for edge in graph.edges if edge.kind == "dangling"] == ["a", "c"]


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        ([pair_tensor("A", "x", "r0", "ab,b")], "explicit"),
        ([pair_tensor("A", "x", "r0", "...a,a->...")], "ellipsis"),
        ([pair_tensor("A", "x", "r0", "aa,a->a")], "repeated"),
        ([pair_tensor("A", "B", "r0", "ab,bc->b")], "both operands"),
        (
            [
                pair_tensor("r1", "A", "r2", "a,a->"),
                pair_tensor("B", "x", "r1", "a,a->"),
            ],
            "before it is defined",
        ),
        (
            [
                pair_tensor("A", "x", "r0", "ab,b->a"),
                pair_tensor("r0", "B", "r0", "a,ac->c"),
            ],
            "must be new",
        ),
        (
            [
                pair_tensor("A", "x", "r0", "ab,b->a"),
                pair_tensor("A", "y", "r1", "ab,b->a"),
            ],
            "not available",
        ),
    ],
)
def test_build_einsum_graph_rejects_invalid_traces(
    trace: list[pair_tensor],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _build_graph(trace)


def test_plot_einsum_network_2d_draws_reconstructed_graph() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]

    fig, ax = plot_einsum_network_2d(trace)

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A0", "x0", "A1", "p<->p", "a<->a", "b"}
    assert len(ax.lines) == 4


def test_einsum_trace_requires_binary_explicit_output_equations() -> None:
    """Einsum backend supports only binary equations with explicit '->' output. Documented limitation."""
    # Non-binary (3 operands in one equation) is rejected
    with pytest.raises(ValueError, match="binary"):
        _build_graph([pair_tensor("A", "B", "r0", "ab,bc,cd->ad")])
    # Missing explicit output '->' is rejected
    with pytest.raises(ValueError, match="explicit"):
        _build_graph([pair_tensor("A", "x", "r0", "ab,b")])


def test_plot_einsum_network_3d_returns_3d_axes() -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = plot_einsum_network_3d(trace)

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) == 2
