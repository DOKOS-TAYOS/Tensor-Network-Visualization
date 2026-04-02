from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plotting_helpers import line_collection_segment_count
from tensor_network_viz import PlotConfig, einsum_trace_step, pair_tensor
from tensor_network_viz._core.layout.body import _compute_layout
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

    visible = sorted(node.name for node in graph.nodes.values() if not node.is_virtual)
    virtual_count = sum(1 for node in graph.nodes.values() if node.is_virtual)

    contraction_edges = [edge for edge in graph.edges if edge.kind == "contraction"]
    dangling = [edge for edge in graph.edges if edge.kind == "dangling"]

    assert visible == ["A0", "A1", "x0", "x1"]
    assert virtual_count == 0
    assert len(contraction_edges) == 3
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

    visible = sorted(n.name for n in graph.nodes.values() if not n.is_virtual)
    assert visible == ["A", "B", "x", "y"]
    assert sum(1 for n in graph.nodes.values() if n.is_virtual) == 0
    assert [edge.kind for edge in graph.edges].count("contraction") == 2
    assert sorted(str(edge.label) for edge in graph.edges if edge.kind == "dangling") == ["a", "c"]


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (
            [
                pair_tensor(
                    "A",
                    "B",
                    "r0",
                    "...ij,...jk",
                    metadata={"left_shape": (2, 3, 4), "right_shape": (2, 4, 5)},
                )
            ],
            "explicit output subscript",
        ),
        (
            [pair_tensor("A", "x", "r0", "...a,a->a")],
            "metadata",
        ),
        (
            [pair_tensor("A", "B", "r0", "ab,bc->b")],
            "Unary reductions",
        ),
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


def test_build_einsum_graph_batch_equation_uses_virtual_hubs() -> None:
    graph = _build_graph(
        [
            pair_tensor(
                "A",
                "B",
                "r0",
                "ab,ab->ab",
                metadata={"left_shape": (2, 3), "right_shape": (2, 3)},
            ),
        ]
    )
    assert sum(1 for n in graph.nodes.values() if n.is_virtual) == 2
    assert sum(1 for e in graph.edges if e.kind == "contraction") == 4
    dangles = [e for e in graph.edges if e.kind == "dangling"]
    assert len(dangles) == 2


def test_build_einsum_graph_expands_ellipsis_with_metadata() -> None:
    graph = _build_graph(
        [
            pair_tensor(
                "A",
                "B",
                "r0",
                "...ij,...jk->...ik",
                metadata={"left_shape": (2, 3, 4), "right_shape": (2, 4, 5)},
            ),
        ]
    )
    # Batch index uses a hub; the contracted matmul axis is a normal tensor–tensor bond.
    assert sum(1 for n in graph.nodes.values() if n.is_virtual) == 1
    a_node = next(n for nid, n in graph.nodes.items() if n.name == "A")
    assert len(a_node.axes_names) == 3


def test_build_einsum_graph_trace_on_single_tensor() -> None:
    graph = _build_graph(
        [
            pair_tensor(
                "A",
                "B",
                "r0",
                "ii,i->i",
                metadata={"left_shape": (4, 4), "right_shape": (4,)},
            ),
        ]
    )
    assert any(n.is_virtual for n in graph.nodes.values())
    assert any(e.kind == "contraction" for e in graph.edges)


def test_plot_einsum_network_2d_draws_reconstructed_graph() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]

    fig, ax = plot_einsum_network_2d(
        trace,
        config=PlotConfig(show_tensor_labels=True, show_index_labels=True),
    )

    labels = {text.get_text() for text in ax.texts}
    assert fig is ax.figure
    assert labels >= {"A0", "x0", "A1", "p", "a", "b"}
    assert line_collection_segment_count(ax) >= 4


def test_einsum_trace_nary_and_implicit_binary_graph() -> None:
    """Ternary equations build a graph; implicit binary pair_tensor is canonicalized."""
    graph = _build_graph(
        [
            einsum_trace_step(
                ("A", "B", "C"),
                "r0",
                "ab,bc,cd->ad",
                metadata={
                    "operand_shapes": ((2, 3), (3, 4), (4, 5)),
                },
            ),
        ]
    )
    visible = sorted(n.name for n in graph.nodes.values() if not n.is_virtual)
    assert visible == ["A", "B", "C"]
    graph2 = _build_graph([pair_tensor("A", "x", "r0", "ab,b")])
    assert any(n.name == "A" for n in graph2.nodes.values())


def test_einsum_trace_unary_graph() -> None:
    graph = _build_graph(
        [
            einsum_trace_step(
                ("M",),
                "r0",
                "ii->i",
                metadata={"operand_shapes": ((4, 4),)},
            ),
        ]
    )
    assert any(n.name == "M" for n in graph.nodes.values() if not n.is_virtual)
    assert any(e.kind == "contraction" for e in graph.edges)


def test_unary_einsum_2d_layout_offsets_virtual_hub_from_tensor() -> None:
    """Regression: single-tensor trace hub must not sit on the physical node (2D)."""
    graph = _build_graph(
        [
            einsum_trace_step(
                ("M",),
                "r0",
                "ii->i",
                metadata={"operand_shapes": ((4, 4),)},
            ),
        ]
    )
    positions = _compute_layout(graph, dimensions=2, seed=0)
    phys_ids = [nid for nid, n in graph.nodes.items() if not n.is_virtual]
    virt_ids = [nid for nid, n in graph.nodes.items() if n.is_virtual]
    assert len(phys_ids) == 1 and len(virt_ids) == 1
    d = float(np.linalg.norm(positions[phys_ids[0]] - positions[virt_ids[0]]))
    assert d > 0.08


def test_plot_einsum_network_3d_returns_3d_axes() -> None:
    trace = [pair_tensor("A", "x", "r0", "ab,b->a")]

    fig, ax = plot_einsum_network_3d(trace)

    assert fig is ax.figure
    assert ax.name == "3d"
    assert len(ax.lines) >= 2
