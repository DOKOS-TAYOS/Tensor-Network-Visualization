from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plotting_helpers import line_collection_segment_count
from tensor_network_viz import PlotConfig, einsum_trace_step, pair_tensor
from tensor_network_viz._core.layout.body import _compute_axis_directions, _compute_layout
from tensor_network_viz._core.layout.positions import (
    _analyze_layout_components_cached,
    _compute_component_layout_2d,
)
from tensor_network_viz._core.renderer import _resolve_draw_scale
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


def _einsum_like_mpo_trace(n_sites: int) -> list[object]:
    phys_dim = 2
    bond_dims = tuple(3 + index for index in range(max(n_sites - 1, 0)))
    if n_sites == 1:
        return [
            einsum_trace_step(
                ("W0", "d0", "u0"),
                "r0",
                "du,d,u->",
                metadata={"operand_shapes": ((phys_dim, phys_dim), (phys_dim,), (phys_dim,))},
            )
        ]

    steps: list[object] = [
        einsum_trace_step(
            operand_names=("W0", "d0", "u0"),
            result_name="r0",
            equation="dub,d,u->b",
            metadata={
                "operand_shapes": ((phys_dim, phys_dim, bond_dims[0]), (phys_dim,), (phys_dim,))
            },
        )
    ]
    current_name = "r0"
    current_shape = (bond_dims[0],)
    for index in range(1, n_sites - 1):
        steps.append(
            einsum_trace_step(
                operand_names=(current_name, f"W{index}", f"d{index}", f"u{index}"),
                result_name=f"r{index}",
                equation="a,adub,d,u->b",
                metadata={
                    "operand_shapes": (
                        current_shape,
                        (bond_dims[index - 1], phys_dim, phys_dim, bond_dims[index]),
                        (phys_dim,),
                        (phys_dim,),
                    )
                },
            )
        )
        current_name = f"r{index}"
        current_shape = (bond_dims[index],)
    steps.append(
        einsum_trace_step(
            operand_names=(current_name, f"W{n_sites - 1}", f"d{n_sites - 1}", f"u{n_sites - 1}"),
            result_name=f"r{n_sites - 1}",
            equation="a,adu,d,u->",
            metadata={
                "operand_shapes": (
                    current_shape,
                    (bond_dims[n_sites - 2], phys_dim, phys_dim),
                    (phys_dim,),
                    (phys_dim,),
                )
            },
        )
    )
    return steps


def test_compute_layout_einsum_disconnected_components_2d_does_not_raise() -> None:
    graph = _build_graph(
        [
            pair_tensor("A", "x", "r0", "ab,b->a"),
            pair_tensor("B", "y", "r1", "cd,d->c"),
        ]
    )

    positions = _compute_layout(graph, dimensions=2, seed=0)

    assert set(positions) == set(graph.nodes)


def test_compute_layout_einsum_mpo_2d_separates_up_and_down_vectors() -> None:
    graph = _build_graph(_einsum_like_mpo_trace(4))

    positions = _compute_layout(graph, dimensions=2, seed=0)
    node_id_by_name = {node.name: node_id for node_id, node in graph.nodes.items()}

    for index in range(4):
        down = np.asarray(positions[node_id_by_name[f"d{index}"]], dtype=float).reshape(-1)[:2]
        up = np.asarray(positions[node_id_by_name[f"u{index}"]], dtype=float).reshape(-1)[:2]
        tensor = np.asarray(positions[node_id_by_name[f"W{index}"]], dtype=float).reshape(-1)[:2]
        assert np.linalg.norm(down - up) > 0.5
        assert float(np.dot(down - tensor, up - tensor)) < 0.0


def test_compute_component_layout_2d_skips_trimmed_leaf_nodes_in_force_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_graph(_einsum_like_mpo_trace(4))
    component = _analyze_layout_components_cached(graph)[0]
    captured_node_ids: list[int] = []

    def fake_force_layout(
        graph_obj: object,
        *,
        node_ids: list[int],
        dimensions: int,
        seed: int,
        iterations: int,
        fixed_positions: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        del graph_obj, dimensions, seed, iterations
        captured_node_ids[:] = list(node_ids)
        positions: dict[int, np.ndarray] = {}
        for index, node_id in enumerate(node_ids):
            if fixed_positions is not None and node_id in fixed_positions:
                positions[node_id] = np.asarray(fixed_positions[node_id], dtype=float).copy()
            else:
                positions[node_id] = np.array([float(index), 0.0], dtype=float)
        return positions

    monkeypatch.setattr(
        "tensor_network_viz._core.layout.positions._compute_force_layout",
        fake_force_layout,
    )

    _compute_component_layout_2d(graph, component, seed=0, iterations=1)

    trimmed_leaf_ids = {leaf_id for leaf_id, _ in component.trimmed_leaf_parents}
    assert trimmed_leaf_ids
    assert trimmed_leaf_ids.isdisjoint(captured_node_ids)


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


def test_batch_virtual_hub_outputs_fan_out_per_shared_visible_neighbor_group() -> None:
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

    positions = _compute_layout(graph, dimensions=2, seed=0)
    draw_scale = _resolve_draw_scale(graph, positions)
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=draw_scale)
    virtual_node_ids = [node_id for node_id, node in graph.nodes.items() if node.is_virtual]

    assert len(virtual_node_ids) == 2
    group_center = np.mean(
        np.stack(
            [
                np.asarray(positions[node_id], dtype=float).reshape(-1)[:2]
                for node_id in virtual_node_ids
            ]
        ),
        axis=0,
    )

    for node_id in virtual_node_ids:
        node = graph.nodes[node_id]
        axis_index = len(node.axes_names) - 1
        direction = np.asarray(directions[(node_id, axis_index)], dtype=float).reshape(-1)[:2]
        direction /= np.linalg.norm(direction)
        outward = np.asarray(positions[node_id], dtype=float).reshape(-1)[:2] - group_center
        outward /= np.linalg.norm(outward)
        assert float(np.dot(direction, outward)) > 0.9, (node.label, direction, outward)


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
