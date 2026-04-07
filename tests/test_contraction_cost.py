"""Naive dense contraction cost helpers for einsum trace steps."""

from __future__ import annotations

import pytest

from tensor_network_viz import pair_tensor
from tensor_network_viz._core.draw.contraction_scheme import _contraction_step_metrics_for_draw
from tensor_network_viz._core.draw.hover import _register_2d_hover_labels
from tensor_network_viz._core.graph import (
    _resolve_contraction_scheme_by_name,
)
from tensor_network_viz.config import PlotConfig
from tensor_network_viz.einsum_module._equation import _ParsedNaryEquation, parse_einsum_equation
from tensor_network_viz.einsum_module.contraction_cost import (
    _wrap_panel_line,
    format_contraction_step_tooltip,
    metrics_for_parsed_step,
)
from tensor_network_viz.einsum_module.graph import _build_graph


def test_metrics_matrix_multiply() -> None:
    parsed = parse_einsum_equation("ij,jk->ik", ((2, 3), (3, 4)))
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3), (3, 4)),
        equation_snippet="ij,jk->ik",
        operand_names=("A", "B"),
    )
    assert m.multiplicative_cost == 2 * 3 * 4
    assert m.flop_mac == 2 * m.multiplicative_cost
    assert dict(m.label_dims) == {"i": 2, "j": 3, "k": 4}
    assert m.operand_names == ("A", "B")
    assert m.operand_shapes == ((2, 3), (3, 4))
    assert m.output_labels == ("i", "k")
    assert m.contracted_labels == ("j",)


def test_metrics_flops_scale_with_operand_count() -> None:
    parsed = parse_einsum_equation("ab,bc,cd->ad", ((2, 3), (3, 5), (5, 7)))
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3), (3, 5), (5, 7)),
        equation_snippet="ab,bc,cd->ad",
        operand_names=("A", "B", "C"),
    )
    assert m.multiplicative_cost == 2 * 3 * 5 * 7
    assert m.flop_mac == 3 * m.multiplicative_cost


def test_metrics_merge_dummy_one_with_real_extent() -> None:
    """Live tensors use placeholder 1s; real operands define bond extent."""
    parsed = parse_einsum_equation("a,ab->b", ((1,), (2, 3)))
    m = metrics_for_parsed_step(parsed, ((1,), (2, 3)))
    assert m.multiplicative_cost == 2 * 3
    assert dict(m.label_dims)["a"] == 2


def test_metrics_rejects_true_mismatch() -> None:
    parsed = _ParsedNaryEquation(
        operand_axes=(("i", "j"), ("j", "k")),
        output_axes=("i", "k"),
    )
    with pytest.raises(ValueError, match="Inconsistent dimension"):
        metrics_for_parsed_step(parsed, ((2, 3), (4, 5)))


def test_format_tooltip_contains_cost_lines() -> None:
    parsed = parse_einsum_equation("ij,jk->ik", ((2, 3), (3, 4)))
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3), (3, 4)),
        equation_snippet="ij,jk->ik",
        operand_names=("A", "B"),
    )
    t = format_contraction_step_tooltip(m)
    assert "Contraction: ij,jk->ik (contracts: j)" in t
    assert "Index sizes: i=2, j=3, k=4" in t
    assert "Tensor shapes: A=[2, 3], B=[3, 4]" in t
    assert "Naive operations: 24 MACs (≈48 FLOPs)" in t
    assert "Complexity: O(N_i N_j N_k)" in t


def test_format_tooltip_omits_naive_footer() -> None:
    parsed = parse_einsum_equation("ij,jk->ik", ((2, 3), (3, 4)))
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3), (3, 4)),
        equation_snippet="ij,jk->ik",
        operand_names=("A", "B"),
    )
    t = format_contraction_step_tooltip(m)
    assert "Naive dense" not in t
    assert "optimized contraction order" not in t


def test_format_tooltip_reports_flops_for_three_operands() -> None:
    parsed = parse_einsum_equation("ab,bc,cd->ad", ((2, 3), (3, 5), (5, 7)))
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3), (3, 5), (5, 7)),
        equation_snippet="ab,bc,cd->ad",
        operand_names=("A", "B", "C"),
    )
    t = format_contraction_step_tooltip(m)
    assert "210 MACs" in t
    assert "630 FLOPs" in t


def test_format_tooltip_wraps_long_lines_in_panel_text() -> None:
    parsed = parse_einsum_equation(
        "abcdef,defghi,ghijkl->abcjkl",
        ((2, 3, 5, 7, 11, 13), (7, 11, 13, 17, 19, 23), (17, 19, 23, 29, 31, 37)),
    )
    m = metrics_for_parsed_step(
        parsed,
        ((2, 3, 5, 7, 11, 13), (7, 11, 13, 17, 19, 23), (17, 19, 23, 29, 31, 37)),
        equation_snippet="abcdef,defghi,ghijkl->abcjkl",
        operand_names=("A", "B", "C"),
    )
    t = format_contraction_step_tooltip(m)
    assert "Complexity: O(" in t
    assert "N_a" in t
    assert "N_l" in t
    assert t.count("\n") >= 4


def test_wrap_panel_line_uses_wider_cost_panel_width() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi"

    assert "\n" not in _wrap_panel_line(text)


def test_build_graph_records_metrics_same_length_as_steps() -> None:
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    ]
    graph = _build_graph(trace)
    assert graph.contraction_steps is not None
    assert graph.contraction_step_metrics is not None
    assert len(graph.contraction_step_metrics) == len(graph.contraction_steps)
    assert all(m is not None for m in graph.contraction_step_metrics)
    first_metric = graph.contraction_step_metrics[0]
    assert first_metric is not None
    assert first_metric.multiplicative_cost == 1


def test_build_graph_records_real_operand_shapes_for_intermediate_steps() -> None:
    trace = [
        pair_tensor(
            "A0",
            "x0",
            "r0",
            "pa,p->a",
            metadata={
                "left_shape": (5, 7),
                "right_shape": (5,),
                "result_shape": (7,),
            },
        ),
        pair_tensor(
            "r0",
            "A1",
            "r1",
            "a,apb->pb",
            metadata={
                "left_shape": (7,),
                "right_shape": (7, 11, 13),
                "result_shape": (11, 13),
            },
        ),
    ]
    graph = _build_graph(trace)
    assert graph.contraction_step_metrics is not None
    second_metric = graph.contraction_step_metrics[1]
    assert second_metric is not None
    assert second_metric.operand_names == ("r0", "A1")
    assert second_metric.operand_shapes == ((7,), (7, 11, 13))
    assert second_metric.output_labels == ("p", "b")
    assert second_metric.contracted_labels == ("a",)


def test_metrics_for_draw_none_when_scheme_overridden_by_name() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    graph = _build_graph(trace)
    override = _resolve_contraction_scheme_by_name(graph, (("A0",),))
    assert graph.contraction_steps != override
    assert _contraction_step_metrics_for_draw(graph, override) is None


def test_register_2d_hover_accepts_scheme_patches_only() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(2, 2))
    patch = Rectangle((0, 0), 1, 1, transform=ax.transData)
    ax.add_patch(patch)
    _register_2d_hover_labels(
        ax,
        node_patch_coll=None,
        visible_node_ids=[],
        tensor_hover={},
        edge_hover=[],
        line_width_px_hint=1.0,
        scheme_hover_patches=((patch, "scheme\nhint"),),
    )
    assert getattr(fig, "_tensor_network_viz_hover_cid", None) is not None
    plt.close(fig)


def test_plot_config_has_cost_hover_flag() -> None:
    assert PlotConfig().contraction_scheme_cost_hover is False
