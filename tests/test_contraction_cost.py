"""Naive dense contraction cost helpers for einsum trace steps."""

from __future__ import annotations

import pytest

from tensor_network_viz import pair_tensor
from tensor_network_viz._core.draw.contraction_scheme import _contraction_step_metrics_for_draw
from tensor_network_viz._core.graph import _GraphData, _make_node, _resolve_contraction_scheme_by_name
from tensor_network_viz._core.draw.hover import _register_2d_hover_labels
from tensor_network_viz.config import PlotConfig
from tensor_network_viz.einsum_module._equation import _ParsedNaryEquation, parse_einsum_equation
from tensor_network_viz.einsum_module.contraction_cost import (
    format_contraction_step_tooltip,
    metrics_for_parsed_step,
)
from tensor_network_viz.einsum_module.graph import _build_graph


def test_metrics_matrix_multiply() -> None:
    parsed = parse_einsum_equation("ij,jk->ik", ((2, 3), (3, 4)))
    m = metrics_for_parsed_step(parsed, ((2, 3), (3, 4)), equation_snippet="ij,jk->ik")
    assert m.multiplicative_cost == 2 * 3 * 4
    assert m.flop_mac == 2 * m.multiplicative_cost
    assert dict(m.label_dims) == {"i": 2, "j": 3, "k": 4}


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
    m = metrics_for_parsed_step(parsed, ((2, 3), (3, 4)), equation_snippet="ij,jk->ik")
    t = format_contraction_step_tooltip(m)
    assert "ij,jk->ik" in t
    assert "FLOPs" in t
    assert "24" in t.replace(",", "")


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
    assert graph.contraction_step_metrics[0].multiplicative_cost == 1


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
    assert PlotConfig().contraction_scheme_cost_hover is True
