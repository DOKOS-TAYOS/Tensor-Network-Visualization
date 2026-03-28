from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from tensor_network_viz._core._label_format import format_tensor_node_label


def _assert_mathtext_parses(s: str) -> None:
    fp = FontProperties(size=10.0)
    TextPath((0.0, 0.0), s, prop=fp)


def test_plain_no_markup_unchanged() -> None:
    assert format_tensor_node_label("A") == "A"
    assert format_tensor_node_label("T12") == "T12"


def test_subscript_single_and_grouped() -> None:
    assert format_tensor_node_label("T_a") == r"$\mathrm{T}_{\mathrm{a}}$"
    assert format_tensor_node_label("rho_(ij)") == r"$\mathrm{rho}_{\mathrm{ij}}$"
    assert format_tensor_node_label("X_{ab}") == r"$\mathrm{X}_{\mathrm{ab}}$"


def test_subscript_double_merged_with_space() -> None:
    assert format_tensor_node_label("T_a_b") == r"$\mathrm{T}_{\mathrm{a}\ \mathrm{b}}$"


def test_superscript() -> None:
    assert format_tensor_node_label("x^2") == r"$\mathrm{x}^{\mathrm{2}}$"
    assert format_tensor_node_label("z^a^b") == r"$\mathrm{z}^{\mathrm{a}\ \mathrm{b}}$"


def test_nested_in_parens() -> None:
    assert format_tensor_node_label("M_(i_j)") == r"$\mathrm{M}_{\mathrm{i}_{\mathrm{j}}}$"


def test_double_underscore_literal_plus_subscript() -> None:
    out = format_tensor_node_label("T__a")
    _assert_mathtext_parses(out)


def test_dollar_latex_passthrough() -> None:
    assert format_tensor_node_label(r"$\alpha$") == r"$\alpha$"
    assert format_tensor_node_label(r"A $\beta$ B") == r"A $\beta$ B"


def test_double_dollar_collapses_to_one_in_plain() -> None:
    out = format_tensor_node_label("a$$b")
    assert out == r"$\mathrm{a\$b}$"
    _assert_mathtext_parses(out)


def test_unclosed_dollar_falls_back_to_plain_markup() -> None:
    out = format_tensor_node_label("x$y")
    _assert_mathtext_parses(out)
