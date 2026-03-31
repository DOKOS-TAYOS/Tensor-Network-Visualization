"""Helpers in ``examples/demo_cli.py`` for demo contraction schedules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from demo_cli import (  # noqa: E402
    cubic_peps_tensor_names,
    cumulative_prefix_contraction_scheme,
    finalize_demo_plot_config,
)


def test_cumulative_prefix_grows_to_full_set() -> None:
    steps = cumulative_prefix_contraction_scheme(("A0", "A1", "A2", "A3"))
    assert steps == (
        ("A0", "A1"),
        ("A0", "A1", "A2"),
        ("A0", "A1", "A2", "A3"),
    )


def test_cumulative_prefix_single_tensor() -> None:
    assert cumulative_prefix_contraction_scheme(("X",)) == (("X",),)


def test_cumulative_prefix_empty() -> None:
    assert cumulative_prefix_contraction_scheme(()) == ()


def test_cubic_peps_names_match_grid_count() -> None:
    names = cubic_peps_tensor_names(2, 3, 2)
    assert len(names) == 12
    assert set(names) == {f"P{i}_{j}_{k}" for i in range(2) for j in range(3) for k in range(2)}


def test_cubic_peps_invalid_grid() -> None:
    with pytest.raises(ValueError, match="must be >="):
        cubic_peps_tensor_names(0, 1, 1)


def _demo_args(*, contraction_scheme: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        contraction_scheme=contraction_scheme,
        hover_labels=False,
        compact=False,
    )


def test_finalize_contraction_playback_true_for_einsum_scheme() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(contraction_scheme=True),
        network="chain",
        engine="einsum",
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_playback is True


def test_finalize_contraction_playback_true_when_manual_scheme() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(contraction_scheme=True),
        network="disconnected",
        engine="quimb",
    )
    assert cfg.contraction_scheme_by_name is not None
    assert cfg.contraction_playback is True


def test_finalize_contraction_playback_false_without_step_schedule() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(contraction_scheme=True),
        network="mera_tree",
        engine="tensornetwork",
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_scheme_by_name is None
    assert cfg.contraction_playback is False
