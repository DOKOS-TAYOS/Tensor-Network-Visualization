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
    demo_runs_headless,
    finalize_demo_plot_config,
    render_demo_tensor_network,
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


def _render_args(*, no_show: bool = False, save: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        no_show=no_show,
        save=save,
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


def test_demo_runs_headless_false_without_save_or_no_show() -> None:
    assert demo_runs_headless(_render_args()) is False


def test_demo_runs_headless_true_with_no_show() -> None:
    assert demo_runs_headless(_render_args(no_show=True)) is True


def test_demo_runs_headless_true_with_save() -> None:
    assert demo_runs_headless(_render_args(save=Path("demo.png"))) is True


def test_render_demo_tensor_network_disables_controls_for_headless_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_show_tensor_network(network: object, **kwargs: object) -> tuple[str, str]:
        calls.append({"network": network, **kwargs})
        return ("figure", "axes")

    monkeypatch.setattr("demo_cli.show_tensor_network", _fake_show_tensor_network)

    result = render_demo_tensor_network(
        object(),
        args=_render_args(no_show=True),
        engine="tensornetwork",
        view="2d",
        config=finalize_demo_plot_config(_demo_args(), network="mps", engine="tensornetwork"),
    )

    assert result == ("figure", "axes")
    assert calls[0]["interactive_controls"] is False
    assert calls[0]["show"] is False


def test_render_demo_tensor_network_keeps_controls_for_interactive_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_show_tensor_network(network: object, **kwargs: object) -> tuple[str, str]:
        calls.append({"network": network, **kwargs})
        return ("figure", "axes")

    monkeypatch.setattr("demo_cli.show_tensor_network", _fake_show_tensor_network)

    result = render_demo_tensor_network(
        object(),
        args=_render_args(),
        engine="tensornetwork",
        view="3d",
        config=finalize_demo_plot_config(_demo_args(), network="mps", engine="tensornetwork"),
    )

    assert result == ("figure", "axes")
    assert calls[0]["interactive_controls"] is True
    assert calls[0]["show"] is False
