from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from demo_cli import (  # noqa: E402
    ExampleCliArgs,
    apply_labels_override,
    auto_save_path,
    cubic_peps_tensor_names,
    cumulative_prefix_contraction_scheme,
    demo_runs_headless,
    finalize_demo_plot_config,
    render_demo_tensor_network,
)


def _load_example_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_apply_labels_override_keeps_specific_values_without_global_override() -> None:
    args = ExampleCliArgs(
        engine="quimb",
        example="mps",
        view="2d",
        labels_nodes=False,
        labels_edges=True,
        labels=None,
        hover_labels=True,
        scheme=False,
        playback=False,
        hover_cost=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
    )

    assert apply_labels_override(args) == (False, True)


def test_apply_labels_override_overwrites_both_when_global_present() -> None:
    args = ExampleCliArgs(
        engine="quimb",
        example="mps",
        view="2d",
        labels_nodes=False,
        labels_edges=True,
        labels=True,
        hover_labels=True,
        scheme=False,
        playback=False,
        hover_cost=False,
        from_scratch=False,
        from_list=False,
        save=None,
        no_show=False,
        n_sites=6,
        lx=3,
        ly=4,
        lz=3,
        mera_log2=3,
        tree_depth=4,
    )

    assert apply_labels_override(args) == (True, True)


def test_auto_save_path_uses_engine_and_example() -> None:
    path = auto_save_path(engine="tenpy", example="imps")

    assert path == Path(".tmp") / "examples" / "tenpy" / "imps.png"


def _demo_args(
    *, scheme: bool = False, playback: bool = False, hover_cost: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=scheme,
        playback=playback,
        hover_cost=hover_cost,
    )


def _render_args(*, no_show: bool = False, save: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        no_show=no_show,
        save=save,
    )


def test_finalize_contraction_playback_stays_false_for_einsum_scheme_by_default() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(scheme=True),
        engine="einsum",
        scheme_tensor_names=None,
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_playback is False


def test_finalize_contraction_playback_true_when_playback_requested_without_scheme() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(playback=True),
        engine="quimb",
        scheme_tensor_names=(("A", "B"),),
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_playback is True


def test_finalize_contraction_cost_hover_auto_enables_scheme() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(hover_cost=True),
        engine="tensornetwork",
        scheme_tensor_names=None,
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_scheme_cost_hover is True


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
        config=finalize_demo_plot_config(
            _demo_args(), engine="tensornetwork", scheme_tensor_names=None
        ),
        show_tensor_labels=True,
        show_index_labels=False,
    )

    assert result == ("figure", "axes")
    assert calls[0]["interactive_controls"] is False
    assert calls[0]["show"] is False
    assert calls[0]["show_tensor_labels"] is True
    assert calls[0]["show_index_labels"] is False


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
        config=finalize_demo_plot_config(
            _demo_args(), engine="tensornetwork", scheme_tensor_names=None
        ),
        show_tensor_labels=False,
        show_index_labels=True,
    )

    assert result == ("figure", "axes")
    assert calls[0]["interactive_controls"] is True
    assert calls[0]["show"] is False
    assert calls[0]["show_tensor_labels"] is False
    assert calls[0]["show_index_labels"] is True


def test_run_demo_parser_defaults_match_cli_contract() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_parser_defaults")

    args = module.parse_args(["quimb", "mps"])

    assert args.engine == "quimb"
    assert args.example == "mps"
    assert args.view == "2d"
    assert args.labels_nodes is True
    assert args.labels_edges is False
    assert args.labels is None
    assert args.hover_labels is True
    assert args.scheme is False
    assert args.playback is False
    assert args.hover_cost is False
    assert args.from_scratch is False
    assert args.from_list is False
    assert args.save is None
    assert args.no_show is False
    assert args.n_sites == 6
    assert args.lx == 3
    assert args.ly == 4
    assert args.lz == 3
    assert args.mera_log2 == 3
    assert args.tree_depth == 4
