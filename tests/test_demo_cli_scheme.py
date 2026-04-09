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
    auto_save_path,
    cubic_peps_tensor_names,
    cumulative_prefix_contraction_scheme,
    demo_runs_headless,
    finalize_demo_plot_config,
    pairwise_merge_contraction_scheme,
    pairwise_merge_group_contraction_scheme,
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


def test_pairwise_merge_contraction_scheme_merges_branches_before_final_join() -> None:
    steps = pairwise_merge_contraction_scheme(("A", "B", "C", "D", "E", "F"))
    assert steps == (
        ("A", "B"),
        ("C", "D"),
        ("E", "F"),
        ("A", "B", "C", "D"),
        ("A", "B", "C", "D", "E", "F"),
    )


def test_pairwise_merge_group_contraction_scheme_keeps_odd_tail_until_it_merges() -> None:
    steps = pairwise_merge_group_contraction_scheme((("A", "x"), ("B", "y"), ("C", "z")))
    assert steps == (
        ("A", "x", "B", "y"),
        ("A", "x", "B", "y", "C", "z"),
    )


def test_cubic_peps_names_match_grid_count() -> None:
    names = cubic_peps_tensor_names(2, 3, 2)
    assert len(names) == 12
    assert set(names) == {f"P{i}_{j}_{k}" for i in range(2) for j in range(3) for k in range(2)}


def test_cubic_peps_invalid_grid() -> None:
    with pytest.raises(ValueError, match="must be >="):
        cubic_peps_tensor_names(0, 1, 1)


def test_auto_save_path_uses_engine_and_example() -> None:
    path = auto_save_path(engine="tenpy", example="imps")

    assert path == Path(".tmp") / "examples" / "tenpy" / "imps.png"


def _demo_args(
    *,
    scheme: bool = False,
    hover_cost: bool = False,
    tensor_inspector: bool = False,
    contracted: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        labels_nodes=True,
        labels_edges=False,
        labels=None,
        hover_labels=True,
        scheme=scheme,
        hover_cost=hover_cost,
        tensor_inspector=tensor_inspector,
        contracted=contracted,
    )


def _render_args(*, no_show: bool = False, save: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        no_show=no_show,
        save=save,
    )


def test_finalize_contraction_scheme_enables_slider_behavior_directly() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(scheme=True),
        engine="einsum",
        scheme_tensor_names=None,
    )
    assert cfg.show_contraction_scheme is True
    assert not hasattr(cfg, "contraction_playback")


def test_finalize_contraction_cost_hover_auto_enables_scheme() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(hover_cost=True),
        engine="tensornetwork",
        scheme_tensor_names=None,
    )
    assert cfg.show_contraction_scheme is True
    assert cfg.contraction_scheme_cost_hover is True


def test_finalize_contraction_tensor_inspector_keeps_scheme_optional() -> None:
    cfg = finalize_demo_plot_config(
        _demo_args(tensor_inspector=True),
        engine="einsum",
        scheme_tensor_names=None,
    )
    assert cfg.show_contraction_scheme is False
    assert cfg.contraction_tensor_inspector is True


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
    )

    assert result == ("figure", "axes")
    assert calls[0]["show_controls"] is False
    assert calls[0]["show"] is False
    assert "show_tensor_labels" not in calls[0]
    assert "show_index_labels" not in calls[0]


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
    )

    assert result == ("figure", "axes")
    assert calls[0]["show_controls"] is True
    assert calls[0]["show"] is False
    assert "show_tensor_labels" not in calls[0]
    assert "show_index_labels" not in calls[0]


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
    assert not hasattr(args, "playback")
    assert args.hover_cost is False
    assert args.tensor_inspector is False
    assert args.contracted is False
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


def test_run_demo_defaults_to_contracted_for_small_tensorkrowch_demo() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_parser_default_tk")

    args = module.parse_args(["tensorkrowch", "mps"])

    assert args.engine == "tensorkrowch"
    assert args.example == "mps"
    assert args.contracted is True


def test_run_demo_allows_disabling_default_contracted_mode() -> None:
    module = _load_example_module(
        Path("examples/run_demo.py"),
        "run_demo_parser_default_tk_disabled",
    )

    args = module.parse_args(["tensorkrowch", "mps", "--no-contracted"])

    assert args.engine == "tensorkrowch"
    assert args.example == "mps"
    assert args.contracted is False


def test_run_demo_rejects_contracted_for_non_tensorkrowch_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_contracted_non_tk")

    with pytest.raises(SystemExit, match="2"):
        module.main(["quimb", "mps", "--contracted"])

    captured = capsys.readouterr()
    assert "only supports --contracted for engine 'tensorkrowch'" in captured.err


def test_run_demo_rejects_contracted_for_large_tensorkrowch_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_contracted_large_tk")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tensorkrowch", "mps", "--contracted", "--n-sites", "7"])

    captured = capsys.readouterr()
    assert "--contracted is limited to small TensorKrowch demos" in captured.err


def test_run_demo_accepts_contracted_for_six_site_tensorkrowch_demo() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_contracted_six_tk")

    args = module.parse_args(["tensorkrowch", "mps", "--contracted", "--n-sites", "6"])

    assert args.engine == "tensorkrowch"
    assert args.example == "mps"
    assert args.contracted is True
    assert args.n_sites == 6


def test_run_demo_rejects_contracted_with_from_list(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_contracted_from_list")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tensorkrowch", "mps", "--contracted", "--from-list"])

    captured = capsys.readouterr()
    assert "--contracted requires the native TensorKrowch network object" in captured.err
