from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


def _require_quimb() -> None:
    pytest.importorskip("quimb.tensor")


def _require_tenpy() -> None:
    pytest.importorskip("tenpy")


def _require_tensornetwork() -> None:
    pytest.importorskip("tensornetwork")


def _require_torch() -> None:
    pytest.importorskip("torch")


def _load_example_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_demo_registry_declares_expected_example_sets() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_registry")

    assert set(module.available_examples("tensorkrowch")) == {
        "cubic_peps",
        "disconnected",
        "ladder",
        "mera",
        "mera_ttn",
        "mps",
        "mpo",
        "peps",
        "weird",
    }
    assert set(module.available_examples("tensornetwork")) == {
        "cubic_peps",
        "disconnected",
        "ladder",
        "mera",
        "mera_ttn",
        "mps",
        "mpo",
        "peps",
        "weird",
    }
    assert set(module.available_examples("quimb")) == {
        "cubic_peps",
        "disconnected",
        "hyper",
        "ladder",
        "mera",
        "mera_ttn",
        "mps",
        "mpo",
        "peps",
        "weird",
    }
    assert set(module.available_examples("tenpy")) == {
        "chain",
        "excitation",
        "hub",
        "hyper",
        "impo",
        "imps",
        "mps",
        "mpo",
        "purification",
        "uniform",
    }
    assert set(module.available_examples("einsum")) == {
        "batch",
        "disconnected",
        "ellipsis",
        "implicit_out",
        "mps",
        "mpo",
        "nway",
        "peps",
        "ternary",
        "trace",
        "unary",
    }


def test_run_demo_alias_tt_resolves_to_mps() -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_alias")

    resolved = module.resolve_requested_example(engine="quimb", example="tt")

    assert resolved == "mps"


def test_run_demo_unknown_engine_lists_valid_engines(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_bad_engine")

    with pytest.raises(SystemExit, match="2"):
        module.main(["bad-engine", "mps"])

    captured = capsys.readouterr()
    assert "Unknown engine 'bad-engine'" in captured.err
    assert "einsum, quimb, tenpy, tensorkrowch, tensornetwork" in captured.err


def test_run_demo_unknown_example_lists_valid_examples_for_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_bad_example")

    with pytest.raises(SystemExit, match="2"):
        module.main(["quimb", "bad-example"])

    captured = capsys.readouterr()
    assert "Unknown example 'bad-example' for engine 'quimb'" in captured.err
    assert "hyper" in captured.err
    assert "mera_ttn" in captured.err


def test_run_demo_rejects_unsupported_from_list(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_from_list_error")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tenpy", "imps", "--from-list"])

    captured = capsys.readouterr()
    assert "does not support --from-list" in captured.err


def test_run_demo_rejects_unsupported_from_scratch(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_from_scratch_error")

    with pytest.raises(SystemExit, match="2"):
        module.main(["tenpy", "uniform", "--from-scratch"])

    captured = capsys.readouterr()
    assert "does not support --from-scratch" in captured.err


def test_run_demo_auto_save_path_is_used_when_save_has_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_auto_save")
    save_paths: list[Path] = []

    def _fake_run_example(args) -> tuple[str, Path]:
        assert args.engine == "einsum"
        assert args.example == "batch"
        assert args.save is not None
        save_paths.append(args.save)
        return ("figure", args.save)

    monkeypatch.setattr(module, "dispatch_run", _fake_run_example)

    exit_code = module.main(["einsum", "batch", "--save", "--no-show"])

    assert exit_code == 0
    assert save_paths == [Path(".tmp") / "examples" / "einsum" / "batch.png"]


def test_quimb_hyper_saves_figure_without_showing() -> None:
    _require_quimb()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_quimb_hyper")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quimb-hyper-demo.png"

    exit_code = module.main(
        ["quimb", "hyper", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_tenpy_imps_saves_figure_without_showing() -> None:
    _require_tenpy()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tenpy_imps")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-imps-demo.png"

    exit_code = module.main(
        ["tenpy", "imps", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_tenpy_chain_saves_figure_without_showing() -> None:
    _require_tenpy()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tenpy_chain")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-chain-demo.png"

    exit_code = module.main(
        ["tenpy", "chain", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_einsum_ellipsis_saves_figure_without_showing() -> None:
    _require_torch()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_einsum_ellipsis")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "einsum-ellipsis-demo.png"

    exit_code = module.main(
        ["einsum", "ellipsis", "--view", "2d", "--save", str(output_path), "--no-show"]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_tensornetwork_mera_ttn_saves_figure_without_showing() -> None:
    _require_tensornetwork()
    module = _load_example_module(Path("examples/run_demo.py"), "run_demo_tn_mera_ttn")
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tensornetwork-mera-ttn-demo.png"

    exit_code = module.main(
        [
            "tensornetwork",
            "mera_ttn",
            "--view",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_run_all_examples_default_2d_matches_new_matrix() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_default")

    commands = module.select_example_commands(group="default", views="2d")
    argvs = {command.argv for command in commands}

    assert ("examples/run_demo.py", "tensorkrowch", "disconnected", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "quimb", "hyper", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "tenpy", "chain", "--view", "2d") in argvs
    assert ("examples/run_demo.py", "einsum", "ellipsis", "--view", "2d") in argvs


def test_run_all_examples_hover_group_appends_hover_flag() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_hover")

    commands = module.select_example_commands(group="hover", views="3d")

    assert commands
    assert all("--hover-labels" in command.argv for command in commands)


def test_run_all_examples_builds_headless_subprocess_command(tmp_path: Path) -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_build")

    command = module.ExampleCommand(
        slug="quimb_hyper_2d",
        argv=("examples/run_demo.py", "quimb", "hyper", "--view", "2d"),
    )
    subprocess_command = module.build_subprocess_command(
        command,
        output_dir=tmp_path,
        python_executable="python-test",
    )

    assert subprocess_command[:6] == [
        "python-test",
        "examples/run_demo.py",
        "quimb",
        "hyper",
        "--view",
        "2d",
    ]
    assert "--no-show" in subprocess_command
    assert "--save" in subprocess_command
    assert str(tmp_path / "quimb_hyper_2d.png") in subprocess_command


def test_run_all_examples_list_mode_prints_without_running_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_list")

    def _fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called in --list mode")

    monkeypatch.setattr(module.subprocess, "run", _fail_run)

    exit_code = module.main(["--group", "contraction", "--views", "2d", "--list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "examples/run_demo.py tensornetwork mera_ttn --view 2d --scheme" in captured.out


def test_run_all_examples_all_group_contains_more_commands_than_default() -> None:
    module = _load_example_module(Path("examples/run_all_examples.py"), "run_all_examples_all")

    default_commands = module.select_example_commands(group="default", views="both")
    all_commands = module.select_example_commands(group="all", views="both")

    assert len(all_commands) > len(default_commands)
