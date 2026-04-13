from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_verify_module() -> ModuleType:
    script_path = Path("scripts/verify.py").resolve()
    spec = importlib.util.spec_from_file_location("verify_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load verify script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_script_exposes_perf_mode_in_parser() -> None:
    module = _load_verify_module()
    parser = module._build_parser()
    mode_action = next(action for action in parser._actions if action.dest == "mode")

    assert mode_action.default == "all"
    assert set(mode_action.choices) == {"all", "quality", "tests", "perf", "smoke", "package"}


def test_verify_script_perf_mode_runs_explicit_perf_pytest_command() -> None:
    module = _load_verify_module()

    (step,) = module._ordered_steps("perf")

    assert step.label == "pytest-perf"
    assert step.command[:4] == (
        sys.executable,
        "-m",
        "pytest",
        "-q",
    )
    assert "-p" in step.command
    assert "no:cacheprovider" in step.command
    assert "--override-ini=addopts=" in step.command
    assert step.command[-2:] == ("-m", "perf")


def test_verify_script_all_mode_keeps_perf_out_of_default_route() -> None:
    module = _load_verify_module()

    labels = [step.label for step in module._ordered_steps("all")]

    assert labels == [
        "ruff-check",
        "ruff-format",
        "pyright",
        "pytest",
        "quimb-smoke",
        "build-dist",
    ]
    assert "pytest-perf" not in labels
