from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _parse_inline_list(raw_value: str) -> list[str]:
    value = raw_value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Expected inline YAML list, got: {raw_value!r}")
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",")]


def _load_workflow() -> dict[str, Any]:
    workflow: dict[str, Any] = {"jobs": {}}
    current_job_name: str | None = None
    current_step: dict[str, Any] | None = None
    in_run_block = False

    for raw_line in Path(".github/workflows/ci.yml").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 2 and stripped.endswith(":") and not stripped.startswith("- "):
            key = stripped[:-1]
            if key in {"smoke-minimal", "wheel-smoke", "lint-and-test"}:
                current_job_name = key
                workflow["jobs"][key] = {"steps": [], "matrix": {}}
                current_step = None
                in_run_block = False
            continue

        if current_job_name == "wheel-smoke" and indent == 8 and stripped.startswith("os: "):
            workflow["jobs"][current_job_name]["matrix"]["os"] = _parse_inline_list(
                stripped.split(": ", 1)[1]
            )
            continue

        if (
            current_job_name == "wheel-smoke"
            and indent == 8
            and stripped.startswith("python-version: ")
        ):
            workflow["jobs"][current_job_name]["matrix"]["python-version"] = _parse_inline_list(
                stripped.split(": ", 1)[1]
            )
            continue

        if current_job_name is None:
            continue

        if indent == 6 and stripped.startswith("- name: "):
            current_step = {"name": stripped[len("- name: ") :]}
            workflow["jobs"][current_job_name]["steps"].append(current_step)
            in_run_block = False
            continue

        if current_step is None:
            continue

        if indent == 8 and stripped == "run: |":
            current_step["run"] = ""
            in_run_block = True
            continue

        if indent == 8 and stripped.startswith("run: "):
            current_step["run"] = stripped[len("run: ") :]
            in_run_block = False
            continue

        if in_run_block and indent >= 10:
            current_step["run"] = (
                f"{current_step['run']}\n{raw_line[10:]}" if current_step["run"] else raw_line[10:]
            )

    return workflow


def _job_step(workflow: dict[str, Any], job_name: str, step_name: str) -> dict[str, Any]:
    steps: Iterable[dict[str, Any]] = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step["name"] == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_ci_workflow_builds_and_smoke_tests_wheels() -> None:
    workflow = _load_workflow()

    assert _job_step(workflow, "wheel-smoke", "Install build tool")["run"] == (
        "python -m pip install build==1.4.2"
    )
    assert _job_step(workflow, "wheel-smoke", "Build package artifacts")["run"] == (
        "python scripts/verify.py package"
    )
    assert _job_step(workflow, "wheel-smoke", "Create wheel test venv")["run"] == (
        "python -m venv .wheel-venv"
    )
    assert _job_step(workflow, "wheel-smoke", "Smoke import and render (Windows)")[
        "run"
    ].startswith(".wheel-venv\\Scripts\\python -c \"import matplotlib; matplotlib.use('Agg'); ")
    assert (
        "show_tensor_network("
        in _job_step(workflow, "wheel-smoke", "Smoke import and render (Windows)")["run"]
    )
    assert _job_step(workflow, "wheel-smoke", "Smoke sdist import and render (Windows)")[
        "run"
    ].startswith(".sdist-venv\\Scripts\\python -c \"import matplotlib; matplotlib.use('Agg'); ")
    assert (
        "show_tensor_network("
        in _job_step(workflow, "wheel-smoke", "Smoke sdist import and render (Windows)")["run"]
    )
    assert workflow["jobs"]["wheel-smoke"]["matrix"]["python-version"] == ["3.12", "3.13"]


def test_ci_workflow_uses_pinned_requirements_and_verify_runner() -> None:
    workflow = _load_workflow()

    assert _job_step(workflow, "smoke-minimal", "Install minimal pinned dependencies")["run"] == (
        "python -m pip install -e . --no-deps\n"
        "python -m pip install matplotlib==3.10.8 networkx==3.6.1 numpy==2.4.3 pytest==9.0.2 "
        "quimb==1.13.0"
    )
    assert _job_step(workflow, "smoke-minimal", "Run headless example")["run"] == (
        "python scripts/verify.py smoke"
    )
    assert _job_step(workflow, "smoke-minimal", "Pytest")["run"] == "python scripts/verify.py tests"
    assert _job_step(workflow, "lint-and-test", "Install PyTorch (CPU)")["run"] == (
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    assert _job_step(workflow, "lint-and-test", "Install pinned dev requirements")["run"] == (
        "python -m pip install -r requirements.dev.txt"
    )
    assert _job_step(workflow, "lint-and-test", "Quality")["run"] == (
        "python scripts/verify.py quality"
    )
    assert _job_step(workflow, "lint-and-test", "Smoke example")["run"] == (
        "python scripts/verify.py smoke"
    )
    assert _job_step(workflow, "lint-and-test", "Pytest")["run"] == "python scripts/verify.py tests"


def test_ci_workflow_covers_python_3_13_for_packaging() -> None:
    workflow = _load_workflow()

    assert workflow["jobs"]["wheel-smoke"]["matrix"]["python-version"] == ["3.12", "3.13"]


def test_ci_workflow_locks_windows_package_smoke_commands() -> None:
    workflow = _load_workflow()

    assert _job_step(workflow, "wheel-smoke", "Install wheel (Windows)")["run"].startswith(
        ".wheel-venv\\Scripts\\python -m pip install "
    )
    assert _job_step(workflow, "wheel-smoke", "Install sdist (Windows)")["run"].startswith(
        ".sdist-venv\\Scripts\\python -m pip install "
    )
    assert "Create wheel test venv" in [
        step["name"] for step in workflow["jobs"]["wheel-smoke"]["steps"]
    ]
    assert "Create sdist test venv" in [
        step["name"] for step in workflow["jobs"]["wheel-smoke"]["steps"]
    ]
    assert "Smoke import and render (Windows)" in [
        step["name"] for step in workflow["jobs"]["wheel-smoke"]["steps"]
    ]
    assert "Smoke sdist import and render (Windows)" in [
        step["name"] for step in workflow["jobs"]["wheel-smoke"]["steps"]
    ]
