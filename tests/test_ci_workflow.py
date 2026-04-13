from __future__ import annotations

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


def _parse_block_list_item(raw_value: str) -> str:
    value = raw_value.strip()
    if not value.startswith("- "):
        raise ValueError(f"Expected YAML list item, got: {raw_value!r}")
    return value[2:].strip().strip('"').strip("'")


def _load_workflow_text(text: str) -> dict[str, Any]:
    workflow: dict[str, Any] = {"jobs": {}}
    current_job_name: str | None = None
    current_step: dict[str, Any] | None = None
    current_matrix_key: str | None = None
    in_run_block = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if current_matrix_key is not None and indent <= 8 and not stripped.startswith("- "):
            current_matrix_key = None

        if indent == 2 and stripped.endswith(":") and not stripped.startswith("- "):
            key = stripped[:-1]
            if key in {"smoke-minimal", "wheel-smoke", "lint-and-test"}:
                current_job_name = key
                workflow["jobs"][key] = {"steps": [], "matrix": {}}
                current_step = None
                in_run_block = False
            continue

        if current_job_name is None:
            continue

        if indent == 8 and stripped.startswith("os: "):
            workflow["jobs"][current_job_name]["matrix"]["os"] = _parse_inline_list(
                stripped.split(": ", 1)[1]
            )
            continue

        if indent == 8 and stripped == "os:":
            workflow["jobs"][current_job_name]["matrix"]["os"] = []
            current_matrix_key = "os"
            continue

        if indent == 8 and stripped.startswith("python-version: "):
            workflow["jobs"][current_job_name]["matrix"]["python-version"] = _parse_inline_list(
                stripped.split(": ", 1)[1]
            )
            continue

        if indent == 8 and stripped == "python-version:":
            workflow["jobs"][current_job_name]["matrix"]["python-version"] = []
            current_matrix_key = "python-version"
            continue

        if current_matrix_key is not None and indent >= 10 and stripped.startswith("- "):
            workflow["jobs"][current_job_name]["matrix"][current_matrix_key].append(
                _parse_block_list_item(stripped)
            )
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


def _load_workflow() -> dict[str, Any]:
    return _load_workflow_text(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _job_step_names(workflow: dict[str, Any], job_name: str) -> list[str]:
    return [step["name"] for step in workflow["jobs"][job_name]["steps"]]


def _job_step(workflow: dict[str, Any], job_name: str, step_name: str) -> dict[str, str]:
    for step in workflow["jobs"][job_name]["steps"]:
        if step["name"] == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_ci_workflow_parser_accepts_block_style_matrix_lists() -> None:
    workflow = _load_workflow_text(
        """
jobs:
  lint-and-test:
    strategy:
      matrix:
        os:
          - ubuntu-latest
          - windows-latest
        python-version:
          - "3.11"
          - "3.12"
          - "3.13"
    steps:
      - name: Quality
        run: python scripts/verify.py quality
"""
    )

    assert workflow["jobs"]["lint-and-test"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert workflow["jobs"]["lint-and-test"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]
    assert (
        _job_step(workflow, "lint-and-test", "Quality")["run"] == "python scripts/verify.py quality"
    )


def test_ci_workflow_declares_expected_jobs_and_runner_matrices() -> None:
    workflow = _load_workflow()

    assert set(workflow["jobs"]) == {"smoke-minimal", "wheel-smoke", "lint-and-test"}
    assert workflow["jobs"]["wheel-smoke"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert workflow["jobs"]["wheel-smoke"]["matrix"]["python-version"] == ["3.12", "3.13"]
    assert workflow["jobs"]["lint-and-test"]["matrix"]["os"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert workflow["jobs"]["lint-and-test"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]


def test_ci_workflow_routes_core_checks_through_verify_script() -> None:
    workflow = _load_workflow()

    assert _job_step(workflow, "smoke-minimal", "Run headless example")["run"] == (
        "python scripts/verify.py smoke"
    )
    assert _job_step(workflow, "smoke-minimal", "Pytest")["run"] == "python scripts/verify.py tests"
    assert _job_step(workflow, "wheel-smoke", "Build package artifacts")["run"] == (
        "python scripts/verify.py package"
    )
    assert _job_step(workflow, "lint-and-test", "Quality")["run"] == (
        "python scripts/verify.py quality"
    )
    assert _job_step(workflow, "lint-and-test", "Smoke example")["run"] == (
        "python scripts/verify.py smoke"
    )
    assert _job_step(workflow, "lint-and-test", "Pytest")["run"] == (
        "python scripts/verify.py tests"
    )


def test_ci_workflow_keeps_minimal_and_packaging_smoke_guards() -> None:
    workflow = _load_workflow()
    minimal_install = _job_step(workflow, "smoke-minimal", "Install minimal pinned dependencies")[
        "run"
    ]

    assert "python -m pip install -e . --no-deps" in minimal_install
    for requirement in ("matplotlib==3.10.8", "networkx==3.6.1", "numpy==2.4.3", "quimb==1.13.0"):
        assert requirement in minimal_install

    wheel_step_names = _job_step_names(workflow, "wheel-smoke")
    for step_name in (
        "Install build tool",
        "Create wheel test venv",
        "Create sdist test venv",
        "Install wheel (Windows)",
        "Smoke import and render (Windows)",
        "Install sdist (Windows)",
        "Smoke sdist import and render (Windows)",
    ):
        assert step_name in wheel_step_names

    wheel_windows_smoke = _job_step(workflow, "wheel-smoke", "Smoke import and render (Windows)")[
        "run"
    ]
    assert "python -c " in wheel_windows_smoke
    assert "matplotlib.use('Agg')" in wheel_windows_smoke
    assert "show_tensor_network(" in wheel_windows_smoke
    assert "engine='einsum'" in wheel_windows_smoke

    sdist_windows_smoke = _job_step(
        workflow,
        "wheel-smoke",
        "Smoke sdist import and render (Windows)",
    )["run"]
    assert "python -c " in sdist_windows_smoke
    assert "matplotlib.use('Agg')" in sdist_windows_smoke
    assert "show_tensor_network(" in sdist_windows_smoke
    assert "engine='einsum'" in sdist_windows_smoke
