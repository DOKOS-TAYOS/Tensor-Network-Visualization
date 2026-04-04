from __future__ import annotations

from pathlib import Path


def test_ci_workflow_builds_and_smoke_tests_wheels() -> None:
    content = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "wheel-smoke:" in content
    assert "python -m pip install build" in content
    assert "scripts/verify.py package" in content
    assert "python -m venv .wheel-venv" in content
    assert "dist/*.whl" in content or "dist\\*.whl" in content
    assert "dist/*.tar.gz" in content or "dist\\*.tar.gz" in content
    assert "import tensor_network_viz" in content
    assert "show_tensor_network(" in content


def test_ci_workflow_uses_pinned_requirements_and_verify_runner() -> None:
    content = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "requirements.dev.txt" in content
    assert 'pip install -e ".[dev]"' not in content
    assert "scripts/verify.py quality" in content
    assert "scripts/verify.py tests" in content
    assert "scripts/verify.py smoke" in content
    assert "scripts/verify.py package" in content


def test_ci_workflow_covers_python_3_13_for_packaging() -> None:
    content = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.12", "3.13"]' in content
