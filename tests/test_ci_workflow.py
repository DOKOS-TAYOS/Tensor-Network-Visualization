from __future__ import annotations

from pathlib import Path


def test_ci_workflow_builds_and_smoke_tests_wheels() -> None:
    content = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "wheel-smoke:" in content
    assert "python -m pip install build" in content
    assert "python -m build --wheel" in content
    assert "python -m venv .wheel-venv" in content
    assert "dist/*.whl" in content or "dist\\*.whl" in content
    assert "import tensor_network_viz" in content
    assert "show_tensor_network(" in content
