from __future__ import annotations

import ast
from pathlib import Path


def _python_source_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_runtime_package_does_not_use_optimized_away_asserts() -> None:
    assert_locations: list[str] = []
    for path in _python_source_files(Path("src/tensor_network_viz")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if isinstance(node, ast.Assert):
                assert_locations.append(f"{path}:{node.lineno}")

    assert assert_locations == []


def test_dependabot_updates_python_and_github_actions_manifests() -> None:
    config_path = Path(".github/dependabot.yml")

    text = config_path.read_text(encoding="utf-8")

    assert 'package-ecosystem: "pip"' in text
    assert 'package-ecosystem: "github-actions"' in text
    assert 'directory: "/"' in text
    assert 'interval: "weekly"' in text
    assert 'timezone: "Europe/Madrid"' in text
