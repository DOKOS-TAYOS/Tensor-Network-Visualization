from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_base_dependencies_include_numpy() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.split(">=", 1)[0].split("==", 1)[0].strip() == "numpy" for dep in dependencies)
