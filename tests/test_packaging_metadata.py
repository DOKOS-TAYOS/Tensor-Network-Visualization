from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_requires_python_is_3_11_or_newer() -> None:
    pyproject = _load_pyproject()

    assert pyproject["project"]["requires-python"] == ">=3.11"


def test_base_dependencies_are_declared_explicitly() -> None:
    pyproject = _load_pyproject()

    dependencies = pyproject["project"]["dependencies"]

    assert dependencies == ["matplotlib>=3.7", "networkx>=3.0", "numpy>=2.0"]


def test_project_uses_spdx_license_metadata() -> None:
    pyproject = _load_pyproject()

    project = pyproject["project"]

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["urls"] == {
        "Homepage": "https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization",
        "Documentation": "https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/blob/main/docs/guide.md",
        "Changelog": "https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/blob/main/CHANGELOG.md",
        "Repository": "https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization",
        "Issues": "https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues",
    }
    assert not any(classifier.startswith("License ::") for classifier in project["classifiers"])
    assert {
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    }.issubset(set(project["classifiers"]))


def test_setuptools_metadata_points_at_src_and_pytyped_marker() -> None:
    pyproject = _load_pyproject()

    setuptools_tool = pyproject["tool"]["setuptools"]

    assert setuptools_tool["packages"]["find"]["where"] == ["src"]
    assert setuptools_tool["package-data"] == {"tensor_network_viz": ["py.typed"]}
