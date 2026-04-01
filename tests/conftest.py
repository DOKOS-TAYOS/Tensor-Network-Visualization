from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if _WORKSPACE_ROOT.parent.name == ".worktrees":
    _REPO_ROOT = _WORKSPACE_ROOT.parents[1]
    _PYTEST_TMP_NAME = f"pytest-temp-{_WORKSPACE_ROOT.name}"
else:
    _REPO_ROOT = _WORKSPACE_ROOT
    _PYTEST_TMP_NAME = "pytest-temp"

_REPO_LOCAL_PYTEST_TMP = _REPO_ROOT / ".tmp" / _PYTEST_TMP_NAME
_REPO_LOCAL_PYTEST_TMP.mkdir(parents=True, exist_ok=True)
for env_name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[env_name] = str(_REPO_LOCAL_PYTEST_TMP)
tempfile.tempdir = str(_REPO_LOCAL_PYTEST_TMP)


@pytest.fixture
def tmp_path() -> Generator[Path, None, None]:
    path = _REPO_LOCAL_PYTEST_TMP / f"pytest-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures_after_test() -> Generator[None, None, None]:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
