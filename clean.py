"""Cross-platform script to remove project caches and build artifacts.

Excludes the .venv directory. Run with: python clean.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _in_venv(path: Path, venv: Path) -> bool:
    """Return True if path is inside the venv directory."""
    try:
        path.resolve().relative_to(venv.resolve())
        return True
    except ValueError:
        return False


def _should_skip(path: Path, venv: Path) -> bool:
    return venv.exists() and _in_venv(path, venv)


def main() -> int:
    root = Path(__file__).resolve().parent
    venv = root / ".venv"

    dir_names = frozenset({
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp",
        ".pip_tmp",
        "build",
        "dist",
    })
    file_suffixes = (".pyc", ".pyo", ".pyd")

    print("Cleaning project caches and temporary files...")
    print("Keeping .venv untouched.")

    dirs_to_remove: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if _should_skip(path, venv):
            continue
        if path.name in dir_names:
            dirs_to_remove.append(path)
        elif path.name.endswith(".egg-info") or path.name.startswith("pytest-cache-files-"):
            dirs_to_remove.append(path)

    # Remove deepest directories first
    dirs_to_remove.sort(key=lambda p: len(p.parts), reverse=True)
    for path in dirs_to_remove:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed directory: {path}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in file_suffixes and not _should_skip(path, venv):
            path.unlink(missing_ok=True)
            print(f"Removed file: {path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
