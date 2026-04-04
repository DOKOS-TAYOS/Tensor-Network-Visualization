from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_verify_script_exists_and_lists_supported_modes() -> None:
    script = Path("scripts/verify.py")

    assert script.is_file()

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "quality" in result.stdout
    assert "tests" in result.stdout
    assert "smoke" in result.stdout
    assert "package" in result.stdout
