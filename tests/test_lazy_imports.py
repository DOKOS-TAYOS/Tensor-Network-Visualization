from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_top_level_import_keeps_heavy_modules_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    code = """
import json
import sys

import tensor_network_viz

heavy = {
    "tensor_network_viz.contraction_viewer": "contraction_viewer",
    "tensor_network_viz.einsum_module.trace": "einsum_trace",
    "tensor_network_viz.tenpy.explicit": "tenpy_explicit",
    "tensor_network_viz.viewer": "viewer",
}
loaded = {label: (module_name in sys.modules) for module_name, label in heavy.items()}
print(json.dumps(loaded, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == (
        '{"contraction_viewer": false, "einsum_trace": false, '
        '"tenpy_explicit": false, "viewer": false}'
    )
