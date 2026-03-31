from __future__ import annotations

import ast
from pathlib import Path


def _top_level_importorskip_targets(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []

    for node in module.body:
        if isinstance(node, ast.Expr | ast.Assign):
            call = node.value
        else:
            continue

        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "pytest":
            continue
        if call.func.attr != "importorskip":
            continue
        if not call.args:
            continue

        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            targets.append(first_arg.value)

    return targets


def test_test_examples_scopes_optional_backend_skips_per_backend() -> None:
    path = Path("tests/test_examples.py")

    assert _top_level_importorskip_targets(path) == []
