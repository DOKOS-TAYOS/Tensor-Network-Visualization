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


def test_engine_demo_modules_do_not_use_shared_graph_blueprints() -> None:
    engine_demo_paths = (
        Path("examples/tensorkrowch_demo.py"),
        Path("examples/tensornetwork_demo.py"),
        Path("examples/quimb_demo.py"),
        Path("examples/tenpy_demo.py"),
    )

    for path in engine_demo_paths:
        source = path.read_text(encoding="utf-8")
        assert "GraphBlueprint" not in source
        assert "_blueprint" not in source
