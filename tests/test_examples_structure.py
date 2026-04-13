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


def test_tenpy_docs_call_out_momentum_compatibility_and_example_fallback() -> None:
    backends_text = Path("docs/backends.md").read_text(encoding="utf-8")
    troubleshooting_text = Path("docs/troubleshooting.md").read_text(encoding="utf-8")
    examples_text = Path("examples/README.md").read_text(encoding="utf-8")

    assert "MomentumMPS" in backends_text
    assert "NumPy" in backends_text
    assert "excitation" in backends_text
    assert "MomentumMPS" in troubleshooting_text
    assert "NumPy" in troubleshooting_text
    assert "excitation" in examples_text
    assert "NumPy" in examples_text
