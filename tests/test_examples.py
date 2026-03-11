from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("quimb")
pytest.importorskip("tenpy")
torch = pytest.importorskip("torch")


def _load_example_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quimb_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quimb-demo.png"
    module = _load_example_module(
        Path("examples/quimb_demo.py"),
        "quimb_demo_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quimb_demo.py",
            "mps",
            "2d",
            "--from-list",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_quimb_demo_hyper_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quimb-hyper-demo.png"
    module = _load_example_module(
        Path("examples/quimb_demo.py"),
        "quimb_demo_hyper_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quimb_demo.py",
            "hyper",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_tenpy_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-demo.png"
    module = _load_example_module(
        Path("examples/tenpy_demo.py"),
        "tenpy_demo_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tenpy_demo.py",
            "mps",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_tenpy_infinite_mps_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-imps-demo.png"
    module = _load_example_module(
        Path("examples/tenpy_demo.py"),
        "tenpy_demo_imps_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tenpy_demo.py",
            "imps",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_tenpy_infinite_mpo_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-impo-demo.png"
    module = _load_example_module(
        Path("examples/tenpy_demo.py"),
        "tenpy_demo_impo_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tenpy_demo.py",
            "impo",
            "3d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_einsum_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "einsum-demo.png"
    module = _load_example_module(
        Path("examples/einsum_demo.py"),
        "einsum_demo_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "einsum_demo.py",
            "mps",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_einsum_peps_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "einsum-peps-demo.png"
    module = _load_example_module(
        Path("examples/einsum_demo.py"),
        "einsum_demo_peps_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "einsum_demo.py",
            "peps",
            "3d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()
