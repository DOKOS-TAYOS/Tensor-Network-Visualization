from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tensor_network_viz import EinsumTrace, pair_tensor

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


def test_total_tests_bat_uses_root_venv_and_covers_example_matrix() -> None:
    script_path = Path("examples/total_tests.bat")
    expected_commands = [
        r"call :run examples\tensorkrowch_demo.py mps 2d",
        r"call :run examples\tensorkrowch_demo.py mps 3d",
        r"call :run examples\tensorkrowch_demo.py mpo 2d",
        r"call :run examples\tensorkrowch_demo.py mpo 3d",
        r"call :run examples\tensorkrowch_demo.py peps 2d",
        r"call :run examples\tensorkrowch_demo.py peps 3d",
        r"call :run examples\tensorkrowch_demo.py weird 2d",
        r"call :run examples\tensorkrowch_demo.py weird 3d",
        r"call :run examples\tensorkrowch_demo.py disconnected 2d",
        r"call :run examples\tensorkrowch_demo.py disconnected 3d",
        r"call :run examples\tensorkrowch_demo.py mps 2d --from-list",
        r"call :run examples\tensornetwork_demo.py mps 2d",
        r"call :run examples\tensornetwork_demo.py mps 3d",
        r"call :run examples\tensornetwork_demo.py mpo 2d",
        r"call :run examples\tensornetwork_demo.py mpo 3d",
        r"call :run examples\tensornetwork_demo.py peps 2d",
        r"call :run examples\tensornetwork_demo.py peps 3d",
        r"call :run examples\tensornetwork_demo.py weird 2d",
        r"call :run examples\tensornetwork_demo.py weird 3d",
        r"call :run examples\tensornetwork_demo.py disconnected 2d",
        r"call :run examples\tensornetwork_demo.py disconnected 3d",
        r"call :run examples\quimb_demo.py mps 2d",
        r"call :run examples\quimb_demo.py mps 3d",
        r"call :run examples\quimb_demo.py hyper 2d",
        r"call :run examples\quimb_demo.py hyper 3d",
        r"call :run examples\quimb_demo.py mpo 2d",
        r"call :run examples\quimb_demo.py mpo 3d",
        r"call :run examples\quimb_demo.py peps 2d",
        r"call :run examples\quimb_demo.py peps 3d",
        r"call :run examples\quimb_demo.py weird 2d",
        r"call :run examples\quimb_demo.py weird 3d",
        r"call :run examples\quimb_demo.py disconnected 2d",
        r"call :run examples\quimb_demo.py disconnected 3d",
        r"call :run examples\quimb_demo.py mps 2d --from-list",
        r"call :run examples\tenpy_demo.py mps 2d",
        r"call :run examples\tenpy_demo.py mps 3d",
        r"call :run examples\tenpy_demo.py mpo 2d",
        r"call :run examples\tenpy_demo.py mpo 3d",
        r"call :run examples\tenpy_demo.py imps 2d",
        r"call :run examples\tenpy_demo.py imps 3d",
        r"call :run examples\tenpy_demo.py impo 2d",
        r"call :run examples\tenpy_demo.py impo 3d",
        r"call :run examples\tenpy_demo.py purification 2d",
        r"call :run examples\tenpy_demo.py purification 3d",
        r"call :run examples\tenpy_demo.py uniform 2d",
        r"call :run examples\tenpy_demo.py uniform 3d",
        r"call :run examples\tenpy_demo.py excitation 2d",
        r"call :run examples\tenpy_demo.py excitation 3d",
        r"call :run examples\tenpy_explicit_tn_demo.py chain 2d",
        r"call :run examples\tenpy_explicit_tn_demo.py chain 3d",
        r"call :run examples\tenpy_explicit_tn_demo.py hub 2d",
        r"call :run examples\tenpy_explicit_tn_demo.py hub 3d",
        r"call :run examples\einsum_demo.py mps 2d",
        r"call :run examples\einsum_demo.py mps 3d",
        r"call :run examples\einsum_demo.py mps 2d --mode manual",
        r"call :run examples\einsum_demo.py mps 3d --mode manual",
        r"call :run examples\einsum_demo.py peps 2d",
        r"call :run examples\einsum_demo.py peps 3d",
        r"call :run examples\einsum_demo.py disconnected 2d",
        r"call :run examples\einsum_demo.py disconnected 3d",
        r"call :run examples\einsum_general.py batch 2d",
        r"call :run examples\einsum_general.py batch 3d",
        r"call :run examples\einsum_general.py ellipsis 2d",
        r"call :run examples\einsum_general.py ellipsis 3d",
        r"call :run examples\einsum_general.py mps_short 2d",
        r"call :run examples\einsum_general.py mps_short 3d",
        r"call :run examples\einsum_general.py nway 2d",
        r"call :run examples\einsum_general.py nway 3d",
        r"call :run examples\einsum_general.py trace 2d",
        r"call :run examples\einsum_general.py trace 3d",
        r"call :run examples\tn_tsp.py -n 4 --view 2d",
        r"call :run examples\tn_tsp.py -n 4 --view 3d",
        r"call :run examples\tn_tsp.py -n 5 --view 2d",
        r"call :run examples\tn_tsp.py -n 5 --view 3d",
        r"call :run examples\tn_tsp.py -n 6 --view 2d",
        r"call :run examples\tn_tsp.py -n 6 --view 3d",
    ]

    content = script_path.read_text(encoding="utf-8")

    assert 'set "PYTHON=%ROOT%\\.venv\\Scripts\\python.exe"' in content
    assert ":find_root" in content
    assert 'if exist "%ROOT%\\.venv\\Scripts\\python.exe"' in content
    assert "if not exist" in content
    assert "goto :error" in content
    for command in expected_commands:
        assert command in content


def test_total_tests_bat_wraps_examples_with_auto_close() -> None:
    content = Path("examples/total_tests.bat").read_text(encoding="utf-8")

    assert 'if not defined PLOT_DELAY_SECONDS set "PLOT_DELAY_SECONDS=2"' in content
    assert 'set "WRAPPER=%ROOT%\\.tmp\\total-tests\\plot_wrapper.py"' in content
    assert 'set "VIEWER=%ROOT%\\.tmp\\total-tests\\image_viewer.ps1"' in content
    assert 'echo matplotlib.use^("Agg"^)' in content
    assert "runpy.run_path" in content
    assert 'savefig^(image_path, bbox_inches="tight"^)' in content
    assert "System.Windows.Forms" in content
    assert "plt.close^('all'^)" in content


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


def test_tenpy_explicit_tn_demo_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tenpy-explicit-tn.png"
    module = _load_example_module(
        Path("examples/tenpy_explicit_tn_demo.py"),
        "tenpy_explicit_tn_demo_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tenpy_explicit_tn_demo.py",
            "chain",
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
            "--mode",
            "auto",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_einsum_general_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "einsum-general.png"
    module = _load_example_module(
        Path("examples/einsum_general.py"),
        "einsum_general_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "einsum_general.py",
            "ellipsis",
            "2d",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_einsum_demo_manual_mode_saves_figure_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path(".tmp") / "example-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "einsum-demo-manual.png"
    module = _load_example_module(
        Path("examples/einsum_demo.py"),
        "einsum_demo_manual_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "einsum_demo.py",
            "mps",
            "2d",
            "--mode",
            "manual",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()


def test_einsum_demo_builders_support_auto_and_manual_modes() -> None:
    module = _load_example_module(
        Path("examples/einsum_demo.py"),
        "einsum_demo_builders_test",
    )

    auto_mps_trace, _ = module.build_mps_trace(mode="auto")
    auto_peps_trace, _ = module.build_peps_trace(mode="auto")
    manual_mps_trace, _ = module.build_mps_trace(mode="manual")
    manual_peps_trace, _ = module.build_peps_trace(mode="manual")

    assert isinstance(auto_mps_trace, EinsumTrace)
    assert isinstance(auto_peps_trace, EinsumTrace)
    assert list(auto_mps_trace)[0].left_name == "A0"
    assert list(auto_peps_trace)[0].left_name == "P00"
    assert isinstance(manual_mps_trace, list)
    assert isinstance(manual_peps_trace, list)
    assert all(isinstance(item, pair_tensor) for item in manual_mps_trace)
    assert all(isinstance(item, pair_tensor) for item in manual_peps_trace)
    assert manual_mps_trace[0].left_name == "A0"
    assert manual_peps_trace[0].left_name == "P00"


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
            "--mode",
            "auto",
            "--save",
            str(output_path),
            "--no-show",
        ],
    )

    module.main()

    assert output_path.exists()
