from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_bench_module() -> ModuleType:
    script_path = Path("scripts/bench_user_workflows.py").resolve()
    spec = importlib.util.spec_from_file_location("bench_user_workflows", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load bench script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bench_user_workflows_parser_exposes_expected_flags() -> None:
    module = _load_bench_module()

    parser = module._build_parser()
    args = parser.parse_args([])

    assert args.surface == "all"
    assert args.size == "all"
    assert args.temperature == "all"
    assert args.output_dir == Path(".tmp/profiling")
    assert args.gui_backend == "auto"
    assert args.cold_samples == 3
    assert args.hot_repeats == 5


def test_bench_user_workflows_enumerates_filtered_scenarios() -> None:
    module = _load_bench_module()

    tensor_elements_small = module._enumerate_scenarios(surface="tensor-elements", size="small")
    tensor_network_small = module._enumerate_scenarios(surface="tensor-network", size="small")

    assert tensor_elements_small
    assert tensor_network_small
    assert all(scenario.surface == "tensor-elements" for scenario in tensor_elements_small)
    assert all(scenario.surface == "tensor-network" for scenario in tensor_network_small)
    assert all(scenario.size_level == "small" for scenario in tensor_elements_small)
    assert all(scenario.size_level == "small" for scenario in tensor_network_small)
    assert any(
        scenario.case == "complex_square" and scenario.action == "mode:real->imag"
        for scenario in tensor_elements_small
    )
    assert any(
        scenario.case == "rank3_analysis" and scenario.action == "analysis:slice_index+1"
        for scenario in tensor_elements_small
    )
    assert not any(
        scenario.case == "real_dense" and scenario.action == "mode:real->imag"
        for scenario in tensor_elements_small
    )
    assert any(
        scenario.backend == "einsum" and scenario.action == "inspector_open"
        for scenario in tensor_network_small
    )
    assert not any(
        scenario.backend == "quimb" and scenario.action == "inspector_open"
        for scenario in tensor_network_small
    )
    structure_kinds = {scenario.structure_kind for scenario in tensor_network_small}
    assert {"linear", "circular", "planar", "generic"} <= structure_kinds


def test_bench_user_workflows_writes_expected_result_schema(tmp_path: Path) -> None:
    module = _load_bench_module()

    result = module.MeasurementResult(
        surface="tensor-elements",
        backend="numpy",
        case="real_dense",
        size_level="small",
        temperature="hot",
        action="initial_render",
        wall_ms=12.5,
        cpu_ms=11.0,
        rss_before_mb=40.0,
        rss_after_mb=44.0,
        peak_rss_mb=45.5,
        approximate_gui=True,
        notes="agg fallback",
    )

    json_path, csv_path = module._write_results([result], tmp_path)

    payload = json_path.read_text(encoding="utf-8")
    assert '"surface": "tensor-elements"' in payload
    assert '"peak_rss_mb": 45.5' in payload
    assert '"approximate_gui": true' in payload

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "surface,backend,case,size_level,temperature,action,wall_ms,cpu_ms" in csv_text
    assert "tensor-elements,numpy,real_dense,small,hot,initial_render,12.5,11.0" in csv_text


def test_bench_user_workflows_runs_agg_worker_smoke_for_tensor_elements() -> None:
    module = _load_bench_module()

    scenario = next(
        scenario
        for scenario in module._enumerate_scenarios(surface="tensor-elements", size="small")
        if scenario.case == "real_dense" and scenario.action == "initial_render"
    )

    result = module._run_worker_subprocess(
        scenario,
        temperature="hot",
        gui_backend="agg",
        cold_samples=1,
        hot_repeats=1,
    )

    assert result.surface == "tensor-elements"
    assert result.case == "real_dense"
    assert result.temperature == "hot"
    assert result.wall_ms >= 0.0
    assert result.peak_rss_mb >= result.rss_before_mb
    assert result.approximate_gui is True


def test_bench_user_workflows_runs_agg_worker_smoke_for_tensor_network() -> None:
    module = _load_bench_module()

    scenario = next(
        scenario
        for scenario in module._enumerate_scenarios(surface="tensor-network", size="small")
        if scenario.backend == "einsum" and scenario.action == "interactive_baseline"
    )

    result = module._run_worker_subprocess(
        scenario,
        temperature="hot",
        gui_backend="agg",
        cold_samples=1,
        hot_repeats=1,
    )

    assert result.surface == "tensor-network"
    assert result.backend == "einsum"
    assert result.temperature == "hot"
    assert result.wall_ms >= 0.0
    assert result.peak_rss_mb >= result.rss_before_mb
    assert result.approximate_gui is True


def test_bench_user_workflows_runs_tkagg_worker_smoke_when_available() -> None:
    module = _load_bench_module()
    if not module._tkagg_available():
        pytest.skip("TkAgg backend is not available in this environment.")

    scenario = next(
        scenario
        for scenario in module._enumerate_scenarios(surface="tensor-network", size="small")
        if scenario.backend == "einsum" and scenario.action == "interactive_baseline"
    )

    result = module._run_worker_subprocess(
        scenario,
        temperature="hot",
        gui_backend="tkagg",
        cold_samples=1,
        hot_repeats=1,
    )

    assert result.surface == "tensor-network"
    assert result.backend == "einsum"
    assert result.approximate_gui is False


def test_bench_user_workflows_cli_smoke_writes_outputs(tmp_path: Path) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "scripts/bench_user_workflows.py",
            "--surface",
            "tensor-elements",
            "--size",
            "small",
            "--temperature",
            "hot",
            "--gui-backend",
            "agg",
            "--case-filter",
            "real_dense",
            "--action-filter",
            "initial_render",
            "--hot-repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ),
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "results.json" in completed.stdout
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
