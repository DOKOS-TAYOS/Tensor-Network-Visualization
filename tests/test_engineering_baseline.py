from __future__ import annotations

import ast
import importlib
import json
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensor_network_viz import PlotConfig
from tensor_network_viz._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from tensor_network_viz._core.layout import _compute_axis_directions


def _build_dense_dangling_chain(length: int) -> _GraphData:
    nodes: dict[int, Any] = {}
    for node_id in range(length):
        axes_names: list[str] = []
        if node_id > 0:
            axes_names.append("left")
        axes_names.append("phys")
        if node_id < length - 1:
            axes_names.append("right")
        nodes[node_id] = _make_node(f"T{node_id}", tuple(axes_names))

    edges = []
    for node_id in range(length):
        phys_axis_index = nodes[node_id].axes_names.index("phys")
        edges.append(
            _make_dangling_edge(
                _EdgeEndpoint(node_id, phys_axis_index, f"p{node_id}"),
                name=f"p{node_id}",
            )
        )
    for node_id in range(length - 1):
        edges.append(
            _make_contraction_edge(
                _EdgeEndpoint(node_id, nodes[node_id].axes_names.index("right"), f"b{node_id}"),
                _EdgeEndpoint(
                    node_id + 1,
                    nodes[node_id + 1].axes_names.index("left"),
                    f"b{node_id}",
                ),
                name=f"b{node_id}",
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _load_pyproject() -> dict[str, Any]:
    import tomllib

    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _load_requirements_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement = stripped.split(" ; ", 1)[0].strip()
        lines.append(requirement)
    return lines


def _top_level_importorskip_targets(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []

    for node in module.body:
        if isinstance(node, (ast.Assign, ast.Expr)):
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


def test_tmp_path_uses_repo_local_tmp_directory(tmp_path: Path) -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    if workspace_root.parent.name == ".worktrees":
        repo_root = workspace_root.parents[1]
    else:
        repo_root = workspace_root
    expected_root = (repo_root / ".tmp").resolve()
    tmp_path_resolved = tmp_path.resolve()

    assert expected_root in tmp_path_resolved.parents


def test_tmp_path_allows_file_writes(tmp_path: Path) -> None:
    output_path = tmp_path / "probe.bin"

    output_path.write_bytes(b"ok")

    assert output_path.read_bytes() == b"ok"


def test_dev_requirements_pin_verification_tools_and_use_editable_install() -> None:
    lines = _load_requirements_lines(Path("requirements.dev.txt"))

    assert "-e ." in lines
    assert {
        "build==1.4.2",
        "ipython==9.10.1",
        "ipython==9.12.0",
        "matplotlib==3.10.8",
        "networkx==3.6.1",
        "numpy==2.4.3",
        "physics-tenpy==1.1.0",
        "pyright==1.1.408",
        "pytest==9.0.2",
        "quimb==1.13.0",
        "ruff==0.15.8",
        "tensorkrowch==1.1.6",
        "tensornetwork==0.4.6",
    }.issubset(set(lines))


def test_pyright_config_escalates_active_type_checks_to_errors() -> None:
    config = json.loads(Path("pyrightconfig.json").read_text(encoding="utf-8"))

    assert config["reportMissingImports"] == "error"
    assert config["reportArgumentType"] == "error"
    assert config["reportAssignmentType"] == "error"
    assert config["reportReturnType"] == "error"
    assert config["reportCallIssue"] == "error"


def test_layout_module_compatibility_exports_survive_internal_split() -> None:
    layout_body = importlib.import_module("tensor_network_viz._core.layout.body")
    draw_edges = importlib.import_module("tensor_network_viz._core.draw.edges")
    draw_pipeline = importlib.import_module("tensor_network_viz._core.draw.graph_pipeline")

    assert hasattr(layout_body, "_compute_axis_directions")
    assert hasattr(layout_body, "_compute_layout")
    assert hasattr(draw_edges, "_draw_edges")
    assert hasattr(draw_edges, "_draw_edges_2d_layered")
    assert hasattr(draw_pipeline, "_draw_graph")


def test_plot_config_accepts_mapping_positions_with_tuple_and_ndarray_values() -> None:
    positions = {
        0: (0.0, 0.0),
        1: np.array([1.0, 0.0], dtype=float),
    }

    config = PlotConfig(positions=positions)

    assert config.positions is positions


def test_compute_axis_directions_chain_2d_is_stable_for_dense_dangling_case() -> None:
    graph = _build_dense_dangling_chain(5)
    positions = {node_id: np.array([float(node_id), 0.0], dtype=float) for node_id in range(5)}

    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)

    for node_id in range(5):
        assert np.allclose(
            directions[(node_id, graph.nodes[node_id].axes_names.index("phys"))],
            np.array([0.0, 1.0], dtype=float),
            atol=1e-9,
        )


@pytest.mark.perf
def test_compute_axis_directions_dense_dangling_chain_completes_quickly() -> None:
    graph = _build_dense_dangling_chain(60)
    positions = {node_id: np.array([float(node_id), 0.0], dtype=float) for node_id in range(60)}

    started = time.perf_counter()
    directions = _compute_axis_directions(graph, positions, dimensions=2, draw_scale=1.0)
    elapsed = time.perf_counter() - started

    assert len(directions) == 178
    assert elapsed < 0.35, f"_compute_axis_directions took {elapsed:.4f}s"


@pytest.mark.perf
def test_compute_axis_directions_dense_dangling_chain_3d_completes_quickly() -> None:
    graph = _build_dense_dangling_chain(160)
    positions = {
        node_id: np.array([float(node_id) * 0.35, 0.0, 0.0], dtype=float) for node_id in range(160)
    }

    started = time.perf_counter()
    directions = _compute_axis_directions(graph, positions, dimensions=3, draw_scale=1.0)
    elapsed = time.perf_counter() - started

    assert len(directions) == 478
    assert elapsed < 0.10, f"_compute_axis_directions(..., dimensions=3) took {elapsed:.4f}s"


def test_pytest_configuration_skips_perf_by_default() -> None:
    pyproject = _load_pyproject()
    pytest_ini = pyproject["tool"]["pytest"]["ini_options"]

    assert shlex.split(pytest_ini["addopts"]) == ["-p", "no:cacheprovider", "-m", "not perf"]
    assert pytest_ini["markers"] == [
        "perf: runtime-sensitive regression checks and throughput guards",
        "smoke: lightweight render smoke checks",
    ]


def test_tensorkrowch_integration_test_skips_when_torch_is_missing() -> None:
    assert _top_level_importorskip_targets(Path("tests/test_integration_tensorkrowch.py")) == [
        "torch",
        "tensorkrowch",
    ]
