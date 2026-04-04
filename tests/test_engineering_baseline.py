from __future__ import annotations

import importlib
import time
from pathlib import Path

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
    nodes = {}
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
                _EdgeEndpoint(
                    node_id,
                    nodes[node_id].axes_names.index("right"),
                    f"b{node_id}",
                ),
                _EdgeEndpoint(
                    node_id + 1,
                    nodes[node_id + 1].axes_names.index("left"),
                    f"b{node_id}",
                ),
                name=f"b{node_id}",
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


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
    content = Path("requirements.dev.txt").read_text(encoding="utf-8")

    assert "-e ." in content
    assert '".[dev]"' not in content
    assert "pytest==" in content
    assert "ruff==" in content
    assert "pyright==" in content
    assert "build==" in content
    assert "matplotlib==" in content
    assert "networkx==" in content
    assert "numpy==" in content


def test_pyright_config_escalates_active_type_checks_to_errors() -> None:
    content = Path("pyrightconfig.json").read_text(encoding="utf-8")

    assert '"reportMissingImports": "error"' in content
    assert '"reportArgumentType": "error"' in content
    assert '"reportAssignmentType": "error"' in content
    assert '"reportReturnType": "error"' in content
    assert '"reportCallIssue": "error"' in content


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


def test_pyproject_declares_smoke_and_perf_markers() -> None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'markers = [' in content
    assert '"perf: runtime-sensitive regression checks and throughput guards"' in content
    assert '"smoke: lightweight render smoke checks"' in content


def test_contributing_references_engine_module_map_instead_of_legacy_engine_config() -> None:
    content = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "_engine_specs.py" in content
    assert "ENGINE_MODULE_MAP" in content
    assert "_ENGINE_CONFIG" not in content


def test_render_benchmark_script_covers_cache_and_control_scenarios() -> None:
    content = Path("scripts/bench_render_workflows.py").read_text(encoding="utf-8")

    assert "first render" in content
    assert "repeated render" in content
    assert "network-object" in content
    assert "tensor-list" in content
    assert "interactive-controls" in content
    assert "static-render" in content
