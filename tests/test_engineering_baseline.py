from __future__ import annotations

import ast
import importlib
import json
import os
import shlex
import subprocess
import sys
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


def _run_module_snapshot(code: str) -> dict[str, bool]:
    workspace_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        cwd=workspace_root,
        env=env,
        text=True,
    )
    return json.loads(result.stdout)


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

    assert config["venvPath"] == "."
    assert config["venv"] == ".venv"
    assert config["reportMissingImports"] == "error"
    assert config["reportArgumentType"] == "error"
    assert config["reportAssignmentType"] == "error"
    assert config["reportReturnType"] == "error"
    assert config["reportCallIssue"] == "error"


def test_input_inspection_reuses_graph_utils_unordered_collection_helper() -> None:
    graph_utils = importlib.import_module("tensor_network_viz._core.graph_utils")
    input_inspection = importlib.import_module("tensor_network_viz._input_inspection")

    assert input_inspection._is_unordered_collection is graph_utils._is_unordered_collection


def test_tenpy_graph_reuses_explicit_leg_label_helper() -> None:
    explicit = importlib.import_module("tensor_network_viz.tenpy.explicit")
    graph = importlib.import_module("tensor_network_viz.tenpy.graph")

    assert graph._leg_labels is explicit._leg_labels


def test_render_prep_does_not_keep_unused_memory_format_helper() -> None:
    render_prep = importlib.import_module("tensor_network_viz._core.draw.render_prep")

    assert not hasattr(render_prep, "_format_memory_estimate")


def test_tensor_elements_data_does_not_keep_unused_spectral_summary_helpers() -> None:
    tensor_elements_data = importlib.import_module("tensor_network_viz._tensor_elements_data")

    assert not hasattr(tensor_elements_data, "_build_spectral_summary_lines")
    assert not hasattr(tensor_elements_data, "_format_name_list")
    assert not hasattr(tensor_elements_data, "_format_percent")
    assert not hasattr(tensor_elements_data, "_topk_singular_value_lines")
    assert not hasattr(tensor_elements_data, "_topk_eigenvalue_lines")


def test_tensorkrowch_history_uses_single_node_tuple_normalizer() -> None:
    history = importlib.import_module("tensor_network_viz.tensorkrowch._history")

    assert hasattr(history, "_normalized_node_tuple")
    assert not hasattr(history, "_normalized_parent_nodes")
    assert not hasattr(history, "_normalized_child_nodes")


def test_top_level_import_keeps_cold_path_modules_lazy() -> None:
    loaded = _run_module_snapshot(
        """
import json
import sys
import tensor_network_viz

targets = [
    "matplotlib.axes",
    "matplotlib.figure",
    "mpl_toolkits.mplot3d.axes3d",
    "numpy",
    "tensor_network_viz.config",
    "tensor_network_viz._core.graph_cache",
]
print(json.dumps({name: name in sys.modules for name in targets}))
"""
    )

    assert loaded == {
        "matplotlib.axes": False,
        "matplotlib.figure": False,
        "mpl_toolkits.mplot3d.axes3d": False,
        "numpy": False,
        "tensor_network_viz.config": False,
        "tensor_network_viz._core.graph_cache": False,
    }


def test_static_tensor_elements_cold_path_skips_controller_and_widgets() -> None:
    loaded = _run_module_snapshot(
        """
import json
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensor_network_viz as tnv

tnv.show_tensor_elements(
    np.arange(6.0).reshape(2, 3),
    show=False,
    show_controls=False,
)
targets = [
    "tensor_network_viz._tensor_elements_controller",
    "matplotlib.widgets",
]
print(json.dumps({name: name in sys.modules for name in targets}))
"""
    )

    assert loaded == {
        "tensor_network_viz._tensor_elements_controller": False,
        "matplotlib.widgets": False,
    }


def test_static_tensor_network_cold_path_skips_interaction_modules() -> None:
    loaded = _run_module_snapshot(
        """
import json
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensor_network_viz as tnv

class Edge:
    def __init__(self, name):
        self.name = name
        self.node1 = None
        self.node2 = None

class Node:
    def __init__(self, name, axis_names):
        self.name = name
        self.axis_names = list(axis_names)
        self.tensor = np.ones((2, 2), dtype=float)
        self.shape = self.tensor.shape
        self.edges = [None] * len(self.axis_names)

left = Node("L", ("a", "b"))
right = Node("R", ("b", "c"))
edge = Edge("bond")
edge.node1 = left
edge.node2 = right
left.edges[1] = edge
right.edges[0] = edge

tnv.show_tensor_network([left, right], show=False, show_controls=False)
targets = [
    "tensor_network_viz.contraction_viewer",
    "tensor_network_viz._interaction.scheme",
    "tensor_network_viz._interactive_scene",
    "matplotlib.widgets",
]
print(json.dumps({name: name in sys.modules for name in targets}))
"""
    )

    assert loaded == {
        "tensor_network_viz.contraction_viewer": False,
        "tensor_network_viz._interaction.scheme": False,
        "tensor_network_viz._interactive_scene": False,
        "matplotlib.widgets": False,
    }


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
