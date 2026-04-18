from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    einsum,
    export_tensor_network_snapshot,
    normalize_tensor_network,
    show_tensor_network,
    translate_tensor_network,
)


def _exec_generated_network(code: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace["network"]


def _graph_signature(graph: Any) -> dict[str, Any]:
    node_name_by_id = {int(node.id): node.name for node in graph.nodes}

    def _normalized_dtype(dtype: str | None) -> str | None:
        if dtype is None:
            return None
        return dtype.split(".")[-1]

    nodes = tuple(
        sorted(
            (
                node.name,
                tuple(node.axes_names),
                None if node.shape is None else tuple(node.shape),
                _normalized_dtype(node.dtype),
            )
            for node in graph.nodes
        )
    )
    edges = tuple(
        sorted(
            (
                edge.kind,
                edge.label,
                tuple(
                    sorted(
                        (
                            node_name_by_id[int(endpoint.node_id)],
                            int(endpoint.axis_index),
                            endpoint.axis_name,
                        )
                        for endpoint in edge.endpoints
                    )
                ),
                edge.bond_dimension,
            )
            for edge in graph.edges
        )
    )
    return {
        "nodes": nodes,
        "edges": edges,
    }


def _snapshot_positions_by_name(snapshot: Any) -> dict[str, tuple[float, ...]]:
    node_name_by_id = {int(node.id): node.name for node in snapshot.graph.nodes}
    return {
        node_name_by_id[int(node_id)]: tuple(float(value) for value in coords)
        for node_id, coords in snapshot.layout.positions.items()
    }


def test_translate_tensor_network_returns_code_string_for_tensornetwork_target() -> None:
    tn = pytest.importorskip("tensornetwork")

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="tensornetwork")

    assert isinstance(code, str)
    assert "def build_tensor_network() -> Any:" in code
    assert "network = build_tensor_network()" in code


def test_translate_tensor_network_writes_same_code_to_path(tmp_path: Path) -> None:
    tn = pytest.importorskip("tensornetwork")

    node = tn.Node(np.ones((2, 2)), name="T", axis_names=("a", "b"))
    output_path = tmp_path / "translated_network.py"

    code = translate_tensor_network([node], target_engine="tensornetwork", path=output_path)

    assert output_path.read_text(encoding="utf-8") == code


def test_translate_tensor_network_rejects_tenpy_as_target() -> None:
    tn = pytest.importorskip("tensornetwork")
    node = tn.Node(np.ones((2, 2)), name="T", axis_names=("a", "b"))

    with pytest.raises(ValueError, match="tenpy"):
        translate_tensor_network([node], target_engine="tenpy")  # type: ignore[arg-type]


def test_translate_tensor_network_round_trips_simple_tensornetwork_structure() -> None:
    tn = pytest.importorskip("tensornetwork")

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="tensornetwork")
    translated = _exec_generated_network(code)

    original_graph = normalize_tensor_network([left, right], engine="tensornetwork")
    translated_graph = normalize_tensor_network(translated, engine="tensornetwork")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)

    fig, _ax = show_tensor_network(translated, engine="tensornetwork", show=False)
    assert fig is not None


def test_translate_tensor_network_supports_quimb_target_round_trip() -> None:
    tn = pytest.importorskip("tensornetwork")
    pytest.importorskip("quimb.tensor")

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="quimb")
    translated = _exec_generated_network(code)

    original_graph = normalize_tensor_network([left, right], engine="tensornetwork")
    translated_graph = normalize_tensor_network(translated, engine="quimb")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_generates_connectivity_only_einsum_when_no_steps_exist() -> None:
    tn = pytest.importorskip("tensornetwork")

    left = tn.Node(np.ones((2, 3)), name="L", axis_names=("a", "b"))
    right = tn.Node(np.ones((3, 4)), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="einsum")

    assert "EinsumTrace()" in code
    assert "einsum(" in code

    translated = _exec_generated_network(code)
    translated_graph = normalize_tensor_network(translated, engine="einsum")
    original_graph = normalize_tensor_network([left, right], engine="tensornetwork")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_preserves_ordered_einsum_trace_when_steps_exist() -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    middle = np.arange(3, dtype=float)
    right = np.arange(8, dtype=float).reshape(2, 4)

    trace.bind("L", left)
    trace.bind("M", middle)
    trace.bind("R", right)
    result = einsum("ab,b->a", left, middle, trace=trace, backend="numpy")
    _ = einsum("a,ac->c", result, right, trace=trace, backend="numpy")

    code = translate_tensor_network(trace, engine="einsum", target_engine="einsum")

    assert code.count(" = einsum(") == 2

    translated = _exec_generated_network(code)
    original_graph = normalize_tensor_network(trace, engine="einsum")
    translated_graph = normalize_tensor_network(translated, engine="einsum")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_supports_tensorkrowch_target_round_trip() -> None:
    tn = pytest.importorskip("tensornetwork")
    pytest.importorskip("torch")
    pytest.importorskip("tensorkrowch")

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="tensorkrowch")
    translated = _exec_generated_network(code)

    original_graph = normalize_tensor_network([left, right], engine="tensornetwork")
    translated_graph = normalize_tensor_network(translated, engine="tensorkrowch")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_rejects_tensorkrowch_target_for_disconnected_outer_product() -> (
    None
):
    tn = pytest.importorskip("tensornetwork")
    pytest.importorskip("torch")
    pytest.importorskip("tensorkrowch")

    left = tn.Node(np.ones((2,)), name="A", axis_names=("a",))
    right = tn.Node(np.ones((3,)), name="B", axis_names=("b",))

    with pytest.raises(ValueError, match="outer product"):
        translate_tensor_network([left, right], target_engine="tensorkrowch")


def test_translate_tensor_network_supports_quimb_source_to_tensornetwork_target() -> None:
    pytest.importorskip("quimb.tensor")
    import quimb.tensor as qtn

    tensors = [
        qtn.Tensor(
            data=cast(Any, np.arange(6, dtype=float).reshape(2, 3)),
            inds=("a", "b"),
            tags={"L"},
        ),
        qtn.Tensor(
            data=cast(Any, np.arange(12, dtype=float).reshape(3, 4)),
            inds=("b", "c"),
            tags={"R"},
        ),
    ]
    network = qtn.TensorNetwork(tensors)

    code = translate_tensor_network(network, target_engine="tensornetwork")
    translated = _exec_generated_network(code)

    original_graph = normalize_tensor_network(network, engine="quimb")
    translated_graph = normalize_tensor_network(translated, engine="tensornetwork")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_supports_einsum_source_to_quimb_target() -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(12, dtype=float).reshape(3, 4)

    trace.bind("L", left)
    trace.bind("R", right)
    _ = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")

    code = translate_tensor_network(trace, engine="einsum", target_engine="quimb")
    translated = _exec_generated_network(code)

    original_graph = normalize_tensor_network(trace, engine="einsum")
    translated_graph = normalize_tensor_network(translated, engine="quimb")

    assert _graph_signature(original_graph) == _graph_signature(translated_graph)


def test_translate_tensor_network_can_reuse_named_positions_for_comparable_view() -> None:
    tn = pytest.importorskip("tensornetwork")
    pytest.importorskip("quimb.tensor")

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]

    code = translate_tensor_network([left, right], target_engine="quimb")
    translated = _exec_generated_network(code)

    source_snapshot = export_tensor_network_snapshot(
        [left, right],
        engine="tensornetwork",
        view="2d",
        seed=0,
    )
    source_positions = _snapshot_positions_by_name(source_snapshot)
    translated_graph = normalize_tensor_network(translated, engine="quimb")
    translated_positions = {
        int(node.id): source_positions[node.name]
        for node in translated_graph.nodes
        if node.name in source_positions
    }
    translated_snapshot = export_tensor_network_snapshot(
        translated,
        engine="quimb",
        view="2d",
        config=PlotConfig(positions=translated_positions),
        seed=0,
    )

    assert _snapshot_positions_by_name(translated_snapshot) == source_positions
