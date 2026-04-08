from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    TensorNetworkDiagnosticsConfig,
    TensorNetworkFocus,
    einsum,
    export_tensor_network_snapshot,
    normalize_tensor_network,
    pair_tensor,
)


def _einsum_network() -> list[object]:
    return [
        pair_tensor("A", "B", "r0", "ab,bc->ac"),
    ]


def _einsum_trace() -> EinsumTrace:
    trace = EinsumTrace()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    mid = np.arange(12, dtype=np.float64).reshape(3, 4)
    right = np.arange(20, dtype=np.float64).reshape(4, 5)

    trace.bind("A", left)
    trace.bind("B", mid)
    trace.bind("C", right)
    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, r0, r1]  # type: ignore[attr-defined]
    return trace


def test_normalize_tensor_network_returns_serializable_graph() -> None:
    graph = normalize_tensor_network(_einsum_network(), engine="einsum")

    payload = graph.to_dict()

    assert payload["engine"] == "einsum"
    assert {node["name"] for node in payload["nodes"]} == {"A", "B"}
    assert {edge["kind"] for edge in payload["edges"]} == {"contraction", "dangling"}
    assert len(payload["contraction_steps"]) == 1


def test_normalize_tensor_network_serializes_diagnostics_for_nodes_and_edges() -> None:
    graph = normalize_tensor_network(_einsum_trace(), engine="einsum")

    payload = graph.to_dict()
    nodes_by_name = {node["name"]: node for node in payload["nodes"]}
    contraction_edges = [edge for edge in payload["edges"] if edge["kind"] == "contraction"]

    assert nodes_by_name["A"]["shape"] == [2, 3]
    assert nodes_by_name["A"]["dtype"] == "float64"
    assert nodes_by_name["A"]["element_count"] == 6
    assert nodes_by_name["A"]["estimated_nbytes"] == 48
    assert nodes_by_name["B"]["shape"] == [3, 4]
    assert nodes_by_name["C"]["shape"] == [4, 5]
    assert {edge["bond_dimension"] for edge in contraction_edges} >= {3, 4}


def test_export_tensor_network_snapshot_includes_layout_data() -> None:
    snapshot = export_tensor_network_snapshot(
        _einsum_network(),
        engine="einsum",
        view="2d",
        config=PlotConfig(),
        seed=0,
    )

    payload = snapshot.to_dict()

    assert payload["graph"]["engine"] == "einsum"
    assert payload["layout"]["view"] == "2d"
    assert payload["layout"]["draw_scale"] > 0.0
    assert payload["layout"]["bond_curve_pad"] >= 0.0
    assert set(payload["layout"]["positions"]) == {
        str(node["id"]) for node in payload["graph"]["nodes"]
    }


def test_export_tensor_network_snapshot_applies_focus_filter_with_stable_positions() -> None:
    trace = _einsum_trace()

    full_snapshot = export_tensor_network_snapshot(
        trace,
        engine="einsum",
        view="2d",
        config=PlotConfig(),
        seed=0,
    )
    focused_snapshot = export_tensor_network_snapshot(
        trace,
        engine="einsum",
        view="2d",
        config=PlotConfig(
            diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=True),
            focus=TensorNetworkFocus(kind="path", endpoints=("A", "C")),
        ),
        seed=0,
    )

    full_payload = full_snapshot.to_dict()
    focused_payload = focused_snapshot.to_dict()
    focused_names = {node["name"] for node in focused_payload["graph"]["nodes"]}
    full_nodes_by_name = {node["name"]: node for node in full_payload["graph"]["nodes"]}
    focused_nodes_by_name = {node["name"]: node for node in focused_payload["graph"]["nodes"]}

    assert focused_names == {"A", "B", "C"}
    for name in focused_names:
        full_id = str(full_nodes_by_name[name]["id"])
        focused_id = str(focused_nodes_by_name[name]["id"])
        assert (
            focused_payload["layout"]["positions"][focused_id]
            == full_payload["layout"]["positions"][full_id]
        )
