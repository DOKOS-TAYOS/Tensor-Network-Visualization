from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from tensor_network_viz import (
    PlotConfig,
    export_tensor_network_snapshot,
    normalize_tensor_network,
    pair_tensor,
)


def _einsum_network() -> list[object]:
    return [
        pair_tensor("A", "B", "r0", "ab,bc->ac"),
    ]


def test_normalize_tensor_network_returns_serializable_graph() -> None:
    graph = normalize_tensor_network(_einsum_network(), engine="einsum")

    payload = graph.to_dict()

    assert payload["engine"] == "einsum"
    assert {node["name"] for node in payload["nodes"]} == {"A", "B"}
    assert {edge["kind"] for edge in payload["edges"]} == {"contraction", "dangling"}
    assert len(payload["contraction_steps"]) == 1


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
