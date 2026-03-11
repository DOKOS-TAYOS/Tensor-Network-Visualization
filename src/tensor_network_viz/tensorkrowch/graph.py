"""TensorKrowch-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core._nodes_edges_common import (
    _build_graph_from_nodes_edges,
)
from .._core.graph import _GraphData


def _build_graph(network: Any) -> _GraphData:
    """Build shared graph from TensorKrowch network."""
    return _build_graph_from_nodes_edges(
        network,
        axis_attr="axes_names",
        node_sources=("nodes", "leaf_nodes"),
        backend_name="TensorKrowch",
    )
