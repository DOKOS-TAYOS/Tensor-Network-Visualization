"""TensorNetwork-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core._nodes_edges_common import (
    _build_graph_from_nodes_edges,
)
from .._core.graph import _GraphData


def _build_graph(nodes_input: Any) -> _GraphData:
    """Build shared graph from TensorNetwork node collection."""
    return _build_graph_from_nodes_edges(
        nodes_input,
        axis_attr="axis_names",
        node_sources=(),
        backend_name="TensorNetwork",
    )
