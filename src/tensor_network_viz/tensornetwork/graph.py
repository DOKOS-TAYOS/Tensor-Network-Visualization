"""TensorNetwork-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core._nodes_edges_common import (
    _build_graph_from_nodes_edges,
)
from .._core._nodes_edges_common import (
    _get_network_nodes as _get_network_nodes_impl,
)
from .._core.graph import _GraphData


def _get_network_nodes(nodes: Any) -> list[Any]:
    """Extract TensorNetwork nodes from an iterable node collection."""
    return _get_network_nodes_impl(
        nodes,
        axis_attr="axis_names",
        node_sources=(),
        backend_name="TensorNetwork",
    )


def _build_graph(nodes_input: Any) -> _GraphData:
    """Build shared graph from TensorNetwork node collection."""
    return _build_graph_from_nodes_edges(
        nodes_input,
        axis_attr="axis_names",
        node_sources=(),
        backend_name="TensorNetwork",
    )
