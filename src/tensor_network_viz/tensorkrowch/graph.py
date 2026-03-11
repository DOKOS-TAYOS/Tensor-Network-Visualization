"""TensorKrowch-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core._nodes_edges_common import (
    _build_graph_from_nodes_edges,
)
from .._core._nodes_edges_common import (
    _get_network_nodes as _get_network_nodes_impl,
)
from .._core.graph import _GraphData


def _get_network_nodes(network: Any) -> list[Any]:
    """Extract TensorKrowch nodes from a network object or iterable."""
    return _get_network_nodes_impl(
        network,
        axis_attr="axes_names",
        node_sources=("nodes", "leaf_nodes"),
        backend_name="TensorKrowch",
    )


def _build_graph(network: Any) -> _GraphData:
    """Build shared graph from TensorKrowch network."""
    return _build_graph_from_nodes_edges(
        network,
        axis_attr="axes_names",
        node_sources=("nodes", "leaf_nodes"),
        backend_name="TensorKrowch",
    )
