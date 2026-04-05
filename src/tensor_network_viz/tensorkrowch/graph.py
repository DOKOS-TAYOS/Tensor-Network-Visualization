"""TensorKrowch-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core._nodes_edges_common import (
    _build_graph_from_nodes_edges,
)
from .._core.graph import _GraphData
from ._history import _recover_contraction_history


def _node_sources_for(network: Any) -> tuple[str, ...]:
    leaf_nodes = getattr(network, "leaf_nodes", None)
    if isinstance(leaf_nodes, dict):
        if leaf_nodes:
            return ("leaf_nodes",)
    elif leaf_nodes:
        return ("leaf_nodes",)
    return ("nodes", "leaf_nodes")


def _build_graph(network: Any) -> _GraphData:
    """Build shared graph from TensorKrowch network."""
    graph = _build_graph_from_nodes_edges(
        network,
        axis_attr="axes_names",
        node_sources=_node_sources_for(network),
        backend_name="TensorKrowch",
    )
    recovered = _recover_contraction_history(network, graph)
    if recovered is None:
        return graph
    return _GraphData(
        nodes=graph.nodes,
        edges=graph.edges,
        contraction_steps=recovered.contraction_steps,
        contraction_step_metrics=recovered.contraction_step_metrics,
    )
