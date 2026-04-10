"""Public immutable snapshot types for normalized tensor-network exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._core.focus import filter_graph_for_focus
from ._core.graph import _ContractionStepMetrics, _GraphData
from ._core.graph_cache import _get_or_build_graph
from ._core.renderer import _resolve_geometry
from ._input_inspection import (
    _detect_network_engine_with_input,
    _merge_grid_positions_into_config,
    _prepare_network_input,
    _validate_grid_engine,
)
from ._registry import _get_graph_builder
from .config import EngineName, PlotConfig, ViewName


def _vector_to_tuple(vector: Any) -> tuple[float, ...]:
    """Convert a NumPy vector-like object to a JSON-friendly float tuple."""
    return tuple(float(value) for value in np.asarray(vector, dtype=float).tolist())


@dataclass(frozen=True)
class NormalizedTensorEndpoint:
    """Serializable endpoint of one normalized edge."""

    node_id: int
    axis_index: int
    axis_name: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "node_id": int(self.node_id),
            "axis_index": int(self.axis_index),
            "axis_name": self.axis_name,
        }


@dataclass(frozen=True)
class NormalizedTensorNode:
    """Serializable normalized tensor node."""

    id: int
    name: str
    axes_names: tuple[str, ...]
    degree: int
    label: str | None
    is_virtual: bool
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    element_count: int | None = None
    estimated_nbytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "id": int(self.id),
            "name": self.name,
            "axes_names": list(self.axes_names),
            "degree": int(self.degree),
            "label": self.label,
            "is_virtual": bool(self.is_virtual),
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "element_count": self.element_count,
            "estimated_nbytes": self.estimated_nbytes,
        }


@dataclass(frozen=True)
class NormalizedTensorEdge:
    """Serializable normalized tensor edge."""

    name: str | None
    kind: str
    node_ids: tuple[int, ...]
    endpoints: tuple[NormalizedTensorEndpoint, ...]
    label: str | None
    bond_dimension: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "node_ids": [int(node_id) for node_id in self.node_ids],
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "label": self.label,
            "bond_dimension": self.bond_dimension,
        }


@dataclass(frozen=True)
class NormalizedContractionStepMetrics:
    """Serializable contraction-cost metrics for one normalized execution step."""

    label_dims: tuple[tuple[str, int], ...]
    multiplicative_cost: int
    flop_mac: int
    equation_snippet: str | None = None
    operand_names: tuple[str, ...] = ()
    operand_shapes: tuple[tuple[int, ...], ...] = ()
    output_labels: tuple[str, ...] = ()
    contracted_labels: tuple[str, ...] = ()
    label_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "label_dims": [[label, int(size)] for label, size in self.label_dims],
            "multiplicative_cost": int(self.multiplicative_cost),
            "flop_mac": int(self.flop_mac),
            "equation_snippet": self.equation_snippet,
            "operand_names": list(self.operand_names),
            "operand_shapes": [list(shape) for shape in self.operand_shapes],
            "output_labels": list(self.output_labels),
            "contracted_labels": list(self.contracted_labels),
            "label_order": list(self.label_order),
        }


@dataclass(frozen=True)
class NormalizedTensorGraph:
    """Serializable backend-agnostic graph snapshot."""

    engine: EngineName
    nodes: tuple[NormalizedTensorNode, ...]
    edges: tuple[NormalizedTensorEdge, ...]
    contraction_steps: tuple[tuple[int, ...], ...] | None = None
    contraction_step_metrics: tuple[NormalizedContractionStepMetrics | None, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "engine": self.engine,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "contraction_steps": (
                None
                if self.contraction_steps is None
                else [[int(node_id) for node_id in step] for step in self.contraction_steps]
            ),
            "contraction_step_metrics": (
                None
                if self.contraction_step_metrics is None
                else [
                    None if metric is None else metric.to_dict()
                    for metric in self.contraction_step_metrics
                ]
            ),
        }


@dataclass(frozen=True)
class TensorNetworkLayoutSnapshot:
    """Serializable layout data for one rendered tensor-network view."""

    view: ViewName
    positions: dict[int, tuple[float, ...]]
    axis_directions: dict[tuple[int, int], tuple[float, ...]]
    draw_scale: float
    bond_curve_pad: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "view": self.view,
            "positions": {str(node_id): list(coords) for node_id, coords in self.positions.items()},
            "axis_directions": {
                f"{node_id}:{axis_index}": list(direction)
                for (node_id, axis_index), direction in self.axis_directions.items()
            },
            "draw_scale": float(self.draw_scale),
            "bond_curve_pad": float(self.bond_curve_pad),
        }


@dataclass(frozen=True)
class TensorNetworkSnapshot:
    """Serializable structural plus layout snapshot."""

    graph: NormalizedTensorGraph
    layout: TensorNetworkLayoutSnapshot

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "graph": self.graph.to_dict(),
            "layout": self.layout.to_dict(),
        }


def _normalized_step_metrics(
    metrics: _ContractionStepMetrics | None,
) -> NormalizedContractionStepMetrics | None:
    if metrics is None:
        return None
    return NormalizedContractionStepMetrics(
        label_dims=tuple((label, int(size)) for label, size in metrics.label_dims),
        multiplicative_cost=int(metrics.multiplicative_cost),
        flop_mac=int(metrics.flop_mac),
        equation_snippet=metrics.equation_snippet,
        operand_names=tuple(str(name) for name in metrics.operand_names),
        operand_shapes=tuple(
            tuple(int(dimension) for dimension in shape) for shape in metrics.operand_shapes
        ),
        output_labels=tuple(str(label) for label in metrics.output_labels),
        contracted_labels=tuple(str(label) for label in metrics.contracted_labels),
        label_order=tuple(str(label) for label in metrics.label_order),
    )


def _normalized_graph_from_internal(
    graph: _GraphData,
    *,
    engine: EngineName,
) -> NormalizedTensorGraph:
    nodes = tuple(
        NormalizedTensorNode(
            id=int(node_id),
            name=node.name,
            axes_names=tuple(str(axis_name) for axis_name in node.axes_names),
            degree=int(node.degree),
            label=node.label,
            is_virtual=bool(node.is_virtual),
            shape=None if node.shape is None else tuple(int(size) for size in node.shape),
            dtype=node.dtype,
            element_count=None if node.element_count is None else int(node.element_count),
            estimated_nbytes=(
                None if node.estimated_nbytes is None else int(node.estimated_nbytes)
            ),
        )
        for node_id, node in sorted(graph.nodes.items(), key=lambda item: int(item[0]))
    )
    edges = tuple(
        NormalizedTensorEdge(
            name=edge.name,
            kind=edge.kind,
            node_ids=tuple(int(node_id) for node_id in edge.node_ids),
            endpoints=tuple(
                NormalizedTensorEndpoint(
                    node_id=int(endpoint.node_id),
                    axis_index=int(endpoint.axis_index),
                    axis_name=endpoint.axis_name,
                )
                for endpoint in edge.endpoints
            ),
            label=edge.label,
            bond_dimension=None if edge.bond_dimension is None else int(edge.bond_dimension),
        )
        for edge in graph.edges
    )
    contraction_steps = None
    if graph.contraction_steps is not None:
        contraction_steps = tuple(
            tuple(sorted(int(node_id) for node_id in step)) for step in graph.contraction_steps
        )
    contraction_step_metrics = None
    if graph.contraction_step_metrics is not None:
        contraction_step_metrics = tuple(
            _normalized_step_metrics(metric) for metric in graph.contraction_step_metrics
        )
    return NormalizedTensorGraph(
        engine=engine,
        nodes=nodes,
        edges=edges,
        contraction_steps=contraction_steps,
        contraction_step_metrics=contraction_step_metrics,
    )


def normalize_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
) -> NormalizedTensorGraph:
    """Export the backend-normalized structural graph for a tensor network."""
    network_input = _prepare_network_input(network)
    resolved_engine = engine
    if resolved_engine is None:
        resolved_engine, network_input = _detect_network_engine_with_input(network_input)
    _validate_grid_engine(network_input, engine=resolved_engine)
    build_graph = _get_graph_builder(resolved_engine)
    graph = _get_or_build_graph(network_input, build_graph)
    return _normalized_graph_from_internal(graph, engine=resolved_engine)


def export_tensor_network_snapshot(
    network: Any,
    *,
    engine: EngineName | None = None,
    view: ViewName = "2d",
    config: PlotConfig | None = None,
    seed: int = 0,
) -> TensorNetworkSnapshot:
    """Export a backend-normalized graph together with the resolved layout snapshot."""
    network_input = _prepare_network_input(network)
    resolved_engine = engine
    if resolved_engine is None:
        resolved_engine, network_input = _detect_network_engine_with_input(network_input)
    _validate_grid_engine(network_input, engine=resolved_engine)
    build_graph = _get_graph_builder(resolved_engine)
    graph = _get_or_build_graph(network_input, build_graph)
    style = config or PlotConfig()
    dimensions = 2 if view == "2d" else 3
    style = _merge_grid_positions_into_config(style, network_input, dimensions=dimensions)
    geometry = _resolve_geometry(
        graph,
        style,
        dimensions=dimensions,
        seed=seed,
    )
    focused_graph = filter_graph_for_focus(graph, style.focus)
    focused_node_ids = set(focused_graph.nodes)
    layout = TensorNetworkLayoutSnapshot(
        view=view,
        positions={
            int(node_id): _vector_to_tuple(coords)
            for node_id, coords in geometry.positions.items()
            if int(node_id) in focused_node_ids
        },
        axis_directions={
            (int(node_id), int(axis_index)): _vector_to_tuple(direction)
            for (node_id, axis_index), direction in geometry.directions.items()
            if int(node_id) in focused_node_ids
        },
        draw_scale=float(geometry.scale),
        bond_curve_pad=float(geometry.bond_curve_pad),
    )
    return TensorNetworkSnapshot(
        graph=_normalized_graph_from_internal(focused_graph, engine=resolved_engine),
        layout=layout,
    )


__all__ = [
    "NormalizedContractionStepMetrics",
    "NormalizedTensorEdge",
    "NormalizedTensorEndpoint",
    "NormalizedTensorGraph",
    "NormalizedTensorNode",
    "TensorNetworkLayoutSnapshot",
    "TensorNetworkSnapshot",
    "export_tensor_network_snapshot",
    "normalize_tensor_network",
]
