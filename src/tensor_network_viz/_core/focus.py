"""Helpers for reproducible subnetwork focus over the normalized graph."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from ..config import TensorNetworkFocus
from .graph import (
    _ContractionStepMetrics,
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_dangling_edge,
)


def _physical_name_index(
    graph: _GraphData,
) -> tuple[dict[str, int], set[str]]:
    """Index non-virtual tensor names and track ambiguous duplicates."""
    mapping: dict[str, int] = {}
    duplicates: set[str] = set()
    for node_id, node in graph.nodes.items():
        if node.is_virtual or not node.name:
            continue
        if node.name in mapping:
            duplicates.add(node.name)
            continue
        mapping[node.name] = int(node_id)
    return mapping, duplicates


def _resolve_focus_name(graph: _GraphData, name: str) -> int:
    """Resolve one public tensor name to a unique normalized node id."""
    mapping, duplicates = _physical_name_index(graph)
    if name in duplicates:
        raise ValueError(f"Focus tensor name {name!r} is ambiguous in this graph.")
    if name not in mapping:
        raise ValueError(f"Focus tensor name {name!r} is not present in this graph.")
    return mapping[name]


def _physical_adjacency(graph: _GraphData) -> dict[int, set[int]]:
    """Build tensor-to-tensor adjacency ignoring virtual hub distance."""
    adjacency = {
        int(node_id): set() for node_id, node in graph.nodes.items() if not node.is_virtual
    }
    virtual_neighbors: dict[int, set[int]] = {}

    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        physical_endpoints = [
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if not graph.nodes[endpoint.node_id].is_virtual
        ]
        virtual_endpoints = [
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if graph.nodes[endpoint.node_id].is_virtual
        ]
        if len(physical_endpoints) == 2:
            left_id, right_id = physical_endpoints
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)
            continue
        if len(physical_endpoints) == 1 and len(virtual_endpoints) == 1:
            hub_id = virtual_endpoints[0]
            virtual_neighbors.setdefault(hub_id, set()).add(physical_endpoints[0])

    for tensor_ids in virtual_neighbors.values():
        for tensor_id in tensor_ids:
            adjacency[tensor_id].update(other for other in tensor_ids if other != tensor_id)
    return adjacency


def _neighborhood_selection(
    graph: _GraphData,
    *,
    center_id: int,
    radius: int,
) -> set[int]:
    """Return the tensor ids within the requested physical-graph radius."""
    adjacency = _physical_adjacency(graph)
    selected = {int(center_id)}
    queue: deque[tuple[int, int]] = deque([(int(center_id), 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= int(radius):
            continue
        for neighbor in adjacency.get(node_id, set()):
            if neighbor in selected:
                continue
            selected.add(neighbor)
            queue.append((neighbor, depth + 1))
    return selected


def _shortest_path_selection(
    graph: _GraphData,
    *,
    start_id: int,
    end_id: int,
) -> set[int]:
    """Return the tensor ids on the physical shortest path between two tensors."""
    if int(start_id) == int(end_id):
        return {int(start_id)}
    adjacency = _physical_adjacency(graph)
    parents: dict[int, int | None] = {int(start_id): None}
    queue: deque[int] = deque([int(start_id)])
    while queue:
        node_id = queue.popleft()
        if node_id == int(end_id):
            break
        for neighbor in adjacency.get(node_id, set()):
            if neighbor in parents:
                continue
            parents[neighbor] = node_id
            queue.append(neighbor)
    if int(end_id) not in parents:
        raise ValueError("No tensor path exists between the requested focus endpoints.")
    path_ids: set[int] = set()
    current_id: int | None = int(end_id)
    while current_id is not None:
        path_ids.add(current_id)
        current_id = parents[current_id]
    return path_ids


def focus_selected_tensor_ids(
    graph: _GraphData,
    focus: TensorNetworkFocus | None,
) -> set[int] | None:
    """Resolve the focused physical tensor ids, or ``None`` when no focus applies."""
    if focus is None:
        return None
    if focus.kind == "neighborhood":
        if focus.center is None:
            return None
        center_id = _resolve_focus_name(graph, focus.center)
        return _neighborhood_selection(graph, center_id=center_id, radius=int(focus.radius))
    if focus.endpoints is None:
        return None
    start_id = _resolve_focus_name(graph, focus.endpoints[0])
    end_id = _resolve_focus_name(graph, focus.endpoints[1])
    return _shortest_path_selection(graph, start_id=start_id, end_id=end_id)


def _focused_virtual_hubs(graph: _GraphData, selected_tensor_ids: set[int]) -> set[int]:
    """Return the virtual hub ids that still connect at least two selected tensors."""
    hub_neighbors: dict[int, set[int]] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        physical_endpoints = [
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if not graph.nodes[endpoint.node_id].is_virtual
            and endpoint.node_id in selected_tensor_ids
        ]
        virtual_endpoints = [
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if graph.nodes[endpoint.node_id].is_virtual
        ]
        if len(physical_endpoints) != 1 or len(virtual_endpoints) != 1:
            continue
        hub_neighbors.setdefault(virtual_endpoints[0], set()).add(physical_endpoints[0])
    return {hub_id for hub_id, neighbors in hub_neighbors.items() if len(neighbors) >= 2}


def _cut_edge_stub(edge: _EdgeData, endpoint: _EdgeEndpoint) -> _EdgeData:
    """Re-express a cut bond as a dangling stub on the kept tensor endpoint."""
    stub = _make_dangling_edge(endpoint, name=edge.name)
    if edge.bond_dimension is None:
        return stub
    return replace(stub, bond_dimension=edge.bond_dimension)


def filter_graph_for_focus(
    graph: _GraphData,
    focus: TensorNetworkFocus | None,
) -> _GraphData:
    """Return a focused graph while preserving the original node ids and coordinates."""
    selected_tensor_ids = focus_selected_tensor_ids(graph, focus)
    if not selected_tensor_ids:
        return graph

    preserve_cut_bonds = focus is not None
    active_hub_ids = _focused_virtual_hubs(graph, selected_tensor_ids)
    kept_edges = []
    for edge in graph.edges:
        selected_physical_endpoints = [
            endpoint
            for endpoint in edge.endpoints
            if (
                not graph.nodes[endpoint.node_id].is_virtual
                and int(endpoint.node_id) in selected_tensor_ids
            )
        ]
        physical_endpoints = {
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if not graph.nodes[endpoint.node_id].is_virtual
        }
        virtual_endpoints = {
            int(endpoint.node_id)
            for endpoint in edge.endpoints
            if graph.nodes[endpoint.node_id].is_virtual
        }
        if edge.kind in {"dangling", "self"}:
            if physical_endpoints and physical_endpoints.issubset(selected_tensor_ids):
                kept_edges.append(edge)
            continue
        if len(physical_endpoints) == 2:
            if physical_endpoints.issubset(selected_tensor_ids):
                kept_edges.append(edge)
            elif preserve_cut_bonds and len(selected_physical_endpoints) == 1:
                kept_edges.append(_cut_edge_stub(edge, selected_physical_endpoints[0]))
            continue
        if (
            len(physical_endpoints) == 1
            and len(virtual_endpoints) == 1
            and next(iter(physical_endpoints)) in selected_tensor_ids
            and next(iter(virtual_endpoints)) in active_hub_ids
        ):
            kept_edges.append(edge)
            continue
        if preserve_cut_bonds and len(selected_physical_endpoints) == 1:
            kept_edges.append(_cut_edge_stub(edge, selected_physical_endpoints[0]))

    kept_node_ids = set(selected_tensor_ids) | active_hub_ids
    kept_nodes = {
        node_id: node for node_id, node in graph.nodes.items() if int(node_id) in kept_node_ids
    }

    kept_steps: tuple[frozenset[int], ...] | None = None
    kept_metrics: tuple[_ContractionStepMetrics | None, ...] | None = None
    if graph.contraction_steps is not None:
        filtered_steps: list[frozenset[int]] = []
        filtered_metrics: list[_ContractionStepMetrics | None] = []
        metrics = graph.contraction_step_metrics
        for index, step in enumerate(graph.contraction_steps):
            filtered_step = frozenset(
                int(node_id) for node_id in step if int(node_id) in selected_tensor_ids
            )
            if not filtered_step:
                continue
            filtered_steps.append(filtered_step)
            if metrics is not None:
                filtered_metrics.append(metrics[index])
        kept_steps = tuple(filtered_steps)
        kept_metrics = None if graph.contraction_step_metrics is None else tuple(filtered_metrics)

    return _GraphData(
        nodes=kept_nodes,
        edges=tuple(kept_edges),
        contraction_steps=kept_steps,
        contraction_step_metrics=kept_metrics,
    )


__all__ = [
    "filter_graph_for_focus",
    "focus_selected_tensor_ids",
]
