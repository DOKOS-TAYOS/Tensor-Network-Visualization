"""TensorKrowch contraction-history recovery for automatic scheme playback."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .._core.graph import _ContractionStepMetrics, _GraphData
from ..einsum_module.contraction_cost import metrics_for_labeled_operands


@dataclass(frozen=True)
class _ProducedNode:
    op_name: str
    parent_ids: tuple[int, ...]
    parents: tuple[Any, ...]


@dataclass(frozen=True)
class _RecoveredHistory:
    contraction_steps: tuple[frozenset[int], ...] | None
    contraction_step_metrics: tuple[_ContractionStepMetrics | None, ...] | None
    step_result_nodes: tuple[Any, ...]


def _recover_contraction_history(
    network: Any,
    graph: _GraphData | None = None,
) -> _RecoveredHistory | None:
    leaf_nodes = _unique_nodes_from_attr(network, "leaf_nodes")
    if not leaf_nodes:
        return None

    resultant_nodes = _unique_nodes_from_attr(network, "resultant_nodes")
    if not resultant_nodes:
        return None

    produced = _build_produced_node_map(network)
    if produced is None:
        return None

    leaf_ids = {id(node) for node in leaf_nodes}
    if graph is not None:
        graph_node_ids = set(graph.nodes)
        if not leaf_ids <= graph_node_ids:
            return None

    contraction_nodes = [
        node
        for node in resultant_nodes
        if (entry := produced.get(id(node))) is not None and _is_contraction_op(entry.op_name)
    ]
    if not contraction_nodes:
        return None

    ordered_contractions = _topological_contraction_order(contraction_nodes, produced)
    if ordered_contractions is None:
        return None

    contributor_cache: dict[int, frozenset[int] | None] = {}
    steps: list[frozenset[int]] = []
    metrics: list[_ContractionStepMetrics | None] = []
    step_result_nodes: list[Any] = []
    for node in ordered_contractions:
        contributors = _resolve_leaf_contributors(
            id(node),
            leaf_ids=leaf_ids,
            produced=produced,
            contributor_cache=contributor_cache,
        )
        if not contributors:
            return None
        if graph is not None:
            steps.append(frozenset(contributors))
        metrics.append(_metric_for_contraction_node(node, produced))
        step_result_nodes.append(node)

    if not step_result_nodes:
        return None
    resolved_metrics: tuple[_ContractionStepMetrics | None, ...] | None = None
    if any(metric is not None for metric in metrics):
        resolved_metrics = tuple(metrics)
    return _RecoveredHistory(
        contraction_steps=tuple(steps) if graph is not None else None,
        contraction_step_metrics=resolved_metrics,
        step_result_nodes=tuple(step_result_nodes),
    )


def _metric_for_contraction_node(
    node: Any,
    produced: dict[int, _ProducedNode],
) -> _ContractionStepMetrics | None:
    entry = produced.get(id(node))
    if entry is None or not _is_contraction_op(entry.op_name):
        return None
    if len(entry.parents) != 2:
        return None

    operand_axes: list[tuple[str, ...]] = []
    operand_shapes: list[tuple[int, ...]] = []
    operand_names: list[str] = []
    edge_label_by_id: dict[int, str] = {}
    next_label_index = 0

    for operand_index, parent in enumerate(entry.parents):
        shape = _node_shape(parent)
        edges = _node_edges(parent)
        if shape is None or edges is None:
            return None
        if len(shape) != len(edges):
            return None

        labels: list[str] = []
        for edge in edges:
            edge_id = id(edge)
            label = edge_label_by_id.get(edge_id)
            if label is None:
                label = _canonical_label(next_label_index)
                next_label_index += 1
                edge_label_by_id[edge_id] = label
            labels.append(label)
        operand_axes.append(tuple(labels))
        operand_shapes.append(shape)
        operand_names.append(_node_display_name(parent, fallback=f"Operand {operand_index}"))

    child_edges = _node_edges(node)
    child_shape = _node_shape(node)
    if child_edges is None or child_shape is None:
        return None
    if len(child_edges) != len(child_shape):
        return None

    output_labels: list[str] = []
    for edge in child_edges:
        label = edge_label_by_id.get(id(edge))
        if label is None:
            return None
        output_labels.append(label)

    label_counts = _label_counts(tuple(operand_axes))
    output_set = set(output_labels)
    for label, count in label_counts.items():
        if label not in output_set and count < 2:
            return None

    try:
        return metrics_for_labeled_operands(
            operand_axes=tuple(operand_axes),
            operand_shapes=tuple(operand_shapes),
            output_axes=tuple(output_labels),
            equation_snippet=(
                f"{_format_equation_operand(operand_axes[0])},"
                f"{_format_equation_operand(operand_axes[1])}->"
                f"{_format_equation_operand(tuple(output_labels))}"
            ),
            operand_names=tuple(operand_names),
        )
    except ValueError:
        return None


def _node_display_name(node: Any, *, fallback: str) -> str:
    name = getattr(node, "name", None)
    return fallback if name is None else str(name)


def _node_shape(node: Any) -> tuple[int, ...] | None:
    raw_shape = getattr(node, "shape", None)
    if raw_shape is None:
        tensor = getattr(node, "tensor", None)
        raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        return None
    try:
        return tuple(int(dim) for dim in raw_shape)
    except TypeError:
        return None


def _node_edges(node: Any) -> tuple[Any, ...] | None:
    raw = getattr(node, "edges", None)
    if raw is None:
        return None
    try:
        edges = tuple(raw)
    except TypeError:
        return None
    if any(edge is None for edge in edges):
        return None
    return edges


def _label_counts(operand_axes: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for axes in operand_axes:
        for label in axes:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _canonical_label(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet_size = len(alphabet)
    if index < alphabet_size:
        return alphabet[index]
    return f"{alphabet[index % alphabet_size]}{index // alphabet_size}"


def _format_equation_operand(labels: tuple[str, ...]) -> str:
    if not labels:
        return ""
    if all(len(label) == 1 for label in labels):
        return "".join(labels)
    return " ".join(labels)


def _unique_nodes_from_attr(source: Any, attr_name: str) -> list[Any]:
    if not hasattr(source, attr_name):
        return []
    raw = getattr(source, attr_name)
    iterable = raw.values() if isinstance(raw, dict) else raw
    if isinstance(iterable, (str, bytes, bytearray)):
        return []
    try:
        items = list(iterable)
    except TypeError:
        return []
    unique: list[Any] = []
    seen: set[int] = set()
    for item in items:
        if item is None:
            continue
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)
    return unique


def _unique_history_nodes(network: Any) -> tuple[Any, ...]:
    nodes: list[Any] = []
    for attr_name in ("nodes", "leaf_nodes", "resultant_nodes"):
        nodes.extend(_unique_nodes_from_attr(network, attr_name))

    unique: list[Any] = []
    seen: set[int] = set()
    for node in nodes:
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(node)
    return tuple(unique)


def _build_produced_node_map(network: Any) -> dict[int, _ProducedNode] | None:
    produced: dict[int, _ProducedNode] = {}
    for node in _unique_history_nodes(network):
        for op_name, successor in _iter_successors(node):
            parents = _normalized_parent_nodes(getattr(successor, "node_ref", None))
            if not parents:
                return None
            children = _normalized_child_nodes(getattr(successor, "child", None))
            if not children:
                return None
            parent_ids = tuple(id(parent) for parent in parents)
            if any(parent is None for parent in parents):
                return None
            for child in children:
                child_id = id(child)
                candidate = _ProducedNode(
                    op_name=str(op_name),
                    parent_ids=parent_ids,
                    parents=parents,
                )
                existing = produced.get(child_id)
                if existing is None:
                    produced[child_id] = candidate
                    continue
                if existing == candidate:
                    continue
                return None
    return produced


def _iter_successors(node: Any) -> tuple[tuple[str, Any], ...]:
    raw = getattr(node, "successors", None)
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        return ()
    out: list[tuple[str, Any]] = []
    for op_name, entries in raw.items():
        if isinstance(entries, dict):
            values = tuple(entries.values())
        elif isinstance(entries, Iterable) and not isinstance(entries, (str, bytes, bytearray)):
            values = tuple(entries)
        else:
            return ()
        for successor in values:
            out.append((str(op_name), successor))
    return tuple(out)


def _normalized_parent_nodes(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(item for item in value if item is not None)
    if isinstance(value, list):
        return tuple(item for item in value if item is not None)
    return (value,)


def _normalized_child_nodes(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(item for item in value if item is not None)
    if isinstance(value, list):
        return tuple(item for item in value if item is not None)
    return (value,)


def _is_contraction_op(op_name: str) -> bool:
    return op_name == "contract_edges" or op_name.startswith("contract_edges_")


def _topological_contraction_order(
    contraction_nodes: list[Any],
    produced: dict[int, _ProducedNode],
) -> list[Any] | None:
    contraction_ids = {id(node) for node in contraction_nodes}
    nodes_by_id = {id(node): node for node in contraction_nodes}
    order: list[Any] = []
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node_id: int) -> bool:
        if node_id in visited:
            return True
        if node_id in visiting:
            return False
        visiting.add(node_id)
        entry = produced.get(node_id)
        if entry is None or not _is_contraction_op(entry.op_name):
            return False
        for parent_id in entry.parent_ids:
            if parent_id not in contraction_ids:
                continue
            if not visit(parent_id):
                return False
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(nodes_by_id[node_id])
        return True

    for node in contraction_nodes:
        if not visit(id(node)):
            return None
    return order


def _resolve_leaf_contributors(
    node_id: int,
    *,
    leaf_ids: set[int],
    produced: dict[int, _ProducedNode],
    contributor_cache: dict[int, frozenset[int] | None],
) -> frozenset[int] | None:
    cached = contributor_cache.get(node_id)
    if cached is not None or node_id in contributor_cache:
        return cached
    if node_id in leaf_ids:
        resolved = frozenset({node_id})
        contributor_cache[node_id] = resolved
        return resolved

    entry = produced.get(node_id)
    if entry is None:
        contributor_cache[node_id] = None
        return None

    merged: set[int] = set()
    for parent_id in entry.parent_ids:
        parent_contributors = _resolve_leaf_contributors(
            parent_id,
            leaf_ids=leaf_ids,
            produced=produced,
            contributor_cache=contributor_cache,
        )
        if not parent_contributors:
            contributor_cache[node_id] = None
            return None
        merged.update(parent_contributors)

    resolved = frozenset(merged)
    contributor_cache[node_id] = resolved
    return resolved


__all__ = ["_RecoveredHistory", "_recover_contraction_history"]
