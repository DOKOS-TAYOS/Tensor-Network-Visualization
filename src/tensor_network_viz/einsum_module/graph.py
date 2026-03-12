"""Einsum trace normalization into the shared graph model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from .trace import _normalize_trace, _parse_equation, pair_tensor


@dataclass(frozen=True)
class _AxisOrigin:
    node_id: int
    axis_index: int
    axis_name: str


def _build_graph(trace_input: Any) -> _GraphData:
    trace = _normalize_trace(trace_input)
    nodes: dict[int, Any] = {}
    node_ids_by_name: dict[str, int] = {}
    live_tensors: dict[str, tuple[_AxisOrigin, ...]] = {}
    edges = []
    produced_names: set[str] = set()
    all_result_names = {pair.result_name for pair in trace}

    for pair in trace:
        if pair.left_name == pair.right_name:
            raise ValueError(
                "Binary einsum traces require distinct tensor names for both operands."
            )

        parsed = _parse_equation(pair.equation)
        if pair.result_name in node_ids_by_name or pair.result_name in produced_names:
            raise ValueError(f"Result name {pair.result_name!r} must be new for each pair_tensor.")
        if pair.result_name in {pair.left_name, pair.right_name}:
            raise ValueError(f"Result name {pair.result_name!r} must be new for each pair_tensor.")

        left_origins = _consume_or_create_tensor(
            pair.left_name,
            parsed.left_axes,
            nodes=nodes,
            node_ids_by_name=node_ids_by_name,
            live_tensors=live_tensors,
            produced_names=produced_names,
            all_result_names=all_result_names,
        )
        right_origins = _consume_or_create_tensor(
            pair.right_name,
            parsed.right_axes,
            nodes=nodes,
            node_ids_by_name=node_ids_by_name,
            live_tensors=live_tensors,
            produced_names=produced_names,
            all_result_names=all_result_names,
        )

        left_by_label = dict(zip(parsed.left_axes, left_origins, strict=True))
        right_by_label = dict(zip(parsed.right_axes, right_origins, strict=True))
        output_set = set(parsed.output_axes)

        _validate_supported_labels(
            pair=pair,
            parsed=parsed,
            left_by_label=left_by_label,
            right_by_label=right_by_label,
            output_set=output_set,
        )

        for label in parsed.left_axes:
            if label in right_by_label and label not in output_set:
                left_origin = left_by_label[label]
                right_origin = right_by_label[label]
                edges.append(
                    _make_contraction_edge(
                        _to_endpoint(left_origin),
                        _to_endpoint(right_origin),
                        name=label,
                    )
                )

        live_tensors[pair.result_name] = tuple(
            left_by_label[label] if label in left_by_label else right_by_label[label]
            for label in parsed.output_axes
        )
        produced_names.add(pair.result_name)

    for origins in live_tensors.values():
        for origin in origins:
            endpoint = _to_endpoint(origin)
            edges.append(
                _make_dangling_edge(
                    endpoint,
                    name=origin.axis_name or None,
                    label=origin.axis_name or None,
                )
            )

    return _GraphData(nodes=nodes, edges=tuple(edges))


def _consume_or_create_tensor(
    tensor_name: str,
    axis_labels: tuple[str, ...],
    *,
    nodes: dict[int, Any],
    node_ids_by_name: dict[str, int],
    live_tensors: dict[str, tuple[_AxisOrigin, ...]],
    produced_names: set[str],
    all_result_names: set[str],
) -> tuple[_AxisOrigin, ...]:
    if tensor_name in live_tensors:
        origins = live_tensors.pop(tensor_name)
        if len(origins) != len(axis_labels):
            raise ValueError(
                f"Tensor {tensor_name!r} exposes {len(origins)} live axes, but "
                f"the equation expects {len(axis_labels)}."
            )
        return origins

    if tensor_name in all_result_names:
        raise ValueError(f"Tensor {tensor_name!r} is referenced before it is defined.")

    if tensor_name in produced_names or tensor_name in node_ids_by_name:
        raise ValueError(
            f"Tensor {tensor_name!r} is not available because it was already consumed."
        )

    node_id = len(nodes)
    node_ids_by_name[tensor_name] = node_id
    nodes[node_id] = _make_node(
        name=tensor_name,
        axes_names=axis_labels,
    )
    return tuple(
        _AxisOrigin(node_id=node_id, axis_index=index, axis_name=label)
        for index, label in enumerate(axis_labels)
    )


def _validate_supported_labels(
    *,
    pair: pair_tensor,
    parsed: Any,
    left_by_label: dict[str, _AxisOrigin],
    right_by_label: dict[str, _AxisOrigin],
    output_set: set[str],
) -> None:
    for label in parsed.left_axes:
        if label not in right_by_label and label not in output_set:
            raise ValueError(
                "Unary reductions are not supported in einsum traces: "
                f"{label!r} from {pair.left_name!r} disappears."
            )
    for label in parsed.right_axes:
        if label not in left_by_label and label not in output_set:
            raise ValueError(
                "Unary reductions are not supported in einsum traces: "
                f"{label!r} from {pair.right_name!r} disappears."
            )


def _to_endpoint(origin: _AxisOrigin) -> _EdgeEndpoint:
    return _EdgeEndpoint(
        node_id=origin.node_id,
        axis_index=origin.axis_index,
        axis_name=origin.axis_name,
    )
