"""Einsum trace normalization into the shared graph model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .._core.graph import (
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
)
from ._equation import _split_raw_equation, parse_equation_for_shapes
from .trace import _normalize_trace, pair_tensor


@dataclass(frozen=True)
class _AxisOrigin:
    node_id: int
    axis_index: int
    axis_name: str


def _operand_shape_for_pair(
    tensor_name: str,
    pair: pair_tensor,
    *,
    live_tensors: dict[str, tuple[_AxisOrigin, ...]],
    side: Literal["left", "right"],
) -> tuple[int, ...]:
    if tensor_name in live_tensors:
        n = len(live_tensors[tensor_name])
        return (1,) * n
    md = pair.metadata or {}
    key = "left_shape" if side == "left" else "right_shape"
    raw = md.get(key)
    if raw is not None:
        return tuple(int(dim) for dim in raw)
    eq = pair.equation.replace(" ", "")
    if "..." in eq:
        raise ValueError(
            f"pair_tensor for operand {tensor_name!r} needs metadata[{key!r}] "
            "when the equation contains ellipsis."
        )
    left_raw, right_raw, _ = _split_raw_equation(eq)
    spec = left_raw if side == "left" else right_raw
    return (1,) * len(spec)


def _axes_grouped(
    axes: tuple[str, ...],
    origins: tuple[_AxisOrigin, ...],
) -> dict[str, list[_AxisOrigin]]:
    by: dict[str, list[_AxisOrigin]] = {}
    for axis_name, origin in zip(axes, origins, strict=True):
        by.setdefault(axis_name, []).append(origin)
    return by


def _build_graph(trace_input: Any) -> _GraphData:
    trace = _normalize_trace(trace_input)
    nodes: dict[int, Any] = {}
    node_ids_by_name: dict[str, int] = {}
    live_tensors: dict[str, tuple[_AxisOrigin, ...]] = {}
    edges: list[Any] = []
    produced_names: set[str] = set()
    all_result_names = {pair.result_name for pair in trace}
    next_virtual_id = -1

    for pair in trace:
        if pair.left_name == pair.right_name:
            raise ValueError(
                "Binary einsum traces require distinct tensor names for both operands."
            )

        left_shape = _operand_shape_for_pair(
            pair.left_name,
            pair,
            live_tensors=live_tensors,
            side="left",
        )
        right_shape = _operand_shape_for_pair(
            pair.right_name,
            pair,
            live_tensors=live_tensors,
            side="right",
        )
        parsed = parse_equation_for_shapes(pair.equation, left_shape, right_shape)

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

        left_by_label = _axes_grouped(parsed.left_axes, left_origins)
        right_by_label = _axes_grouped(parsed.right_axes, right_origins)
        output_set = set(parsed.output_axes)

        operand_labels = set(left_by_label) | set(right_by_label)
        hub_output_registry: dict[str, _AxisOrigin] = {}

        for label in sorted(operand_labels):
            lhs = list(left_by_label.get(label, []))
            rhs_list = list(right_by_label.get(label, []))
            lhs_all = lhs + rhs_list
            n = len(lhs_all)
            if n == 0:
                continue
            if n == 1 and label not in output_set:
                side = pair.left_name if lhs else pair.right_name
                raise ValueError(
                    "Unary reductions are not supported in einsum traces: "
                    f"{label!r} from {side!r} disappears."
                )
            if n == 1:
                continue

            in_output = label in output_set
            # Normal pairwise contraction: one leg per operand, summation index (not in output).
            if n == 2 and not in_output:
                origin_a, origin_b = lhs_all[0], lhs_all[1]
                if origin_a.node_id != origin_b.node_id:
                    edges.append(
                        _make_contraction_edge(
                            _to_endpoint(origin_a),
                            _to_endpoint(origin_b),
                            name=label,
                        )
                    )
                    continue

            n_tensor_legs = n
            axis_names = tuple(f"{label}__branch_{k}" for k in range(n_tensor_legs))
            if in_output:
                # Open leg uses the equation index letter (no "__out" suffix).
                axis_names = axis_names + (label,)

            hub_id = next_virtual_id
            next_virtual_id -= 1
            nodes[hub_id] = _make_node(
                name="",
                axes_names=axis_names,
                label=label,
                is_virtual=True,
            )

            for branch_index, origin in enumerate(lhs_all):
                hub_ep = _EdgeEndpoint(
                    node_id=hub_id,
                    axis_index=branch_index,
                    axis_name=axis_names[branch_index],
                )
                edges.append(
                    _make_contraction_edge(
                        _to_endpoint(origin),
                        hub_ep,
                        name=label,
                    )
                )

            if in_output:
                out_idx = n_tensor_legs
                out_origin = _AxisOrigin(node_id=hub_id, axis_index=out_idx, axis_name=label)
                hub_output_registry[label] = out_origin

        live_tensors[pair.result_name] = tuple(
            _origin_for_output_label(
                label,
                hub_output_registry=hub_output_registry,
                left_by_label=left_by_label,
                right_by_label=right_by_label,
            )
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


def _origin_for_output_label(
    label: str,
    *,
    hub_output_registry: dict[str, _AxisOrigin],
    left_by_label: dict[str, list[_AxisOrigin]],
    right_by_label: dict[str, list[_AxisOrigin]],
) -> _AxisOrigin:
    if label in hub_output_registry:
        return hub_output_registry[label]
    lhs = left_by_label.get(label, []) + right_by_label.get(label, [])
    if len(lhs) == 1:
        return lhs[0]
    raise RuntimeError(
        f"Internal error: cannot resolve output axis {label!r} (expected hub or single occurrence)."
    )


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


def _to_endpoint(origin: _AxisOrigin) -> _EdgeEndpoint:
    return _EdgeEndpoint(
        node_id=origin.node_id,
        axis_index=origin.axis_index,
        axis_name=origin.axis_name,
    )
