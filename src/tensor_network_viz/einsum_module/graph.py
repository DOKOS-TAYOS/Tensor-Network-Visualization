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
from ._equation import _binary_operand_specs_before_arrow, parse_einsum_equation
from .trace import _normalize_trace, einsum_trace_step, pair_tensor


@dataclass(frozen=True)
class _AxisOrigin:
    node_id: int
    axis_index: int
    axis_name: str


def _trace_step_operand_names(step: pair_tensor | einsum_trace_step) -> tuple[str, ...]:
    if isinstance(step, pair_tensor):
        return (step.left_name, step.right_name)
    return step.operand_names


def _equation_string(step: pair_tensor | einsum_trace_step) -> str:
    if isinstance(step, pair_tensor):
        return str(step)
    return step.equation


def _operand_shape_for_step(
    tensor_name: str,
    step: pair_tensor | einsum_trace_step,
    *,
    live_tensors: dict[str, tuple[_AxisOrigin, ...]],
    operand_index: int,
) -> tuple[int, ...]:
    if tensor_name in live_tensors:
        n = len(live_tensors[tensor_name])
        return (1,) * n
    md = step.metadata or {}
    raw_list = md.get("operand_shapes")
    if raw_list is not None:
        raw = raw_list[operand_index]
        return tuple(int(dim) for dim in raw)
    if isinstance(step, pair_tensor):
        key = "left_shape" if operand_index == 0 else "right_shape"
        raw = md.get(key)
        if raw is not None:
            return tuple(int(dim) for dim in raw)
    eq = _equation_string(step).replace(" ", "")
    if "..." in eq:
        side = "left_shape" if operand_index == 0 else "right_shape"
        raise ValueError(
            f"pair_tensor for operand {tensor_name!r} needs metadata[{side!r}] "
            "when the equation contains ellipsis."
        )
    operands_part = eq.split("->", 1)[0] if "->" in eq else eq
    specs = operands_part.split(",")
    if len(specs) == 2 and operand_index < 2:
        left_raw, right_raw = _binary_operand_specs_before_arrow(eq)
        spec = left_raw if operand_index == 0 else right_raw
    else:
        spec = specs[operand_index]
    return (1,) * len(spec)


def _build_graph(trace_input: Any) -> _GraphData:
    trace = _normalize_trace(trace_input)
    nodes: dict[int, Any] = {}
    node_ids_by_name: dict[str, int] = {}
    live_tensors: dict[str, tuple[_AxisOrigin, ...]] = {}
    edges: list[Any] = []
    produced_names: set[str] = set()
    all_result_names = {_step_result_name(s) for s in trace}
    next_virtual_id = -1
    contraction_scheme: list[frozenset[int]] = []
    # Full physical lineage per tensor (for the final step’s global highlight).
    physical_contributors: dict[str, frozenset[int]] = {}

    n_steps = len(trace)
    for step_index, step in enumerate(trace):
        operand_names = _trace_step_operand_names(step)
        if len(set(operand_names)) != len(operand_names):
            raise ValueError("Einsum traces require distinct tensor names for all operands.")

        shapes = tuple(
            _operand_shape_for_step(
                operand_names[i],
                step,
                live_tensors=live_tensors,
                operand_index=i,
            )
            for i in range(len(operand_names))
        )
        eq_str = _equation_string(step)
        parsed = parse_einsum_equation(eq_str, shapes)

        res_name = _step_result_name(step)
        if res_name in node_ids_by_name or res_name in produced_names:
            raise ValueError(f"Result name {res_name!r} must be new for each trace step.")
        if res_name in set(operand_names):
            raise ValueError(f"Result name {res_name!r} must be new (not an operand name).")

        operand_origins: list[tuple[_AxisOrigin, ...]] = []
        for i, name in enumerate(operand_names):
            operand_origins.append(
                _consume_or_create_tensor(
                    name,
                    parsed.operand_axes[i],
                    nodes=nodes,
                    node_ids_by_name=node_ids_by_name,
                    live_tensors=live_tensors,
                    produced_names=produced_names,
                    all_result_names=all_result_names,
                    physical_contributors=physical_contributors,
                )
            )

        if step_index == n_steps - 1:
            step_physical_last: set[int] = set()
            for name in operand_names:
                step_physical_last.update(physical_contributors[name])
            contraction_scheme.append(frozenset(step_physical_last))
        elif _trace_step_needs_peps_lineage_union(operand_names):
            # PEPS-style sweep: keep environment tensors (x**) in lineage when contracting with P**
            # or when applying the next local x** (MPS uses x0/x1 without P and stays on immediate).
            step_peps: set[int] = set()
            for name in operand_names:
                step_peps.update(physical_contributors[name])
            contraction_scheme.append(frozenset(step_peps))
        else:
            contraction_scheme.append(
                _participant_ids_for_contraction_step(operand_origins, nodes)
            )

        by_label: dict[str, list[_AxisOrigin]] = {}
        for i, axes in enumerate(parsed.operand_axes):
            for axis_name, origin in zip(axes, operand_origins[i], strict=True):
                by_label.setdefault(axis_name, []).append(origin)

        output_set = set(parsed.output_axes)
        hub_output_registry: dict[str, _AxisOrigin] = {}

        for label in sorted(by_label):
            lhs_all = list(by_label[label])
            n = len(lhs_all)
            if n == 0:
                continue
            if n == 1 and label not in output_set:
                raise ValueError(
                    "Unary reductions are not supported in einsum traces: "
                    f"{label!r} disappears (no corresponding output index)."
                )
            if n == 1:
                continue

            in_output = label in output_set
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

        live_tensors[res_name] = tuple(
            _origin_for_output_label(
                label,
                hub_output_registry=hub_output_registry,
                by_label=by_label,
            )
            for label in parsed.output_axes
        )
        merged_lineage: set[int] = set()
        for name in operand_names:
            merged_lineage.update(physical_contributors[name])
        physical_contributors[res_name] = frozenset(merged_lineage)
        produced_names.add(res_name)

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

    return _GraphData(
        nodes=nodes,
        edges=tuple(edges),
        contraction_steps=tuple(contraction_scheme),
    )


def _trace_step_has_peps_plaquette_operand(operand_names: tuple[str, ...]) -> bool:
    """True if a plaquette-like tensor name (``P`` + digits, e.g. ``P00``, ``P01``) is an operand."""
    for n in operand_names:
        if len(n) < 2 or n[0] != "P":
            continue
        if any(ch.isdigit() for ch in n[1:]):
            return True
    return False


def _trace_step_has_peps_environment_operand(operand_names: tuple[str, ...]) -> bool:
    """True if an operand looks like PEPS row/col env vectors (``x`` + digits, e.g. ``x00``)."""
    for n in operand_names:
        if len(n) < 2 or n[0].lower() != "x":
            continue
        if any(ch.isdigit() for ch in n[1:]):
            return True
    return False


def _trace_step_needs_peps_lineage_union(operand_names: tuple[str, ...]) -> bool:
    return _trace_step_has_peps_plaquette_operand(
        operand_names
    ) or _trace_step_has_peps_environment_operand(operand_names)


def _participant_ids_for_contraction_step(
    operand_origins: list[tuple[_AxisOrigin, ...]],
    nodes: dict[int, Any],
) -> frozenset[int]:
    """Non-virtual node ids appearing on operand axes for this einsum only (immediate footprint)."""
    all_ids: set[int] = set()
    for origins in operand_origins:
        for origin in origins:
            all_ids.add(origin.node_id)
    non_virtual = {nid for nid in all_ids if not nodes[nid].is_virtual}
    if non_virtual:
        return frozenset(non_virtual)
    return frozenset(all_ids)


def _step_result_name(step: pair_tensor | einsum_trace_step) -> str:
    return step.result_name


def _origin_for_output_label(
    label: str,
    *,
    hub_output_registry: dict[str, _AxisOrigin],
    by_label: dict[str, list[_AxisOrigin]],
) -> _AxisOrigin:
    if label in hub_output_registry:
        return hub_output_registry[label]
    group = by_label.get(label, [])
    if len(group) == 1:
        return group[0]
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
    physical_contributors: dict[str, frozenset[int]],
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
    physical_contributors[tensor_name] = frozenset({node_id})
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
