"""Public code generation helpers for tensor-network translations."""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

from ._core.graph_cache import _get_or_build_graph
from ._engine_specs import EngineName, TranslationTargetName
from ._input_inspection import (
    _detect_network_engine_with_input,
    _prepare_network_input,
    _validate_grid_engine,
)
from ._registry import _get_graph_builder
from ._tensor_elements_inputs import _extract_tensor_records
from ._translation_codegen import _generate_translation_code
from ._translation_models import _TranslatedEdge, _TranslatedNetwork, _TranslatedTensor
from .einsum_module.trace import _normalize_trace
from .exceptions import TensorDataError, TensorDataTypeError


def _sanitized_variable_name(name: str, *, index: int) -> str:
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not candidate:
        candidate = f"tensor_{index}"
    if candidate[0].isdigit():
        candidate = f"tensor_{candidate}"
    return candidate


def _resolved_source_engine(
    network: Any,
    *,
    engine: EngineName | None,
) -> tuple[EngineName, Any]:
    prepared_input = _prepare_network_input(network)
    resolved_engine = engine
    if resolved_engine is None:
        resolved_engine, prepared_input = _detect_network_engine_with_input(prepared_input)
    _validate_grid_engine(prepared_input, engine=resolved_engine)
    return resolved_engine, prepared_input


def _tensor_arrays_by_name(
    prepared_input: Any,
    *,
    engine: EngineName,
) -> dict[str, list[Any]]:
    try:
        _, records = _extract_tensor_records(prepared_input, engine=engine)
    except (TensorDataError, TensorDataTypeError):
        return {}
    arrays_by_name: dict[str, list[Any]] = {}
    for record in records:
        arrays_by_name.setdefault(record.name, []).append(record.array)
    return arrays_by_name


def _index_labels_for_edges(
    graph_edges: tuple[Any, ...],
) -> tuple[dict[tuple[int, int], str], tuple[_TranslatedEdge, ...], tuple[str, ...]]:
    endpoint_labels: dict[tuple[int, int], str] = {}
    translated_edges: list[_TranslatedEdge] = []
    open_labels: list[str] = []
    used_labels: set[str] = set()
    for edge_index, edge in enumerate(graph_edges):
        axis_names = tuple(
            str(endpoint.axis_name) for endpoint in edge.endpoints if endpoint.axis_name is not None
        )
        if edge.label is not None and str(edge.label):
            label = str(edge.label)
        elif axis_names and len(set(axis_names)) == 1:
            label = axis_names[0]
        elif edge.name is not None and str(edge.name):
            label = str(edge.name)
        else:
            label = f"i{edge_index}"
        if label in used_labels:
            label = f"{label}_{edge_index}"
        used_labels.add(label)
        translated_edge = _TranslatedEdge(
            kind=edge.kind,
            label=label,
            endpoints=(),
        )
        translated_edges.append(translated_edge)
        endpoint_labels.update(
            {
                (int(endpoint.node_id), int(endpoint.axis_index)): label
                for endpoint in edge.endpoints
            }
        )
        if edge.kind == "dangling":
            open_labels.append(label)
    return endpoint_labels, tuple(translated_edges), tuple(open_labels)


def _build_translation_model(
    prepared_input: Any,
    *,
    engine: EngineName,
) -> _TranslatedNetwork:
    build_graph = _get_graph_builder(engine)
    graph = _get_or_build_graph(prepared_input, build_graph)
    arrays_by_name = _tensor_arrays_by_name(prepared_input, engine=engine)
    endpoint_labels, translated_edges_base, open_labels = _index_labels_for_edges(graph.edges)

    tensors: list[_TranslatedTensor] = []
    node_name_to_var: dict[int, str] = {}
    for index, (node_id, node) in enumerate(graph.nodes.items()):
        variable_name = _sanitized_variable_name(node.name, index=index)
        node_name_to_var[node_id] = variable_name
        array = None
        if node.name in arrays_by_name and arrays_by_name[node.name]:
            array = arrays_by_name[node.name].pop(0)
        index_labels = tuple(
            endpoint_labels.get((int(node_id), axis_index), f"open_{index}_{axis_index}")
            for axis_index, _axis_name in enumerate(node.axes_names)
        )
        tensors.append(
            _TranslatedTensor(
                node_id=int(node_id),
                variable_name=variable_name,
                display_name=node.name,
                axis_names=tuple(str(axis_name) for axis_name in node.axes_names),
                index_labels=index_labels,
                shape=None if node.shape is None else tuple(int(size) for size in node.shape),
                dtype_text=node.dtype,
                array=array,
            )
        )

    translated_edges: list[_TranslatedEdge] = []
    for edge, translated_edge_base in zip(graph.edges, translated_edges_base, strict=True):
        translated_edges.append(
            _TranslatedEdge(
                kind=translated_edge_base.kind,
                label=translated_edge_base.label,
                endpoints=tuple(
                    (f"node_{node_name_to_var[int(endpoint.node_id)]}", int(endpoint.axis_index))
                    for endpoint in edge.endpoints
                ),
            )
        )

    contraction_steps = None
    if graph.contraction_steps is not None:
        contraction_steps = tuple(
            tuple(node_name_to_var[int(node_id)] for node_id in sorted(step))
            for step in graph.contraction_steps
        )
    return _TranslatedNetwork(
        source_engine=engine,
        tensors=tuple(tensors),
        edges=tuple(translated_edges),
        open_labels=open_labels,
        contraction_steps=contraction_steps,
    )


def _array_literal(array: Any) -> str:
    return f"np.array({array.tolist()!r}, dtype=np.dtype({str(array.dtype)!r}))"


def _ordered_einsum_code(prepared_input: Any) -> str:
    steps = _normalize_trace(prepared_input)
    arrays_by_name = _tensor_arrays_by_name(prepared_input, engine="einsum")
    result_names = {step.result_name for step in steps}
    initial_names: list[str] = []
    for step in steps:
        for operand_name in step.operand_names:
            if operand_name not in result_names and operand_name not in initial_names:
                initial_names.append(operand_name)

    if not all(name in arrays_by_name and arrays_by_name[name] for name in initial_names):
        return ""

    used_vars: set[str] = set()

    def variable_name(name: str) -> str:
        candidate = _sanitized_variable_name(name, index=len(used_vars))
        while candidate in used_vars:
            candidate = f"{candidate}_{len(used_vars)}"
        used_vars.add(candidate)
        return candidate

    initial_vars = {name: variable_name(f"tensor_{name}") for name in initial_names}
    result_vars = {
        step.result_name: variable_name(step.result_name)
        for step in steps
        if step.result_name not in initial_vars
    }
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import numpy as np",
        "from tensor_network_viz import EinsumTrace, einsum",
        "",
        "",
        "def build_tensor_network() -> Any:",
        "    trace = EinsumTrace()",
    ]
    for name in initial_names:
        array = arrays_by_name[name].pop(0)
        var_name = initial_vars[name]
        lines.extend(
            [
                f"    {var_name} = {_array_literal(array)}",
                f"    trace.bind({name!r}, {var_name})",
            ]
        )
    available_vars = dict(initial_vars)
    available_vars.update(result_vars)
    for step in steps:
        metadata = {} if step.metadata is None else dict(step.metadata)
        backend = str(metadata.get("backend", "numpy"))
        operand_vars = ", ".join(available_vars[name] for name in step.operand_names)
        result_var = available_vars[step.result_name]
        lines.append(
            f"    {result_var} = einsum("
            f"{step.equation!r}, {operand_vars}, trace=trace, backend={backend!r})"
        )
    lines.extend(
        [
            "    return trace",
            "",
            "",
            "network = build_tensor_network()",
            "",
        ]
    )
    return "\n".join(lines)


def translate_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
    target_engine: TranslationTargetName,
    path: str | PathLike[str] | None = None,
) -> str:
    """Generate Python code that reconstructs a tensor network in another engine."""
    if target_engine == "tenpy":
        raise ValueError("translate_tensor_network does not support target_engine='tenpy'.")
    resolved_engine, prepared_input = _resolved_source_engine(network, engine=engine)
    code = ""
    if target_engine == "einsum" and resolved_engine == "einsum":
        code = _ordered_einsum_code(prepared_input)
    if not code:
        model = _build_translation_model(prepared_input, engine=resolved_engine)
        code = _generate_translation_code(model, target_engine=target_engine)
    if path is not None:
        Path(path).write_text(code, encoding="utf-8")
    return code


__all__ = ["translate_tensor_network"]
