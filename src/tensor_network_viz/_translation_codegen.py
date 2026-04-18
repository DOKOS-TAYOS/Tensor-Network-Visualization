"""Private code generators for translated tensor-network exports."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._engine_specs import TranslationTargetName
from ._translation_models import _TranslatedEdge, _TranslatedNetwork, _TranslatedTensor


def _shape_literal(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "()"
    return repr(shape)


def _dtype_literal(dtype_text: str | None) -> str:
    if dtype_text is None:
        return 'np.dtype("float64")'
    return f"np.dtype({dtype_text!r})"


def _array_literal(array: np.ndarray[Any, Any]) -> str:
    return f"np.array({array.tolist()!r}, dtype={_dtype_literal(str(array.dtype))})"


def _placeholder_literal(tensor: _TranslatedTensor) -> str:
    if tensor.shape is None:
        return f"np.array(1.0, dtype={_dtype_literal(tensor.dtype_text)})"
    return f"np.ones({_shape_literal(tensor.shape)}, dtype={_dtype_literal(tensor.dtype_text)})"


def _tensor_value_literal(tensor: _TranslatedTensor) -> str:
    if tensor.array is not None:
        return _array_literal(tensor.array)
    return _placeholder_literal(tensor)


def _node_var(tensor: _TranslatedTensor) -> str:
    return f"node_{tensor.variable_name}"


def _data_var(tensor: _TranslatedTensor) -> str:
    return f"tensor_{tensor.variable_name}"


def _tensor_creation_lines(tensor: _TranslatedTensor) -> list[str]:
    return [
        f"    {_data_var(tensor)} = {_tensor_value_literal(tensor)}",
        (
            f"    {_node_var(tensor)} = tn.Node("
            f"{_data_var(tensor)}, name={tensor.display_name!r}, axis_names={tensor.axis_names!r})"
        ),
    ]


def _connection_line(edge: _TranslatedEdge) -> str | None:
    if edge.kind not in {"contraction", "self"} or len(edge.endpoints) != 2:
        return None
    (left_var, left_axis), (right_var, right_axis) = edge.endpoints
    return f"    {left_var}[{left_axis}] ^ {right_var}[{right_axis}]"


def _generate_tensornetwork_code(model: _TranslatedNetwork) -> str:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import numpy as np",
        "import tensornetwork as tn",
        "",
        "",
        "def build_tensor_network() -> Any:",
    ]
    for tensor in model.tensors:
        lines.extend(_tensor_creation_lines(tensor))
    for edge in model.edges:
        connection = _connection_line(edge)
        if connection is not None:
            lines.append(connection)
    node_vars = ", ".join(_node_var(tensor) for tensor in model.tensors)
    lines.extend(
        [
            f"    return [{node_vars}]",
            "",
            "",
            "network = build_tensor_network()",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_quimb_code(model: _TranslatedNetwork) -> str:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import numpy as np",
        "import quimb.tensor as qtn",
        "",
        "",
        "def build_tensor_network() -> Any:",
        "    tensors = []",
    ]
    for tensor in model.tensors:
        lines.extend(
            [
                f"    {_data_var(tensor)} = {_tensor_value_literal(tensor)}",
                (
                    f"    qtn_{tensor.variable_name} = qtn.Tensor("
                    f"{_data_var(tensor)}, "
                    f"inds={tensor.index_labels!r}, "
                    f"tags={{{tensor.display_name!r}}})"
                ),
                f"    tensors.append(qtn_{tensor.variable_name})",
            ]
        )
    lines.extend(
        [
            "    return qtn.TensorNetwork(tensors)",
            "",
            "",
            "network = build_tensor_network()",
            "",
        ]
    )
    return "\n".join(lines)


def _connected_component_count(model: _TranslatedNetwork) -> int:
    parents = {tensor.variable_name: tensor.variable_name for tensor in model.tensors}

    def find(name: str) -> str:
        parent = parents[name]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        parents[name] = parent
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for edge in model.edges:
        if edge.kind != "contraction" or len(edge.endpoints) != 2:
            continue
        (left_var, _left_axis), (right_var, _right_axis) = edge.endpoints
        if left_var != right_var:
            union(left_var.removeprefix("node_"), right_var.removeprefix("node_"))
    return len({find(tensor.variable_name) for tensor in model.tensors})


def _torch_dtype_literal(dtype_text: str | None) -> str:
    normalized = "" if dtype_text is None else dtype_text.split(".")[-1]
    mapping = {
        "bool": "torch.bool",
        "complex128": "torch.complex128",
        "complex64": "torch.complex64",
        "float16": "torch.float16",
        "float32": "torch.float32",
        "float64": "torch.float64",
        "int16": "torch.int16",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "int8": "torch.int8",
        "uint8": "torch.uint8",
    }
    return mapping.get(normalized, "torch.float64")


def _torch_value_literal(tensor: _TranslatedTensor) -> str:
    dtype_literal = _torch_dtype_literal(tensor.dtype_text)
    if tensor.array is not None:
        return f"torch.tensor({tensor.array.tolist()!r}, dtype={dtype_literal})"
    if tensor.shape is None:
        return f"torch.tensor(1.0, dtype={dtype_literal})"
    return f"torch.ones({_shape_literal(tensor.shape)}, dtype={dtype_literal})"


def _generate_tensorkrowch_code(model: _TranslatedNetwork) -> str:
    if _connected_component_count(model) > 1:
        raise ValueError(
            "translate_tensor_network cannot export to tensorkrowch when the structure would "
            "require an outer product across disconnected components."
        )

    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import tensorkrowch as tk",
        "import torch",
        "",
        "",
        "def build_tensor_network() -> Any:",
        "    network = tk.TensorNetwork(name='translated')",
    ]
    for tensor in model.tensors:
        lines.extend(
            [
                f"    {_data_var(tensor)} = {_torch_value_literal(tensor)}",
                (
                    f"    {_node_var(tensor)} = tk.Node("
                    f"shape=tuple({_data_var(tensor)}.shape), "
                    f"axes_names={tensor.axis_names!r}, "
                    f"name={tensor.display_name!r}, "
                    f"network=network, "
                    f"tensor={_data_var(tensor)})"
                ),
            ]
        )
    for edge in model.edges:
        connection = _connection_line(edge)
        if connection is not None:
            lines.append(connection)
    lines.extend(
        [
            "    return network",
            "",
            "",
            "network = build_tensor_network()",
            "",
        ]
    )
    return "\n".join(lines)


def _einsum_symbol_pool() -> tuple[str, ...]:
    return tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _einsum_symbol_map(model: _TranslatedNetwork) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    pool_iter = iter(_einsum_symbol_pool())
    for tensor in model.tensors:
        for label in tensor.index_labels:
            if label in mapping:
                continue
            if len(label) == 1 and label.isalpha() and label not in used:
                mapping[label] = label
                used.add(label)
                continue
            for candidate in pool_iter:
                if candidate not in used:
                    mapping[label] = candidate
                    used.add(candidate)
                    break
            else:
                raise ValueError("Not enough distinct einsum labels for translation.")
    return mapping


def _generate_einsum_connectivity_code(model: _TranslatedNetwork) -> str:
    symbol_map = _einsum_symbol_map(model)
    operand_specs = [
        "".join(symbol_map[label] for label in tensor.index_labels) for tensor in model.tensors
    ]
    output_spec = "".join(symbol_map[label] for label in model.open_labels)
    equation = f"{','.join(operand_specs)}->{output_spec}"
    operand_vars = ", ".join(_data_var(tensor) for tensor in model.tensors)

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
    for tensor in model.tensors:
        lines.extend(
            [
                f"    {_data_var(tensor)} = {_tensor_value_literal(tensor)}",
                f"    trace.bind({tensor.display_name!r}, {_data_var(tensor)})",
            ]
        )
    lines.extend(
        [
            f"    _ = einsum({equation!r}, {operand_vars}, trace=trace, backend='numpy')",
            "    return trace",
            "",
            "",
            "network = build_tensor_network()",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_translation_code(
    model: _TranslatedNetwork,
    *,
    target_engine: TranslationTargetName,
) -> str:
    if target_engine == "tensornetwork":
        return _generate_tensornetwork_code(model)
    if target_engine == "quimb":
        return _generate_quimb_code(model)
    if target_engine == "einsum":
        return _generate_einsum_connectivity_code(model)
    if target_engine == "tensorkrowch":
        return _generate_tensorkrowch_code(model)
    raise ValueError(
        f"translate_tensor_network does not support target_engine={target_engine!r} yet."
    )


__all__ = ["_generate_translation_code"]
