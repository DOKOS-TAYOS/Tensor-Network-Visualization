from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from itertools import tee
from typing import Any

from ._engine_specs import EngineName
from .einsum_module.trace import EinsumTrace, einsum_trace_step, pair_tensor
from .tenpy.explicit import TenPyTensorNetwork


def _first_non_none(items: Iterable[Any]) -> Any | None:
    for item in items:
        if item is not None:
            return item
    return None


def _peek_input_item(source: Any) -> tuple[Any | None, Any]:
    if isinstance(source, dict):
        return _first_non_none(source.values()), source
    if isinstance(source, (str, bytes, bytearray)) or not isinstance(source, Iterable):
        return None, source

    iterator = iter(source)
    if iterator is source:
        probe_iter, runtime_iter = tee(iterator)
        return _first_non_none(probe_iter), runtime_iter
    return _first_non_none(iterator), source


def _is_unordered_collection(value: Any) -> bool:
    return isinstance(value, AbstractSet) and not isinstance(value, (str, bytes, bytearray))


def _is_tenpy_tensor(value: Any) -> bool:
    return callable(getattr(value, "get_leg_labels", None)) and callable(
        getattr(value, "to_ndarray", None)
    )


def _is_tenpy_like(value: Any, *, include_tensor: bool = False) -> bool:
    return (
        isinstance(value, TenPyTensorNetwork)
        or callable(getattr(value, "get_W", None))
        or (callable(getattr(value, "get_X", None)) and hasattr(value, "uMPS_GS"))
        or callable(getattr(value, "get_B", None))
        or (include_tensor and _is_tenpy_tensor(value))
    )


def _detect_engine_from_network_sample(sample_item: Any | None) -> EngineName | None:
    if isinstance(sample_item, (pair_tensor, einsum_trace_step)):
        return "einsum"
    if hasattr(sample_item, "inds"):
        return "quimb"
    if hasattr(sample_item, "axes_names"):
        return "tensorkrowch"
    if hasattr(sample_item, "axis_names"):
        return "tensornetwork"
    return None


def _detect_engine_from_tensor_sample(sample_item: Any | None) -> EngineName | None:
    if isinstance(sample_item, (pair_tensor, einsum_trace_step)):
        return "einsum"
    if _is_tenpy_like(sample_item, include_tensor=True):
        return "tenpy"
    if hasattr(sample_item, "data") and hasattr(sample_item, "inds"):
        return "quimb"
    if hasattr(sample_item, "axes_names"):
        return "tensorkrowch"
    if hasattr(sample_item, "axis_names") and hasattr(sample_item, "tensor"):
        return "tensornetwork"
    return None


def _detect_network_engine_with_input(network: Any) -> tuple[EngineName, Any]:
    if isinstance(network, EinsumTrace):
        return "einsum", network
    if _is_tenpy_like(network):
        return "tenpy", network

    if hasattr(network, "tensors"):
        tensor_sample, _ = _peek_input_item(network.tensors)
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            return "quimb", network

    if hasattr(network, "leaf_nodes"):
        leaf_sample, _ = _peek_input_item(network.leaf_nodes)
        if leaf_sample is None or hasattr(leaf_sample, "axes_names"):
            return "tensorkrowch", network
    if hasattr(network, "nodes"):
        node_sample, _ = _peek_input_item(network.nodes)
        if node_sample is None or hasattr(node_sample, "axes_names"):
            return "tensorkrowch", network

    sample_item, sampled_network = _peek_input_item(network)
    detected_engine = _detect_engine_from_network_sample(sample_item)
    if detected_engine is not None:
        return detected_engine, sampled_network

    raise ValueError(
        "Could not infer tensor network engine from input of type "
        f"{type(network).__name__!r}. Pass engine= explicitly or provide a supported backend input."
    )


def _detect_tensor_engine_with_input(data: Any) -> tuple[EngineName, Any]:
    if isinstance(data, EinsumTrace):
        return "einsum", data
    if _is_tenpy_like(data, include_tensor=True):
        return "tenpy", data

    direct_engine = _detect_engine_from_tensor_sample(data)
    if direct_engine is not None:
        return direct_engine, data

    if hasattr(data, "tensors"):
        tensor_sample, _ = _peek_input_item(data.tensors)
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            return "quimb", data

    if hasattr(data, "leaf_nodes"):
        sample, _ = _peek_input_item(data.leaf_nodes)
        if sample is None or hasattr(sample, "axes_names"):
            return "tensorkrowch", data
    if hasattr(data, "nodes"):
        sample, _ = _peek_input_item(data.nodes)
        if sample is None or hasattr(sample, "axes_names"):
            return "tensorkrowch", data
        if sample is None or hasattr(sample, "axis_names"):
            return "tensornetwork", data

    sample_item, sampled_input = _peek_input_item(data)
    detected_engine = _detect_engine_from_tensor_sample(sample_item)
    if detected_engine is not None:
        return detected_engine, sampled_input

    raise ValueError(
        "Could not infer tensor engine from input of type "
        f"{type(data).__name__!r}. Pass engine= explicitly or provide a supported tensor input."
    )


__all__ = [
    "_detect_engine_from_network_sample",
    "_detect_engine_from_tensor_sample",
    "_detect_network_engine_with_input",
    "_detect_tensor_engine_with_input",
    "_first_non_none",
    "_is_tenpy_like",
    "_is_tenpy_tensor",
    "_is_unordered_collection",
    "_peek_input_item",
]
