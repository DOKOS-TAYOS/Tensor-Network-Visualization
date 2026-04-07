from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from itertools import tee
from typing import Any

from ._engine_specs import EngineName
from ._logging import package_logger
from .einsum_module.trace import EinsumTrace, einsum_trace_step, pair_tensor
from .exceptions import TensorDataError, VisualizationInputError
from .tenpy.explicit import TenPyTensorNetwork


class _PreparedInputProxy:
    """Delegate input access while swapping in replayable single-pass attributes."""

    def __init__(self, source: Any, overrides: dict[str, Any] | None = None) -> None:
        self._source = source
        self._overrides = {} if overrides is None else dict(overrides)

    def with_override(self, attr_name: str, attr_value: Any) -> _PreparedInputProxy:
        overrides = dict(self._overrides)
        overrides[attr_name] = attr_value
        return _PreparedInputProxy(self._source, overrides)

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self._overrides:
            return self._overrides[attr_name]
        return getattr(self._source, attr_name)

    def __iter__(self) -> Any:
        return iter(self._source)


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


def _peek_input_attr(source: Any, attr_name: str) -> tuple[Any | None, Any]:
    raw_value = getattr(source, attr_name)
    sample_item, prepared_value = _peek_input_item(raw_value)
    if prepared_value is raw_value:
        return sample_item, source
    if isinstance(source, _PreparedInputProxy):
        return sample_item, source.with_override(attr_name, prepared_value)
    return sample_item, _PreparedInputProxy(source, {attr_name: prepared_value})


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
        package_logger.debug("Auto-detected tensor network engine='einsum' from EinsumTrace.")
        return "einsum", network
    if _is_tenpy_like(network):
        package_logger.debug(
            "Auto-detected tensor network engine='tenpy' from %s.", type(network).__name__
        )
        return "tenpy", network

    prepared_network = network

    if hasattr(prepared_network, "tensors"):
        tensor_sample, prepared_network = _peek_input_attr(prepared_network, "tensors")
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            package_logger.debug(
                "Auto-detected tensor network engine='quimb' from %s.", type(network).__name__
            )
            return "quimb", prepared_network

    if hasattr(prepared_network, "leaf_nodes"):
        leaf_sample, prepared_network = _peek_input_attr(prepared_network, "leaf_nodes")
        if leaf_sample is None or hasattr(leaf_sample, "axes_names"):
            package_logger.debug(
                "Auto-detected tensor network engine='tensorkrowch' from leaf_nodes."
            )
            return "tensorkrowch", prepared_network
    if hasattr(prepared_network, "nodes"):
        node_sample, prepared_network = _peek_input_attr(prepared_network, "nodes")
        if node_sample is None or hasattr(node_sample, "axes_names"):
            package_logger.debug("Auto-detected tensor network engine='tensorkrowch' from nodes.")
            return "tensorkrowch", prepared_network

    sample_item, sampled_network = _peek_input_item(prepared_network)
    detected_engine = _detect_engine_from_network_sample(sample_item)
    if detected_engine is not None:
        package_logger.debug(
            "Auto-detected tensor network engine='%s' from sample type %s.",
            detected_engine,
            type(sample_item).__name__,
        )
        return detected_engine, sampled_network

    raise VisualizationInputError(
        "Could not infer tensor network engine from input of type "
        f"{type(network).__name__!r}. Pass engine= explicitly or provide a supported backend input."
    )


def _detect_tensor_engine_with_input(data: Any) -> tuple[EngineName, Any]:
    if isinstance(data, EinsumTrace):
        package_logger.debug("Auto-detected tensor engine='einsum' from EinsumTrace.")
        return "einsum", data
    if _is_tenpy_like(data, include_tensor=True):
        package_logger.debug("Auto-detected tensor engine='tenpy' from %s.", type(data).__name__)
        return "tenpy", data

    direct_engine = _detect_engine_from_tensor_sample(data)
    if direct_engine is not None:
        package_logger.debug(
            "Auto-detected tensor engine='%s' from direct input type %s.",
            direct_engine,
            type(data).__name__,
        )
        return direct_engine, data

    prepared_data = data

    if hasattr(prepared_data, "tensors"):
        tensor_sample, prepared_data = _peek_input_attr(prepared_data, "tensors")
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            package_logger.debug("Auto-detected tensor engine='quimb' from tensor container.")
            return "quimb", prepared_data

    if hasattr(prepared_data, "leaf_nodes"):
        sample, prepared_data = _peek_input_attr(prepared_data, "leaf_nodes")
        if sample is None or hasattr(sample, "axes_names"):
            package_logger.debug("Auto-detected tensor engine='tensorkrowch' from leaf_nodes.")
            return "tensorkrowch", prepared_data
    if hasattr(prepared_data, "nodes"):
        sample, prepared_data = _peek_input_attr(prepared_data, "nodes")
        if sample is None or hasattr(sample, "axes_names"):
            package_logger.debug("Auto-detected tensor engine='tensorkrowch' from nodes.")
            return "tensorkrowch", prepared_data
        if sample is None or hasattr(sample, "axis_names"):
            package_logger.debug("Auto-detected tensor engine='tensornetwork' from nodes.")
            return "tensornetwork", prepared_data

    sample_item, sampled_input = _peek_input_item(prepared_data)
    detected_engine = _detect_engine_from_tensor_sample(sample_item)
    if detected_engine is not None:
        package_logger.debug(
            "Auto-detected tensor engine='%s' from sample type %s.",
            detected_engine,
            type(sample_item).__name__,
        )
        return detected_engine, sampled_input

    raise TensorDataError(
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
