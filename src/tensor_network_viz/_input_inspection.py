from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import cache
from itertools import tee
from typing import Any

from ._core.graph_utils import _is_unordered_collection
from ._engine_specs import EngineName
from ._logging import package_logger
from ._typing import PositionMapping
from .config import PlotConfig, ViewName
from .exceptions import TensorDataError, VisualizationInputError


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


class _PreparedGridInput:
    """Replayable flattened view of a nested 2D or 3D grid input."""

    def __init__(
        self,
        source: Any,
        *,
        flat_items: tuple[Any, ...],
        grid_dimensions: int,
        positions_2d: dict[int, tuple[float, float]],
        positions_3d: dict[int, tuple[float, float, float]],
    ) -> None:
        self._source = source
        self._flat_items = flat_items
        self.grid_dimensions = int(grid_dimensions)
        self.positions_2d = dict(positions_2d)
        self.positions_3d = dict(positions_3d)
        self._graph_cache_source = source

    def __iter__(self) -> Any:
        return iter(self._flat_items)


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


def _sample_matches(sample_item: Any | None, predicate: Callable[[Any], bool]) -> bool:
    return sample_item is not None and bool(predicate(sample_item))


def _peek_matching_attr(
    source: Any,
    attr_name: str,
    *,
    predicate: Callable[[Any], bool],
) -> tuple[bool, Any]:
    if not hasattr(source, attr_name):
        return False, source
    sample_item, prepared_source = _peek_input_attr(source, attr_name)
    return _sample_matches(sample_item, predicate), prepared_source


def _is_grid_container(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _raise_grid_error(message: str) -> None:
    raise VisualizationInputError(message)


def _append_grid_item(
    item: Any,
    *,
    flat_items: list[Any],
    seen_item_ids: set[int],
    positions_2d: dict[int, tuple[float, float]],
    positions_3d: dict[int, tuple[float, float, float]],
    position_2d: tuple[float, float],
    position_3d: tuple[float, float, float],
) -> None:
    if item is None:
        return
    item_id = id(item)
    if item_id in seen_item_ids:
        _raise_grid_error(
            "The same tensor or node object appears more than once in the grid input."
        )
    seen_item_ids.add(item_id)
    flat_items.append(item)
    positions_2d[item_id] = position_2d
    positions_3d[item_id] = position_3d


def _prepare_grid_input_2d(rows: tuple[Any, ...], *, source: Any) -> _PreparedGridInput:
    flat_items: list[Any] = []
    seen_item_ids: set[int] = set()
    positions_2d: dict[int, tuple[float, float]] = {}
    positions_3d: dict[int, tuple[float, float, float]] = {}

    for row_index, row in enumerate(rows):
        if row is None:
            _raise_grid_error(
                "Nested tensor-network grid input must use None only for cells, not for rows."
            )
        if not _is_grid_container(row):
            _raise_grid_error(
                "Mixed flat and nested tensor-network inputs are not supported. "
                "Use either a flat iterable, a 2D grid, or a 3D grid."
            )
        for col_index, item in enumerate(row):
            if _is_grid_container(item):
                _raise_grid_error("Nested tensor-network grids support only 2D or 3D cell layouts.")
            x_coord = float(col_index)
            y_coord = float(-row_index)
            _append_grid_item(
                item,
                flat_items=flat_items,
                seen_item_ids=seen_item_ids,
                positions_2d=positions_2d,
                positions_3d=positions_3d,
                position_2d=(x_coord, y_coord),
                position_3d=(x_coord, y_coord, 0.0),
            )

    if not flat_items:
        _raise_grid_error("Nested tensor-network grid input does not contain any tensors or nodes.")
    return _PreparedGridInput(
        source,
        flat_items=tuple(flat_items),
        grid_dimensions=2,
        positions_2d=positions_2d,
        positions_3d=positions_3d,
    )


def _prepare_grid_input_3d(layers: tuple[Any, ...], *, source: Any) -> _PreparedGridInput:
    from ._core.layout_structure import _GRID3D_PROJECTION_X, _GRID3D_PROJECTION_Y

    flat_items: list[Any] = []
    seen_item_ids: set[int] = set()
    positions_2d: dict[int, tuple[float, float]] = {}
    positions_3d: dict[int, tuple[float, float, float]] = {}

    for layer_index, layer in enumerate(layers):
        if layer is None:
            _raise_grid_error(
                "Nested tensor-network grid input must use None only for cells, not for layers."
            )
        if not _is_grid_container(layer):
            _raise_grid_error(
                "Mixed nested tensor-network grid depths are not supported. "
                "Use either a 2D grid or a 3D grid with consistent nesting."
            )
        for row_index, row in enumerate(layer):
            if row is None:
                _raise_grid_error(
                    "Nested tensor-network grid input must use None only for cells, not for rows."
                )
            if not _is_grid_container(row):
                _raise_grid_error(
                    "Mixed nested tensor-network grid depths are not supported. "
                    "Use either a 2D grid or a 3D grid with consistent nesting."
                )
            for col_index, item in enumerate(row):
                if _is_grid_container(item):
                    _raise_grid_error("Nested tensor-network grids support only 2D or 3D inputs.")
                x_coord = float(col_index)
                y_coord = float(-row_index)
                z_coord = float(layer_index)
                _append_grid_item(
                    item,
                    flat_items=flat_items,
                    seen_item_ids=seen_item_ids,
                    positions_2d=positions_2d,
                    positions_3d=positions_3d,
                    position_2d=(
                        x_coord + float(_GRID3D_PROJECTION_X) * z_coord,
                        y_coord + float(_GRID3D_PROJECTION_Y) * z_coord,
                    ),
                    position_3d=(x_coord, y_coord, z_coord),
                )

    if not flat_items:
        _raise_grid_error("Nested tensor-network grid input does not contain any tensors or nodes.")
    return _PreparedGridInput(
        source,
        flat_items=tuple(flat_items),
        grid_dimensions=3,
        positions_2d=positions_2d,
        positions_3d=positions_3d,
    )


def _prepare_network_input(network: Any) -> Any:
    if isinstance(network, _PreparedGridInput):
        return network
    if not _is_grid_container(network) or not network:
        return network

    top_level = tuple(network)
    has_nested_items = any(_is_grid_container(item) for item in top_level)
    if not has_nested_items:
        return network
    if any(item is None for item in top_level):
        _raise_grid_error(
            "Nested tensor-network grid input must use None only for cells, not for rows or layers."
        )
    if not all(_is_grid_container(item) for item in top_level):
        _raise_grid_error(
            "Mixed flat and nested tensor-network inputs are not supported. "
            "Use either a flat iterable, a 2D grid, or a 3D grid."
        )

    second_level_has_nested = False
    second_level_has_cells = False
    for row_or_layer in top_level:
        for item in row_or_layer:
            if _is_grid_container(item):
                second_level_has_nested = True
            else:
                second_level_has_cells = True

    if second_level_has_nested and second_level_has_cells:
        _raise_grid_error(
            "Mixed nested tensor-network grid depths are not supported. "
            "Use either a 2D grid or a 3D grid with consistent nesting."
        )
    if second_level_has_nested:
        return _prepare_grid_input_3d(top_level, source=network)
    return _prepare_grid_input_2d(top_level, source=network)


def _default_view_for_network_input(network: Any) -> ViewName | None:
    if isinstance(network, _PreparedGridInput) and network.grid_dimensions == 3:
        return "3d"
    return None


def _grid_positions_for_network_input(
    network: Any,
    *,
    dimensions: int,
) -> PositionMapping | None:
    if not isinstance(network, _PreparedGridInput):
        return None
    if dimensions == 2:
        return dict(network.positions_2d)
    return dict(network.positions_3d)


def _merge_grid_positions_into_config(
    config: PlotConfig,
    network: Any,
    *,
    dimensions: int,
) -> PlotConfig:
    grid_positions = _grid_positions_for_network_input(network, dimensions=dimensions)
    if grid_positions is None:
        return config
    merged_positions = dict(grid_positions)
    if config.positions is not None:
        merged_positions.update(config.positions)
    return replace(config, positions=merged_positions)


def _validate_grid_engine(network: Any, *, engine: EngineName) -> None:
    if not isinstance(network, _PreparedGridInput):
        return
    if engine in {"tensorkrowch", "tensornetwork", "quimb"}:
        return
    raise VisualizationInputError(
        f"Nested grid tensor-network inputs are not supported for engine {engine!r}."
    )


def _is_tenpy_tensor(value: Any) -> bool:
    return callable(getattr(value, "get_leg_labels", None)) and callable(
        getattr(value, "to_ndarray", None)
    )


@cache
def _einsum_trace_type() -> type[Any]:
    from .einsum_module.trace import EinsumTrace

    return EinsumTrace


@cache
def _einsum_trace_entry_types() -> tuple[type[Any], type[Any]]:
    from .einsum_module._trace_types import einsum_trace_step, pair_tensor

    return pair_tensor, einsum_trace_step


@cache
def _explicit_tenpy_network_type() -> type[Any]:
    from .tenpy.explicit import TenPyTensorNetwork

    return TenPyTensorNetwork


def _is_einsum_trace(value: Any) -> bool:
    return isinstance(value, _einsum_trace_type())


def _is_einsum_trace_entry(value: Any) -> bool:
    return isinstance(value, _einsum_trace_entry_types())


def _is_explicit_tenpy_network(value: Any) -> bool:
    return isinstance(value, _explicit_tenpy_network_type())


def _is_tenpy_like(value: Any, *, include_tensor: bool = False) -> bool:
    return (
        _is_explicit_tenpy_network(value)
        or callable(getattr(value, "get_W", None))
        or (callable(getattr(value, "get_X", None)) and hasattr(value, "uMPS_GS"))
        or callable(getattr(value, "get_B", None))
        or (include_tensor and _is_tenpy_tensor(value))
    )


def _detect_engine_from_network_sample(sample_item: Any | None) -> EngineName | None:
    if _is_einsum_trace_entry(sample_item):
        return "einsum"
    if hasattr(sample_item, "inds"):
        return "quimb"
    if hasattr(sample_item, "axes_names"):
        return "tensorkrowch"
    if hasattr(sample_item, "axis_names"):
        return "tensornetwork"
    return None


def _detect_engine_from_tensor_sample(sample_item: Any | None) -> EngineName | None:
    if _is_einsum_trace_entry(sample_item):
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
    if _is_einsum_trace(network):
        package_logger.debug("Auto-detected tensor network engine='einsum' from EinsumTrace.")
        return "einsum", network
    if _is_tenpy_like(network):
        package_logger.debug(
            "Auto-detected tensor network engine='tenpy' from %s.", type(network).__name__
        )
        return "tenpy", network

    prepared_network = _prepare_network_input(network)

    matched, prepared_network = _peek_matching_attr(
        prepared_network,
        "tensors",
        predicate=lambda item: hasattr(item, "inds"),
    )
    if matched:
        package_logger.debug(
            "Auto-detected tensor network engine='quimb' from %s.", type(network).__name__
        )
        return "quimb", prepared_network

    matched, prepared_network = _peek_matching_attr(
        prepared_network,
        "leaf_nodes",
        predicate=lambda item: hasattr(item, "axes_names"),
    )
    if matched:
        package_logger.debug("Auto-detected tensor network engine='tensorkrowch' from leaf_nodes.")
        return "tensorkrowch", prepared_network

    matched, prepared_network = _peek_matching_attr(
        prepared_network,
        "nodes",
        predicate=lambda item: hasattr(item, "axes_names"),
    )
    if matched:
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
    if _is_einsum_trace(data):
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

    matched, prepared_data = _peek_matching_attr(
        prepared_data,
        "tensors",
        predicate=lambda item: hasattr(item, "inds"),
    )
    if matched:
        package_logger.debug("Auto-detected tensor engine='quimb' from tensor container.")
        return "quimb", prepared_data

    matched, prepared_data = _peek_matching_attr(
        prepared_data,
        "leaf_nodes",
        predicate=lambda item: hasattr(item, "axes_names"),
    )
    if matched:
        package_logger.debug("Auto-detected tensor engine='tensorkrowch' from leaf_nodes.")
        return "tensorkrowch", prepared_data

    matched, prepared_data = _peek_matching_attr(
        prepared_data,
        "nodes",
        predicate=lambda item: hasattr(item, "axes_names"),
    )
    if matched:
        package_logger.debug("Auto-detected tensor engine='tensorkrowch' from nodes.")
        return "tensorkrowch", prepared_data

    matched, prepared_data = _peek_matching_attr(
        prepared_data,
        "nodes",
        predicate=lambda item: hasattr(item, "axis_names"),
    )
    if matched:
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
    "_default_view_for_network_input",
    "_first_non_none",
    "_grid_positions_for_network_input",
    "_is_tenpy_like",
    "_is_tenpy_tensor",
    "_is_unordered_collection",
    "_merge_grid_positions_into_config",
    "_peek_input_item",
    "_prepare_network_input",
    "_validate_grid_engine",
]
