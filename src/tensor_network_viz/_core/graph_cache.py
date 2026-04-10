"""Reuse normalized graphs across repeated draws of the same network object.

Covers preview + export and 2D + 3D.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable
from typing import Any

from .._logging import package_logger
from .graph import _GraphData

_graph_weak_cache: weakref.WeakKeyDictionary[Any, dict[int, _GraphData]] = (
    weakref.WeakKeyDictionary()
)
_builtin_container_cache: dict[tuple[int, int], tuple[tuple[Any, ...], _GraphData]] = {}
_MAX_BUILTIN_CONTAINER_CACHE_ENTRIES: int = 128

# On objects without weak-key support (e.g. plain list), fall back to an instance attribute dict.
_CACHE_ATTR: str = "_tensor_network_viz_graph_cache_by_builder"


def _cache_source(network: Any) -> Any:
    source = network
    while hasattr(source, "_graph_cache_source"):
        next_source = source._graph_cache_source
        if next_source is source:
            break
        source = next_source
    return source


def _builtin_container_signature_inner(network: Any) -> tuple[Any, ...] | None:
    if isinstance(network, list):
        items: list[tuple[Any, ...]] = []
        for item in network:
            nested = _builtin_container_signature_inner(item)
            items.append(nested if nested is not None else ("leaf", id(item)))
        return (
            "list",
            tuple(items),
        )
    if isinstance(network, tuple):
        items = []
        for item in network:
            nested = _builtin_container_signature_inner(item)
            items.append(nested if nested is not None else ("leaf", id(item)))
        return (
            "tuple",
            tuple(items),
        )
    if isinstance(network, dict):
        items = []
        for item in network.values():
            nested = _builtin_container_signature_inner(item)
            items.append(nested if nested is not None else ("leaf", id(item)))
        return (
            "dict",
            tuple(items),
        )
    return None


def _builtin_container_signature(network: Any) -> tuple[Any, ...] | None:
    return _builtin_container_signature_inner(_cache_source(network))


def _builtin_container_cache_get(
    network: Any,
    *,
    builder_id: int,
) -> _GraphData | None:
    signature = _builtin_container_signature(network)
    if signature is None:
        return None
    hit = _builtin_container_cache.get((id(_cache_source(network)), builder_id))
    if hit is None:
        return None
    cached_signature, graph = hit
    if cached_signature != signature:
        return None
    return graph


def _builtin_container_cache_put(
    network: Any,
    *,
    builder_id: int,
    graph: _GraphData,
) -> bool:
    signature = _builtin_container_signature(network)
    if signature is None:
        return False
    cache_key = (id(_cache_source(network)), builder_id)
    _builtin_container_cache[cache_key] = (signature, graph)
    while len(_builtin_container_cache) > _MAX_BUILTIN_CONTAINER_CACHE_ENTRIES:
        oldest_key = next(iter(_builtin_container_cache))
        del _builtin_container_cache[oldest_key]
    return True


def _get_or_build_graph(
    network: Any,
    builder: Callable[[Any], _GraphData],
) -> _GraphData:
    """Return a cached :class:`_GraphData` for *network* when *builder* matches a previous build."""
    b_id = id(builder)
    try:
        bucket = _graph_weak_cache.get(network)
        if bucket is not None:
            hit = bucket.get(b_id)
            if hit is not None:
                package_logger.debug("Graph cache hit via weak cache for builder_id=%s.", b_id)
                return hit
    except TypeError:
        pass

    builtin_hit = _builtin_container_cache_get(network, builder_id=b_id)
    if builtin_hit is not None:
        package_logger.debug("Graph cache hit via builtin container cache for builder_id=%s.", b_id)
        return builtin_hit

    attr_bucket = getattr(network, _CACHE_ATTR, None)
    if isinstance(attr_bucket, dict):
        hit = attr_bucket.get(b_id)
        if isinstance(hit, _GraphData):
            package_logger.debug(
                "Graph cache hit via object attribute cache for builder_id=%s.",
                b_id,
            )
            return hit

    package_logger.debug("Graph cache miss for builder_id=%s; rebuilding graph.", b_id)
    graph = builder(network)

    try:
        bucket = _graph_weak_cache.get(network)
        if bucket is None:
            bucket = {}
            _graph_weak_cache[network] = bucket
        bucket[b_id] = graph
    except TypeError:
        if _builtin_container_cache_put(network, builder_id=b_id, graph=graph):
            package_logger.debug("Stored graph in builtin container cache for builder_id=%s.", b_id)
            return graph
        try:
            if not isinstance(attr_bucket, dict):
                attr_bucket = {}
            attr_bucket[b_id] = graph
            setattr(network, _CACHE_ATTR, attr_bucket)
        except (TypeError, AttributeError):
            pass

    return graph


def clear_tensor_network_graph_cache(
    network: Any,
    *,
    builder: Callable[[Any], _GraphData] | None = None,
) -> None:
    """Drop cached :class:`_GraphData` for *network* (all builders, or only *builder*).

    Call this after in-place edits to a tensor network so the next draw re-extracts the structure.
    """
    b_id = id(builder) if builder is not None else None
    package_logger.debug("Clearing tensor network graph cache for builder_id=%s.", b_id)
    try:
        bucket = _graph_weak_cache.get(network)
        if bucket is not None:
            if b_id is None:
                bucket.clear()
            else:
                bucket.pop(b_id, None)
            if not bucket:
                del _graph_weak_cache[network]
    except (TypeError, KeyError):
        pass

    builtin_key_prefix = id(_cache_source(network))
    if b_id is None:
        for key in tuple(_builtin_container_cache):
            if key[0] == builtin_key_prefix:
                del _builtin_container_cache[key]
    else:
        _builtin_container_cache.pop((builtin_key_prefix, b_id), None)

    attr_bucket = getattr(network, _CACHE_ATTR, None)
    if not isinstance(attr_bucket, dict):
        return
    if b_id is None:
        attr_bucket.clear()
    else:
        attr_bucket.pop(b_id, None)
    if not attr_bucket:
        with contextlib.suppress(AttributeError):
            delattr(network, _CACHE_ATTR)


__all__ = [
    "_get_or_build_graph",
    "clear_tensor_network_graph_cache",
]
