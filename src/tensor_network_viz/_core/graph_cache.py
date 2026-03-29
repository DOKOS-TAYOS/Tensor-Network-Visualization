"""Reuse normalized graphs across repeated draws of the same network object.

Covers preview + export and 2D + 3D.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable
from typing import Any

from .graph import _GraphData

_graph_weak_cache: weakref.WeakKeyDictionary[Any, dict[int, _GraphData]] = (
    weakref.WeakKeyDictionary()
)

# On objects without weak-key support (e.g. plain list), fall back to an instance attribute dict.
_CACHE_ATTR: str = "_tensor_network_viz_graph_cache_by_builder"


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
                return hit
    except TypeError:
        pass

    attr_bucket = getattr(network, _CACHE_ATTR, None)
    if isinstance(attr_bucket, dict):
        hit = attr_bucket.get(b_id)
        if isinstance(hit, _GraphData):
            return hit

    graph = builder(network)

    try:
        bucket = _graph_weak_cache.get(network)
        if bucket is None:
            bucket = {}
            _graph_weak_cache[network] = bucket
        bucket[b_id] = graph
    except TypeError:
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
