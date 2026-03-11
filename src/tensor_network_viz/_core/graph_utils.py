"""Shared utilities for graph building from backend-specific node/edge structures."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from typing import Any


def _extract_unique_items(
    source: Any,
    *,
    attr_sources: tuple[str, ...],
    sort_key: Callable[[Any], Any] | None,
    backend_name: str,
    type_name: str = "nodes",
) -> list[Any]:
    """Extract unique items from network object, dict, or iterable. Deduplicates by id."""
    if isinstance(source, dict):
        raw_items = source.values()
        should_sort = False
    elif attr_sources and any(hasattr(source, attr) for attr in attr_sources):
        raw_items = None
        for attr in attr_sources:
            if hasattr(source, attr):
                raw_items = getattr(source, attr)
                break
        if raw_items is None:
            raise TypeError(
                f"Input must be an iterable of {backend_name} {type_name}, or an object with "
                f"'{attr_sources[0]}' attribute."
            )
        should_sort = _is_unordered_collection(raw_items)
    elif isinstance(source, (str, bytes, bytearray)):
        raise TypeError(f"Input must be an iterable of {backend_name} {type_name}.")
    elif isinstance(source, Iterable):
        raw_items = source
        should_sort = _is_unordered_collection(raw_items)
    else:
        extra = (
            f", or an object with '{attr_sources[0]}' attribute."
            if attr_sources
            else "."
        )
        raise TypeError(f"Input must be an iterable of {backend_name} {type_name}{extra}.")

    iterable = raw_items.values() if isinstance(raw_items, dict) else raw_items

    try:
        items = list(iterable)
    except TypeError as exc:
        raise TypeError(f"{backend_name} {type_name} must be iterable.") from exc

    unique: list[Any] = []
    seen: set[int] = set()
    for item in items:
        if item is None:
            continue
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)

    if should_sort and sort_key is not None:
        unique.sort(key=sort_key)
    return unique


def _is_unordered_collection(value: Any) -> bool:
    return isinstance(value, AbstractSet) and not isinstance(value, (str, bytes, bytearray))


def _stringify(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _require_attr(obj: Any, attr_name: str, object_name: str) -> Any:
    if not hasattr(obj, attr_name):
        raise TypeError(f"{object_name.capitalize()} is missing required attribute '{attr_name}'.")
    return getattr(obj, attr_name)


def _iterable_attr(obj: Any, attr_name: str, object_name: str) -> list[Any]:
    value = _require_attr(obj, attr_name, object_name)
    if isinstance(value, dict):
        return list(value.values())
    try:
        return list(value)
    except TypeError as exc:
        msg = f"{object_name.capitalize()} attribute '{attr_name}' must be iterable."
        raise TypeError(msg) from exc
