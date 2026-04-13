"""Shared helpers for auto-resolving display toggles."""

from __future__ import annotations

from dataclasses import replace

from ._core.graph import _GraphData
from .config import PlotConfig

AUTO_VISIBLE_TENSOR_THRESHOLD: int = 25


def count_visible_tensors(graph: _GraphData) -> int:
    """Count visible non-virtual tensors, falling back to total nodes when needed."""
    visible_tensors = sum(1 for node in graph.nodes.values() if not node.is_virtual)
    return visible_tensors or len(graph.nodes)


def resolve_auto_display_flag(
    value: bool | None,
    *,
    visible_tensor_count: int,
) -> bool:
    """Resolve one optional display flag using the shared visibility heuristic."""
    if value is not None:
        return bool(value)
    return int(visible_tensor_count) < AUTO_VISIBLE_TENSOR_THRESHOLD


def resolve_auto_display_config(
    config: PlotConfig,
    *,
    visible_tensor_count: int,
) -> PlotConfig:
    """Freeze auto display flags into concrete booleans for the current render."""
    return replace(
        config,
        show_nodes=resolve_auto_display_flag(
            config.show_nodes,
            visible_tensor_count=visible_tensor_count,
        ),
        show_tensor_labels=resolve_auto_display_flag(
            config.show_tensor_labels,
            visible_tensor_count=visible_tensor_count,
        ),
    )


__all__ = [
    "AUTO_VISIBLE_TENSOR_THRESHOLD",
    "count_visible_tensors",
    "resolve_auto_display_config",
    "resolve_auto_display_flag",
]
