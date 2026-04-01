"""Shared helpers for axis-direction assignment."""

from __future__ import annotations

import numpy as np

from ..axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from ..graph import _GraphData
from .parameters import _FREE_DIR_OVERLAP_THRESHOLD
from .types import AxisDirections, Vector


def _is_dangling_leg_axis(graph: _GraphData, node_id: int, axis_index: int) -> bool:
    """True when this node axis is the tensor endpoint of a dangling edge."""
    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        endpoint = edge.endpoints[0]
        if endpoint.node_id == int(node_id) and endpoint.axis_index == int(axis_index):
            return True
    return False


def _direction_from_axis_name(
    axis_name: str | None,
    *,
    dimensions: int,
) -> np.ndarray | None:
    """Return direction vector for an axis name in 2D or 3D."""
    if not axis_name:
        return None
    key = axis_name.lower().strip()
    if dimensions == 2 and key in _AXIS_DIR_2D:
        return np.array(_AXIS_DIR_2D[key], dtype=float)
    if dimensions == 3 and key in _AXIS_DIR_3D:
        return np.array(_AXIS_DIR_3D[key], dtype=float)
    return None


def _used_axis_directions(
    directions: AxisDirections,
    *,
    node_id: int,
    axis_count: int,
) -> list[np.ndarray]:
    return [
        directions[(node_id, axis_index)]
        for axis_index in range(axis_count)
        if (node_id, axis_index) in directions
    ]


def _direction_has_space(direction: np.ndarray, used_dirs: list[np.ndarray]) -> bool:
    overlap = sum(
        max(0.0, float(np.dot(direction, used_direction))) for used_direction in used_dirs
    )
    return overlap < _FREE_DIR_OVERLAP_THRESHOLD


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)


__all__ = [
    "_direction_from_axis_name",
    "_direction_has_space",
    "_is_dangling_leg_axis",
    "_orthogonal_unit",
    "_used_axis_directions",
]
