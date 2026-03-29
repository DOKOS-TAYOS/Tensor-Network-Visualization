from __future__ import annotations

from typing import Literal

import numpy as np


def _perpendicular_2d(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]], dtype=float)


def _perpendicular_3d(direction: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    perp = np.cross(direction, reference)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
    return perp / np.linalg.norm(perp)


def _bond_perpendicular_unoriented(
    delta: np.ndarray,
    dimensions: Literal[2, 3],
) -> np.ndarray:
    dist = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / dist
    return _perpendicular_3d(direction) if dimensions == 3 else _perpendicular_2d(direction)


def _signed_bond_perpendicular(
    delta: np.ndarray,
    dimensions: Literal[2, 3],
) -> np.ndarray:
    perpendicular = _bond_perpendicular_unoriented(delta, dimensions)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    if (dimensions == 2 and perpendicular[1] < 0) or (dimensions == 3 and perpendicular[2] < 0):
        perpendicular = -perpendicular
    return perpendicular


__all__ = [
    "_bond_perpendicular_unoriented",
    "_perpendicular_2d",
    "_perpendicular_3d",
    "_signed_bond_perpendicular",
]
