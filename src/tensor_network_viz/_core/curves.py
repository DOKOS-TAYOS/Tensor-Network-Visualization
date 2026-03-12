"""Curve geometry utilities."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

from .graph import _EdgeData, _EdgeEndpoint

Vector: TypeAlias = np.ndarray


def _quadratic_curve(
    start: Vector,
    control: Vector,
    end: Vector,
    samples: int = 40,
) -> Vector:
    t = np.linspace(0.0, 1.0, samples)
    return (
        ((1.0 - t) ** 2)[:, None] * start
        + (2.0 * (1.0 - t) * t)[:, None] * control
        + (t**2)[:, None] * end
    )


def _ellipse_points(
    center: Vector,
    direction: Vector,
    normal: Vector,
    *,
    width: float,
    height: float,
    samples: int = 60,
) -> Vector:
    theta = np.linspace(0.0, 2.0 * math.pi, samples)
    return (
        center
        + np.outer(np.cos(theta), direction) * width
        + np.outer(np.sin(theta), normal) * height
    )


def _ellipse_points_3d(
    center: Vector,
    axis_a: Vector,
    axis_b: Vector,
    *,
    width: float,
    height: float,
    samples: int = 60,
) -> Vector:
    theta = np.linspace(0.0, 2.0 * math.pi, samples)
    return (
        center + np.outer(np.cos(theta), axis_a) * width + np.outer(np.sin(theta), axis_b) * height
    )


def _require_self_endpoints(edge: _EdgeData) -> tuple[_EdgeEndpoint, _EdgeEndpoint]:
    if len(edge.endpoints) < 2:
        raise TypeError("Self-edges must expose two endpoints.")
    return edge.endpoints[0], edge.endpoints[1]
