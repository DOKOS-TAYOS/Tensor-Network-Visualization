"""Curve geometry and edge grouping utilities."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

from .axis_directions import _AXIS_OFFSET_SIGN
from .graph import _EdgeData, _EdgeEndpoint, _GraphData

Vector: TypeAlias = np.ndarray


def _offset_sign_from_axis_name(axis_name: str | None) -> int:
    if not axis_name:
        return 0
    return _AXIS_OFFSET_SIGN.get(axis_name.lower().strip(), 0)


def _group_contractions(graph: _GraphData) -> dict[tuple[int, int], list[_EdgeData]]:
    groups: dict[tuple[int, int], list[_EdgeData]] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        key = tuple(sorted(edge.node_ids))
        groups.setdefault(key, []).append(edge)
    for group in groups.values():
        group.sort(
            key=lambda edge: _offset_sign_from_axis_name(
                edge.endpoints[0].axis_name if edge.endpoints else None
            )
        )
    return groups


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
        center
        + np.outer(np.cos(theta), axis_a) * width
        + np.outer(np.sin(theta), axis_b) * height
    )


def _require_self_endpoints(edge: _EdgeData) -> tuple[_EdgeEndpoint, _EdgeEndpoint]:
    if len(edge.endpoints) < 2:
        raise TypeError("Self-edges must expose two endpoints.")
    return edge.endpoints[0], edge.endpoints[1]
