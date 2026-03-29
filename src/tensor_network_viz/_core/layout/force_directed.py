"""Force-directed placement (2D/3D within component)."""

from __future__ import annotations

import math

import numpy as np

from ..contractions import _iter_contractions
from ..graph import _GraphData
from .parameters import _FORCE_LAYOUT_COOLING_FACTOR, _FORCE_LAYOUT_K, _LAYOUT_TARGET_NORM
from .types import NodePositions, Vector

_REPULSION_DENSE_NODE_CAP: int = 72
_REPULSION_SAMPLE_MIN: int = 48
_REPULSION_SAMPLE_LINEAR: float = 12.0


def _pair_weights_for_node_ids(
    graph: _GraphData,
    *,
    node_ids: list[int],
) -> dict[tuple[int, int], int]:
    node_id_set = set(node_ids)
    counts: dict[tuple[int, int], int] = {}
    for record in _iter_contractions(graph):
        a, b = record.key
        if a in node_id_set and b in node_id_set:
            counts[record.key] = counts.get(record.key, 0) + 1
    return counts


def _repulsion_displacement(
    positions: np.ndarray,
    *,
    k: float,
    rng: np.random.Generator,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Repulsion between nodes; full O(n²) when n is modest, sampled pairs when n is large."""
    n = int(positions.shape[0])
    if out is None:
        out = np.zeros_like(positions)
    else:
        out.fill(0.0)

    if n <= _REPULSION_DENSE_NODE_CAP:
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, 1.0)
        directions = deltas / np.maximum(distances[..., None], 1e-6)
        repulsion = (k * k / np.maximum(distances, 1e-6) ** 2)[..., None] * directions
        out[:] = repulsion.sum(axis=1)
        return out

    m_sample = min(n - 1, max(_REPULSION_SAMPLE_MIN, int(_REPULSION_SAMPLE_LINEAR * math.sqrt(n))))
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        if len(pool) <= m_sample:
            js = pool
        else:
            pick = rng.choice(len(pool), size=m_sample, replace=False)
            js = [pool[int(t)] for t in pick]
        for j in js:
            delta = positions[i] - positions[j]
            dist = max(float(np.linalg.norm(delta)), 1e-6)
            direction = delta / dist
            out[i] += (k * k / (dist * dist)) * direction
    return out


def _accumulate_attraction_forces(
    displacement: np.ndarray,
    positions: np.ndarray,
    *,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    edge_weights: np.ndarray,
    k: float,
) -> None:
    if left_idx.size == 0:
        return
    dvec = positions[right_idx] - positions[left_idx]
    dist = np.linalg.norm(dvec, axis=1)
    dist = np.maximum(dist, 1e-6)
    direction = dvec / dist[:, np.newaxis]
    mag = (edge_weights * dist * dist / k)[:, np.newaxis] * direction
    np.add.at(displacement, left_idx, mag)
    np.add.at(displacement, right_idx, -mag)


def _apply_attraction_forces(
    displacement: np.ndarray,
    positions: np.ndarray,
    *,
    index_by_node: dict[int, int],
    pair_weights: dict[tuple[int, int], int],
    k: float,
) -> None:
    if not pair_weights:
        return
    items = sorted(pair_weights.items(), key=lambda item: item[0])
    left_idx = np.array([index_by_node[a] for (a, _b), _w in items], dtype=np.int64)
    right_idx = np.array([index_by_node[b] for (_a, b), _w in items], dtype=np.int64)
    edge_weights = np.array([float(w) for _pair, w in items], dtype=np.float64)
    _accumulate_attraction_forces(
        displacement,
        positions,
        left_idx=left_idx,
        right_idx=right_idx,
        edge_weights=edge_weights,
        k=k,
    )


def _apply_force_step(
    positions: np.ndarray,
    displacement: np.ndarray,
    *,
    temperature: float,
    fixed_mask: np.ndarray | None = None,
    fixed_positions: np.ndarray | None = None,
) -> None:
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    step = displacement / np.maximum(norms, 1e-6) * temperature
    if fixed_mask is None or fixed_positions is None:
        positions += step
        positions -= positions.mean(axis=0, keepdims=True)
        max_norm = np.linalg.norm(positions, axis=1).max()
        if max_norm > _LAYOUT_TARGET_NORM:
            positions /= max_norm / _LAYOUT_TARGET_NORM
        return

    movable_mask = ~fixed_mask
    positions[movable_mask] += step[movable_mask]
    positions[fixed_mask] = fixed_positions[fixed_mask]


def _initial_positions(node_ids: list[int], dimensions: int, seed: int) -> Vector:
    count = len(node_ids)
    rng = np.random.default_rng(seed)

    if dimensions == 2:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        indices = np.arange(count, dtype=float)
        denom = max(count - 1, 1)
        y = 1.0 - (2.0 * indices) / denom
        radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        theta = golden_angle * indices
        positions = np.column_stack(
            (np.cos(theta) * radius, y, np.sin(theta) * radius),
        ).astype(float)

    positions += rng.normal(loc=0.0, scale=0.03, size=positions.shape)
    return positions


def _compute_force_layout(
    graph: _GraphData,
    *,
    node_ids: list[int],
    dimensions: int,
    seed: int,
    iterations: int,
    fixed_positions: NodePositions | None = None,
) -> NodePositions:
    rng_rep = np.random.default_rng(seed + 917_733)
    positions = _initial_positions(node_ids, dimensions=dimensions, seed=seed)
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    pair_weights = _pair_weights_for_node_ids(graph, node_ids=node_ids)
    fixed_mask = np.zeros(len(node_ids), dtype=bool)
    fixed_array = np.zeros_like(positions)
    displacement = np.zeros_like(positions)

    if fixed_positions:
        fixed_centroid = np.mean(np.stack(list(fixed_positions.values())), axis=0)
        for node_id, fixed_position in fixed_positions.items():
            if node_id not in index_by_node:
                continue
            index = index_by_node[node_id]
            fixed_mask[index] = True
            fixed_array[index] = fixed_position
            positions[index] = fixed_position
        for node_id in node_ids:
            index = index_by_node[node_id]
            if fixed_mask[index]:
                continue
            positions[index] = fixed_centroid + (positions[index] * 0.35)

    temperature = 0.12
    for _ in range(iterations):
        _repulsion_displacement(positions, k=_FORCE_LAYOUT_K, rng=rng_rep, out=displacement)
        _apply_attraction_forces(
            displacement,
            positions,
            index_by_node=index_by_node,
            pair_weights=pair_weights,
            k=_FORCE_LAYOUT_K,
        )
        if fixed_positions:
            displacement[fixed_mask] = 0.0
        _apply_force_step(
            positions,
            displacement,
            temperature=temperature,
            fixed_mask=fixed_mask if fixed_positions else None,
            fixed_positions=fixed_array if fixed_positions else None,
        )
        temperature *= _FORCE_LAYOUT_COOLING_FACTOR

    return {node_id: positions[index].copy() for index, node_id in enumerate(node_ids)}


__all__ = [
    "_accumulate_attraction_forces",
    "_apply_attraction_forces",
    "_apply_force_step",
    "_compute_force_layout",
    "_initial_positions",
    "_pair_weights_for_node_ids",
    "_repulsion_displacement",
]
