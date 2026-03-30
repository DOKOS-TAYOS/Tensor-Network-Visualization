"""Layout orchestration (components, axis directions, stubs)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from ...config import PlotConfig
from ..axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from ..contractions import _ContractionGroups, _group_contractions, _iter_contractions
from ..curves import _quadratic_curve
from ..graph import _GraphData
from ..layout_structure import (
    _analyze_layout_components,
    _component_orthogonal_basis,
    _LayoutComponent,
    _leaf_nodes,
    _specialized_anchor_positions,
)
from .force_directed import _compute_force_layout
from .parameters import *
from .types import AxisDirections, NodePositions, Vector


def _is_dangling_leg_axis(graph: _GraphData, node_id: int, axis_index: int) -> bool:
    """True when this node axis is the tensor endpoint of a dangling edge.

    Open legs are the only indices treated as physical for stub clearance and draw bond styling.
    """
    for edge in graph.edges:
        if edge.kind != "dangling":
            continue
        ep = edge.endpoints[0]
        if ep.node_id == int(node_id) and ep.axis_index == int(axis_index):
            return True
    return False


def _segment_point_min_distance_sq_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a2 = np.asarray(a, dtype=float).reshape(-1)[:2]
    b2 = np.asarray(b, dtype=float).reshape(-1)[:2]
    c2 = np.asarray(c, dtype=float).reshape(-1)[:2]
    ab = b2 - a2
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.sum((c2 - a2) ** 2))
    t = float(np.dot(c2 - a2, ab)) / denom
    t = max(0.0, min(1.0, t))
    closest = a2 + t * ab
    return float(np.sum((c2 - closest) ** 2))


def _segment_point_min_distance_sq_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a3 = np.asarray(a, dtype=float).reshape(-1)[:3]
    b3 = np.asarray(b, dtype=float).reshape(-1)[:3]
    c3 = np.asarray(c, dtype=float).reshape(-1)[:3]
    ab = b3 - a3
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.sum((c3 - a3) ** 2))
    t = float(np.dot(c3 - a3, ab)) / denom
    t = max(0.0, min(1.0, t))
    closest = a3 + t * ab
    return float(np.sum((c3 - closest) ** 2))


def _normalize_positions(
    positions: NodePositions,
    node_ids: list[int],
    target_norm: float = _LAYOUT_TARGET_NORM,
) -> NodePositions:
    """Center and scale positions to target norm. Returns new dict."""
    arr = np.array([positions[node_id] for node_id in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(arr, axis=1).max()
    if max_norm > 1e-6:
        arr /= max_norm / target_norm
    return {node_id: arr[index].copy() for index, node_id in enumerate(node_ids)}


def _direction_from_axis_name(
    axis_name: str | None,
    dimensions: int,
) -> np.ndarray | None:
    """Return direction vector for axis name in 2D or 3D, or None if unknown."""
    if not axis_name:
        return None
    key = axis_name.lower().strip()
    if dimensions == 2 and key in _AXIS_DIR_2D:
        return np.array(_AXIS_DIR_2D[key], dtype=float)
    if dimensions == 3 and key in _AXIS_DIR_3D:
        return np.array(_AXIS_DIR_3D[key], dtype=float)
    return None


def _compute_layout_from_components(
    graph: _GraphData,
    components: tuple[_LayoutComponent, ...],
    dimensions: int,
    seed: int,
    *,
    iterations: int = 220,
) -> NodePositions:
    node_ids = list(graph.nodes)
    component_positions: list[NodePositions] = []
    for index, component in enumerate(components):
        positions_2d = _compute_component_layout_2d(
            graph,
            component,
            seed=seed + index,
            iterations=iterations,
        )
        if dimensions == 2:
            component_positions.append(positions_2d)
            continue
        component_positions.append(_lift_component_layout_3d(graph, component, positions_2d))

    packed_positions = _pack_component_positions(component_positions, dimensions=dimensions)
    return _normalize_positions(packed_positions, node_ids)


def _compute_layout(
    graph: _GraphData,
    dimensions: int,
    seed: int,
    *,
    iterations: int = 220,
) -> NodePositions:
    node_ids = list(graph.nodes)
    if len(node_ids) == 1:
        return {node_ids[0]: np.zeros(dimensions, dtype=float)}

    components = _analyze_layout_components(graph)
    return _compute_layout_from_components(
        graph,
        components,
        dimensions,
        seed,
        iterations=iterations,
    )


def _compute_component_layout_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    seed: int,
    iterations: int,
) -> NodePositions:
    node_ids = list(component.node_ids)
    if len(node_ids) == 1:
        return {node_ids[0]: np.zeros(2, dtype=float)}

    fixed_positions = _specialized_anchor_positions(component)
    if fixed_positions:
        positions = _compute_force_layout(
            graph,
            node_ids=node_ids,
            dimensions=2,
            seed=seed,
            iterations=iterations,
            fixed_positions=fixed_positions,
        )
    else:
        positions = _compute_force_layout(
            graph,
            node_ids=node_ids,
            dimensions=2,
            seed=seed,
            iterations=iterations,
        )

    _snap_virtual_nodes_to_barycenters(component, positions)
    _spread_colocated_virtual_hubs_2d(component, positions)
    _place_trimmed_leaf_nodes_2d(component, positions)
    return _center_positions(positions, node_ids=node_ids)


def _lift_component_layout_3d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions_2d: NodePositions,
) -> NodePositions:
    positions = {
        node_id: np.array([coords[0], coords[1], 0.0], dtype=float)
        for node_id, coords in positions_2d.items()
    }
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        for node_id, (i, j, k) in component.grid3d_mapping.items():
            positions[node_id] = np.array([float(i), float(j), float(k)], dtype=float)
    _place_trimmed_leaf_nodes_3d(component, positions)
    if component.structure_kind != "grid3d" or component.grid3d_mapping is None:
        _promote_3d_layers(graph, component, positions)
    return positions


def _center_positions(positions: NodePositions, *, node_ids: list[int]) -> NodePositions:
    arr = np.array([positions[node_id] for node_id in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    return {node_id: arr[index].copy() for index, node_id in enumerate(node_ids)}


def _pack_component_positions(
    component_positions: list[NodePositions],
    *,
    dimensions: int,
) -> NodePositions:
    if len(component_positions) == 1:
        return component_positions[0]

    packed: NodePositions = {}
    cursor_x = 0.0
    for positions in component_positions:
        node_ids = sorted(positions)
        coords = np.array([positions[node_id] for node_id in node_ids], dtype=float)
        min_x = float(coords[:, 0].min())
        max_x = float(coords[:, 0].max())
        shift = np.zeros(dimensions, dtype=float)
        shift[0] = cursor_x - min_x
        if dimensions > 1:
            shift[1:] = -coords[:, 1:].mean(axis=0)

        packed_coords = coords + shift
        for index, node_id in enumerate(node_ids):
            packed[node_id] = packed_coords[index].copy()

        cursor_x += max(max_x - min_x, 0.8) + _COMPONENT_GAP

    return packed


def _snap_virtual_nodes_to_barycenters(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    for node_id in component.virtual_node_ids:
        neighbors = sorted(component.contraction_graph.neighbors(node_id))
        if not neighbors:
            continue
        positions[node_id] = np.mean(
            np.stack([positions[neighbor_id] for neighbor_id in neighbors]),
            axis=0,
        )


def _spread_colocated_virtual_hubs_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    """Separate virtual hubs that share the same visible neighbors (identical barycenters)."""
    groups: defaultdict[frozenset[int], list[int]] = defaultdict(list)
    graph_nx = component.contraction_graph
    for vid in component.virtual_node_ids:
        groups[frozenset(graph_nx.neighbors(vid))].append(vid)

    spacing = float(_VIRTUAL_HUB_MIN_SEPARATION)
    for neighbor_set, vids in groups.items():
        if len(vids) < 2:
            continue
        neighbors = sorted(neighbor_set, key=lambda nid: float(positions[nid][0]))
        base = np.mean(
            np.stack([positions[nid] for nid in neighbors]),
            axis=0,
        )
        if len(neighbors) >= 2:
            p0 = np.asarray(positions[neighbors[0]], dtype=float).reshape(-1)[:2]
            p1 = np.asarray(positions[neighbors[-1]], dtype=float).reshape(-1)[:2]
            chord = p1 - p0
            chord_len = float(np.linalg.norm(chord))
            if chord_len > 1e-9:
                perp = np.array([-chord[1], chord[0]], dtype=float) / chord_len
            else:
                perp = np.array([0.0, 1.0], dtype=float)
        else:
            perp = np.array([0.0, 1.0], dtype=float)

        vids_sorted = sorted(vids)
        n_v = len(vids_sorted)
        offsets = np.linspace(-0.5 * (n_v - 1), 0.5 * (n_v - 1), n_v)
        for vid, off in zip(vids_sorted, offsets, strict=True):
            pos = np.asarray(positions[vid], dtype=float).copy()
            pos[:2] = base[:2] + perp * spacing * float(off)
            positions[vid] = pos


def _place_trimmed_leaf_nodes_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    if not component.trimmed_leaf_parents:
        return

    axis = _component_main_axis_2d(component, positions)
    perpendicular = np.array([-axis[1], axis[0]], dtype=float)
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.array([0.0, 1.0], dtype=float)
    else:
        perpendicular /= np.linalg.norm(perpendicular)
    core_node_ids = [
        node_id
        for node_id in component.visible_node_ids
        if node_id not in {leaf_id for leaf_id, _ in component.trimmed_leaf_parents}
    ]
    if not core_node_ids:
        return
    centroid = np.mean(
        np.stack([positions[node_id] for node_id in core_node_ids]),
        axis=0,
    )

    assigned_targets: list[np.ndarray] = []
    for leaf_id, parent_id in component.trimmed_leaf_parents:
        positions[leaf_id] = _best_attachment_position_2d(
            component=component,
            origin=positions[parent_id],
            parent_id=parent_id,
            leaf_id=leaf_id,
            candidates=(perpendicular, -perpendicular),
            axis=axis,
            centroid=centroid,
            assigned_targets=assigned_targets,
            positions=positions,
        )
        assigned_targets.append(positions[leaf_id].copy())


def _place_trimmed_leaf_nodes_3d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    if not component.trimmed_leaf_parents:
        return

    _, lateral, normal = _component_orthogonal_basis(component, positions)
    candidates = (normal, -normal, lateral, -lateral)
    assigned_targets: list[np.ndarray] = []
    for leaf_id, parent_id in component.trimmed_leaf_parents:
        positions[leaf_id] = _best_attachment_position_3d(
            origin=positions[parent_id],
            candidates=candidates,
            assigned_targets=assigned_targets,
        )
        assigned_targets.append(positions[leaf_id].copy())


def _component_main_axis_2d(
    component: _LayoutComponent,
    positions: NodePositions,
) -> np.ndarray:
    anchor_node_ids = component.anchor_node_ids or component.visible_node_ids
    if component.structure_kind == "chain" and len(component.chain_order) >= 2:
        start = positions[component.chain_order[0]]
        end = positions[component.chain_order[-1]]
        axis = end - start
    elif len(anchor_node_ids) >= 2:
        coords = np.stack([positions[node_id] for node_id in anchor_node_ids])
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    else:
        axis = np.array([1.0, 0.0], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return axis / norm


def _best_attachment_position_2d(
    *,
    component: _LayoutComponent,
    origin: np.ndarray,
    parent_id: int,
    leaf_id: int,
    candidates: tuple[np.ndarray, ...],
    axis: np.ndarray,
    centroid: np.ndarray,
    assigned_targets: list[np.ndarray],
    positions: NodePositions,
) -> np.ndarray:
    leaf_node_ids = {node_id for node_id, _ in component.trimmed_leaf_parents}
    direction_options = (
        candidates if component.structure_kind == "chain" else (*candidates, axis, -axis)
    )
    used_dirs = []
    for neighbor_id in component.contraction_graph.neighbors(parent_id):
        if neighbor_id == leaf_id or neighbor_id in leaf_node_ids:
            continue
        delta = positions[neighbor_id][:2] - origin[:2]
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            used_dirs.append(delta / norm)

    outward = origin[:2] - centroid[:2]
    outward_norm = np.linalg.norm(outward)
    outward_dir = outward / outward_norm if outward_norm > 1e-6 else np.zeros(2, dtype=float)

    existing_segments = []
    for left_id, right_id in component.contraction_graph.edges():
        if leaf_id in (left_id, right_id):
            continue
        if left_id in leaf_node_ids or right_id in leaf_node_ids:
            continue
        if parent_id in (left_id, right_id):
            continue
        existing_segments.append((positions[left_id][:2], positions[right_id][:2]))

    distance = 1.0
    best_score = -np.inf
    best_candidate = origin[:2] + candidates[0] * distance
    for direction in direction_options:
        candidate = origin + direction * distance
        score = 0.8 * float(np.dot(direction[:2], outward_dir))
        score -= 2.5 * sum(
            max(0.0, float(np.dot(direction[:2], used_dir))) for used_dir in used_dirs
        )
        score -= 2.0 * sum(
            1.0 for target in assigned_targets if np.linalg.norm(candidate[:2] - target[:2]) < 0.3
        )
        score -= 4.0 * sum(
            1.0
            for node_id, position in positions.items()
            if node_id not in {leaf_id, parent_id}
            and np.linalg.norm(candidate[:2] - position[:2]) < 0.25
        )
        score -= 4.0 * sum(
            1.0
            for start, end in existing_segments
            if _segment_hits_existing_geometry_2d(origin[:2], candidate[:2], start, end)
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate[:2].copy()
    return best_candidate


def _best_attachment_position_3d(
    *,
    origin: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    assigned_targets: list[np.ndarray],
) -> np.ndarray:
    distance = 1.0
    for direction in candidates:
        candidate = origin + direction * distance
        if all(np.linalg.norm(candidate - target) >= 0.3 for target in assigned_targets):
            return candidate
    return origin + candidates[0] * distance


def _segment_hits_existing_geometry_2d(
    start: np.ndarray,
    end: np.ndarray,
    other_start: np.ndarray,
    other_end: np.ndarray,
) -> bool:
    if _segments_cross_2d(start, end, other_start, other_end):
        return True
    midpoint = (start + end) / 2.0
    return _point_segment_distance_2d(midpoint, other_start, other_end) < 0.15


def _point_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _promote_3d_layers(
    graph: _GraphData,
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    layer_indices = dict.fromkeys(component.node_ids, 0)

    for node_id in component.virtual_node_ids:
        if _node_overlaps_component(node_id, component, positions):
            layer_indices[node_id] = _next_layer(used_layers=layer_indices.values())

    edges = [
        tuple(sorted(record.node_ids))
        for record in _iter_contractions(graph)
        if all(node_id in component.contraction_graph for node_id in record.node_ids)
    ]

    for index, left_edge in enumerate(edges):
        for right_edge in edges[index + 1 :]:
            if set(left_edge) & set(right_edge):
                continue
            if not _segments_cross_2d(
                positions[left_edge[0]][:2],
                positions[left_edge[1]][:2],
                positions[right_edge[0]][:2],
                positions[right_edge[1]][:2],
            ):
                continue
            promoted = _choose_promotable_node(
                graph,
                component,
                node_ids=left_edge + right_edge,
            )
            if promoted is None or layer_indices[promoted] != 0:
                continue
            layer_indices[promoted] = _next_layer(used_layers=layer_indices.values())

    for node_id, layer_index in layer_indices.items():
        positions[node_id][2] += layer_index * _LAYER_SPACING


def _next_layer(*, used_layers: Iterable[int | float]) -> int:
    used = {int(layer) for layer in used_layers}
    for candidate in _LAYER_SEQUENCE[1:]:
        if candidate not in used:
            return candidate
    return max(used or {0}, key=abs) + 1


def _node_overlaps_component(
    node_id: int,
    component: _LayoutComponent,
    positions: NodePositions,
) -> bool:
    point = positions[node_id][:2]
    for other_id in component.node_ids:
        if other_id == node_id:
            continue
        if np.linalg.norm(point - positions[other_id][:2]) < 0.18:
            return True
    return False


def _choose_promotable_node(
    graph: _GraphData,
    component: _LayoutComponent,
    *,
    node_ids: tuple[int, ...],
) -> int | None:
    leaf_node_ids = set(_leaf_nodes(component))
    anchor_node_ids = set(component.anchor_node_ids)

    def priority(node_id: int) -> tuple[int, int]:
        if graph.nodes[node_id].is_virtual:
            return (0, node_id)
        if node_id not in anchor_node_ids:
            return (1, node_id)
        if node_id in leaf_node_ids:
            return (2, node_id)
        return (3, node_id)

    ordered = sorted(dict.fromkeys(node_ids), key=priority)
    if not ordered:
        return None
    return ordered[0]


def _segments_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _normalize_2d(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(-1)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def _dangling_stub_segment_2d(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    *,
    draw_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Polyline for a dangling leg in layout space (approx. rim → stub end).

    Lengths scale with *draw_scale* so conflict checks match `_draw_graph(..., scale=...)`.
    """
    o = np.asarray(origin, dtype=float).reshape(-1)[:2]
    d = _normalize_2d(direction_unit)
    s = max(float(draw_scale), 1e-6)
    return o + d * (_STUB_LAYOUT_R0 * s), o + d * (_STUB_LAYOUT_R1 * s)


def _dangling_stub_segment_3d(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    *,
    draw_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(origin, dtype=float).reshape(-1)[:3]
    d = np.asarray(direction_unit, dtype=float).reshape(-1)[:3]
    n = float(np.linalg.norm(d))
    d = np.array([0.0, 0.0, 1.0], dtype=float) if n < 1e-9 else d / n
    s = max(float(draw_scale), 1e-6)
    return o + d * (_STUB_LAYOUT_R0 * s), o + d * (_STUB_LAYOUT_R1 * s)


def _bond_perpendicular_unoriented_2d(delta: np.ndarray) -> np.ndarray:
    direction = delta / max(float(np.linalg.norm(delta)), 1e-6)
    return np.array([-direction[1], direction[0]], dtype=float)


def _layout_quadratic_bond_polyline_2d(
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
    *,
    scale: float,
) -> np.ndarray:
    """2D bond curve matching the draw path (quadratic Bezier), in layout coordinates."""
    start2 = np.asarray(start, dtype=float).reshape(-1)[:2]
    end2 = np.asarray(end, dtype=float).reshape(-1)[:2]
    midpoint = (start2 + end2) / 2.0
    delta = end2 - start2
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    perpendicular = _bond_perpendicular_unoriented_2d(delta)
    ref_len = _LAYOUT_BOND_CURVE_NEAR_PAIR_REF * scale
    effective_chord = float(math.hypot(distance, ref_len))
    offset = (
        (offset_index - (edge_count - 1) / 2.0)
        * _LAYOUT_BOND_CURVE_OFFSET_FACTOR
        * scale
        * effective_chord
    )
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start2, control, end2, samples=_LAYOUT_BOND_CURVE_SAMPLES)


def _planar_contraction_bond_segments_2d(
    graph: _GraphData,
    positions: NodePositions,
    *,
    scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Short segments along each contraction's rendered 2D bond (for stub–bond crossing tests)."""
    groups = contraction_groups if contraction_groups is not None else _group_contractions(graph)
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        if left_id == right_id:
            continue
        offset_index, edge_count = groups.offsets[id(record.edge)]
        start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:2]
        end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:2]
        poly = _layout_quadratic_bond_polyline_2d(start, end, offset_index, edge_count, scale=scale)
        for i in range(int(poly.shape[0]) - 1):
            out.append((left_id, right_id, poly[i].copy(), poly[i + 1].copy()))
    return out


def _direction_conflicts_2d(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_stub_segments: list[tuple[np.ndarray, np.ndarray]],
    bond_segments: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> bool:
    """True if a candidate dangling direction crosses another stub or a non-incident bond."""
    d = _normalize_2d(direction)
    o2 = np.asarray(origin, dtype=float).reshape(-1)[:2]
    p0, p1 = _dangling_stub_segment_2d(origin, d, draw_scale=draw_scale)
    s = max(float(draw_scale), 1e-6)

    for a, b, ba, bb in bond_segments:
        if a == node_id or b == node_id:
            continue
        if _segments_cross_2d(p0, p1, ba, bb):
            return True

    if strict_physical_node_clearance:
        r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * s * 1.08
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            op = np.asarray(other_position, dtype=float).reshape(-1)[:2]
            if math.sqrt(_segment_point_min_distance_sq_2d(p0, p1, op)) < r_disk:
                return True
    else:
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            op = np.asarray(other_position, dtype=float).reshape(-1)[:2]
            if float(np.linalg.norm(p1 - op)) < _STUB_TIP_NODE_CLEAR:
                return True

    for q0, q1 in assigned_stub_segments:
        if _segments_cross_2d(p0, p1, q0, q1):
            return True
        if float(np.linalg.norm(p1 - q1)) < _STUB_TIP_TIP_CLEAR:
            return True
        d_other = _normalize_2d(q1 - q0)
        if (
            float(np.linalg.norm(o2 - q0)) < _STUB_ORIGIN_PAIR_CLEAR
            and float(np.dot(d, d_other)) > _STUB_PARALLEL_DOT
        ):
            return True

    return False


def _compute_axis_directions(
    graph: _GraphData,
    positions: NodePositions,
    dimensions: int,
    *,
    draw_scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
    layout_components: tuple[_LayoutComponent, ...] | None = None,
) -> AxisDirections:
    directions: AxisDirections = {}
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        left_endpoint, right_endpoint = record.endpoints
        left_position = positions[left_id]
        right_position = positions[right_id]
        delta = right_position - left_position
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        toward_right = delta / distance
        directions[(left_id, left_endpoint.axis_index)] = toward_right
        directions[(right_id, right_endpoint.axis_index)] = -toward_right

    if dimensions == 2:
        _compute_free_directions_2d(
            graph,
            positions,
            directions,
            draw_scale=draw_scale,
            contraction_groups=contraction_groups,
        )
    else:
        components_3d = (
            layout_components
            if layout_components is not None
            else _analyze_layout_components(graph)
        )
        _compute_free_directions_3d(
            graph,
            positions,
            directions,
            draw_scale=draw_scale,
            layout_components=components_3d,
        )

    return directions


def _compute_free_directions_2d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    contraction_groups: _ContractionGroups | None = None,
) -> None:
    node_list = sorted(positions.keys())
    n_nodes = len(node_list)
    index_of: dict[int, int] = {node_id: idx for idx, node_id in enumerate(node_list)}
    coords_xy = np.stack(
        [np.asarray(positions[nid], dtype=float).reshape(-1)[:2] for nid in node_list],
    )
    angles = np.linspace(0.0, 2.0 * math.pi, _FREE_DIR_SAMPLES_2D, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    neighbor_midpoints: dict[int, list[np.ndarray]] = {node_id: [] for node_id in node_list}
    for record in _iter_contractions(graph):
        left_id, right_id = record.node_ids
        midpoint = (positions[left_id] + positions[right_id]) / 2.0
        neighbor_midpoints[left_id].append(midpoint)
        neighbor_midpoints[right_id].append(midpoint)

    assigned_stub_segments: list[tuple[np.ndarray, np.ndarray]] = []
    bond_segments = _planar_contraction_bond_segments_2d(
        graph,
        positions,
        scale=draw_scale,
        contraction_groups=contraction_groups,
    )

    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        origin = positions[node_id]
        obstacle_parts: list[np.ndarray] = []
        if n_nodes > 1:
            self_idx = index_of[node_id]
            mask = np.ones(n_nodes, dtype=bool)
            mask[self_idx] = False
            other_rows = coords_xy[mask]
            if other_rows.size:
                obstacle_parts.append(other_rows)
        if neighbor_midpoints[node_id]:
            obstacle_parts.append(np.array(neighbor_midpoints[node_id], dtype=float))
        obstacles = (
            np.concatenate(obstacle_parts, axis=0)
            if obstacle_parts
            else np.array([[origin[0] + 1.0, origin[1]]], dtype=float)
        )

        vecs_to_obstacles = obstacles - origin
        dists = np.linalg.norm(vecs_to_obstacles, axis=1, keepdims=True)
        dirs_to_obstacles = vecs_to_obstacles / np.maximum(dists, 1e-6)

        axis_count = max(node.degree, 1)
        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=2)
            strict_phys = _is_dangling_leg_axis(graph, node_id, axis_index)

            toward = dirs_to_obstacles @ unit_circle.T
            away_scores = -np.min(toward, axis=0)
            if used_dirs:
                used_stack = np.stack(
                    [used_direction[:2].astype(float) for used_direction in used_dirs],
                )
                overlap = unit_circle @ used_stack.T
                separation = np.maximum(0.0, overlap).sum(axis=1) * 2.0
            else:
                separation = np.zeros(unit_circle.shape[0], dtype=float)
            scores = away_scores - separation
            order = np.argsort(-scores)

            picked: np.ndarray | None = None
            if named_direction is not None and _direction_has_space(named_direction, used_dirs):
                nd = _normalize_2d(named_direction[:2])
                if not _direction_conflicts_2d(
                    node_id=node_id,
                    origin=origin,
                    direction=nd,
                    assigned_stub_segments=assigned_stub_segments,
                    bond_segments=bond_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                ):
                    picked = nd

            if picked is None:
                for idx in order:
                    cand = unit_circle[int(idx)].copy()
                    if not _direction_has_space(cand, used_dirs):
                        continue
                    if _direction_conflicts_2d(
                        node_id=node_id,
                        origin=origin,
                        direction=cand,
                        assigned_stub_segments=assigned_stub_segments,
                        bond_segments=bond_segments,
                        positions=positions,
                        draw_scale=draw_scale,
                        strict_physical_node_clearance=strict_phys,
                    ):
                        continue
                    picked = cand
                    break

            if picked is None and not strict_phys:
                for idx in order:
                    cand = unit_circle[int(idx)].copy()
                    if not _direction_has_space(cand, used_dirs):
                        continue
                    picked = cand
                    break

            if picked is None and strict_phys:
                r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
                other_ids = [oid for oid in positions if oid != node_id]
                best_idx: int | None = None
                best_margin = -1e300
                for idx in range(int(unit_circle.shape[0])):
                    cand = unit_circle[idx].copy()
                    p0s, p1s = _dangling_stub_segment_2d(origin, cand, draw_scale=draw_scale)
                    if other_ids:
                        margin = min(
                            math.sqrt(
                                _segment_point_min_distance_sq_2d(
                                    p0s,
                                    p1s,
                                    np.asarray(positions[oid], dtype=float).reshape(-1)[:2],
                                )
                            )
                            - r_disk
                            for oid in other_ids
                        )
                    else:
                        margin = 0.0
                    if margin > best_margin:
                        best_margin = margin
                        best_idx = idx
                if best_idx is not None:
                    picked = unit_circle[int(best_idx)].copy()

            if picked is None:
                picked = unit_circle[int(np.argmax(scores))].copy()

            directions[axis_key] = picked
            assigned_stub_segments.append(
                _dangling_stub_segment_2d(origin, picked, draw_scale=draw_scale)
            )


def _compute_free_directions_3d(
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    *,
    draw_scale: float = 1.0,
    layout_components: tuple[_LayoutComponent, ...],
) -> None:
    assigned_segments: list[tuple[np.ndarray, np.ndarray]] = []
    component_by_node = {
        node_id: component for component in layout_components for node_id in component.node_ids
    }

    for node_id, node in graph.nodes.items():
        component = component_by_node[node_id]
        axis_count = max(node.degree, 1)
        axis, lateral, normal = _component_orthogonal_basis(component, positions)
        candidate_directions = [normal, -normal, lateral, -lateral, axis, -axis]

        for axis_index in range(axis_count):
            axis_key = (node_id, axis_index)
            if axis_key in directions:
                continue

            axis_name = node.axes_names[axis_index] if axis_index < len(node.axes_names) else None
            named_direction = _direction_from_axis_name(axis_name, dimensions=3)
            strict_phys = _is_dangling_leg_axis(graph, node_id, axis_index)
            used_dirs = _used_axis_directions(directions, node_id=node_id, axis_count=axis_count)
            origin = positions[node_id]
            if (
                named_direction is not None
                and _direction_has_space(named_direction, used_dirs)
                and not _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=named_direction,
                    assigned_segments=assigned_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                )
            ):
                directions[axis_key] = named_direction
                assigned_segments.append((origin.copy(), named_direction.copy()))
                continue
            for direction in candidate_directions:
                if not _direction_has_space(direction, used_dirs):
                    continue
                if _direction_conflicts_3d(
                    node_id=node_id,
                    origin=origin,
                    direction=direction,
                    assigned_segments=assigned_segments,
                    positions=positions,
                    draw_scale=draw_scale,
                    strict_physical_node_clearance=strict_phys,
                ):
                    continue
                directions[axis_key] = direction
                assigned_segments.append((origin.copy(), direction.copy()))
                break
            else:
                if strict_phys:
                    dirs_try: list[np.ndarray] = []
                    if named_direction is not None:
                        dirs_try.append(np.asarray(named_direction, dtype=float))
                    dirs_try.extend(list(candidate_directions))
                    r_disk = (
                        float(PlotConfig.DEFAULT_NODE_RADIUS) * max(float(draw_scale), 1e-6) * 1.08
                    )
                    o3 = np.asarray(origin, dtype=float).reshape(-1)[:3]
                    other_ids = [oid for oid in positions if oid != node_id]
                    best_d: np.ndarray | None = None
                    best_margin = -1e300
                    for raw in dirs_try:
                        dvec = np.asarray(raw, dtype=float).reshape(-1)[:3]
                        nn = float(np.linalg.norm(dvec))
                        if nn < 1e-9:
                            continue
                        d_unit = dvec / nn
                        p0s, p1s = _dangling_stub_segment_3d(o3, d_unit, draw_scale=draw_scale)
                        if other_ids:
                            margin = min(
                                math.sqrt(
                                    _segment_point_min_distance_sq_3d(
                                        p0s,
                                        p1s,
                                        np.asarray(positions[oid], dtype=float).reshape(-1)[:3],
                                    )
                                )
                                - r_disk
                                for oid in other_ids
                            )
                        else:
                            margin = 0.0
                        if margin > best_margin:
                            best_margin = margin
                            best_d = d_unit
                    if best_d is not None:
                        directions[axis_key] = best_d
                        assigned_segments.append((origin.copy(), best_d.copy()))
                        continue
                fallback = (
                    named_direction if named_direction is not None else _orthogonal_unit(axis)
                )
                if not _direction_has_space(fallback, used_dirs):
                    fallback = -fallback
                directions[axis_key] = fallback / np.linalg.norm(fallback)
                assigned_segments.append((origin.copy(), directions[axis_key].copy()))


def _direction_conflicts_3d(
    *,
    node_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    assigned_segments: list[tuple[np.ndarray, np.ndarray]],
    positions: NodePositions,
    draw_scale: float = 1.0,
    strict_physical_node_clearance: bool = False,
) -> bool:
    tip = origin + direction * 0.45

    if strict_physical_node_clearance:
        d = np.asarray(direction, dtype=float).reshape(-1)[:3]
        nrm = float(np.linalg.norm(d))
        d_unit = d / nrm if nrm >= 1e-9 else np.array([0.0, 0.0, 1.0], dtype=float)
        s = max(float(draw_scale), 1e-6)
        p0, p1 = _dangling_stub_segment_3d(origin, d_unit, draw_scale=draw_scale)
        r_disk = float(PlotConfig.DEFAULT_NODE_RADIUS) * s * 1.08
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            op = np.asarray(other_position, dtype=float).reshape(-1)[:3]
            if math.sqrt(_segment_point_min_distance_sq_3d(p0, p1, op)) < r_disk:
                return True
    else:
        for other_id, other_position in positions.items():
            if other_id == node_id:
                continue
            if np.linalg.norm(tip - other_position) < 0.26:
                return True

    for other_origin, other_direction in assigned_segments:
        other_tip = other_origin + other_direction * 0.45
        if np.linalg.norm(tip - other_tip) < 0.26:
            return True
        if (
            np.linalg.norm(origin - other_origin) < 0.12
            and float(np.dot(direction, other_direction)) > 0.92
        ):
            return True

    return False


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
    "_best_attachment_position_2d",
    "_best_attachment_position_3d",
    "_bond_perpendicular_unoriented_2d",
    "_center_positions",
    "_choose_promotable_node",
    "_component_main_axis_2d",
    "_compute_axis_directions",
    "_compute_component_layout_2d",
    "_compute_layout_from_components",
    "_compute_free_directions_2d",
    "_compute_free_directions_3d",
    "_compute_layout",
    "_direction_conflicts_2d",
    "_direction_conflicts_3d",
    "_direction_from_axis_name",
    "_direction_has_space",
    "_dangling_stub_segment_2d",
    "_dangling_stub_segment_3d",
    "_is_dangling_leg_axis",
    "_layout_quadratic_bond_polyline_2d",
    "_lift_component_layout_3d",
    "_next_layer",
    "_node_overlaps_component",
    "_normalize_2d",
    "_normalize_positions",
    "_orthogonal_unit",
    "_pack_component_positions",
    "_place_trimmed_leaf_nodes_2d",
    "_place_trimmed_leaf_nodes_3d",
    "_planar_contraction_bond_segments_2d",
    "_point_segment_distance_2d",
    "_promote_3d_layers",
    "_segment_hits_existing_geometry_2d",
    "_segment_point_min_distance_sq_2d",
    "_segment_point_min_distance_sq_3d",
    "_segments_cross_2d",
    "_snap_virtual_nodes_to_barycenters",
    "_spread_colocated_virtual_hubs_2d",
    "_used_axis_directions",
]
