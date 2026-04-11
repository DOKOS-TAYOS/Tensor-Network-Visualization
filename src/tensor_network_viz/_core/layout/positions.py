"""Layout position resolution and component packing helpers."""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from ..contractions import _iter_contractions
from ..graph import _GraphData
from ..layout_structure import (
    _analyze_layout_components,
    _component_orthogonal_basis,
    _layout_tube_3d,
    _LayoutComponent,
    _leaf_nodes,
    _specialized_anchor_positions,
)
from .direction_common import _preferred_component_directions_2d
from .force_directed import _compute_force_layout
from .geometry import (
    _BondSegment2D,
    _planar_contraction_bond_segment_records_2d,
    _segment_bboxes_overlap_2d,
    _segment_hits_existing_geometry_2d,
    _segment_segment_min_distance_2d,
    _segments_cross_2d,
)
from .parameters import (
    _ATTACHMENT_NODE_CLEAR_2D,
    _ATTACHMENT_PARALLEL_DOT_2D,
    _ATTACHMENT_RADII_2D,
    _ATTACHMENT_TARGET_CLEAR_2D,
    _COMPONENT_GAP,
    _FREE_DIR_SAMPLES_2D,
    _LAYER_SEQUENCE,
    _LAYER_SPACING,
    _LAYOUT_TARGET_NORM,
    _SEGMENT_BOND_CLEAR_2D,
    _VIRTUAL_HUB_CHORD_CLEARANCE,
    _VIRTUAL_HUB_MIN_SEPARATION,
)
from .types import NodePositions

_layout_components_by_id: dict[int, tuple[_LayoutComponent, ...]] = {}


def _analyze_layout_components_cached(graph: _GraphData) -> tuple[_LayoutComponent, ...]:
    """Reuse layout-component analysis for repeated renders of the same normalized graph."""
    key = id(graph)
    cached = _layout_components_by_id.get(key)
    if cached is not None:
        return cached

    components = _analyze_layout_components(graph)
    _layout_components_by_id[key] = components

    def _evict() -> None:
        _layout_components_by_id.pop(key, None)

    weakref.finalize(graph, _evict)
    return components


def _normalize_positions(
    positions: NodePositions,
    node_ids: list[int],
    target_norm: float = _LAYOUT_TARGET_NORM,
) -> NodePositions:
    """Center and scale positions to target norm."""
    arr = np.array([positions[node_id] for node_id in node_ids], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(arr, axis=1).max()
    if max_norm > 1e-6:
        arr /= max_norm / target_norm
    return {node_id: arr[index].copy() for index, node_id in enumerate(node_ids)}


def _compute_layout_from_components(
    graph: _GraphData,
    components: tuple[_LayoutComponent, ...],
    dimensions: int,
    seed: int,
    *,
    iterations: int = 220,
) -> NodePositions:
    """Lay out each connected component independently and then pack them together."""
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
    """Compute normalized 2D or 3D positions for the full tensor-network graph."""
    node_ids = list(graph.nodes)
    if len(node_ids) == 1:
        return {node_ids[0]: np.zeros(dimensions, dtype=float)}

    components = _analyze_layout_components_cached(graph)
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
    """Lay out one component in 2D, then restore virtual hubs and trimmed leaves."""
    node_ids = list(component.node_ids)
    trimmed_leaf_ids = {leaf_id for leaf_id, _ in component.trimmed_leaf_parents}
    layout_node_ids = [node_id for node_id in node_ids if node_id not in trimmed_leaf_ids]
    if not layout_node_ids:
        layout_node_ids = node_ids.copy()
    fixed_positions = {
        node_id: np.asarray(position, dtype=float).copy()
        for node_id, position in _specialized_anchor_positions(component).items()
        if node_id in layout_node_ids
    }

    if len(fixed_positions) == len(layout_node_ids):
        positions = {node_id: fixed_positions[node_id].copy() for node_id in layout_node_ids}
    elif len(layout_node_ids) == 1:
        positions = {
            layout_node_ids[0]: fixed_positions.get(
                layout_node_ids[0],
                np.zeros(2, dtype=float),
            ).copy()
        }
    else:
        if fixed_positions:
            positions = _compute_force_layout(
                graph,
                node_ids=layout_node_ids,
                dimensions=2,
                seed=seed,
                iterations=iterations,
                fixed_positions=fixed_positions,
            )
        else:
            positions = _compute_force_layout(
                graph,
                node_ids=layout_node_ids,
                dimensions=2,
                seed=seed,
                iterations=iterations,
            )

    _snap_virtual_nodes_to_barycenters(component, positions)
    _spread_colocated_virtual_hubs_2d(component, positions)
    _nudge_singleton_attachment_virtual_hubs_2d(graph, component, positions)
    _offset_virtual_hubs_off_direct_tensor_chords_2d(graph, component, positions)
    _place_trimmed_leaf_nodes_2d(graph, component, positions)
    return _center_positions(positions, node_ids=node_ids)


def _lift_component_layout_3d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions_2d: NodePositions,
) -> NodePositions:
    """Lift a 2D component layout into 3D and add structure-specific depth cues."""
    positions = {
        node_id: np.array([coords[0], coords[1], 0.0], dtype=float)
        for node_id, coords in positions_2d.items()
    }
    if component.structure_kind == "grid3d" and component.grid3d_mapping is not None:
        for node_id, (i, j, k) in component.grid3d_mapping.items():
            positions[node_id] = np.array([float(i), float(j), float(k)], dtype=float)
    if component.structure_kind == "tube" and component.grid_mapping is not None:
        positions.update(_layout_tube_3d(component.grid_mapping))
    _place_trimmed_leaf_nodes_3d(component, positions)
    if (
        component.structure_kind not in {"grid3d", "tube"}
        or (component.structure_kind == "grid3d" and component.grid3d_mapping is None)
        or (component.structure_kind == "tube" and component.grid_mapping is None)
    ):
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
    """Pack already-laid-out components along the x-axis with a stable gap."""
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
    """Separate virtual hubs that share the same visible neighbors."""
    groups: defaultdict[frozenset[int], list[int]] = defaultdict(list)
    graph_nx = component.contraction_graph
    for node_id in component.virtual_node_ids:
        groups[frozenset(graph_nx.neighbors(node_id))].append(node_id)

    spacing = float(_VIRTUAL_HUB_MIN_SEPARATION)
    for neighbor_set, node_ids in groups.items():
        if len(node_ids) < 2:
            continue
        neighbors = sorted(neighbor_set, key=lambda node_id: float(positions[node_id][0]))
        base = np.mean(
            np.stack([positions[node_id] for node_id in neighbors]),
            axis=0,
        )
        if len(neighbors) >= 2:
            start = np.asarray(positions[neighbors[0]], dtype=float).reshape(-1)[:2]
            end = np.asarray(positions[neighbors[-1]], dtype=float).reshape(-1)[:2]
            chord = end - start
            chord_len = float(np.linalg.norm(chord))
            if chord_len > 1e-9:
                perpendicular = np.array([-chord[1], chord[0]], dtype=float) / chord_len
            else:
                perpendicular = np.array([0.0, 1.0], dtype=float)
        else:
            perpendicular = np.array([0.0, 1.0], dtype=float)

        sorted_node_ids = sorted(node_ids)
        offsets = np.linspace(
            -0.5 * (len(sorted_node_ids) - 1),
            0.5 * (len(sorted_node_ids) - 1),
            len(sorted_node_ids),
        )
        for node_id, offset in zip(sorted_node_ids, offsets, strict=True):
            pos = np.asarray(positions[node_id], dtype=float).copy()
            pos[:2] = base[:2] + perpendicular * spacing * float(offset)
            positions[node_id] = pos


def _nudge_singleton_attachment_virtual_hubs_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    """Move singleton virtual hubs off their lone visible neighbor in 2D."""
    contraction_graph = component.contraction_graph
    by_neighbors: defaultdict[frozenset[int], list[int]] = defaultdict(list)
    for node_id in component.virtual_node_ids:
        by_neighbors[frozenset(contraction_graph.neighbors(node_id))].append(node_id)

    distance = float(_VIRTUAL_HUB_MIN_SEPARATION)
    for neighbor_key, node_ids in by_neighbors.items():
        if len(node_ids) != 1:
            continue
        node_id = node_ids[0]
        neighbors = sorted(neighbor_key)
        if len(neighbors) != 1:
            continue
        neighbor_id = neighbors[0]
        if graph.nodes[neighbor_id].is_virtual:
            continue

        neighbor_position = np.asarray(positions[neighbor_id], dtype=float).reshape(-1)
        hub_position = np.asarray(positions[node_id], dtype=float).copy().reshape(-1)

        visible_ids = [
            visible_id
            for visible_id in component.visible_node_ids
            if not graph.nodes[visible_id].is_virtual and visible_id in positions
        ]
        if len(visible_ids) <= 1:
            tangent = np.array([0.0, 1.0], dtype=float)
        else:
            centroid = np.mean(
                np.stack([positions[visible_id] for visible_id in visible_ids]),
                axis=0,
            ).reshape(-1)[:2]
            radial = neighbor_position[:2] - centroid
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-9:
                tangent = np.array([-radial[1], radial[0]], dtype=float) / radial_norm
            else:
                tangent = np.array([0.0, 1.0], dtype=float)

        hub_position[:2] = neighbor_position[:2] + tangent * distance
        positions[node_id] = hub_position


def _tensor_tensor_contraction_pairs(graph: _GraphData) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for edge in graph.edges:
        if edge.kind != "contraction" or len(edge.node_ids) != 2:
            continue
        left_id, right_id = int(edge.node_ids[0]), int(edge.node_ids[1])
        if graph.nodes[left_id].is_virtual or graph.nodes[right_id].is_virtual:
            continue
        pairs.add(frozenset((left_id, right_id)))
    return pairs


def _offset_virtual_hubs_off_direct_tensor_chords_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    """Nudge hyperedge hubs off the UV segment when U and V also have a direct bond."""
    direct_pairs = _tensor_tensor_contraction_pairs(graph)
    contraction_graph = component.contraction_graph
    margin = float(_VIRTUAL_HUB_CHORD_CLEARANCE)
    for node_id in component.virtual_node_ids:
        neighbors = frozenset(contraction_graph.neighbors(node_id))
        if len(neighbors) != 2 or neighbors not in direct_pairs:
            continue
        left_id, right_id = sorted(neighbors)
        start = np.asarray(positions[left_id], dtype=float).reshape(-1)[:2]
        end = np.asarray(positions[right_id], dtype=float).reshape(-1)[:2]
        chord = end - start
        chord_len = float(np.linalg.norm(chord))
        if chord_len > 1e-9:
            perpendicular = np.array([-chord[1], chord[0]], dtype=float) / chord_len
        else:
            perpendicular = np.array([0.0, 1.0], dtype=float)
        pos = np.asarray(positions[node_id], dtype=float).copy()
        pos[:2] = pos[:2] + perpendicular * margin
        positions[node_id] = pos


def _place_trimmed_leaf_nodes_2d(
    graph: _GraphData,
    component: _LayoutComponent,
    positions: NodePositions,
) -> None:
    if not component.trimmed_leaf_parents:
        return

    leaf_node_ids = {leaf_id for leaf_id, _ in component.trimmed_leaf_parents}
    component_node_ids = frozenset(component.node_ids)
    core_segments = tuple(
        record
        for record in _planar_contraction_bond_segment_records_2d(
            graph,
            positions,
            scale=1.0,
            node_filter=component_node_ids,
        )
        if record.node_ids[0] not in leaf_node_ids and record.node_ids[1] not in leaf_node_ids
    )
    assigned_targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for leaf_id, parent_id in component.trimmed_leaf_parents:
        positions[leaf_id] = _best_attachment_position_2d(
            component=component,
            origin=positions[parent_id],
            parent_id=parent_id,
            leaf_id=leaf_id,
            assigned_targets=assigned_targets,
            core_segments=core_segments,
            positions=positions,
        )
        direction = positions[leaf_id][:2] - positions[parent_id][:2]
        norm = float(np.linalg.norm(direction))
        normalized = direction / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=float)
        assigned_targets.append(
            (
                parent_id,
                np.asarray(positions[parent_id], dtype=float).copy(),
                positions[leaf_id].copy(),
                normalized,
            )
        )


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
    anchor_node_ids = [
        node_id
        for node_id in (component.anchor_node_ids or component.visible_node_ids)
        if node_id in positions
    ]
    if not anchor_node_ids:
        anchor_node_ids = [node_id for node_id in component.node_ids if node_id in positions]
    chain_node_ids = [node_id for node_id in component.chain_order if node_id in positions]
    if component.structure_kind == "chain" and len(chain_node_ids) >= 2:
        start = positions[chain_node_ids[0]]
        end = positions[chain_node_ids[-1]]
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
    assigned_targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    core_segments: tuple[_BondSegment2D, ...],
    positions: NodePositions,
) -> np.ndarray:
    leaf_node_ids = {node_id for node_id, _ in component.trimmed_leaf_parents}
    direction_options = _preferred_component_directions_2d(
        component,
        positions,
        node_id=parent_id,
        origin=origin,
        prefer_outward_chain_axis=False,
    )
    if not direction_options:
        direction_options = (np.array([0.0, 1.0], dtype=float), np.array([0.0, -1.0], dtype=float))
    preferred_directions = _dedupe_attachment_directions_2d(direction_options)
    angles = np.linspace(0.0, 2.0 * np.pi, _FREE_DIR_SAMPLES_2D, endpoint=False)
    raw_fallback_directions = tuple(
        np.array([float(np.cos(angle)), float(np.sin(angle))], dtype=float) for angle in angles
    )
    all_directions = _dedupe_attachment_directions_2d(
        (*preferred_directions, *raw_fallback_directions),
    )
    fallback_directions = all_directions[len(preferred_directions) :]
    used_dirs: list[np.ndarray] = []
    for neighbor_id in component.contraction_graph.neighbors(parent_id):
        if neighbor_id == leaf_id or neighbor_id in leaf_node_ids:
            continue
        delta = positions[neighbor_id][:2] - origin[:2]
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            used_dirs.append(delta / norm)

    origin_2d = np.asarray(origin, dtype=float).reshape(-1)[:2]
    for directions in (preferred_directions, fallback_directions):
        for radius in _ATTACHMENT_RADII_2D:
            for direction in directions:
                direction_2d = np.asarray(direction, dtype=float).reshape(-1)[:2]
                candidate = origin_2d + direction_2d * float(radius)
                if _attachment_candidate_conflicts_2d(
                    candidate=candidate,
                    direction=direction_2d,
                    parent_id=parent_id,
                    leaf_id=leaf_id,
                    origin=origin_2d,
                    assigned_targets=assigned_targets,
                    core_segments=core_segments,
                    positions=positions,
                    used_dirs=used_dirs,
                ):
                    continue
                return candidate

    best_candidate = origin_2d + (
        np.asarray(preferred_directions[0], dtype=float).reshape(-1)[:2]
        * float(_ATTACHMENT_RADII_2D[-1])
    )
    best_margin = -1e300
    for radius in _ATTACHMENT_RADII_2D:
        for direction in all_directions:
            direction_2d = np.asarray(direction, dtype=float).reshape(-1)[:2]
            candidate = origin_2d + direction_2d * float(radius)
            margin = _attachment_candidate_margin_2d(
                candidate=candidate,
                direction=direction_2d,
                parent_id=parent_id,
                leaf_id=leaf_id,
                origin=origin_2d,
                assigned_targets=assigned_targets,
                core_segments=core_segments,
                positions=positions,
                used_dirs=used_dirs,
            )
            if margin > best_margin:
                best_margin = margin
                best_candidate = candidate.copy()
    return best_candidate


def _dedupe_attachment_directions_2d(
    directions: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    unique: list[np.ndarray] = []
    for direction in directions:
        normalized = np.asarray(direction, dtype=float).reshape(-1)[:2]
        norm = float(np.linalg.norm(normalized))
        if norm < 1e-9:
            continue
        normalized = normalized / norm
        if any(float(np.dot(normalized, other)) > 0.995 for other in unique):
            continue
        unique.append(normalized)
    return tuple(unique)


def _attachment_candidate_conflicts_2d(
    *,
    candidate: np.ndarray,
    direction: np.ndarray,
    parent_id: int,
    leaf_id: int,
    origin: np.ndarray,
    assigned_targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    core_segments: tuple[_BondSegment2D, ...],
    positions: NodePositions,
    used_dirs: list[np.ndarray],
) -> bool:
    return (
        _attachment_candidate_margin_2d(
            candidate=candidate,
            direction=direction,
            parent_id=parent_id,
            leaf_id=leaf_id,
            origin=origin,
            assigned_targets=assigned_targets,
            core_segments=core_segments,
            positions=positions,
            used_dirs=used_dirs,
        )
        < 0.0
    )


def _attachment_candidate_margin_2d(
    *,
    candidate: np.ndarray,
    direction: np.ndarray,
    parent_id: int,
    leaf_id: int,
    origin: np.ndarray,
    assigned_targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    core_segments: tuple[_BondSegment2D, ...],
    positions: NodePositions,
    used_dirs: list[np.ndarray],
) -> float:
    candidate_2d = np.asarray(candidate, dtype=float).reshape(-1)[:2]
    direction_2d = np.asarray(direction, dtype=float).reshape(-1)[:2]
    origin_2d = np.asarray(origin, dtype=float).reshape(-1)[:2]
    direction_norm = float(np.linalg.norm(direction_2d))
    if direction_norm > 1e-9:
        direction_2d = direction_2d / direction_norm
    margin = float("inf")

    for used_dir in used_dirs:
        margin = min(
            margin,
            _ATTACHMENT_PARALLEL_DOT_2D - float(np.dot(direction_2d, used_dir[:2])),
        )

    for other_parent_id, parent_origin, target, target_direction in assigned_targets:
        parent_origin_2d = np.asarray(parent_origin, dtype=float).reshape(-1)[:2]
        target_2d = np.asarray(target, dtype=float).reshape(-1)[:2]
        margin = min(
            margin,
            float(np.linalg.norm(candidate_2d - target_2d)) - _ATTACHMENT_TARGET_CLEAR_2D,
        )
        if _segments_cross_2d(origin_2d, candidate_2d, parent_origin_2d, target_2d):
            margin = min(margin, -1.0)
        if other_parent_id == parent_id:
            margin = min(
                margin,
                _ATTACHMENT_PARALLEL_DOT_2D
                - float(
                    np.dot(
                        direction_2d,
                        np.asarray(target_direction, dtype=float).reshape(-1)[:2],
                    )
                ),
            )

    for node_id, position in positions.items():
        if node_id in {leaf_id, parent_id}:
            continue
        other = np.asarray(position, dtype=float).reshape(-1)[:2]
        margin = min(
            margin,
            float(np.linalg.norm(candidate_2d - other)) - _ATTACHMENT_NODE_CLEAR_2D,
        )

    for record in core_segments:
        if parent_id in record.node_ids:
            continue
        if not _segment_bboxes_overlap_2d(
            origin_2d,
            candidate_2d,
            record.bbox,
            padding=_SEGMENT_BOND_CLEAR_2D,
        ):
            continue
        start = record.start
        end = record.end
        if _segment_hits_existing_geometry_2d(origin_2d, candidate_2d, start, end):
            margin = min(margin, -1.0)
            continue
        margin = min(
            margin,
            _segment_segment_min_distance_2d(origin_2d, candidate_2d, start, end)
            - _SEGMENT_BOND_CLEAR_2D,
        )

    return margin


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


__all__ = [
    "_analyze_layout_components_cached",
    "_best_attachment_position_2d",
    "_best_attachment_position_3d",
    "_center_positions",
    "_choose_promotable_node",
    "_component_main_axis_2d",
    "_compute_component_layout_2d",
    "_compute_layout",
    "_compute_layout_from_components",
    "_lift_component_layout_3d",
    "_next_layer",
    "_node_overlaps_component",
    "_normalize_positions",
    "_nudge_singleton_attachment_virtual_hubs_2d",
    "_offset_virtual_hubs_off_direct_tensor_chords_2d",
    "_pack_component_positions",
    "_place_trimmed_leaf_nodes_2d",
    "_place_trimmed_leaf_nodes_3d",
    "_promote_3d_layers",
    "_snap_virtual_nodes_to_barycenters",
    "_spread_colocated_virtual_hubs_2d",
]
