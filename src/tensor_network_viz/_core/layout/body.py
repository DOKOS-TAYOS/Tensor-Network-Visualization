"""Layout orchestration compatibility facade."""

from __future__ import annotations

import numpy as np

from ..contractions import _ContractionGroups, _iter_contractions
from ..graph import _GraphData
from ..layout_structure import _LayoutComponent
from .direction_common import (
    _direction_from_axis_name,
    _direction_has_space,
    _is_dangling_leg_axis,
    _orthogonal_unit,
    _used_axis_directions,
)
from .free_directions_2d import _compute_free_directions_2d, _direction_conflicts_2d
from .free_directions_3d import _compute_free_directions_3d, _direction_conflicts_3d
from .geometry import (
    _bond_perpendicular_unoriented_2d,
    _dangling_stub_segment_2d,
    _dangling_stub_segment_3d,
    _layout_quadratic_bond_polyline_2d,
    _normalize_2d,
    _planar_contraction_bond_segments_2d,
    _point_segment_distance_2d,
    _segment_hits_existing_geometry_2d,
    _segment_point_min_distance_sq_2d,
    _segment_point_min_distance_sq_3d,
    _segments_cross_2d,
)
from .positions import (
    _analyze_layout_components_cached,
    _best_attachment_position_2d,
    _best_attachment_position_3d,
    _center_positions,
    _choose_promotable_node,
    _component_main_axis_2d,
    _compute_component_layout_2d,
    _compute_layout,
    _compute_layout_from_components,
    _lift_component_layout_3d,
    _next_layer,
    _node_overlaps_component,
    _normalize_positions,
    _nudge_singleton_attachment_virtual_hubs_2d,
    _offset_virtual_hubs_off_direct_tensor_chords_2d,
    _pack_component_positions,
    _place_trimmed_leaf_nodes_2d,
    _place_trimmed_leaf_nodes_3d,
    _promote_3d_layers,
    _snap_virtual_nodes_to_barycenters,
    _spread_colocated_virtual_hubs_2d,
)
from .types import AxisDirections, NodePositions


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
            else _analyze_layout_components_cached(graph)
        )
        _compute_free_directions_3d(
            graph,
            positions,
            directions,
            draw_scale=draw_scale,
            layout_components=components_3d,
        )

    return directions


__all__ = [
    "_analyze_layout_components_cached",
    "_best_attachment_position_2d",
    "_best_attachment_position_3d",
    "_bond_perpendicular_unoriented_2d",
    "_center_positions",
    "_choose_promotable_node",
    "_component_main_axis_2d",
    "_compute_axis_directions",
    "_compute_component_layout_2d",
    "_compute_free_directions_2d",
    "_compute_free_directions_3d",
    "_compute_layout",
    "_compute_layout_from_components",
    "_dangling_stub_segment_2d",
    "_dangling_stub_segment_3d",
    "_direction_conflicts_2d",
    "_direction_conflicts_3d",
    "_direction_from_axis_name",
    "_direction_has_space",
    "_is_dangling_leg_axis",
    "_layout_quadratic_bond_polyline_2d",
    "_lift_component_layout_3d",
    "_next_layer",
    "_node_overlaps_component",
    "_normalize_2d",
    "_normalize_positions",
    "_nudge_singleton_attachment_virtual_hubs_2d",
    "_offset_virtual_hubs_off_direct_tensor_chords_2d",
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
