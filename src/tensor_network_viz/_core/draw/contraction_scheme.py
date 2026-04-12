"""Dynamic contraction-slider state reconstruction and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..._interactive_scene import _bring_scene_label_artists_to_front
from ...config import PlotConfig
from ..graph import (
    _ContractionStepMetrics,
    _EdgeData,
    _GraphData,
    _resolve_contraction_scheme_by_name,
)
from .constants import (
    _EDGE_LINE_CAP_STYLE,
    _EDGE_LINE_JOIN_STYLE,
    _OCTAHEDRON_EDGE_LINEWIDTH_FACTOR,
    _OCTAHEDRON_EDGE_LINEWIDTH_MIN,
    _OCTAHEDRON_TRI_COUNT,
    _UNIT_NODE_TRIS,
    _ZORDER_NODE_DISK,
)
from .plotter import _edge_outline_effects, _node_edge_degrees
from .scene_state import _InteractiveSceneState

_DEFAULT_SCHEME_COLORS: tuple[str, ...] = (
    "#C4B5FD",
    "#86EFAC",
    "#FDBA74",
    "#F0ABFC",
    "#A78BFA",
    "#6EE7B7",
    "#FCD34D",
    "#F9A8D4",
)

_CONTRACTION_SCHEME_GID: str = "tnv_contraction_scheme"
_CONTRACTION_SCHEME_LABEL_GID: str = "tnv_contraction_scheme_label"
_EDGE_GID: str = "tnv_contraction_edges"
_NODE_GID_CIRCLE: str = "tnv_tensor_nodes_circle"
_NODE_GID_SQUARE: str = "tnv_tensor_nodes_square"
_NODE_GID_OCTAHEDRON: str = "tnv_tensor_nodes_octahedron"
_NODE_GID_CUBE: str = "tnv_tensor_nodes_cube"
_NODE_EDGE_DARKEN_FACTOR: float = 0.62

_CubeFace = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _PlaybackGroup:
    """Persistent color assignment for tensors that already contracted together."""

    members: frozenset[int]
    color: tuple[float, float, float, float]
    birth_step: int
    color_index: int


@dataclass(frozen=True)
class _NodePlaybackState:
    """Per-node playback status for one contraction step."""

    contracted: bool
    color: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class _ContractionPlaybackState:
    """Per-step scene state used by the contraction playback renderer."""

    current_event: frozenset[int]
    node_states: dict[int, _NodePlaybackState]
    edge_colors: dict[int, tuple[float, float, float, float]]


@dataclass(frozen=True)
class _ManualCostOperand:
    """Live operand state for best-effort graph-derived scheme costs."""

    members: frozenset[int]
    axes: tuple[str, ...]
    shape: tuple[int, ...] | None
    name: str


def _effective_contraction_steps(
    graph: _GraphData,
    config: PlotConfig,
) -> tuple[frozenset[int], ...] | None:
    """Return the contraction scheme that should drive playback for this graph."""
    if config.contraction_scheme_by_name is not None:
        return _resolve_contraction_scheme_by_name(graph, config.contraction_scheme_by_name)
    return graph.contraction_steps


def _contraction_step_metrics_for_draw(
    graph: _GraphData,
    scheme_steps: tuple[frozenset[int], ...],
) -> tuple[_ContractionStepMetrics | None, ...] | None:
    """Return exact trace metrics, or best-effort graph metrics for explicit schemes."""
    metrics = graph.contraction_step_metrics
    base = graph.contraction_steps
    if metrics is not None and base is not None:
        if (
            len(metrics) == len(scheme_steps)
            and len(base) == len(scheme_steps)
            and base == scheme_steps
        ):
            return metrics
        return _manual_contraction_step_metrics_for_draw(graph, scheme_steps)
    if base is not None and base == scheme_steps:
        return None
    return _manual_contraction_step_metrics_for_draw(graph, scheme_steps)


def _manual_contraction_step_metrics_for_draw(
    graph: _GraphData,
    scheme_steps: tuple[frozenset[int], ...],
) -> tuple[_ContractionStepMetrics | None, ...] | None:
    """Infer cost rows for explicit schemes from visible tensor labels and shapes."""
    from ...einsum_module.contraction_cost import metrics_for_labeled_operands

    active_operands: dict[int, _ManualCostOperand] = {}
    node_to_operand: dict[int, int] = {}
    for node_id, node in graph.nodes.items():
        if node.is_virtual:
            continue
        shape = tuple(int(dim) for dim in node.shape) if node.shape is not None else None
        if shape is not None and len(shape) != len(node.axes_names):
            shape = None
        active_operands[int(node_id)] = _ManualCostOperand(
            members=frozenset({int(node_id)}),
            axes=tuple(str(axis) for axis in node.axes_names),
            shape=shape,
            name=node.name or f"Tensor {node_id}",
        )
        node_to_operand[int(node_id)] = int(node_id)

    metrics: list[_ContractionStepMetrics | None] = []
    any_metric = False
    for step_index, step in enumerate(scheme_steps):
        selected_operand_ids = _selected_manual_cost_operand_ids(
            step,
            node_to_operand=node_to_operand,
        )
        selected_operands = tuple(
            active_operands[operand_id] for operand_id in selected_operand_ids
        )
        unselected_operands = tuple(
            operand
            for operand_id, operand in active_operands.items()
            if operand_id not in set(selected_operand_ids)
        )
        output_axes = _manual_cost_output_axes(
            selected_operands,
            unselected_operands=unselected_operands,
        )

        metric: _ContractionStepMetrics | None = None
        if len(selected_operands) >= 2 and all(
            operand.shape is not None for operand in selected_operands
        ):
            try:
                metric = metrics_for_labeled_operands(
                    operand_axes=tuple(operand.axes for operand in selected_operands),
                    operand_shapes=tuple(
                        operand.shape for operand in selected_operands if operand.shape is not None
                    ),
                    output_axes=output_axes,
                    operand_names=tuple(operand.name for operand in selected_operands),
                )
            except ValueError:
                metric = None
        if metric is not None:
            any_metric = True
        metrics.append(metric)

        result_shape = _manual_cost_result_shape(metric)
        merged_members = frozenset(
            member for operand in selected_operands for member in operand.members
        )
        result_operand_id = -(step_index + 1)
        for operand_id in selected_operand_ids:
            active_operands.pop(operand_id, None)
        active_operands[result_operand_id] = _ManualCostOperand(
            members=merged_members,
            axes=output_axes,
            shape=result_shape,
            name="+".join(operand.name for operand in selected_operands),
        )
        for node_id in merged_members:
            node_to_operand[node_id] = result_operand_id

    return tuple(metrics) if any_metric else None


def _selected_manual_cost_operand_ids(
    step: frozenset[int],
    *,
    node_to_operand: dict[int, int],
) -> tuple[int, ...]:
    selected: list[int] = []
    for node_id in sorted(int(node_id) for node_id in step):
        operand_id = node_to_operand.get(node_id)
        if operand_id is None or operand_id in selected:
            continue
        selected.append(operand_id)
    return tuple(selected)


def _manual_cost_output_axes(
    selected_operands: tuple[_ManualCostOperand, ...],
    *,
    unselected_operands: tuple[_ManualCostOperand, ...],
) -> tuple[str, ...]:
    selected_counts: dict[str, int] = {}
    for operand in selected_operands:
        for label in operand.axes:
            selected_counts[label] = selected_counts.get(label, 0) + 1
    outside_labels = {label for operand in unselected_operands for label in operand.axes}

    output_axes: list[str] = []
    seen: set[str] = set()
    for operand in selected_operands:
        for label in operand.axes:
            if label in seen:
                continue
            if selected_counts[label] == 1 or label in outside_labels:
                output_axes.append(label)
                seen.add(label)
    return tuple(output_axes)


def _manual_cost_result_shape(
    metric: _ContractionStepMetrics | None,
) -> tuple[int, ...] | None:
    if metric is None:
        return None
    label_dims = dict(metric.label_dims)
    return tuple(int(label_dims[label]) for label in metric.output_labels)


def _scheme_color_rgba(
    step_color_index: int,
    *,
    config: PlotConfig,
) -> tuple[float, float, float, float]:
    palette = config.contraction_scheme_colors or _DEFAULT_SCHEME_COLORS
    r, g, b, _ = to_rgba(palette[step_color_index % len(palette)])
    return (float(r), float(g), float(b), 1.0)


def _darken_rgba(
    color: tuple[float, float, float, float],
    *,
    factor: float = _NODE_EDGE_DARKEN_FACTOR,
) -> tuple[float, float, float, float]:
    r, g, b, a = color
    return (
        float(np.clip(r * factor, 0.0, 1.0)),
        float(np.clip(g * factor, 0.0, 1.0)),
        float(np.clip(b * factor, 0.0, 1.0)),
        float(a),
    )


def _visible_node_ids(graph: _GraphData) -> tuple[int, ...]:
    return tuple(node_id for node_id, node in graph.nodes.items() if not node.is_virtual)


def _visible_neighbors_by_virtual_node(graph: _GraphData) -> dict[int, frozenset[int]]:
    neighbors: dict[int, set[int]] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        endpoint_ids = tuple(endpoint.node_id for endpoint in edge.endpoints)
        virtual_ids = [node_id for node_id in endpoint_ids if graph.nodes[node_id].is_virtual]
        visible_ids = [node_id for node_id in endpoint_ids if not graph.nodes[node_id].is_virtual]
        if not virtual_ids or not visible_ids:
            continue
        for virtual_id in virtual_ids:
            bucket = neighbors.setdefault(virtual_id, set())
            bucket.update(int(node_id) for node_id in visible_ids)
    return {node_id: frozenset(node_ids) for node_id, node_ids in neighbors.items()}


def _edge_color_for_state(
    edge: _EdgeData,
    *,
    graph: _GraphData,
    node_to_group: dict[int, int],
    groups: dict[int, _PlaybackGroup],
    virtual_visible_neighbors: dict[int, frozenset[int]],
) -> tuple[float, float, float, float] | None:
    """Choose the playback color for one edge, if the edge is currently internal."""
    if edge.kind == "dangling":
        return None
    if edge.kind == "self":
        node_id = int(edge.node_ids[0])
        group_id = node_to_group.get(node_id)
        if group_id is None:
            return None
        return groups[group_id].color

    endpoint_ids = tuple(int(endpoint.node_id) for endpoint in edge.endpoints)
    visible_ids = [node_id for node_id in endpoint_ids if not graph.nodes[node_id].is_virtual]
    if len(visible_ids) == 2:
        left_group = node_to_group.get(visible_ids[0])
        right_group = node_to_group.get(visible_ids[1])
        if left_group is None or left_group != right_group:
            return None
        return groups[left_group].color
    if len(visible_ids) != 1:
        return None

    visible_id = visible_ids[0]
    group_id = node_to_group.get(visible_id)
    if group_id is None:
        return None
    group = groups[group_id]
    virtual_ids = [node_id for node_id in endpoint_ids if graph.nodes[node_id].is_virtual]
    for virtual_id in virtual_ids:
        neighbors = virtual_visible_neighbors.get(virtual_id)
        if neighbors and neighbors <= group.members:
            return group.color
    return None


def _build_playback_state(
    *,
    graph: _GraphData,
    current_event: frozenset[int],
    node_to_group_before: dict[int, int],
    node_to_group_after: dict[int, int],
    groups_after: dict[int, _PlaybackGroup],
    virtual_visible_neighbors: dict[int, frozenset[int]],
) -> _ContractionPlaybackState:
    previous_nodes = frozenset(int(node_id) for node_id in node_to_group_before)
    node_states: dict[int, _NodePlaybackState] = {}
    for node_id in _visible_node_ids(graph):
        group_id = node_to_group_after.get(node_id)
        color = groups_after[group_id].color if group_id is not None else None
        node_states[node_id] = _NodePlaybackState(
            contracted=node_id in previous_nodes,
            color=color,
        )

    edge_colors: dict[int, tuple[float, float, float, float]] = {}
    for edge in graph.edges:
        color = _edge_color_for_state(
            edge,
            graph=graph,
            node_to_group=node_to_group_after,
            groups=groups_after,
            virtual_visible_neighbors=virtual_visible_neighbors,
        )
        if color is not None:
            edge_colors[id(edge)] = color

    return _ContractionPlaybackState(
        current_event=current_event,
        node_states=node_states,
        edge_colors=edge_colors,
    )


def _build_contraction_playback_states(
    *,
    graph: _GraphData,
    steps: tuple[frozenset[int], ...],
    config: PlotConfig,
) -> tuple[_ContractionPlaybackState, ...]:
    """Build the cumulative playback state sequence for every contraction step."""
    virtual_visible_neighbors = _visible_neighbors_by_virtual_node(graph)
    active_groups: dict[int, _PlaybackGroup] = {}
    node_to_group: dict[int, int] = {}
    next_color_index = 0
    states: list[_ContractionPlaybackState] = [
        _build_playback_state(
            graph=graph,
            current_event=frozenset(),
            node_to_group_before={},
            node_to_group_after={},
            groups_after={},
            virtual_visible_neighbors=virtual_visible_neighbors,
        )
    ]

    for step_index, step in enumerate(steps):
        before_node_to_group = dict(node_to_group)
        touched_group_ids: list[int] = []
        for node_id in step:
            group_id = before_node_to_group.get(int(node_id))
            if group_id is not None and group_id not in touched_group_ids:
                touched_group_ids.append(group_id)

        touched_groups = [active_groups[group_id] for group_id in touched_group_ids]
        if touched_groups:
            oldest_group = min(
                touched_groups,
                key=lambda group: (int(group.birth_step), int(group.color_index)),
            )
            color_index = int(oldest_group.color_index)
            birth_step = int(oldest_group.birth_step)
        else:
            color_index = int(next_color_index)
            next_color_index += 1
            birth_step = int(step_index)

        merged_members: set[int] = {int(node_id) for node_id in step}
        for group in touched_groups:
            merged_members.update(int(node_id) for node_id in group.members)
        for group_id in touched_group_ids:
            active_groups.pop(group_id, None)

        new_group = _PlaybackGroup(
            members=frozenset(merged_members),
            color=_scheme_color_rgba(color_index, config=config),
            birth_step=birth_step,
            color_index=color_index,
        )
        new_group_id = int(step_index)
        active_groups[new_group_id] = new_group
        for node_id in merged_members:
            node_to_group[node_id] = new_group_id

        states.append(
            _build_playback_state(
                graph=graph,
                current_event=step,
                node_to_group_before=before_node_to_group,
                node_to_group_after=node_to_group,
                groups_after=active_groups,
                virtual_visible_neighbors=virtual_visible_neighbors,
            )
        )

    return tuple(states)


def _remove_artists(artists: list[Artist]) -> None:
    for artist in artists:
        remover = getattr(artist, "remove", None)
        if callable(remover):
            try:
                remover()
                continue
            except NotImplementedError:
                pass
        setter = getattr(artist, "set_visible", None)
        if callable(setter):
            setter(False)
    artists.clear()


def _set_artists_visible(artists: tuple[Artist, ...], visible: bool) -> None:
    for artist in artists:
        setter = getattr(artist, "set_visible", None)
        if callable(setter):
            setter(bool(visible))


def _scene_base_node_artists(scene: _InteractiveSceneState) -> tuple[Artist, ...]:
    bundle = scene.node_artist_bundles.get(scene.active_node_mode)
    if bundle is None:
        return ()
    return tuple(artist for artist in bundle.artists if isinstance(artist, Artist))


def _scene_base_node_hover_target(scene: _InteractiveSceneState) -> object | None:
    bundle = scene.node_artist_bundles.get(scene.active_node_mode)
    if bundle is None:
        return None
    return bundle.hover_target


def _set_scene_base_graph_visible(scene: _InteractiveSceneState, visible: bool) -> None:
    _set_artists_visible(_scene_base_node_artists(scene), visible)
    _set_artists_visible(tuple(scene.edge_artists), visible)


def _refresh_scene_hover(scene: _InteractiveSceneState) -> None:
    controls = scene.contraction_controls
    refresh_hover = getattr(controls, "_refresh_hover", None)
    if callable(refresh_hover):
        refresh_hover()


def _neutral_node_facecolor(
    *,
    config: PlotConfig,
    degree_one: bool,
) -> tuple[float, float, float, float]:
    return to_rgba(config.node_color_degree_one if degree_one else config.node_color)


def _neutral_node_edgecolor(
    *,
    config: PlotConfig,
    degree_one: bool,
) -> tuple[float, float, float, float]:
    return to_rgba(
        config.node_edge_color_degree_one if degree_one else config.node_edge_color,
    )


def _draw_scheme_edges_2d(
    *,
    scene: _InteractiveSceneState,
    state: _ContractionPlaybackState,
) -> list[Artist]:
    groups: dict[tuple[float, float, float, float], list[np.ndarray]] = {}
    neutral: dict[tuple[float, float, float, float], list[np.ndarray]] = {}
    neutral_bond = to_rgba(scene.config.bond_edge_color)
    neutral_dangling = to_rgba(scene.config.dangling_edge_color)
    for entry in scene.edge_geometry:
        color = state.edge_colors.get(id(entry.edge))
        if color is None:
            base_color = neutral_dangling if entry.edge.kind == "dangling" else neutral_bond
            neutral.setdefault(base_color, []).append(entry.polyline[:, :2])
            continue
        groups.setdefault(color, []).append(entry.polyline[:, :2])

    artists: list[Artist] = []
    for color_map in (neutral, groups):
        for color, polylines in color_map.items():
            coll = LineCollection(
                polylines,
                colors=[color],
                linewidths=float(scene.params.lw),
                zorder=float(_ZORDER_NODE_DISK - 1),
                capstyle=_EDGE_LINE_CAP_STYLE,
                joinstyle=_EDGE_LINE_JOIN_STYLE,
            )
            coll.set_gid(_EDGE_GID)
            coll.set_path_effects(_edge_outline_effects(float(scene.params.lw)))
            scene.ax.add_collection(coll)
            artists.append(coll)
    return artists


def _draw_scheme_edges_3d(
    *,
    scene: _InteractiveSceneState,
    state: _ContractionPlaybackState,
) -> list[Artist]:
    groups: dict[tuple[float, float, float, float], list[np.ndarray]] = {}
    neutral: dict[tuple[float, float, float, float], list[np.ndarray]] = {}
    neutral_bond = to_rgba(scene.config.bond_edge_color)
    neutral_dangling = to_rgba(scene.config.dangling_edge_color)
    for entry in scene.edge_geometry:
        color = state.edge_colors.get(id(entry.edge))
        if color is None:
            base_color = neutral_dangling if entry.edge.kind == "dangling" else neutral_bond
            neutral.setdefault(base_color, []).append(entry.polyline[:, :3])
            continue
        groups.setdefault(color, []).append(entry.polyline[:, :3])

    artists: list[Artist] = []
    for color_map in (neutral, groups):
        for color, polylines in color_map.items():
            coll = Line3DCollection(
                polylines,
                colors=[color],
                linewidths=float(scene.params.lw),
            )
            coll.set_gid(_EDGE_GID)
            coll.set_zorder(float(_ZORDER_NODE_DISK - 1))
            scene.ax.add_collection3d(coll)
            artists.append(coll)
    return artists


def _shape_gid(
    *,
    dimensions: Literal[2, 3],
    contracted: bool,
) -> str:
    if dimensions == 2:
        return _NODE_GID_SQUARE if contracted else _NODE_GID_CIRCLE
    return _NODE_GID_CUBE if contracted else _NODE_GID_OCTAHEDRON


def _node_shape_key(
    *,
    dimensions: Literal[2, 3],
    contracted: bool,
) -> Literal["circle", "square", "octahedron", "cube"]:
    if dimensions == 2:
        return "square" if contracted else "circle"
    return "cube" if contracted else "octahedron"


def _cube_faces_for_node(
    center: np.ndarray,
    *,
    radius: float,
) -> tuple[_CubeFace, ...]:
    half = float(radius) / float(np.sqrt(3.0))
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    corners = np.asarray(
        [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ],
        dtype=float,
    )
    faces: tuple[tuple[int, int, int, int], ...] = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    )
    typed_faces: list[_CubeFace] = []
    for face in faces:
        typed_face: list[tuple[float, float, float]] = []
        for index in face:
            x_coord, y_coord, z_coord = corners[index]
            typed_face.append((float(x_coord), float(y_coord), float(z_coord)))
        typed_faces.append(tuple(typed_face))
    return tuple(typed_faces)


def _draw_scheme_nodes_2d(
    *,
    scene: _InteractiveSceneState,
    state: _ContractionPlaybackState,
) -> tuple[list[Artist], PatchCollection | None]:
    node_degrees = _node_edge_degrees(scene.graph)
    grouped: dict[
        tuple[str, tuple[float, float, float, float], tuple[float, float, float, float]],
        list[np.ndarray],
    ] = {}
    for node_id in scene.visible_node_ids:
        node_state = state.node_states.get(int(node_id))
        if node_state is None:
            continue
        degree_one = node_degrees.get(int(node_id), 0) == 1
        face_color = (
            _neutral_node_facecolor(config=scene.config, degree_one=degree_one)
            if node_state.color is None
            else node_state.color
        )
        edge_color = (
            _neutral_node_edgecolor(config=scene.config, degree_one=degree_one)
            if node_state.color is None
            else _darken_rgba(node_state.color)
        )
        key = (
            _node_shape_key(dimensions=2, contracted=node_state.contracted),
            face_color,
            edge_color,
        )
        grouped.setdefault(key, []).append(np.asarray(scene.positions[node_id], dtype=float))

    artists: list[Artist] = []
    for (shape, face_color, edge_color), coords in grouped.items():
        patches = []
        for coord in coords:
            x = float(coord[0])
            y = float(coord[1])
            if shape == "square":
                side = float(scene.params.r) * 2.0
                patches.append(
                    Rectangle(
                        (x - float(scene.params.r), y - float(scene.params.r)),
                        side,
                        side,
                    )
                )
            else:
                patches.append(Circle((x, y), radius=float(scene.params.r)))
        coll = PatchCollection(
            patches,
            facecolors=[face_color],
            edgecolors=[edge_color],
            linewidths=float(scene.params.lw),
            zorder=float(_ZORDER_NODE_DISK),
            match_original=False,
        )
        coll.set_gid(_shape_gid(dimensions=2, contracted=shape == "square"))
        coll._tnv_node_count = len(coords)  # type: ignore[attr-defined]
        scene.ax.add_collection(coll)
        artists.append(coll)
    hover_patches: list[Circle | Rectangle] = []
    transparent = [(0.0, 0.0, 0.0, 0.0)]
    for node_id in scene.visible_node_ids:
        node_state = state.node_states.get(int(node_id))
        if node_state is None:
            continue
        coord = np.asarray(scene.positions[node_id], dtype=float)
        x = float(coord[0])
        y = float(coord[1])
        if node_state.contracted:
            side = float(scene.params.r) * 2.0
            hover_patches.append(
                Rectangle(
                    (x - float(scene.params.r), y - float(scene.params.r)),
                    side,
                    side,
                )
            )
        else:
            hover_patches.append(Circle((x, y), radius=float(scene.params.r)))
    if not hover_patches:
        return artists, None
    hover_target = PatchCollection(
        hover_patches,
        facecolors=transparent,
        edgecolors=transparent,
        linewidths=0.0,
        zorder=float(_ZORDER_NODE_DISK),
        match_original=False,
    )
    hover_target.set_gid("tnv_contraction_scheme_hover_target")
    scene.ax.add_collection(hover_target)
    artists.append(hover_target)
    return artists, hover_target


def _draw_scheme_nodes_3d(
    *,
    scene: _InteractiveSceneState,
    state: _ContractionPlaybackState,
) -> list[Artist]:
    node_degrees = _node_edge_degrees(scene.graph)
    grouped: dict[
        tuple[str, tuple[float, float, float, float], tuple[float, float, float, float]],
        list[np.ndarray],
    ] = {}
    for node_id in scene.visible_node_ids:
        node_state = state.node_states.get(int(node_id))
        if node_state is None:
            continue
        degree_one = node_degrees.get(int(node_id), 0) == 1
        face_color = (
            _neutral_node_facecolor(config=scene.config, degree_one=degree_one)
            if node_state.color is None
            else node_state.color
        )
        edge_color = (
            _neutral_node_edgecolor(config=scene.config, degree_one=degree_one)
            if node_state.color is None
            else _darken_rgba(node_state.color)
        )
        key = (
            _node_shape_key(dimensions=3, contracted=node_state.contracted),
            face_color,
            edge_color,
        )
        grouped.setdefault(key, []).append(np.asarray(scene.positions[node_id], dtype=float))

    artists: list[Artist] = []
    node_edge_lw = max(
        float(scene.params.lw) * _OCTAHEDRON_EDGE_LINEWIDTH_FACTOR,
        _OCTAHEDRON_EDGE_LINEWIDTH_MIN,
    )
    for (shape, face_color, edge_color), coords in grouped.items():
        if shape == "cube":
            polys = [
                face
                for coord in coords
                for face in _cube_faces_for_node(coord, radius=float(scene.params.r))
            ]
            coll = Poly3DCollection(
                polys,
                facecolors=[face_color] * (6 * len(coords)),
                edgecolors=[edge_color] * (6 * len(coords)),
                linewidths=node_edge_lw,
            )
            coll.set_gid(_NODE_GID_CUBE)
            coll.set_zorder(float(_ZORDER_NODE_DISK))
        else:
            scaled = _UNIT_NODE_TRIS * float(scene.params.r)
            centers = np.stack(coords, axis=0)
            stacked = scaled[np.newaxis, :, :, :] + centers[:, np.newaxis, np.newaxis, :]
            polys = stacked.reshape(-1, 3, 3)
            coll = Poly3DCollection(
                polys,
                facecolors=[face_color] * (_OCTAHEDRON_TRI_COUNT * len(coords)),
                edgecolors=[edge_color] * (_OCTAHEDRON_TRI_COUNT * len(coords)),
                linewidths=node_edge_lw,
            )
            coll.set_gid(_NODE_GID_OCTAHEDRON)
            coll.set_zorder(float(_ZORDER_NODE_DISK))
        coll._tnv_node_count = len(coords)  # type: ignore[attr-defined]
        coll.set_sort_zpos(float(_ZORDER_NODE_DISK))
        scene.ax.add_collection3d(coll)
        artists.append(coll)
    return artists


def _draw_scene_playback_state(
    scene: _InteractiveSceneState,
    state: _ContractionPlaybackState,
) -> tuple[list[Artist], PatchCollection | None]:
    if scene.dimensions == 2:
        edge_artists = _draw_scheme_edges_2d(scene=scene, state=state)
        node_artists, hover_target = _draw_scheme_nodes_2d(scene=scene, state=state)
        return [*edge_artists, *node_artists], hover_target
    edge_artists = _draw_scheme_edges_3d(scene=scene, state=state)
    node_artists = _draw_scheme_nodes_3d(scene=scene, state=state)
    return [*edge_artists, *node_artists], None


class _ContractionSceneApplier:
    """Apply dynamic contraction-slider states to an interactive scene."""

    def __init__(
        self,
        *,
        states: tuple[_ContractionPlaybackState, ...],
    ) -> None:
        self._states = states
        self._scene: _InteractiveSceneState | None = None
        self._enabled: bool = False
        self._current_step: int = 0

    def bind_scene(self, scene: _InteractiveSceneState) -> None:
        self._scene = scene
        if self._enabled:
            self.apply_step(self._current_step)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self._scene is None:
            return
        if not self._enabled:
            _remove_artists(self._scene.scheme_artists)
            _set_scene_base_graph_visible(self._scene, True)
            self._scene.node_patch_coll = _scene_base_node_hover_target(self._scene)
            _bring_scene_label_artists_to_front(self._scene)
            _refresh_scene_hover(self._scene)
            self._scene.ax.figure.canvas.draw_idle()
            return
        self.apply_step(self._current_step)

    def apply_step(self, step: int) -> None:
        self._current_step = int(np.clip(step, 0, len(self._states) - 1))
        if not self._enabled or self._scene is None:
            return
        _remove_artists(self._scene.scheme_artists)
        _set_scene_base_graph_visible(self._scene, False)
        state = self._states[self._current_step]
        scheme_artists, hover_target = _draw_scene_playback_state(self._scene, state)
        self._scene.scheme_artists = scheme_artists
        if self._scene.dimensions == 2:
            self._scene.node_patch_coll = (
                hover_target
                if hover_target is not None
                else _scene_base_node_hover_target(self._scene)
            )
        _bring_scene_label_artists_to_front(self._scene)
        _refresh_scene_hover(self._scene)
        self._scene.ax.figure.canvas.draw_idle()


__all__ = [
    "_CONTRACTION_SCHEME_GID",
    "_CONTRACTION_SCHEME_LABEL_GID",
    "_ContractionPlaybackState",
    "_ContractionSceneApplier",
    "_build_contraction_playback_states",
    "_contraction_step_metrics_for_draw",
    "_draw_scene_playback_state",
    "_effective_contraction_steps",
]
