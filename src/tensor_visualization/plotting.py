from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

Vector: TypeAlias = np.ndarray
EdgeKind = Literal["contraction", "dangling", "self"]
NodePositions: TypeAlias = dict[int, Vector]
AxisDirections: TypeAlias = dict[tuple[int, int], Vector]


_NODE_COLOR = "#2D6A9F"
_EDGE_COLOR = "#202B33"
_LABEL_COLOR = "#0C1319"
_NODE_RADIUS = 0.08
_STUB_LENGTH = 0.34
_SELF_LOOP_RADIUS = 0.2


@dataclass(frozen=True)
class _EdgeEndpoint:
    node_id: int
    axis_index: int
    axis_name: str | None


@dataclass(frozen=True)
class _NodeData:
    ref: Any
    name: str
    axes_names: tuple[str, ...]
    degree: int


@dataclass(frozen=True)
class _EdgeData:
    ref: Any
    name: str | None
    kind: EdgeKind
    node_ids: tuple[int, ...]
    endpoints: tuple[_EdgeEndpoint, ...]
    label: str | None


@dataclass(frozen=True)
class _GraphData:
    nodes: dict[int, _NodeData]
    edges: tuple[_EdgeData, ...]


def plot_tensor_network_2d(
    network: Any,
    *,
    ax: Axes | None = None,
    show_tensor_labels: bool = True,
    show_index_labels: bool = True,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    graph = _build_graph(network)
    fig, ax = _prepare_axes_2d(ax=ax)
    positions = _compute_layout(graph, dimensions=2, seed=seed)
    directions = _compute_axis_directions(graph, positions, dimensions=2)
    _draw_2d(
        ax=ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
    )
    return fig, ax


def plot_tensor_network_3d(
    network: Any,
    *,
    ax: Axes | Axes3D | None = None,
    show_tensor_labels: bool = True,
    show_index_labels: bool = True,
    seed: int = 0,
) -> tuple[Figure, Axes3D]:
    graph = _build_graph(network)
    fig, ax = _prepare_axes_3d(ax=ax)
    positions = _compute_layout(graph, dimensions=3, seed=seed)
    directions = _compute_axis_directions(graph, positions, dimensions=3)
    _draw_3d(
        ax=ax,
        graph=graph,
        positions=positions,
        directions=directions,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
    )
    return fig, ax


def _build_graph(network: Any) -> _GraphData:
    node_refs = _get_network_nodes(network)
    if not node_refs:
        raise ValueError("The tensor network does not expose any nodes to visualize.")

    nodes: dict[int, _NodeData] = {}
    edge_refs: dict[int, Any] = {}
    edge_endpoints: dict[int, list[_EdgeEndpoint]] = {}

    for node in node_refs:
        name = _stringify(_require_attr(node, "name", "node"))
        edges = tuple(_iterable_attr(node, "edges", "node"))
        axes_names = tuple(_stringify(item) for item in _iterable_attr(node, "axes_names", "node"))
        if len(edges) != len(axes_names):
            raise TypeError(
                f"Node {name!r} has {len(edges)} edges but {len(axes_names)} axes_names."
            )

        node_id = id(node)
        nodes[node_id] = _NodeData(
            ref=node,
            name=name,
            axes_names=axes_names,
            degree=len(edges),
        )

        for axis_index, edge in enumerate(edges):
            if edge is None:
                continue
            edge_id = id(edge)
            edge_refs[edge_id] = edge
            edge_endpoints.setdefault(edge_id, []).append(
                _EdgeEndpoint(
                    node_id=node_id,
                    axis_index=axis_index,
                    axis_name=axes_names[axis_index],
                )
            )

    edges: list[_EdgeData] = []
    for edge_id, edge in edge_refs.items():
        name = _optional_string(_require_attr(edge, "name", "edge"))
        node1 = _require_attr(edge, "node1", "edge")
        node2 = _require_attr(edge, "node2", "edge")

        node1_id = id(node1) if node1 is not None and id(node1) in nodes else None
        node2_id = id(node2) if node2 is not None and id(node2) in nodes else None
        endpoints = tuple(edge_endpoints.get(edge_id, ()))
        if not endpoints:
            continue
        if len(endpoints) > 2:
            raise TypeError("Edges with more than two endpoints are not supported.")

        if node1_id is not None and node2_id is not None:
            kind: EdgeKind
            if node1_id == node2_id:
                kind = "self"
                node_ids = (node1_id,)
            else:
                kind = "contraction"
                node_ids = (node1_id, node2_id)
        elif node1_id is not None:
            kind = "dangling"
            node_ids = (node1_id,)
        elif node2_id is not None:
            kind = "dangling"
            node_ids = (node2_id,)
        else:
            continue

        edges.append(
            _EdgeData(
                ref=edge,
                name=name,
                kind=kind,
                node_ids=node_ids,
                endpoints=endpoints,
                label=_build_edge_label(kind=kind, endpoints=endpoints, edge_name=name),
            )
        )

    return _GraphData(nodes=nodes, edges=tuple(edges))


def _prepare_axes_2d(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        return fig, ax

    if getattr(ax, "name", "") == "3d":
        raise ValueError("plot_tensor_network_2d requires a 2D Matplotlib axis.")
    return ax.figure, ax


def _prepare_axes_3d(ax: Axes | Axes3D | None) -> tuple[Figure, Axes3D]:
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        created_ax = fig.add_subplot(111, projection="3d")
        return fig, cast(Axes3D, created_ax)

    if getattr(ax, "name", "") != "3d":
        raise ValueError("plot_tensor_network_3d requires a 3D Matplotlib axis.")
    return ax.figure, cast(Axes3D, ax)


def _get_network_nodes(network: Any) -> list[Any]:
    if hasattr(network, "nodes"):
        raw_nodes = network.nodes
    elif hasattr(network, "leaf_nodes"):
        raw_nodes = network.leaf_nodes
    else:
        raise TypeError(
            "Tensor network must expose either a 'nodes' attribute or a 'leaf_nodes' attribute."
        )

    if isinstance(raw_nodes, dict):
        iterable = raw_nodes.values()
    else:
        iterable = raw_nodes

    try:
        items = list(iterable)
    except TypeError as exc:
        raise TypeError("Tensor network nodes must be iterable.") from exc

    unique_nodes: list[Any] = []
    seen: set[int] = set()
    for node in items:
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        unique_nodes.append(node)
    return unique_nodes


def _iterable_attr(obj: Any, attr_name: str, object_name: str) -> list[Any]:
    value = _require_attr(obj, attr_name, object_name)
    if isinstance(value, dict):
        return list(value.values())
    try:
        return list(value)
    except TypeError as exc:
        msg = f"{object_name.capitalize()} attribute '{attr_name}' must be iterable."
        raise TypeError(msg) from exc


def _require_attr(obj: Any, attr_name: str, object_name: str) -> Any:
    if not hasattr(obj, attr_name):
        raise TypeError(f"{object_name.capitalize()} is missing required attribute '{attr_name}'.")
    return getattr(obj, attr_name)


def _stringify(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _build_edge_label(
    kind: EdgeKind,
    endpoints: tuple[_EdgeEndpoint, ...],
    edge_name: str | None,
) -> str | None:
    axis_names = [endpoint.axis_name for endpoint in endpoints if endpoint.axis_name]
    if kind == "dangling":
        return axis_names[0] if axis_names else edge_name
    if len(axis_names) >= 2:
        return f"{axis_names[0]}<->{axis_names[1]}"
    return edge_name


def _compute_layout(graph: _GraphData, dimensions: int, seed: int) -> NodePositions:
    node_ids = list(graph.nodes)
    if len(node_ids) == 1:
        origin = np.zeros(dimensions, dtype=float)
        return {node_ids[0]: origin}

    positions = _initial_positions(node_ids, dimensions=dimensions, seed=seed)
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}

    pair_weights: dict[tuple[int, int], int] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        left, right = edge.node_ids
        key = tuple(sorted((left, right)))
        pair_weights[key] = pair_weights.get(key, 0) + 1

    k = 1.6
    temperature = 0.12
    for _ in range(220):
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, 1.0)
        directions = deltas / np.maximum(distances[..., None], 1e-6)
        repulsion = (k * k / np.maximum(distances, 1e-6) ** 2)[..., None] * directions
        displacement = repulsion.sum(axis=1)

        for (left_id, right_id), weight in pair_weights.items():
            left_index = index_by_node[left_id]
            right_index = index_by_node[right_id]
            delta = positions[right_index] - positions[left_index]
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            direction = delta / distance
            attraction = weight * distance * distance / k * direction
            displacement[left_index] += attraction
            displacement[right_index] -= attraction

        norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        positions += displacement / np.maximum(norms, 1e-6) * temperature
        positions -= positions.mean(axis=0, keepdims=True)
        max_norm = np.linalg.norm(positions, axis=1).max()
        if max_norm > 1.6:
            positions /= max_norm / 1.6
        temperature *= 0.985

    return {node_id: positions[index] for index, node_id in enumerate(node_ids)}


def _initial_positions(node_ids: list[int], dimensions: int, seed: int) -> Vector:
    count = len(node_ids)
    rng = np.random.default_rng(seed)

    if dimensions == 2:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        positions = np.zeros((count, 3), dtype=float)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(count):
            y = 1.0 - (2.0 * index) / max(count - 1, 1)
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * index
            positions[index] = np.array(
                [math.cos(theta) * radius, y, math.sin(theta) * radius],
                dtype=float,
            )

    positions += rng.normal(loc=0.0, scale=0.03, size=positions.shape)
    return positions


def _compute_axis_directions(
    graph: _GraphData,
    positions: NodePositions,
    dimensions: int,
) -> AxisDirections:
    center = np.mean(np.stack(list(positions.values())), axis=0)
    directions: AxisDirections = {}

    for node_id, node in graph.nodes.items():
        origin = positions[node_id]
        radial = origin - center
        if np.linalg.norm(radial) < 1e-6:
            radial = np.zeros(dimensions, dtype=float)
            radial[0] = 1.0
        radial = radial / np.linalg.norm(radial)

        if dimensions == 2:
            base_angle = math.atan2(radial[1], radial[0])
            for axis_index in range(max(node.degree, 1)):
                angle = base_angle + 2.0 * math.pi * axis_index / max(node.degree, 1)
                directions[(node_id, axis_index)] = np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=float,
                )
        else:
            basis_a = _orthogonal_unit(radial)
            basis_b = np.cross(radial, basis_a)
            basis_b = basis_b / np.linalg.norm(basis_b)
            for axis_index in range(max(node.degree, 1)):
                angle = 2.0 * math.pi * axis_index / max(node.degree, 1)
                direction = radial + 0.55 * (
                    math.cos(angle) * basis_a + math.sin(angle) * basis_b
                )
                directions[(node_id, axis_index)] = direction / np.linalg.norm(direction)

    return directions


def _orthogonal_unit(vector: Vector) -> Vector:
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    orthogonal = np.cross(vector, reference)
    return orthogonal / np.linalg.norm(orthogonal)


def _draw_2d(
    *,
    ax: Axes,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
) -> None:
    ax.cla()
    pair_groups = _group_contractions(graph)

    for edge in graph.edges:
        if edge.kind == "dangling":
            endpoint = edge.endpoints[0]
            direction = directions[(endpoint.node_id, endpoint.axis_index)]
            start = positions[endpoint.node_id] + direction * _NODE_RADIUS
            end = start + direction * _STUB_LENGTH
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=_EDGE_COLOR,
                linewidth=1.8,
                zorder=2,
            )
            if show_index_labels and edge.label:
                label_pos = end + direction * 0.08
                ax.text(label_pos[0], label_pos[1], edge.label, color=_LABEL_COLOR, fontsize=9)
        elif edge.kind == "self":
            endpoint_a, endpoint_b = _require_self_endpoints(edge)
            direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
            direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
            orientation = direction_a + direction_b
            if np.linalg.norm(orientation) < 1e-6:
                orientation = np.array([1.0, 0.0], dtype=float)
            orientation = orientation / np.linalg.norm(orientation)
            normal = np.array([-orientation[1], orientation[0]], dtype=float)
            center = (
                positions[endpoint_a.node_id]
                + orientation * (_NODE_RADIUS + _SELF_LOOP_RADIUS)
            )
            curve = _ellipse_points(center, orientation, normal, width=0.16, height=0.12)
            ax.plot(curve[:, 0], curve[:, 1], color=_EDGE_COLOR, linewidth=1.8, zorder=2)
            if show_index_labels and edge.label:
                label_pos = center + normal * 0.16
                ax.text(label_pos[0], label_pos[1], edge.label, color=_LABEL_COLOR, fontsize=9)
        else:
            key = tuple(sorted(edge.node_ids))
            group = pair_groups[key]
            offset_index = group.index(edge)
            curve = _curved_edge_points_2d(
                start=positions[edge.node_ids[0]],
                end=positions[edge.node_ids[1]],
                offset_index=offset_index,
                edge_count=len(group),
            )
            ax.plot(curve[:, 0], curve[:, 1], color=_EDGE_COLOR, linewidth=1.8, zorder=1)
            if show_index_labels and edge.label:
                midpoint = curve[len(curve) // 2]
                ax.text(midpoint[0], midpoint[1], edge.label, color=_LABEL_COLOR, fontsize=9)

    coords = np.stack(list(positions.values()))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=900,
        c=_NODE_COLOR,
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )

    if show_tensor_labels:
        for node_id, node in graph.nodes.items():
            x, y = positions[node_id]
            ax.text(x, y, node.name, color="white", ha="center", va="center", fontsize=10, zorder=4)

    _style_2d_axes(ax, coords)


def _draw_3d(
    *,
    ax: Axes3D,
    graph: _GraphData,
    positions: NodePositions,
    directions: AxisDirections,
    show_tensor_labels: bool,
    show_index_labels: bool,
) -> None:
    ax.cla()
    pair_groups = _group_contractions(graph)

    for edge in graph.edges:
        if edge.kind == "dangling":
            endpoint = edge.endpoints[0]
            direction = directions[(endpoint.node_id, endpoint.axis_index)]
            start = positions[endpoint.node_id] + direction * _NODE_RADIUS
            end = start + direction * _STUB_LENGTH
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=_EDGE_COLOR,
                linewidth=1.6,
                zorder=2,
            )
            if show_index_labels and edge.label:
                label_pos = end + direction * 0.08
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    edge.label,
                    color=_LABEL_COLOR,
                    fontsize=9,
                )
        elif edge.kind == "self":
            endpoint_a, endpoint_b = _require_self_endpoints(edge)
            direction_a = directions[(endpoint_a.node_id, endpoint_a.axis_index)]
            direction_b = directions[(endpoint_b.node_id, endpoint_b.axis_index)]
            orientation = direction_a + direction_b
            if np.linalg.norm(orientation) < 1e-6:
                orientation = np.array([1.0, 0.0, 0.0], dtype=float)
            orientation = orientation / np.linalg.norm(orientation)
            normal = _orthogonal_unit(orientation)
            binormal = np.cross(orientation, normal)
            binormal = binormal / np.linalg.norm(binormal)
            center = (
                positions[endpoint_a.node_id]
                + orientation * (_NODE_RADIUS + _SELF_LOOP_RADIUS)
            )
            curve = _ellipse_points_3d(center, normal, binormal, width=0.16, height=0.12)
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color=_EDGE_COLOR,
                linewidth=1.6,
                zorder=2,
            )
            if show_index_labels and edge.label:
                label_pos = center + binormal * 0.16
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    edge.label,
                    color=_LABEL_COLOR,
                    fontsize=9,
                )
        else:
            key = tuple(sorted(edge.node_ids))
            group = pair_groups[key]
            offset_index = group.index(edge)
            curve = _curved_edge_points_3d(
                start=positions[edge.node_ids[0]],
                end=positions[edge.node_ids[1]],
                offset_index=offset_index,
                edge_count=len(group),
            )
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color=_EDGE_COLOR,
                linewidth=1.6,
                zorder=1,
            )
            if show_index_labels and edge.label:
                midpoint = curve[len(curve) // 2]
                ax.text(
                    midpoint[0],
                    midpoint[1],
                    midpoint[2],
                    edge.label,
                    color=_LABEL_COLOR,
                    fontsize=9,
                )

    coords = np.stack(list(positions.values()))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=120, c=_NODE_COLOR, depthshade=False)

    if show_tensor_labels:
        for node_id, node in graph.nodes.items():
            x, y, z = positions[node_id]
            ax.text(x, y, z, node.name, color="white", fontsize=10, ha="center", va="center")

    _style_3d_axes(ax, coords)


def _group_contractions(graph: _GraphData) -> dict[tuple[int, int], list[_EdgeData]]:
    groups: dict[tuple[int, int], list[_EdgeData]] = {}
    for edge in graph.edges:
        if edge.kind != "contraction":
            continue
        key = tuple(sorted(edge.node_ids))
        groups.setdefault(key, []).append(edge)
    return groups


def _curved_edge_points_2d(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    offset = (offset_index - (edge_count - 1) / 2.0) * 0.18 * distance
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


def _curved_edge_points_3d(
    *,
    start: np.ndarray,
    end: np.ndarray,
    offset_index: int,
    edge_count: int,
) -> np.ndarray:
    midpoint = (start + end) / 2.0
    delta = end - start
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    perpendicular = np.cross(direction, reference)
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    offset = (offset_index - (edge_count - 1) / 2.0) * 0.18 * distance
    control = midpoint + perpendicular * offset
    return _quadratic_curve(start, control, end)


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


def _style_2d_axes(ax: Axes, coords: Vector) -> None:
    span = np.ptp(coords, axis=0)
    span = np.maximum(span, 1.0)
    center = coords.mean(axis=0)
    ax.set_xlim(center[0] - span[0] * 0.9, center[0] + span[0] * 0.9)
    ax.set_ylim(center[1] - span[1] * 0.9, center[1] + span[1] * 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _style_3d_axes(ax: Axes3D, coords: Vector) -> None:
    span = np.ptp(coords, axis=0)
    span = np.maximum(span, 1.0)
    center = coords.mean(axis=0)
    ax.set_xlim(center[0] - span[0] * 0.9, center[0] + span[0] * 0.9)
    ax.set_ylim(center[1] - span[1] * 0.9, center[1] + span[1] * 0.9)
    ax.set_zlim(center[2] - span[2] * 0.9, center[2] + span[2] * 0.9)
    ax.set_box_aspect(span)
    ax.set_axis_off()


def _require_self_endpoints(edge: _EdgeData) -> tuple[_EdgeEndpoint, _EdgeEndpoint]:
    if len(edge.endpoints) < 2:
        raise TypeError("Self-edges must expose two endpoints.")
    return edge.endpoints[0], edge.endpoints[1]
