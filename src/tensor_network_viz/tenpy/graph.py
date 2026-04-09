"""TeNPy-specific normalization into the shared graph model."""

from __future__ import annotations

from typing import Any

from .._core.graph import (
    _coerce_shape,
    _EdgeEndpoint,
    _element_count_for_shape,
    _estimated_nbytes_for_node,
    _finalize_graph_diagnostics,
    _GraphData,
    _make_contraction_edge,
    _make_dangling_edge,
    _make_node,
    _NodeData,
)
from .._core.graph_utils import _stringify
from .explicit import TenPyTensorNetwork

_SUPPORTED_NETWORKS_HINT: str = (
    "Expected TenPyTensorNetwork (explicit npc.Array list + bonds), a TeNPy tensor chain: "
    "MPO (get_W), MomentumMPS-like (get_X + uMPS_GS), or MPS-like (get_B)."
)


def _boundary_mode_from_geometry(geometry: Any) -> str:
    if not hasattr(geometry, "bc") or not hasattr(geometry, "finite"):
        raise TypeError(
            "TeNPy network geometry must expose boundary metadata via attributes 'bc' and 'finite'."
        )
    boundary_condition = _stringify(getattr(geometry, "bc", None)).lower()
    if boundary_condition in {"finite", "segment"}:
        return "open"
    if boundary_condition == "infinite" and not bool(geometry.finite):
        return "periodic"
    raise ValueError("TeNPy visualization supports finite, segment, and infinite networks.")


def _leg_labels(tensor: Any) -> tuple[str, ...]:
    if not hasattr(tensor, "get_leg_labels"):
        raise TypeError("TeNPy tensors must expose leg labels via 'get_leg_labels()'.")
    return tuple(_stringify(label) for label in tensor.get_leg_labels())


def _find_leg_index(labels: tuple[str, ...], leg_name: str, *, node_name: str) -> int:
    try:
        return labels.index(leg_name)
    except ValueError as exc:
        raise TypeError(f"Tensor {node_name!r} is missing required leg {leg_name!r}.") from exc


def _build_hyperedge_hub(
    *,
    hub_label: str,
    endpoints: list[_EdgeEndpoint],
) -> _NodeData:
    axis_names = tuple(f"{hub_label}__branch_{index}" for index in range(len(endpoints)))
    return _make_node(
        name="",
        axes_names=axis_names,
        label=hub_label or None,
        is_virtual=True,
    )


def _tensor_chain_bonds(
    length: int,
    boundary_mode: str,
    left_leg: str,
    right_leg: str,
    names: list[str],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    bonds: list[tuple[tuple[str, str], ...]] = []
    for i in range(length - 1):
        bonds.append(((names[i], right_leg), (names[i + 1], left_leg)))
    if boundary_mode == "periodic" and length > 0:
        bonds.append(((names[length - 1], right_leg), (names[0], left_leg)))
    return tuple(bonds)


def _build_explicit_tn_graph(tn: TenPyTensorNetwork) -> _GraphData:
    nodes: dict[int, Any] = {}
    labels_by_node: dict[int, tuple[str, ...]] = {}
    id_to_int: dict[str, int] = {}

    for index, (tensor_id, array) in enumerate(tn.nodes):
        tid = str(tensor_id)
        id_to_int[tid] = index
        labels = _leg_labels(array)
        labels_by_node[index] = labels
        shape = _coerce_shape(getattr(array, "shape", None))
        dtype_attr = getattr(array, "dtype", None)
        dtype_text = None if dtype_attr is None else str(dtype_attr)
        element_count = _element_count_for_shape(shape)
        nodes[index] = _make_node(
            name=tid,
            axes_names=labels,
            shape=shape,
            dtype=dtype_text,
            element_count=element_count,
            estimated_nbytes=_estimated_nbytes_for_node(
                shape,
                dtype_text,
                element_count=element_count,
            ),
        )

    used_axes: set[tuple[int, int]] = set()
    edges: list[Any] = []
    next_hub_id = -1

    for bond_index, bond in enumerate(tn.bonds):
        endpoints: list[_EdgeEndpoint] = []
        for tid, leg in bond:
            ni = id_to_int[str(tid)]
            labels = labels_by_node[ni]
            axis_index = _find_leg_index(labels, str(leg), node_name=str(tid))
            used_axes.add((ni, axis_index))
            endpoints.append(
                _EdgeEndpoint(node_id=ni, axis_index=axis_index, axis_name=str(leg)),
            )

        bond_name = f"b{bond_index}"
        if len(endpoints) == 2:
            edges.append(
                _make_contraction_edge(endpoints[0], endpoints[1], name=bond_name),
            )
            continue

        hub_id = next_hub_id
        next_hub_id -= 1
        nodes[hub_id] = _build_hyperedge_hub(hub_label=bond_name, endpoints=endpoints)
        hub_axes = nodes[hub_id].axes_names
        for branch_index, endpoint in enumerate(endpoints):
            hub_ep = _EdgeEndpoint(
                node_id=hub_id,
                axis_index=branch_index,
                axis_name=hub_axes[branch_index],
            )
            edges.append(
                _make_contraction_edge(endpoint, hub_ep, name=bond_name),
            )

    for ni, labels in labels_by_node.items():
        for axis_index, label in enumerate(labels):
            if (ni, axis_index) in used_axes:
                continue
            endpoint = _EdgeEndpoint(
                node_id=ni,
                axis_index=axis_index,
                axis_name=label,
            )
            edges.append(_make_dangling_edge(endpoint, name=label or None, label=label or None))

    return _finalize_graph_diagnostics(_GraphData(nodes=nodes, edges=tuple(edges)))


def _build_mps_graph(network: Any) -> _GraphData:
    boundary_mode = _boundary_mode_from_geometry(network)
    length = int(network.L)
    names = [f"B{i}" for i in range(length)]
    nodes = tuple((names[i], network.get_B(i, form=None)) for i in range(length))
    bonds = _tensor_chain_bonds(length, boundary_mode, "vL", "vR", names)
    return _build_explicit_tn_graph(TenPyTensorNetwork(nodes=nodes, bonds=bonds))


def _build_mpo_graph(network: Any) -> _GraphData:
    boundary_mode = _boundary_mode_from_geometry(network)
    length = int(network.L)
    names = [f"W{i}" for i in range(length)]
    nodes = tuple((names[i], network.get_W(i)) for i in range(length))
    bonds = _tensor_chain_bonds(length, boundary_mode, "wL", "wR", names)
    return _build_explicit_tn_graph(TenPyTensorNetwork(nodes=nodes, bonds=bonds))


def _build_momentum_mps_graph(network: Any) -> _GraphData:
    geometry = network.uMPS_GS
    boundary_mode = _boundary_mode_from_geometry(geometry)
    length = int(geometry.L)
    names = [f"X{i}" for i in range(length)]
    nodes = tuple((names[i], network.get_X(i)) for i in range(length))
    bonds = _tensor_chain_bonds(length, boundary_mode, "vL", "vR", names)
    return _build_explicit_tn_graph(TenPyTensorNetwork(nodes=nodes, bonds=bonds))


def _is_momentum_mps_like(network: Any) -> bool:
    return callable(getattr(network, "get_X", None)) and hasattr(network, "uMPS_GS")


def _build_graph(network: Any) -> _GraphData:
    if isinstance(network, TenPyTensorNetwork):
        return _build_explicit_tn_graph(network)
    if callable(getattr(network, "get_W", None)):
        return _build_mpo_graph(network)
    if _is_momentum_mps_like(network):
        return _build_momentum_mps_graph(network)
    if callable(getattr(network, "get_B", None)):
        return _build_mps_graph(network)
    raise TypeError(
        f"Unsupported TeNPy input: {type(network).__name__!r}. {_SUPPORTED_NETWORKS_HINT}"
    )
