"""Explicit TeNPy tensor networks: named ``npc.Array`` tensors plus bond specifications.

TeNPy does not provide a generic graph container; this package type is only for visualization.
Each bond lists **two or more** ``(tensor_id, leg_label)`` pairs sharing one contracted index.
Binary bonds use two pairs; *n*-way junctions use a virtual hub (same model as the Quimb backend).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._core.graph_utils import _stringify


def _leg_labels(tensor: Any) -> tuple[str, ...]:
    if not hasattr(tensor, "get_leg_labels"):
        raise TypeError("TeNPy tensors must expose leg labels via 'get_leg_labels()'.")
    return tuple(_stringify(label) for label in tensor.get_leg_labels())


@dataclass(frozen=True)
class TenPyTensorNetwork:
    """Hand-made TeNPy tensor network for ``engine='tenpy'``.

    ``nodes`` preserves drawing order (stable node ids 0..n-1). ``bonds`` groups legs that belong
    to the same contraction index; dangling legs are every axis not listed in any bond.
    """

    nodes: tuple[tuple[str, Any], ...]
    bonds: tuple[tuple[tuple[str, str], ...], ...]


def make_tenpy_tensor_network(
    nodes: Sequence[tuple[str, Any]],
    bonds: Sequence[Sequence[tuple[str, str]]],
) -> TenPyTensorNetwork:
    """Validate and freeze an explicit TeNPy tensor network."""
    node_list = tuple((str(nid), t) for nid, t in nodes)
    if not node_list:
        raise ValueError("TenPyTensorNetwork requires at least one tensor.")

    ids = [nid for nid, _ in node_list]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate tensor ids: {ids!r}.")

    id_to_tensor = dict(node_list)
    bond_tuple: list[tuple[tuple[str, str], ...]] = []
    seen_legs: set[tuple[str, str]] = set()

    for bond_index, raw in enumerate(bonds):
        legs = tuple((str(tid), _stringify(leg)) for tid, leg in raw)
        if len(legs) < 2:
            raise ValueError(f"Bond {bond_index} needs at least two legs, got {legs!r}.")
        for tid, leg in legs:
            if tid not in id_to_tensor:
                raise ValueError(f"Unknown tensor id {tid!r} in bond {bond_index}.")
            labels = _leg_labels(id_to_tensor[tid])
            if leg not in labels:
                raise ValueError(
                    f"Tensor {tid!r} has no leg {leg!r} (available: {labels}) in bond {bond_index}."
                )
            key = (tid, leg)
            if key in seen_legs:
                raise ValueError(f"Leg {key!r} appears in more than one bond.")
            seen_legs.add(key)
        bond_tuple.append(legs)

    return TenPyTensorNetwork(nodes=node_list, bonds=tuple(bond_tuple))


__all__ = ["TenPyTensorNetwork", "make_tenpy_tensor_network"]
