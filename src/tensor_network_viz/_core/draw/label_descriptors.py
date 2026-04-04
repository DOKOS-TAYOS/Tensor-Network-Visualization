from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


@dataclass(frozen=True)
class _TextLabelDescriptor:
    position: np.ndarray
    text: str
    kwargs: dict[str, Any]
    node_id: int | None = None


@dataclass(frozen=True)
class _DeferredBondLabelDescriptor:
    text: str
    point: np.ndarray
    tangent_geom: np.ndarray
    tangent_align: np.ndarray
    bond_start: np.ndarray
    bond_end: np.ndarray
    text_endpoint: Literal["left", "right"]
    stub_kind: Literal["bond", "dangling"]
    is_physical: bool
    peer_captions_for_width: tuple[str, ...] | None = None
    zorder: float | None = None


@dataclass(frozen=True)
class _DeferredSelfLoopLabelDescriptor:
    text: str
    point: np.ndarray
    tangent: np.ndarray
    bond_start: np.ndarray
    bond_end: np.ndarray
    offset_direction: np.ndarray
    offset_scale: float
    text_endpoint: Literal["left", "right"]
    peer_captions_for_width: tuple[str, ...] | None = None
    zorder: float | None = None


_AnyLabelDescriptor: TypeAlias = (
    _TextLabelDescriptor | _DeferredBondLabelDescriptor | _DeferredSelfLoopLabelDescriptor
)


__all__ = [
    "_AnyLabelDescriptor",
    "_DeferredBondLabelDescriptor",
    "_DeferredSelfLoopLabelDescriptor",
    "_TextLabelDescriptor",
]
