"""Private model types used by the tensor-network translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._engine_specs import EngineName


@dataclass(frozen=True)
class _TranslatedTensor:
    node_id: int
    variable_name: str
    display_name: str
    axis_names: tuple[str, ...]
    index_labels: tuple[str, ...]
    shape: tuple[int, ...] | None
    dtype_text: str | None
    array: np.ndarray[Any, Any] | None


@dataclass(frozen=True)
class _TranslatedEdge:
    kind: str
    label: str
    endpoints: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _TranslatedNetwork:
    source_engine: EngineName
    tensors: tuple[_TranslatedTensor, ...]
    edges: tuple[_TranslatedEdge, ...]
    open_labels: tuple[str, ...]
    contraction_steps: tuple[tuple[str, ...], ...] | None


__all__ = ["_TranslatedEdge", "_TranslatedNetwork", "_TranslatedTensor"]
