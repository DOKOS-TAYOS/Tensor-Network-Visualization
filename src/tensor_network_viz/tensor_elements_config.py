from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TensorElementsMode: TypeAlias = Literal[
    "auto",
    "elements",
    "magnitude",
    "distribution",
    "data",
    "real",
    "imag",
    "phase",
    "sign",
    "signed_value",
]
TensorAxisSelector: TypeAlias = int | str


@dataclass(frozen=True)
class TensorElementsConfig:
    """Configuration for ``show_tensor_elements``."""

    mode: TensorElementsMode = "auto"
    figsize: tuple[float, float] | None = (7.2, 6.4)
    row_axes: tuple[TensorAxisSelector, ...] | None = None
    col_axes: tuple[TensorAxisSelector, ...] | None = None
    max_matrix_shape: tuple[int, int] = (256, 256)
    histogram_bins: int = 40
    histogram_max_samples: int = 100_000


__all__ = ["TensorAxisSelector", "TensorElementsConfig", "TensorElementsMode"]
