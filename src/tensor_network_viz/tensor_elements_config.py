from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TensorElementsMode: TypeAlias = Literal[
    "auto",
    "elements",
    "magnitude",
    "log_magnitude",
    "distribution",
    "data",
    "real",
    "imag",
    "phase",
    "sign",
    "signed_value",
    "sparsity",
    "nan_inf",
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
    topk_count: int = 8
    zero_threshold: float = 1e-12
    log_magnitude_floor: float = 1e-12
    robust_percentiles: tuple[float, float] | None = None
    shared_color_scale: bool = False
    highlight_outliers: bool = False
    outlier_zscore: float = 3.5

    def __post_init__(self) -> None:
        if int(self.topk_count) <= 0:
            raise ValueError("topk_count must be positive.")
        if float(self.zero_threshold) <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        if float(self.log_magnitude_floor) <= 0.0:
            raise ValueError("log_magnitude_floor must be positive.")
        if float(self.outlier_zscore) <= 0.0:
            raise ValueError("outlier_zscore must be positive.")
        if self.robust_percentiles is None:
            return
        low, high = (float(self.robust_percentiles[0]), float(self.robust_percentiles[1]))
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("robust_percentiles must satisfy 0 <= low < high <= 100.")
        object.__setattr__(self, "robust_percentiles", (low, high))


__all__ = ["TensorAxisSelector", "TensorElementsConfig", "TensorElementsMode"]
