"""Public configuration types for tensor-elements inspection figures."""

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
    "singular_values",
    "eigen_real",
    "eigen_imag",
]
TensorAxisSelector: TypeAlias = int | str


@dataclass(frozen=True)
class TensorElementsConfig:
    """Configuration for ``show_tensor_elements``.

    The constructor is ordered so that the first arguments choose *what* to inspect and
    the later ones tune rendering details.

    Attributes:
        mode: Initial inspection mode. ``"auto"`` chooses a sensible default from the
            tensor dtype.
        row_axes: Optional axes to group into the matrix rows when rank is greater than 2.
        col_axes: Optional axes to group into the matrix columns when rank is greater than 2.
        figsize: Figure size in inches. ``None`` leaves Matplotlib's default.
        max_matrix_shape: Maximum matrix size used by heatmaps and spectral views after
            downsampling.
        shared_color_scale: Whether compatible heatmap modes reuse one color scale across
            the tensor slider.
        robust_percentiles: Optional percentile-based color scaling for continuous heatmaps.
        highlight_outliers: Whether to overlay outlier markers on continuous heatmaps.
        outlier_zscore: Threshold used by the outlier overlay.
        zero_threshold: Values with magnitude at or below this threshold are treated as zero
            in zero-aware modes such as ``"sparsity"`` and singular-value display.
        log_magnitude_floor: Positive floor used by log-based views so zeros can still be
            displayed safely.
        histogram_bins: Number of bins used in ``"distribution"`` mode.
        histogram_max_samples: Maximum number of values sampled for histograms.
        topk_count: Number of entries shown in the textual ``"data"`` summary.
    """

    mode: TensorElementsMode = "auto"
    row_axes: tuple[TensorAxisSelector, ...] | None = None
    col_axes: tuple[TensorAxisSelector, ...] | None = None
    figsize: tuple[float, float] | None = (7.2, 6.4)
    max_matrix_shape: tuple[int, int] = (256, 256)
    shared_color_scale: bool = False
    robust_percentiles: tuple[float, float] | None = None
    highlight_outliers: bool = False
    outlier_zscore: float = 3.5
    zero_threshold: float = 1e-12
    log_magnitude_floor: float = 1e-12
    histogram_bins: int = 40
    histogram_max_samples: int = 100_000
    topk_count: int = 8

    def __post_init__(self) -> None:
        """Validate numeric configuration values and normalize percentile input."""
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
