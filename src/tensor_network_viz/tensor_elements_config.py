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
    "slice",
    "reduce",
    "profiles",
]
TensorAxisSelector: TypeAlias = int | str
TensorElementsTheme: TypeAlias = Literal[
    "default",
    "grayscale",
    "contrast",
    "categorical",
    "paper",
    "colorblind",
    "rainbow",
    "spectral",
]

_TENSOR_ELEMENTS_THEME_ORDER: tuple[TensorElementsTheme, ...] = (
    "default",
    "grayscale",
    "contrast",
    "categorical",
    "paper",
    "colorblind",
    "rainbow",
    "spectral",
)
_TENSOR_ELEMENTS_THEME_NAMES: frozenset[str] = frozenset(_TENSOR_ELEMENTS_THEME_ORDER)
_TENSOR_ELEMENTS_THEME_OVERRIDES: dict[str, dict[str, object]] = {
    "grayscale": {
        "continuous_cmap": "gray",
        "log_magnitude_cmap": "Greys",
        "phase_cmap": "twilight_shifted",
        "diverging_cmap": "Greys",
        "sign_colors": ("#111111", "#D4D4D4", "#FFFFFF"),
        "sparsity_colors": ("#111111", "#FFFFFF"),
        "nan_inf_colors": ("#111111", "#7A7A7A", "#C7C7C7", "#EFEFEF"),
        "series_color": "#111111",
        "histogram_color": "#444444",
        "histogram_edge_color": "#111111",
        "zero_marker_color": "#000000",
        "hover_facecolor": "#FFFFFF",
        "hover_edgecolor": "#111111",
        "summary_facecolor": "#FFFFFF",
        "summary_edgecolor": "#111111",
    },
    "contrast": {
        "continuous_cmap": "CMRmap",
        "log_magnitude_cmap": "Greys",
        "phase_cmap": "twilight",
        "diverging_cmap": "RdGy",
        "sign_colors": ("#B91C1C", "#E5E7EB", "#1D4ED8"),
        "sparsity_colors": ("#111827", "#FFFFFF"),
        "nan_inf_colors": ("#0F766E", "#D97706", "#7C3AED", "#B91C1C"),
        "series_color": "#111827",
        "histogram_color": "#64748B",
        "histogram_edge_color": "#0F172A",
        "zero_marker_color": "#991B1B",
        "hover_facecolor": "#FFFFFF",
        "hover_edgecolor": "#CBD5E1",
        "summary_facecolor": "#FFFFFF",
        "summary_edgecolor": "#CBD5E1",
    },
    "categorical": {
        "continuous_cmap": "turbo",
        "log_magnitude_cmap": "nipy_spectral",
        "phase_cmap": "hsv",
        "diverging_cmap": "Spectral",
        "sign_colors": ("#E11D48", "#FDE68A", "#2563EB"),
        "sparsity_colors": ("#7C3AED", "#22C55E"),
        "nan_inf_colors": ("#06B6D4", "#F59E0B", "#8B5CF6", "#EF4444"),
        "series_color": "#7C3AED",
        "histogram_color": "#0EA5E9",
        "histogram_edge_color": "#312E81",
        "zero_marker_color": "#DC2626",
        "hover_facecolor": "#FEFCE8",
        "hover_edgecolor": "#A16207",
        "summary_facecolor": "#F8FAFC",
        "summary_edgecolor": "#7C3AED",
    },
    "paper": {
        "continuous_cmap": "inferno",
        "log_magnitude_cmap": "inferno",
        "phase_cmap": "twilight",
        "diverging_cmap": "coolwarm",
        "sign_colors": ("#7F1D1D", "#F8FAFC", "#0C4A6E"),
        "sparsity_colors": ("#0F172A", "#FACC15"),
        "nan_inf_colors": ("#0F766E", "#F97316", "#2563EB", "#B91C1C"),
        "series_color": "#0F172A",
        "histogram_color": "#0369A1",
        "histogram_edge_color": "#0F172A",
        "zero_marker_color": "#B91C1C",
        "hover_facecolor": "#F8FAFC",
        "hover_edgecolor": "#0F172A",
        "summary_facecolor": "#F8FAFC",
        "summary_edgecolor": "#0F172A",
    },
    "colorblind": {
        "continuous_cmap": "cividis",
        "log_magnitude_cmap": "cividis",
        "phase_cmap": "twilight_shifted",
        "diverging_cmap": "coolwarm",
        "sign_colors": ("#D55E00", "#F0E442", "#0072B2"),
        "sparsity_colors": ("#111827", "#F7F7F7"),
        "nan_inf_colors": ("#009E73", "#E69F00", "#56B4E9", "#CC79A7"),
        "series_color": "#0072B2",
        "histogram_color": "#56B4E9",
        "histogram_edge_color": "#111827",
        "zero_marker_color": "#D55E00",
        "hover_facecolor": "#FFFFFF",
        "hover_edgecolor": "#111827",
        "summary_facecolor": "#FFFFFF",
        "summary_edgecolor": "#111827",
    },
    "rainbow": {
        "continuous_cmap": "gist_rainbow",
        "log_magnitude_cmap": "gist_rainbow",
        "phase_cmap": "hsv",
        "diverging_cmap": "gist_rainbow",
        "sign_colors": ("#FF0000", "#FFFF00", "#0000FF"),
        "sparsity_colors": ("#6D28D9", "#10B981"),
        "nan_inf_colors": ("#00FFFF", "#FFFF00", "#FF00FF", "#FF0000"),
        "series_color": "#FF00FF",
        "histogram_color": "#00AEEF",
        "histogram_edge_color": "#6B21A8",
        "zero_marker_color": "#DC2626",
        "hover_facecolor": "#F8FAFC",
        "hover_edgecolor": "#6B21A8",
        "summary_facecolor": "#F8FAFC",
        "summary_edgecolor": "#6B21A8",
    },
    "spectral": {
        "continuous_cmap": "nipy_spectral",
        "log_magnitude_cmap": "nipy_spectral",
        "phase_cmap": "twilight_shifted",
        "diverging_cmap": "Spectral",
        "sign_colors": ("#C026D3", "#FDE68A", "#2563EB"),
        "sparsity_colors": ("#111827", "#67E8F9"),
        "nan_inf_colors": ("#22C55E", "#F59E0B", "#7C3AED", "#DC2626"),
        "series_color": "#7C3AED",
        "histogram_color": "#0891B2",
        "histogram_edge_color": "#312E81",
        "zero_marker_color": "#BE123C",
        "hover_facecolor": "#F8FAFC",
        "hover_edgecolor": "#7C3AED",
        "summary_facecolor": "#F8FAFC",
        "summary_edgecolor": "#7C3AED",
    },
}


@dataclass(frozen=True)
class TensorAnalysisConfig:
    """Configuration for the analytical tensor views."""

    slice_axis: TensorAxisSelector | None = None
    slice_index: int = 0
    reduce_axes: tuple[TensorAxisSelector, ...] | None = None
    reduce_method: Literal["mean", "norm"] = "mean"
    profile_axis: TensorAxisSelector | None = None
    profile_method: Literal["mean", "norm"] = "mean"

    def __post_init__(self) -> None:
        """Validate the analytical selector inputs."""
        slice_index = int(self.slice_index)
        if slice_index < 0:
            raise ValueError("analysis.slice_index must be non-negative.")
        object.__setattr__(self, "slice_index", slice_index)
        if self.reduce_method not in {"mean", "norm"}:
            raise ValueError("analysis.reduce_method must be 'mean' or 'norm'.")
        if self.profile_method not in {"mean", "norm"}:
            raise ValueError("analysis.profile_method must be 'mean' or 'norm'.")
        if self.reduce_axes is not None:
            object.__setattr__(self, "reduce_axes", tuple(self.reduce_axes))


@dataclass(frozen=True)
class TensorElementsConfig:
    """Configuration for ``show_tensor_elements``.

    The constructor is ordered so that the first arguments choose *what* to inspect, the
    next ones define the public style theme and its optional overrides, and the final ones
    tune rendering details.
    """

    mode: TensorElementsMode = "auto"
    row_axes: tuple[TensorAxisSelector, ...] | None = None
    col_axes: tuple[TensorAxisSelector, ...] | None = None
    analysis: TensorAnalysisConfig | None = None
    theme: TensorElementsTheme = "default"
    continuous_cmap: str = "viridis"
    log_magnitude_cmap: str = "magma"
    phase_cmap: str = "twilight"
    diverging_cmap: str = "RdBu_r"
    sign_colors: tuple[str, str, str] = ("#B91C1C", "#E2E8F0", "#0369A1")
    sparsity_colors: tuple[str, str] = ("#0F172A", "#F8FAFC")
    nan_inf_colors: tuple[str, str, str, str] = ("#0F766E", "#D97706", "#7C3AED", "#B91C1C")
    series_color: str = "#0369A1"
    histogram_color: str = "#0369A1"
    histogram_edge_color: str = "#0F172A"
    zero_marker_color: str = "#7F1D1D"
    hover_facecolor: str = "#F8FAFC"
    hover_edgecolor: str = "#CBD5E1"
    summary_facecolor: str = "#F8FAFC"
    summary_edgecolor: str = "#CBD5E1"
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
        """Validate public theme/style inputs and numeric rendering options."""
        theme = str(self.theme)
        if theme not in _TENSOR_ELEMENTS_THEME_NAMES:
            available = ", ".join(f"{name!r}" for name in sorted(_TENSOR_ELEMENTS_THEME_NAMES))
            raise ValueError(f"theme must be one of {available}.")
        object.__setattr__(self, "theme", theme)
        if theme != "default":
            for field_name, themed_value in _TENSOR_ELEMENTS_THEME_OVERRIDES[theme].items():
                base_value = getattr(type(self), field_name)
                if getattr(self, field_name) == base_value:
                    object.__setattr__(self, field_name, themed_value)

        object.__setattr__(self, "sign_colors", tuple(self.sign_colors))
        object.__setattr__(self, "sparsity_colors", tuple(self.sparsity_colors))
        object.__setattr__(self, "nan_inf_colors", tuple(self.nan_inf_colors))
        if len(self.sign_colors) != 3:
            raise ValueError("sign_colors must contain exactly three colors.")
        if len(self.sparsity_colors) != 2:
            raise ValueError("sparsity_colors must contain exactly two colors.")
        if len(self.nan_inf_colors) != 4:
            raise ValueError("nan_inf_colors must contain exactly four colors.")

        try:
            max_rows_raw, max_cols_raw = self.max_matrix_shape
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_matrix_shape must contain exactly two positive integers."
            ) from exc
        max_rows = int(max_rows_raw)
        max_cols = int(max_cols_raw)
        if max_rows <= 0 or max_cols <= 0:
            raise ValueError("max_matrix_shape must contain exactly two positive integers.")
        object.__setattr__(self, "max_matrix_shape", (max_rows, max_cols))

        histogram_bins = int(self.histogram_bins)
        if histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive.")
        object.__setattr__(self, "histogram_bins", histogram_bins)

        histogram_max_samples = int(self.histogram_max_samples)
        if histogram_max_samples <= 0:
            raise ValueError("histogram_max_samples must be positive.")
        object.__setattr__(self, "histogram_max_samples", histogram_max_samples)

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


__all__ = [
    "TensorAnalysisConfig",
    "TensorAxisSelector",
    "TensorElementsConfig",
    "TensorElementsMode",
    "TensorElementsTheme",
]
