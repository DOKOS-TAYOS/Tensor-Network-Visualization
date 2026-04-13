"""Shared tensor-elements model types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from ._engine_specs import EngineName

NumericArray: TypeAlias = np.ndarray[Any, Any]
TensorElementsSourceName: TypeAlias = EngineName | Literal["numpy"]


@dataclass(frozen=True)
class _TensorRecord:
    """Normalized tensor entry extracted from a supported backend.

    Attributes:
        array: Concrete NumPy array used by the rendering pipeline.
        axis_names: Axis labels associated with ``array``.
        engine: Backend that produced the tensor.
        name: Stable display name shown in controls and summaries.
    """

    array: NumericArray
    axis_names: tuple[str, ...]
    engine: TensorElementsSourceName
    name: str


@dataclass(frozen=True)
class _TensorStats:
    """Human-readable summary statistics for one tensor.

    Attributes:
        dtype_text: Display-ready dtype description.
        element_count: Number of scalar entries in the tensor.
        is_complex: Whether the tensor contains complex values.
        shape: Tensor shape used in the summary.
        text: Multiline textual summary rendered in ``data`` mode.
    """

    dtype_text: str
    element_count: int
    is_complex: bool
    shape: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class _MatrixMetadata:
    """Metadata describing how a tensor was matrixized for inspection views.

    Attributes:
        col_axes: Axis indices grouped into the matrix columns.
        col_names: Display names for the column axes.
        original_shape: Shape of the source tensor before reshaping.
        row_axes: Axis indices grouped into the matrix rows.
        row_names: Display names for the row axes.
    """

    col_axes: tuple[int, ...]
    col_names: tuple[str, ...]
    original_shape: tuple[int, ...]
    row_axes: tuple[int, ...]
    row_names: tuple[str, ...]


@dataclass(frozen=True)
class _SpectralAnalysis:
    """Derived singular-value and eigenvalue information for one tensor.

    Attributes:
        analysis_shape: Shape of the matrix actually analyzed after optional downsampling.
        col_names: Column-axis names used in the matrix view.
        eigenvalues: Eigenvalues when the analysis matrix is square, else ``None``.
        issue: Reason why the spectral analysis is unavailable, if any.
        matrix_shape: Shape of the full matrixized tensor before reduction.
        row_names: Row-axis names used in the matrix view.
        singular_values: Singular values for the analysis matrix, if available.
        used_reduced_matrix: Whether downsampling changed the matrix before analysis.
    """

    analysis_shape: tuple[int, int]
    col_names: tuple[str, ...]
    eigenvalues: NumericArray | None
    issue: str | None
    matrix_shape: tuple[int, int]
    row_names: tuple[str, ...]
    singular_values: NumericArray | None
    used_reduced_matrix: bool


@dataclass(frozen=True)
class _ResolvedTensorAnalysis:
    """Concrete analytical selectors for one tensor record and one analysis mode."""

    original_axis_names: tuple[str, ...]
    post_slice_axis_names: tuple[str, ...]
    profile_axis: int | None
    profile_axis_name: str | None
    profile_method: Literal["mean", "norm"]
    reduce_axes: tuple[int, ...]
    reduce_axis_names: tuple[str, ...]
    reduce_method: Literal["mean", "norm"]
    slice_active: bool
    slice_axis: int | None
    slice_axis_name: str | None
    slice_axis_size: int
    slice_index: int


@dataclass(frozen=True)
class _PlaybackStepRecord:
    """Tensor payload associated with one playback step result.

    Attributes:
        result_name: Name of the result tensor produced by the step.
        record: Normalized tensor data for the result, when available.
    """

    result_name: str
    record: _TensorRecord | None


__all__ = [
    "NumericArray",
    "TensorElementsSourceName",
    "_MatrixMetadata",
    "_PlaybackStepRecord",
    "_ResolvedTensorAnalysis",
    "_SpectralAnalysis",
    "_TensorRecord",
    "_TensorStats",
]
