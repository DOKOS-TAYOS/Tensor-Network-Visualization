"""Package-specific exception hierarchy for public visualization entry points."""

from __future__ import annotations


class TensorNetworkVizError(Exception):
    """Base class for package-specific exceptions."""


class VisualizationInputError(TensorNetworkVizError, ValueError):
    """Invalid visualization input or configuration value."""


class VisualizationTypeError(TensorNetworkVizError, TypeError):
    """Invalid visualization input type."""


class UnsupportedEngineError(VisualizationInputError):
    """Unsupported engine or backend name."""


class AxisConfigurationError(VisualizationInputError):
    """Invalid or incompatible Matplotlib axis configuration."""


class TensorDataError(VisualizationInputError):
    """Tensor data is missing, invalid, or unsupported for visualization."""


class TensorDataTypeError(TensorNetworkVizError, TypeError):
    """Tensor data has an incompatible runtime type."""


class MissingOptionalDependencyError(TensorNetworkVizError, ImportError):
    """Optional dependency required by a backend is not installed."""


__all__ = [
    "AxisConfigurationError",
    "MissingOptionalDependencyError",
    "TensorDataError",
    "TensorDataTypeError",
    "TensorNetworkVizError",
    "UnsupportedEngineError",
    "VisualizationInputError",
    "VisualizationTypeError",
]
