"""Public package surface with lazy imports for optional backends and viewers."""

from __future__ import annotations

from importlib import import_module
from os import PathLike
from typing import TYPE_CHECKING, Any

from . import _logging as _package_logging

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from ._core.graph_cache import clear_tensor_network_graph_cache
    from .config import (
        EngineName,
        PlotConfig,
        PlotTheme,
        TensorNetworkDiagnosticsConfig,
        TensorNetworkFocus,
        TranslationTargetName,
        ViewName,
    )
    from .contraction_viewer import ContractionViewer2D, ContractionViewer3D
    from .einsum_module.trace import EinsumTrace, einsum, einsum_trace_step, pair_tensor
    from .exceptions import (
        AxisConfigurationError,
        MissingOptionalDependencyError,
        TensorDataError,
        TensorDataTypeError,
        TensorNetworkVizError,
        UnsupportedEngineError,
        VisualizationInputError,
        VisualizationTypeError,
    )
    from .snapshot import (
        NormalizedContractionStepMetrics,
        NormalizedTensorEdge,
        NormalizedTensorEndpoint,
        NormalizedTensorGraph,
        NormalizedTensorNode,
        TensorNetworkLayoutSnapshot,
        TensorNetworkSnapshot,
        export_tensor_network_snapshot,
        normalize_tensor_network,
    )
    from .tenpy.explicit import TenPyTensorNetwork, make_tenpy_tensor_network
    from .tensor_comparison import show_tensor_comparison
    from .tensor_comparison_config import TensorComparisonConfig
    from .tensor_elements import show_tensor_elements
    from .tensor_elements_config import (
        TensorAnalysisConfig,
        TensorElementsConfig,
        TensorElementsTheme,
    )
    from .translation import translate_tensor_network
    from .viewer import show_tensor_network
else:

    def show_tensor_network(
        network: Any,
        *,
        engine: EngineName | None = None,
        view: ViewName | None = None,
        config: PlotConfig | None = None,
        ax: Axes | Axes3D | None = None,
        show_controls: bool = True,
        show: bool = True,
    ) -> tuple[Figure, Axes | Axes3D]:
        """Lazily dispatch to :func:`tensor_network_viz.viewer.show_tensor_network`.

        Args:
            network: Tensor-network input accepted by the public viewer entry point,
                including flat iterables and nested 2D/3D grids with ``None`` holes.
            engine: Optional backend override.
            view: Optional initial view name.
            config: Optional plotting configuration.
            ax: Optional Matplotlib axes to render into.
            show_controls: Whether to attach embedded interactive controls.
            show: Whether to display the figure automatically.

        Returns:
            The ``(figure, axes)`` tuple returned by the concrete viewer implementation.
        """
        from .viewer import show_tensor_network as _show_tensor_network

        return _show_tensor_network(
            network,
            engine=engine,
            view=view,
            config=config,
            ax=ax,
            show_controls=show_controls,
            show=show,
        )

    def show_tensor_elements(
        data: Any,
        *,
        engine: EngineName | None = None,
        config: TensorElementsConfig | None = None,
        ax: Axes | None = None,
        show_controls: bool = True,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Lazily dispatch to :func:`tensor_network_viz.tensor_elements.show_tensor_elements`.

        Args:
            data: Tensor data accepted by the public tensor-elements entry point, including direct
                numeric arrays and iterables of tensors.
            engine: Optional backend override.
            config: Optional tensor-inspection configuration.
            ax: Optional Matplotlib axes for single-tensor rendering.
            show_controls: Whether to attach grouped controls and the tensor slider.
            show: Whether to display the figure automatically.

        Returns:
            The ``(figure, axes)`` tuple returned by the tensor-elements implementation.
        """
        from .tensor_elements import show_tensor_elements as _show_tensor_elements

        return _show_tensor_elements(
            data,
            engine=engine,
            config=config,
            ax=ax,
            show_controls=show_controls,
            show=show,
        )

    def show_tensor_comparison(
        data: Any,
        reference: Any,
        *,
        engine: EngineName | None = None,
        config: TensorElementsConfig | None = None,
        comparison_config: TensorComparisonConfig | None = None,
        ax: Axes | None = None,
        show_controls: bool = True,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Lazily dispatch to ``tensor_network_viz.tensor_comparison.show_tensor_comparison``."""
        from .tensor_comparison import show_tensor_comparison as _show_tensor_comparison

        return _show_tensor_comparison(
            data,
            reference,
            engine=engine,
            config=config,
            comparison_config=comparison_config,
            ax=ax,
            show_controls=show_controls,
            show=show,
        )

    def translate_tensor_network(
        network: Any,
        *,
        engine: EngineName | None = None,
        target_engine: TranslationTargetName,
        path: str | PathLike[str] | None = None,
    ) -> str:
        """Lazily dispatch to :func:`tensor_network_viz.translation.translate_tensor_network`."""
        from .translation import translate_tensor_network as _translate_tensor_network

        return _translate_tensor_network(
            network,
            engine=engine,
            target_engine=target_engine,
            path=path,
        )


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AxisConfigurationError": (".exceptions", "AxisConfigurationError"),
    "ContractionViewer2D": (".contraction_viewer", "ContractionViewer2D"),
    "ContractionViewer3D": (".contraction_viewer", "ContractionViewer3D"),
    "EngineName": (".config", "EngineName"),
    "EinsumTrace": (".einsum_module.trace", "EinsumTrace"),
    "MissingOptionalDependencyError": (".exceptions", "MissingOptionalDependencyError"),
    "NormalizedContractionStepMetrics": (".snapshot", "NormalizedContractionStepMetrics"),
    "NormalizedTensorEdge": (".snapshot", "NormalizedTensorEdge"),
    "NormalizedTensorEndpoint": (".snapshot", "NormalizedTensorEndpoint"),
    "NormalizedTensorGraph": (".snapshot", "NormalizedTensorGraph"),
    "NormalizedTensorNode": (".snapshot", "NormalizedTensorNode"),
    "PlotConfig": (".config", "PlotConfig"),
    "PlotTheme": (".config", "PlotTheme"),
    "TranslationTargetName": ("._engine_specs", "TranslationTargetName"),
    "TenPyTensorNetwork": (".tenpy.explicit", "TenPyTensorNetwork"),
    "TensorComparisonConfig": (".tensor_comparison_config", "TensorComparisonConfig"),
    "TensorAnalysisConfig": (".tensor_elements_config", "TensorAnalysisConfig"),
    "TensorDataError": (".exceptions", "TensorDataError"),
    "TensorDataTypeError": (".exceptions", "TensorDataTypeError"),
    "TensorElementsConfig": (".tensor_elements_config", "TensorElementsConfig"),
    "TensorElementsTheme": (".tensor_elements_config", "TensorElementsTheme"),
    "TensorNetworkDiagnosticsConfig": (".config", "TensorNetworkDiagnosticsConfig"),
    "TensorNetworkFocus": (".config", "TensorNetworkFocus"),
    "TensorNetworkLayoutSnapshot": (".snapshot", "TensorNetworkLayoutSnapshot"),
    "TensorNetworkSnapshot": (".snapshot", "TensorNetworkSnapshot"),
    "TensorNetworkVizError": (".exceptions", "TensorNetworkVizError"),
    "UnsupportedEngineError": (".exceptions", "UnsupportedEngineError"),
    "ViewName": (".config", "ViewName"),
    "VisualizationInputError": (".exceptions", "VisualizationInputError"),
    "VisualizationTypeError": (".exceptions", "VisualizationTypeError"),
    "clear_tensor_network_graph_cache": ("._core.graph_cache", "clear_tensor_network_graph_cache"),
    "einsum": (".einsum_module.trace", "einsum"),
    "einsum_trace_step": (".einsum_module.trace", "einsum_trace_step"),
    "export_tensor_network_snapshot": (".snapshot", "export_tensor_network_snapshot"),
    "make_tenpy_tensor_network": (".tenpy.explicit", "make_tenpy_tensor_network"),
    "normalize_tensor_network": (".snapshot", "normalize_tensor_network"),
    "pair_tensor": (".einsum_module.trace", "pair_tensor"),
}


def __getattr__(name: str) -> Any:
    """Resolve a lazily exported public symbol on first attribute access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including the lazily exported public names."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AxisConfigurationError",
    "ContractionViewer2D",
    "ContractionViewer3D",
    "EngineName",
    "EinsumTrace",
    "MissingOptionalDependencyError",
    "NormalizedContractionStepMetrics",
    "NormalizedTensorEdge",
    "NormalizedTensorEndpoint",
    "NormalizedTensorGraph",
    "NormalizedTensorNode",
    "PlotConfig",
    "PlotTheme",
    "TranslationTargetName",
    "TensorComparisonConfig",
    "TensorAnalysisConfig",
    "TensorDataError",
    "TensorDataTypeError",
    "TensorElementsConfig",
    "TensorElementsTheme",
    "TensorNetworkDiagnosticsConfig",
    "TensorNetworkFocus",
    "TensorNetworkLayoutSnapshot",
    "TensorNetworkSnapshot",
    "TensorNetworkVizError",
    "TenPyTensorNetwork",
    "UnsupportedEngineError",
    "ViewName",
    "VisualizationInputError",
    "VisualizationTypeError",
    "clear_tensor_network_graph_cache",
    "einsum",
    "einsum_trace_step",
    "export_tensor_network_snapshot",
    "make_tenpy_tensor_network",
    "normalize_tensor_network",
    "pair_tensor",
    "show_tensor_comparison",
    "show_tensor_elements",
    "show_tensor_network",
    "translate_tensor_network",
]

_ = _package_logging
