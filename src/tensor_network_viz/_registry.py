"""Engine registry for lazy-loaded tensor-network helpers."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core.graph import _GraphData
from ._engine_specs import ENGINE_GRAPH_BUILDER_MAP, ENGINE_MODULE_MAP
from ._logging import package_logger
from .config import EngineName, PlotConfig
from .exceptions import UnsupportedEngineError, VisualizationTypeError


class _Plot2D(Protocol):
    def __call__(
        self,
        network: Any,
        *,
        ax: Axes | None = None,
        config: PlotConfig | None = None,
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
        _build_scene_state: bool = True,
    ) -> tuple[Figure, Axes]: ...


class _Plot3D(Protocol):
    def __call__(
        self,
        network: Any,
        *,
        ax: Axes | Axes3D | None = None,
        config: PlotConfig | None = None,
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
        _build_scene_state: bool = True,
    ) -> tuple[Figure, Axes3D]: ...


class _GraphBuilder(Protocol):
    def __call__(self, network: Any) -> _GraphData: ...


def _get_plotters(engine: EngineName) -> tuple[_Plot2D, _Plot3D]:
    """Get plot_2d and plot_3d for an engine, loading the module if needed."""
    try:
        module_path, name_2d, name_3d = ENGINE_MODULE_MAP[engine]
    except KeyError as exc:
        raise UnsupportedEngineError(f"Unsupported tensor network engine: {engine}") from exc

    package_logger.debug("Loading plotters for engine='%s' from module='%s'.", engine, module_path)
    module = importlib.import_module(module_path)
    plot_2d = getattr(module, name_2d)
    plot_3d = getattr(module, name_3d)
    if not callable(plot_2d) or not callable(plot_3d):
        raise VisualizationTypeError(f"Engine {engine!r} does not expose callable plotters.")
    return cast(_Plot2D, plot_2d), cast(_Plot3D, plot_3d)


def _get_graph_builder(engine: EngineName) -> _GraphBuilder:
    """Get the normalized-graph builder for one backend engine."""
    try:
        module_path, attr_name = ENGINE_GRAPH_BUILDER_MAP[engine]
    except KeyError as exc:
        raise UnsupportedEngineError(f"Unsupported tensor network engine: {engine}") from exc

    package_logger.debug(
        "Loading graph builder for engine='%s' from module='%s'.",
        engine,
        module_path,
    )
    module = importlib.import_module(module_path)
    build_graph = getattr(module, attr_name)
    if not callable(build_graph):
        raise VisualizationTypeError(f"Engine {engine!r} does not expose a callable graph builder.")
    return cast(_GraphBuilder, build_graph)
