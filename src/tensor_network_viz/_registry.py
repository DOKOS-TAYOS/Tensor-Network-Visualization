"""Engine registry for lazy-loaded tensor network plotters."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._engine_specs import ENGINE_MODULE_MAP
from .config import EngineName, PlotConfig


class _Plot2D(Protocol):
    def __call__(
        self,
        network: Any,
        *,
        ax: Axes | None = None,
        config: PlotConfig | None = None,
        show_tensor_labels: bool | None = None,
        show_index_labels: bool | None = None,
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
    ) -> tuple[Figure, Axes]: ...


class _Plot3D(Protocol):
    def __call__(
        self,
        network: Any,
        *,
        ax: Axes | Axes3D | None = None,
        config: PlotConfig | None = None,
        show_tensor_labels: bool | None = None,
        show_index_labels: bool | None = None,
        seed: int = 0,
        _build_contraction_controls: bool = True,
        _contraction_controls_build_ui: bool = True,
        _register_contraction_controls_on_figure: bool = True,
    ) -> tuple[Figure, Axes3D]: ...


def _get_plotters(engine: EngineName) -> tuple[_Plot2D, _Plot3D]:
    """Get plot_2d and plot_3d for an engine, loading the module if needed."""
    try:
        module_path, name_2d, name_3d = ENGINE_MODULE_MAP[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor network engine: {engine}") from exc

    module = importlib.import_module(module_path)
    plot_2d = getattr(module, name_2d)
    plot_3d = getattr(module, name_3d)
    if not callable(plot_2d) or not callable(plot_3d):
        raise TypeError(f"Engine {engine!r} does not expose callable plotters.")
    return cast(_Plot2D, plot_2d), cast(_Plot3D, plot_3d)
