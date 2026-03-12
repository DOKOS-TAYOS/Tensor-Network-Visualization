"""Engine registry for lazy-loaded tensor network plotters."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

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
    ) -> tuple[Figure, Axes3D]: ...


_ENGINE_CONFIG: dict[EngineName, tuple[str, str, str]] = {
    "tensorkrowch": (
        "tensor_network_viz.tensorkrowch",
        "plot_tensorkrowch_network_2d",
        "plot_tensorkrowch_network_3d",
    ),
    "tensornetwork": (
        "tensor_network_viz.tensornetwork",
        "plot_tensornetwork_network_2d",
        "plot_tensornetwork_network_3d",
    ),
    "quimb": (
        "tensor_network_viz.quimb",
        "plot_quimb_network_2d",
        "plot_quimb_network_3d",
    ),
    "tenpy": (
        "tensor_network_viz.tenpy",
        "plot_tenpy_network_2d",
        "plot_tenpy_network_3d",
    ),
    "einsum": (
        "tensor_network_viz.einsum_module",
        "plot_einsum_network_2d",
        "plot_einsum_network_3d",
    ),
}


def _get_plotters(engine: EngineName) -> tuple[_Plot2D, _Plot3D]:
    """Get plot_2d and plot_3d for an engine, loading the module if needed."""
    try:
        module_path, name_2d, name_3d = _ENGINE_CONFIG[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor network engine: {engine}") from exc

    module = importlib.import_module(module_path)
    plot_2d = getattr(module, name_2d)
    plot_3d = getattr(module, name_3d)
    if not callable(plot_2d) or not callable(plot_3d):
        raise TypeError(f"Engine {engine!r} does not expose callable plotters.")
    return cast(_Plot2D, plot_2d), cast(_Plot3D, plot_3d)
