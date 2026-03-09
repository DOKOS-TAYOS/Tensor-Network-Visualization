from __future__ import annotations

from typing import Any, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .config import EngineName, PlotConfig, ViewName
from .tensorkrowch import (
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
)

RenderedAxes: TypeAlias = Axes | Axes3D


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig | None = None,
    show: bool = True,
) -> tuple[Figure, RenderedAxes]:
    """Render a tensor network and optionally display the figure.

    Args:
        network: Tensor network object (TensorNetwork with 'nodes'/'leaf_nodes',
            or a list/tuple of nodes with 'edges', 'axes_names', and 'name').
        engine: Rendering engine; currently only "tensorkrowch" is supported.
        view: "2d" or "3d" visualization mode.
        config: Optional styling; uses defaults if None.
        show: If True, call plt.show() to display the figure. Set False when
            integrating into other applications (e.g. adding a title before showing).

    Returns:
        Tuple of (Figure, Axes) for further customization.

    Example:
        >>> config = PlotConfig(figsize=(8, 6))
        >>> fig, ax = show_tensor_network(network, engine="tensorkrowch", view="2d", config=config)
    """
    style = config or PlotConfig()

    if engine == "tensorkrowch":
        if view == "2d":
            fig, ax = plot_tensorkrowch_network_2d(network, config=style)
        else:
            fig, ax = plot_tensorkrowch_network_3d(network, config=style)
    else:
        raise ValueError(f"Unsupported tensor network engine: {engine}")

    if show:
        plt.show()
    return fig, ax
