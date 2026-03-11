from __future__ import annotations

from typing import Any, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._registry import _get_plotters
from .config import EngineName, PlotConfig, ViewName

RenderedAxes: TypeAlias = Axes | Axes3D


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    show: bool = True,
) -> tuple[Figure, RenderedAxes]:
    """Render a tensor network and optionally display the figure.

    Args:
        network: Tensor network object (with 'nodes'/'leaf_nodes'), or an
            iterable of nodes with 'edges', 'axes_names' or 'axis_names',
            and 'name'.
        engine: Rendering engine; supported values are "tensorkrowch",
            "tensornetwork", "quimb", "tenpy", and "einsum".
        view: "2d" or "3d" visualization mode.
        config: Optional styling; uses defaults if None.
        show_tensor_labels: Whether to display tensor names on nodes. None uses
            config default.
        show_index_labels: Whether to display axis names on edges. None uses
            config default.
        show: If True, call plt.show() to display the figure. Set False when
            integrating into other applications (e.g. adding a title before showing).

    Returns:
        Tuple of (Figure, Axes) for further customization.

    Example:
        >>> config = PlotConfig(figsize=(8, 6))
        >>> fig, ax = show_tensor_network(network, engine="tensorkrowch", view="2d", config=config)
    """
    style = config or PlotConfig()
    plot_2d, plot_3d = _get_plotters(engine)
    if view == "2d":
        fig, ax = plot_2d(
            network,
            config=style,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
        )
    else:
        fig, ax = plot_3d(
            network,
            config=style,
            show_tensor_labels=show_tensor_labels,
            show_index_labels=show_index_labels,
        )
    if show:
        plt.show()
    return fig, ax
