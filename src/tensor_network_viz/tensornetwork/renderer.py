"""Public entry points for google/TensorNetwork plotting."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..config import PlotConfig
from ..tensorkrowch.renderer import _plot_network_2d, _plot_network_3d


def plot_tensornetwork_network_2d(
    nodes: Any,
    *,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    """Plot a TensorNetwork node collection in 2D.

    Args:
        nodes: Iterable of TensorNetwork nodes (for example a list or set).
        ax: Matplotlib 2D axes; if None, creates a new figure.
        config: Styling options; uses defaults if None.
        show_tensor_labels: Override config; None uses config value.
        show_index_labels: Override config; None uses config value.
        seed: Random seed for layout when using force-directed positioning.

    Returns:
        Tuple of (Figure, Axes) for further customization.
    """
    return _plot_network_2d(
        nodes,
        ax=ax,
        config=config,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        seed=seed,
        renderer_name="plot_tensornetwork_network_2d",
    )


def plot_tensornetwork_network_3d(
    nodes: Any,
    *,
    ax: Axes | Axes3D | None = None,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes3D]:
    """Plot a TensorNetwork node collection in 3D.

    Args:
        nodes: Iterable of TensorNetwork nodes (for example a list or set).
        ax: Matplotlib 3D axes; if None, creates a new figure with 3D projection.
        config: Styling options; uses defaults if None.
        show_tensor_labels: Override config; None uses config value.
        show_index_labels: Override config; None uses config value.
        seed: Random seed for layout when using force-directed positioning.

    Returns:
        Tuple of (Figure, Axes3D) for further customization.
    """
    return _plot_network_3d(
        nodes,
        ax=ax,
        config=config,
        show_tensor_labels=show_tensor_labels,
        show_index_labels=show_index_labels,
        seed=seed,
        renderer_name="plot_tensornetwork_network_3d",
    )
