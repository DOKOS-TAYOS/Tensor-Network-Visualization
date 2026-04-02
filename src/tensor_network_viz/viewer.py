from __future__ import annotations

from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._registry import _get_plotters
from ._typing import FigureLike, root_figure
from .config import EngineName, PlotConfig, ViewName
from .interactive_viewer import show_tensor_network_interactive

RenderedAxes: TypeAlias = Axes | Axes3D


def _show_figure(fig: FigureLike) -> None:
    """Show *fig* in a Jupyter kernel via IPython display, else ``plt.show()``."""
    display_figure = root_figure(fig)
    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import display
    except ImportError:
        plt.show()
        return
    ip = get_ipython()
    if ip is not None and getattr(ip, "kernel", None) is not None:
        display(display_figure)
        return
    plt.show()


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName,
    view: ViewName | None = None,
    config: PlotConfig | None = None,
    ax: RenderedAxes | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    interactive_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, RenderedAxes]:
    """Render a tensor network and optionally display the figure.

    Args:
        network: Tensor network object (with 'nodes'/'leaf_nodes'), or an
            iterable of nodes with 'edges', 'axes_names' or 'axis_names',
            and 'name'.
        engine: Rendering engine; supported values are "tensorkrowch",
            "tensornetwork", "quimb", "tenpy", and "einsum".
        view: "2d" or "3d" visualization mode. ``None`` defaults to ``"2d"``
            unless ``ax`` is a 3D axes, in which case the view is inferred.
        config: Optional styling; uses defaults if None. Use ``PlotConfig`` for
            colors, layout, ``hover_labels`` (interactive hover tooltips), etc.
        ax: Optional Matplotlib axes to render into. When passed, the 2D/3D
            selector is suppressed and the view is fixed to that axes.
        show_tensor_labels: Whether to display tensor names on nodes. ``None``
            uses ``config.show_tensor_labels``.
        show_index_labels: Whether to display axis names on edges. ``None``
            uses ``config.show_index_labels``.
        interactive_controls: If True, attach figure-level controls for view,
            hover, and label toggles when the renderer exposes the required
            scene cache. Set False for a static render without widgets.
        show: If True, display the figure. In a Jupyter kernel this uses
            ``IPython.display.display(fig)`` (use ``pip install
            "tensor-network-visualization[jupyter]"`` and ``%matplotlib widget``
            for interactive figures). Otherwise ``plt.show()`` is used.

    Note:
        Repeated calls with the **same** ``network`` instance reuse the normalized
        graph structure until the object is collected or you call
        ``clear_tensor_network_graph_cache(network)`` after in-place changes.

    Returns:
        Tuple of (Figure, Axes) for further customization.

    Example:
        >>> config = PlotConfig(figsize=(8, 6))
        >>> fig, ax = show_tensor_network(network, engine="tensorkrowch", config=config)
    """
    style = config or PlotConfig()
    resolved_tensor_labels = (
        style.show_tensor_labels if show_tensor_labels is None else show_tensor_labels
    )
    resolved_index_labels = (
        style.show_index_labels if show_index_labels is None else show_index_labels
    )
    ax_view: ViewName | None = None
    if ax is not None:
        ax_view = "3d" if getattr(ax, "name", "") == "3d" else "2d"
    resolved_view = ax_view or "2d" if view is None else view
    if ax_view is not None and resolved_view != ax_view:
        raise ValueError(f"Provided ax is {ax_view}, but view={resolved_view!r} was requested.")
    if resolved_view not in ("2d", "3d"):
        raise ValueError(f"Unsupported tensor network view: {resolved_view}")
    plot_2d, plot_3d = _get_plotters(engine)
    if not interactive_controls:
        if resolved_view == "2d":
            fig, ax_out = plot_2d(
                network,
                ax=cast(Axes, ax) if ax is not None else None,
                config=style,
                show_tensor_labels=show_tensor_labels,
                show_index_labels=show_index_labels,
            )
        elif resolved_view == "3d":
            fig, ax_out = plot_3d(
                network,
                ax=ax,
                config=style,
                show_tensor_labels=show_tensor_labels,
                show_index_labels=show_index_labels,
            )
        else:
            raise ValueError(f"Unsupported tensor network view: {resolved_view}")
    else:
        fig, ax_out = show_tensor_network_interactive(
            network,
            engine=engine,
            view=resolved_view,
            config=style,
            show_tensor_labels=resolved_tensor_labels,
            show_index_labels=resolved_index_labels,
            ax=ax,
        )
    if show:
        _show_figure(fig)
    return root_figure(fig), ax_out
