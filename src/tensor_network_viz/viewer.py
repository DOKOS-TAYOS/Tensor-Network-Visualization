from __future__ import annotations

from collections.abc import Iterable
from itertools import tee
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._registry import _get_plotters
from ._typing import FigureLike, root_figure
from .config import EngineName, PlotConfig, ViewName
from .einsum_module.trace import EinsumTrace, einsum_trace_step, pair_tensor
from .interactive_viewer import show_tensor_network_interactive
from .tenpy.explicit import TenPyTensorNetwork

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


def _first_non_none(items: Iterable[Any]) -> Any | None:
    """Return the first non-None item from an iterable, if any."""
    for item in items:
        if item is not None:
            return item
    return None


def _peek_network_item(source: Any) -> tuple[Any | None, Any]:
    """Peek one item from iterable input while preserving single-pass iterators."""
    if isinstance(source, dict):
        return _first_non_none(source.values()), source
    if isinstance(source, (str, bytes, bytearray)) or not isinstance(source, Iterable):
        return None, source

    iterator = iter(source)
    if iterator is source:
        probe_iter, runtime_iter = tee(iterator)
        return _first_non_none(probe_iter), runtime_iter
    return _first_non_none(iterator), source


def _is_tenpy_like(network: Any) -> bool:
    """Return True when *network* matches one of the supported TeNPy entry points."""
    return (
        isinstance(network, TenPyTensorNetwork)
        or callable(getattr(network, "get_W", None))
        or (
            callable(getattr(network, "get_X", None))
            and hasattr(network, "uMPS_GS")
        )
        or callable(getattr(network, "get_B", None))
    )


def _detect_engine_from_sample(sample_item: Any | None) -> EngineName | None:
    """Infer an engine from one representative item when possible."""
    if isinstance(sample_item, (pair_tensor, einsum_trace_step)):
        return "einsum"
    if hasattr(sample_item, "inds"):
        return "quimb"
    if hasattr(sample_item, "axes_names"):
        return "tensorkrowch"
    if hasattr(sample_item, "axis_names"):
        return "tensornetwork"
    return None


def _detect_engine_with_network(network: Any) -> tuple[EngineName, Any]:
    """Infer the backend engine and preserve single-pass iterables when needed."""
    if isinstance(network, EinsumTrace):
        return "einsum", network
    if _is_tenpy_like(network):
        return "tenpy", network

    if hasattr(network, "tensors"):
        tensor_sample, _ = _peek_network_item(network.tensors)
        if tensor_sample is None or hasattr(tensor_sample, "inds"):
            return "quimb", network

    if hasattr(network, "leaf_nodes"):
        leaf_sample, _ = _peek_network_item(network.leaf_nodes)
        if leaf_sample is None or hasattr(leaf_sample, "axes_names"):
            return "tensorkrowch", network
    if hasattr(network, "nodes"):
        node_sample, _ = _peek_network_item(network.nodes)
        if node_sample is None or hasattr(node_sample, "axes_names"):
            return "tensorkrowch", network

    sample_item, sampled_network = _peek_network_item(network)
    detected_engine = _detect_engine_from_sample(sample_item)
    if detected_engine is not None:
        return detected_engine, sampled_network

    raise ValueError(
        "Could not infer tensor network engine from input of type "
        f"{type(network).__name__!r}. Pass engine= explicitly or provide a supported backend input."
    )


def _detect_engine(network: Any) -> EngineName:
    """Infer the backend engine from the received tensor-network object."""
    detected_engine, _ = _detect_engine_with_network(network)
    return detected_engine


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
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
            "tensornetwork", "quimb", "tenpy", and "einsum". When omitted,
            ``show_tensor_network`` infers the engine from ``network``.
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
        >>> fig, ax = show_tensor_network(network, config=config)
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
    network_input = network
    if engine is None:
        resolved_engine, network_input = _detect_engine_with_network(network)
    else:
        resolved_engine = engine
    plot_2d, plot_3d = _get_plotters(resolved_engine)
    if not interactive_controls:
        if resolved_view == "2d":
            fig, ax_out = plot_2d(
                network_input,
                ax=cast(Axes, ax) if ax is not None else None,
                config=style,
                show_tensor_labels=show_tensor_labels,
                show_index_labels=show_index_labels,
            )
        elif resolved_view == "3d":
            fig, ax_out = plot_3d(
                network_input,
                ax=ax,
                config=style,
                show_tensor_labels=show_tensor_labels,
                show_index_labels=show_index_labels,
            )
        else:
            raise ValueError(f"Unsupported tensor network view: {resolved_view}")
    else:
        fig, ax_out = show_tensor_network_interactive(
            network_input,
            engine=resolved_engine,
            view=resolved_view,
            config=style,
            show_tensor_labels=resolved_tensor_labels,
            show_index_labels=resolved_index_labels,
            ax=ax,
        )
    if show:
        _show_figure(fig)
    return root_figure(fig), ax_out
