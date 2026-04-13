"""Public tensor-network rendering entry point and figure display helpers."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._import_state import _preserve_sys_module_entry
from ._input_inspection import (
    _default_view_for_network_input,
    _detect_network_engine_with_input,
    _merge_grid_positions_into_config,
    _prepare_network_input,
    _validate_grid_engine,
)
from ._logging import package_logger
from ._registry import _get_plotters
from ._typing import FigureLike, root_figure
from .config import EngineName, PlotConfig, ViewName, _theme_background_colors
from .exceptions import AxisConfigurationError

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


def _detect_engine_with_network(network: Any) -> tuple[EngineName, Any]:
    """Infer the backend engine and preserve single-pass iterables when needed."""
    return _detect_network_engine_with_input(network)


def _detect_engine(network: Any) -> EngineName:
    """Infer the backend engine from the received tensor-network object."""
    detected_engine, _ = _detect_engine_with_network(network)
    return detected_engine


def _apply_theme_background(
    fig: FigureLike,
    ax: RenderedAxes,
    *,
    theme: str,
) -> None:
    """Apply theme-driven figure and axes background colors."""
    figure_color, axes_color = _theme_background_colors(theme)
    root_figure(fig).patch.set_facecolor(figure_color)
    ax.set_facecolor(axes_color)


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
    view: ViewName | None = None,
    config: PlotConfig | None = None,
    ax: RenderedAxes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, RenderedAxes]:
    """Render a tensor network and optionally display the figure.

    Args:
        network: Tensor network object (with 'nodes'/'leaf_nodes'), an
            iterable of nodes with 'edges', 'axes_names' or 'axis_names',
            and 'name', or a nested 2D/3D ``list``/``tuple`` grid of those
            same node/tensor objects. In grid mode, ``None`` leaves a hole,
            ``network[row][col]`` fixes a regular 2D placement, and
            ``network[layer][row][col]`` fixes a regular 3D placement.
        engine: Rendering engine; supported values are "tensorkrowch",
            "tensornetwork", "quimb", "tenpy", and "einsum". When omitted,
            ``show_tensor_network`` infers the engine from ``network``.
        view: "2d" or "3d" visualization mode. ``None`` defaults to ``"2d"``
            unless ``ax`` is a 3D axes or ``network`` is a 3D grid, in which
            case the view is inferred.
        config: Optional plot configuration. When omitted, ``PlotConfig()`` is used.
            ``PlotConfig`` groups the most visible toggles first
            (nodes, labels, hover, contraction playback) and leaves geometry/styling
            options later. It also includes optional ``tensor_label_fontsize`` and
            ``edge_label_fontsize`` overrides. When a grid input is used, its fixed
            positions become the base layout and ``config.positions`` overrides only
            the node ids you pass explicitly.
        ax: Optional Matplotlib axes to render into. When passed, the 2D/3D
            selector is suppressed and the view is fixed to that axes.
        show_controls: If True, attach figure-level controls for view, hover,
            and label toggles when the renderer exposes the required scene
            cache. Set False for a static render without widgets.
        show: If True, display the figure. In a Jupyter kernel this uses
            ``IPython.display.display(fig)`` (use ``pip install
            "tensor-network-visualization[jupyter]"`` and ``%matplotlib widget``
            for interactive figures). Otherwise ``plt.show()`` is used.

    Notes:
        Repeated calls with the **same** ``network`` instance reuse the normalized
        graph structure for regular objects and for re-iterable builtin containers
        such as ``list``, ``tuple``, and ``dict``. One-shot iterators are rebuilt on
        each call. After in-place changes, call
        ``clear_tensor_network_graph_cache(network)`` so the next draw re-extracts the
        structure.

        For dangling/free indices, axis names such as ``left``, ``right``, ``up``,
        ``down``, ``front``, ``back``, ``xp/xm/yp/ym/zp/zm``, ``north/south/east/west``,
        and ``in/out`` are tried first as directional hints, then fall back to the
        regular candidate search if they conflict with existing geometry.

    Returns:
        Tuple of (Figure, Axes) for further customization.

    Raises:
        AxisConfigurationError: If ``ax`` and ``view`` request incompatible dimensions, or
            if the resolved view name is unsupported.
        ValueError: If ``show_contraction_scheme=True`` is requested together with
            ``show_controls=False``.
        VisualizationInputError: If a nested 2D/3D grid input is malformed or unsupported.

    Examples:
        >>> config = PlotConfig(show_tensor_labels=True, hover_labels=True, figsize=(8, 6))
        >>> fig, ax = show_tensor_network(network, config=config)
    """
    style = config or PlotConfig()
    package_logger.debug(
        "show_tensor_network called with engine=%r view=%r show_controls=%s show=%s.",
        engine,
        view,
        show_controls,
        show,
    )
    network_input = _prepare_network_input(network)
    ax_view: ViewName | None = None
    if ax is not None:
        ax_view = "3d" if getattr(ax, "name", "") == "3d" else "2d"
    resolved_view = view
    if resolved_view is None:
        resolved_view = ax_view or _default_view_for_network_input(network_input) or "2d"
    if ax_view is not None and resolved_view != ax_view:
        raise AxisConfigurationError(
            f"Provided ax is {ax_view}, but view={resolved_view!r} was requested."
        )
    if resolved_view not in ("2d", "3d"):
        raise AxisConfigurationError(f"Unsupported tensor network view: {resolved_view}")
    if engine is None:
        resolved_engine, network_input = _detect_engine_with_network(network_input)
    else:
        resolved_engine = engine
    _validate_grid_engine(network_input, engine=resolved_engine)
    package_logger.debug(
        "Rendering tensor network with engine=%r resolved_view=%r.", resolved_engine, resolved_view
    )
    plot_2d, plot_3d = _get_plotters(resolved_engine)
    if style.show_contraction_scheme and not show_controls:
        raise ValueError(
            "show_contraction_scheme=True requires show_controls=True because the contraction "
            "scheme is dynamic-only; it cannot be rendered with show_controls=False."
        )
    if not show_controls:
        dimensions = 2 if resolved_view == "2d" else 3
        style = _merge_grid_positions_into_config(style, network_input, dimensions=dimensions)
        static_style = style
        with _preserve_sys_module_entry("matplotlib.widgets"):
            if resolved_view == "2d":
                fig, ax_out = plot_2d(
                    network_input,
                    ax=cast(Axes, ax) if ax is not None else None,
                    config=static_style,
                    _build_contraction_controls=False,
                    _build_scene_state=False,
                )
            elif resolved_view == "3d":
                fig, ax_out = plot_3d(
                    network_input,
                    ax=ax,
                    config=static_style,
                    _build_contraction_controls=False,
                    _build_scene_state=False,
                )
            else:
                raise AxisConfigurationError(f"Unsupported tensor network view: {resolved_view}")
    else:
        from .interactive_viewer import show_tensor_network_interactive

        fig, ax_out = show_tensor_network_interactive(
            network_input,
            engine=resolved_engine,
            view=resolved_view,
            config=style,
            ax=ax,
        )
    _apply_theme_background(fig, ax_out, theme=style.theme)
    if show:
        _show_figure(fig)
    return root_figure(fig), ax_out
