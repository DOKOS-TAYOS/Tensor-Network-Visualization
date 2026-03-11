"""Public entry points for google/TensorNetwork plotting."""

from __future__ import annotations

from .._core.renderer import _make_plot_functions
from .graph import _build_graph

(
    plot_tensornetwork_network_2d,
    plot_tensornetwork_network_3d,
) = _make_plot_functions(
    _build_graph,
    "plot_tensornetwork_network_2d",
    "plot_tensornetwork_network_3d",
    "Plot a TensorNetwork node collection in 2D.",
    "Plot a TensorNetwork node collection in 3D.",
)
