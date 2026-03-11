"""Public entry points for plotting traced einsum networks."""

from __future__ import annotations

from .._core.renderer import _make_plot_functions
from .graph import _build_graph

(
    plot_einsum_network_2d,
    plot_einsum_network_3d,
) = _make_plot_functions(
    _build_graph,
    "plot_einsum_network_2d",
    "plot_einsum_network_3d",
    "Plot an ordered einsum trace in 2D.",
    "Plot an ordered einsum trace in 3D.",
)
