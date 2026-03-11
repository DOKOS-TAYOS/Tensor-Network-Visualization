"""Public entry points for TensorKrowch plotting."""

from __future__ import annotations

from .._core.renderer import _make_plot_functions
from .graph import _build_graph

(
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
) = _make_plot_functions(
    _build_graph,
    "plot_tensorkrowch_network_2d",
    "plot_tensorkrowch_network_3d",
    "Plot a TensorKrowch tensor network in 2D.",
    "Plot a TensorKrowch tensor network in 3D.",
)
