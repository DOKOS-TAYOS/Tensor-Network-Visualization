from __future__ import annotations

from typing import Any, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tensorkrowch_engine import (
    plot_tensorkrowch_network_2d,
    plot_tensorkrowch_network_3d,
)

from .config import EngineName, PlotConfig, ViewName

RenderedAxes: TypeAlias = Axes | Axes3D


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig | None = None,
) -> tuple[Figure, RenderedAxes]:
    style = config or PlotConfig()

    if engine == "tensorkrowch":
        if view == "2d":
            fig, ax = plot_tensorkrowch_network_2d(network, config=style)
        else:
            fig, ax = plot_tensorkrowch_network_3d(network, config=style)
    else:
        raise ValueError(f"Unsupported tensor network engine: {engine}")

    plt.show()
    return fig, ax
