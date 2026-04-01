from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from matplotlib.figure import Figure, SubFigure

CoordinateLike: TypeAlias = Sequence[float] | np.ndarray[Any, Any]
PositionMapping: TypeAlias = Mapping[int, CoordinateLike]
FigureLike: TypeAlias = Figure | SubFigure


def root_figure(fig: FigureLike) -> Figure:
    if isinstance(fig, Figure):
        return fig
    return cast(Figure, fig.figure)


__all__ = [
    "CoordinateLike",
    "FigureLike",
    "PositionMapping",
    "root_figure",
]
