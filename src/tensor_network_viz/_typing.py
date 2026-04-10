from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure, SubFigure

    CoordinateLike: TypeAlias = Sequence[float] | np.ndarray[Any, Any]
    FigureLike: TypeAlias = Figure | SubFigure
else:
    CoordinateLike: TypeAlias = Sequence[float] | Any
    FigureLike: TypeAlias = Any

PositionMapping: TypeAlias = Mapping[int, CoordinateLike]


def root_figure(fig: FigureLike) -> Figure:
    parent_figure = getattr(fig, "figure", None)
    if parent_figure is None:
        return cast("Figure", fig)
    return cast("Figure", parent_figure)


__all__ = [
    "CoordinateLike",
    "FigureLike",
    "PositionMapping",
    "root_figure",
]
