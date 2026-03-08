from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

EngineName: TypeAlias = Literal["tensorkrowch"]
ViewName: TypeAlias = Literal["2d", "3d"]


@dataclass(frozen=True)
class PlotConfig:
    node_color: str = "#2D6A9F"
    edge_color: str = "#202B33"
    label_color: str = "#0C1319"
    figsize: tuple[float, float] | None = None
    show_tensor_labels: bool = True
    show_index_labels: bool = True
