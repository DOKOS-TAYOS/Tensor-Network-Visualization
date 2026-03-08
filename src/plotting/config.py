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
    bond_edge_color: str = "#00008B"
    dangling_edge_color: str = "#8B0000"
    figsize: tuple[float, float] | None = (8, 6)
    show_tensor_labels: bool = True
    show_index_labels: bool = True
