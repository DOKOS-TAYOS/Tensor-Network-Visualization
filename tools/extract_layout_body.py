from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
p = REPO / "src/tensor_network_viz/_core/layout.py"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
CORE = REPO / "src/tensor_network_viz/_core/layout"
CORE.mkdir(exist_ok=True)
a = "".join(lines[49:232])
b = "".join(lines[373:1178])
header = '''"""Layout orchestration (components, axis directions, stubs)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from ..config import PlotConfig
from .axis_directions import _AXIS_DIR_2D, _AXIS_DIR_3D
from .contractions import _contraction_weights, _group_contractions, _iter_contractions
from .curves import _quadratic_curve
from .graph import _GraphData
from .layout_structure import (
    _analyze_layout_components,
    _component_orthogonal_basis,
    _LayoutComponent,
    _leaf_nodes,
    _specialized_anchor_positions,
)
from .force_directed import _compute_force_layout
from .parameters import *
from .types import AxisDirections, NodePositions, Vector

'''
(CORE / "body.py").write_text(header + a + b, encoding="utf-8")
print("wrote body.py")
