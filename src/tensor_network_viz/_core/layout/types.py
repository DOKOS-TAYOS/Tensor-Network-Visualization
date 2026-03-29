"""Layout type aliases."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

Vector: TypeAlias = np.ndarray
NodePositions: TypeAlias = dict[int, Vector]
AxisDirections: TypeAlias = dict[tuple[int, int], Vector]

__all__ = ["AxisDirections", "NodePositions", "Vector"]
