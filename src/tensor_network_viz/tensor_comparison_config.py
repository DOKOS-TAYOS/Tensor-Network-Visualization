"""Public configuration types for tensor-to-tensor comparison views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TensorComparisonMode: TypeAlias = Literal[
    "reference",
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
]


@dataclass(frozen=True)
class TensorComparisonConfig:
    """Configuration for ``show_tensor_comparison``.

    Attributes:
        mode: Comparison transform to display.
        zero_threshold: Positive floor used when masking near-zero denominators in
            relative-difference and ratio views.
        topk_count: Number of ranked entries shown by ``"topk_changes"`` mode.
    """

    mode: TensorComparisonMode = "reference"
    zero_threshold: float = 1e-12
    topk_count: int = 8

    def __post_init__(self) -> None:
        """Validate numeric comparison parameters."""
        if float(self.zero_threshold) <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        if int(self.topk_count) <= 0:
            raise ValueError("topk_count must be positive.")


__all__ = ["TensorComparisonConfig", "TensorComparisonMode"]
