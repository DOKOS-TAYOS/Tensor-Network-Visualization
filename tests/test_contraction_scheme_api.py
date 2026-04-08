from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network


def test_show_tensor_network_scheme_requires_controls() -> None:
    trace = [pair_tensor("A0", "x0", "r0", "pa,p->a")]

    with pytest.raises(ValueError, match="show_controls=False"):
        show_tensor_network(
            trace,
            engine="einsum",
            config=PlotConfig(show_contraction_scheme=True),
            show_controls=False,
            show=False,
        )
