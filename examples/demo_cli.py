"""Shared CLI helpers for example scripts (import from the ``examples`` directory)."""

from __future__ import annotations

import argparse

from tensor_network_viz import PlotConfig


def add_hover_labels_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hover-labels",
        action="store_true",
        help=(
            "Show tensor names and bond index labels on pointer hover (2D axes or 3D projection). "
            "Requires an interactive window (useless with --no-show / non-interactive only)."
        ),
    )


def demo_plot_config(args: argparse.Namespace) -> PlotConfig | None:
    if getattr(args, "hover_labels", False):
        return PlotConfig(hover_labels=True)
    return None
