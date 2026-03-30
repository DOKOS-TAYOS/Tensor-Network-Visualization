"""Shared CLI helpers for example scripts (import from the ``examples`` directory)."""

from __future__ import annotations

import argparse
from dataclasses import replace

from matplotlib.figure import Figure

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


def add_compact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Smaller figure and lighter layout work (default line widths / layout iterations; "
            "faster CI or quick checks)."
        ),
    )


def showcase_plot_config() -> PlotConfig:
    """Styling tuned for demos: modest canvas, slightly thicker lines, more layout iterations."""
    return PlotConfig(
        figsize=(8.5, 6.25),
        line_width_2d=1.05,
        line_width_3d=0.95,
        layout_iterations=300,
    )


def demo_plot_config(args: argparse.Namespace) -> PlotConfig:
    hover = getattr(args, "hover_labels", False)
    compact = getattr(args, "compact", False)
    cfg = PlotConfig() if compact else showcase_plot_config()
    if hover:
        return replace(cfg, hover_labels=True)
    return cfg


def apply_demo_caption(
    fig: Figure,
    *,
    title: str,
    subtitle: str | None = None,
    footer: str | None = None,
) -> None:
    fig.suptitle(title, fontsize=15, fontweight="semibold", color="#0F172A", y=0.965)
    if subtitle:
        fig.text(
            0.5,
            0.918,
            subtitle,
            ha="center",
            va="top",
            fontsize=11,
            color="#475569",
            transform=fig.transFigure,
        )
    if footer:
        fig.text(
            0.5,
            0.02,
            footer,
            ha="center",
            fontsize=9.5,
            color="#64748B",
            style="italic",
            transform=fig.transFigure,
        )
