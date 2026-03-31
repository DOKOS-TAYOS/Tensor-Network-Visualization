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


def add_contraction_scheme_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contraction-scheme",
        action="store_true",
        help=(
            "Overlay contraction-step highlights. Einsum examples use steps from the trace; "
            "other engines use an illustrative manual schedule when one is defined for this demo."
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
    scheme = getattr(args, "contraction_scheme", False)
    if compact:
        cfg = PlotConfig(show_contraction_scheme=scheme)
    else:
        cfg = replace(showcase_plot_config(), show_contraction_scheme=scheme)
    if hover:
        return replace(cfg, hover_labels=True)
    return cfg


def optional_backend_contraction_scheme_by_name(
    *,
    network: str,
    engine: str,
) -> tuple[tuple[str, ...], ...] | None:
    """Illustrative ``contraction_scheme_by_name`` for non-einsum demos (tensor names must match)."""
    if engine == "einsum":
        return None
    if network == "mps" and engine in ("tensorkrowch", "tensornetwork", "quimb"):
        return (
            ("A0", "A1"),
            ("A1", "A2"),
            ("A0", "A1", "A2"),
        )
    if network == "mpo" and engine in ("tensorkrowch", "tensornetwork", "quimb"):
        return (
            ("W0", "W1"),
            ("W1", "W2"),
            ("W0", "W1", "W2"),
        )
    if network == "peps" and engine in ("tensorkrowch", "tensornetwork", "quimb"):
        return (
            ("P00", "P01"),
            ("P10", "P11"),
            ("P00", "P01", "P10", "P11"),
        )
    if network == "disconnected" and engine in ("tensorkrowch", "tensornetwork", "quimb"):
        return (("A", "B"), ("C", "D", "E"))
    if network == "hyper" and engine == "quimb":
        return (("A", "B", "C"),)
    if network == "cubic_peps" and engine == "tensornetwork":
        return (
            ("P0_0_0", "P1_0_0"),
            ("P0_0_1", "P1_0_1"),
            ("P0_0_0", "P1_0_0", "P0_0_1", "P1_0_1"),
        )
    if network == "chain" and engine == "tenpy_explicit":
        return (
            ("T0", "T1"),
            ("T1", "T2"),
            ("T0", "T1", "T2"),
        )
    if network == "hub" and engine == "tenpy_explicit":
        return (("A", "B", "C"),)
    return None


def finalize_demo_plot_config(
    args: argparse.Namespace,
    *,
    network: str | None,
    engine: str,
) -> PlotConfig:
    """Merge hover/compact/scheme flags; attach manual contraction groups for supported backends."""
    cfg = demo_plot_config(args)
    if not getattr(args, "contraction_scheme", False):
        return cfg
    if engine == "einsum":
        return cfg
    manual = optional_backend_contraction_scheme_by_name(network=network or "", engine=engine)
    if manual is not None:
        return replace(cfg, contraction_scheme_by_name=manual)
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
