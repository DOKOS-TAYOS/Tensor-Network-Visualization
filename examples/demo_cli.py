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


def mps_demo_tensor_names(length: int = 11) -> tuple[str, ...]:
    """Tensor names in chain order for MPS demos (matches ``A{i}`` in example builders)."""
    return tuple(f"A{i}" for i in range(length))


def mpo_demo_tensor_names(length: int = 7) -> tuple[str, ...]:
    """Tensor names in chain order for MPO demos (matches ``W{i}`` in example builders)."""
    return tuple(f"W{i}" for i in range(length))


def peps_demo_tensor_names(rows: int = 4, cols: int = 5) -> tuple[str, ...]:
    """Row-major PEPS site names (matches ``P{row}{col}`` in example builders)."""
    return tuple(f"P{row}{col}" for row in range(rows) for col in range(cols))


def cubic_peps_tensor_names(lx: int, ly: int, lz: int) -> tuple[str, ...]:
    """Lexicographic grid order (matches ``build_cubic_peps``: x, then y, then z)."""
    if min(lx, ly, lz) < 1:
        raise ValueError("lx, ly, lz must be >= 1")
    return tuple(f"P{i}_{j}_{k}" for i in range(lx) for j in range(ly) for k in range(lz))


def cumulative_prefix_contraction_scheme(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Growing hull steps: (a,b), (a,b,c), … so the last step includes every tensor."""
    if not names:
        return ()
    if len(names) == 1:
        return (names,)
    return tuple(tuple(names[:k]) for k in range(2, len(names) + 1))


def demo_scheme_tensor_names_for_network(network: str) -> tuple[str, ...] | None:
    """Full tensor name lists for structured demos (MPS/MPO/PEPS with default builder sizes)."""
    if network == "mps":
        return mps_demo_tensor_names()
    if network == "mpo":
        return mpo_demo_tensor_names()
    if network == "peps":
        return peps_demo_tensor_names()
    return None


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
    """Illustrative ``contraction_scheme_by_name`` for non-einsum demos.

    Tensor names must match the ones created by the demo builder.
    """
    if engine == "einsum":
        return None
    if network == "disconnected" and engine in ("tensorkrowch", "tensornetwork", "quimb"):
        return (
            ("A", "B"),
            ("A", "B", "C", "D", "E"),
        )
    if network == "hyper" and engine == "quimb":
        return (("A", "B", "C"),)
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
    scheme_tensor_names: tuple[str, ...] | None = None,
) -> PlotConfig:
    """Merge hover/compact/scheme flags; attach manual contraction groups for supported backends."""
    cfg = demo_plot_config(args)
    if not getattr(args, "contraction_scheme", False):
        return cfg
    if engine == "einsum":
        return cfg
    if scheme_tensor_names is not None:
        return replace(
            cfg,
            contraction_scheme_by_name=cumulative_prefix_contraction_scheme(scheme_tensor_names),
        )
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
