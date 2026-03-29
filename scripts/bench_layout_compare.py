#!/usr/bin/env python3
"""Median timings for layout / axis+scale / full _plot_graph.

Works with both the pre-optimization API (``_resolve_draw_scale`` only) and the
post-optimization API (``_resolve_draw_scale_and_bond_curve_pad``).

Usage:
  python scripts/bench_layout_compare.py --tag before --repeats 5 --warmup 1
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Repo root on sys.path when run as python scripts/bench_layout_compare.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)


def _chain_graph(n_nodes: int) -> _GraphData:
    nodes = {i: _make_node(f"T{i}", ("L", "R")) for i in range(n_nodes)}
    edges: list[_EdgeData] = []
    for i in range(n_nodes - 1):
        edges.append(
            _make_contraction_edge(
                _EdgeEndpoint(i, 1, f"b{i}"),
                _EdgeEndpoint(i + 1, 0, f"b{i}"),
                name=f"b{i}",
            )
        )
    return _GraphData(nodes=nodes, edges=tuple(edges))


def _median_wall_seconds(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="", help="Label prepended to each line")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from tensor_network_viz._core import renderer as renderer_mod
    from tensor_network_viz._core.contractions import _group_contractions
    from tensor_network_viz._core.layout import (
        NodePositions,
        _compute_axis_directions,
        _compute_layout,
    )
    from tensor_network_viz._core.renderer import _plot_graph, _resolve_draw_scale
    from tensor_network_viz.config import PlotConfig

    prefix = f"[{args.tag}] " if args.tag else ""

    use_pad = hasattr(renderer_mod, "_resolve_draw_scale_and_bond_curve_pad")
    resolve_pad = getattr(renderer_mod, "_resolve_draw_scale_and_bond_curve_pad", None)
    api_note = "API: unified_scale_curve" if use_pad else "API: separate_resolve_draw_scale"

    suites: list[tuple[str, _GraphData]] = [
        ("chain_n24", _chain_graph(24)),
        ("chain_n48", _chain_graph(48)),
        ("chain_n72", _chain_graph(72)),
    ]
    plot_config = PlotConfig(figsize=(6, 4), refine_tensor_labels=False)

    print(f"{prefix}warmup={args.warmup} repeats={args.repeats} backend=Agg {api_note}\n")

    for label, graph in suites:
        print(f"{prefix}=== {label} |V|={len(graph.nodes)} |E|={len(graph.edges)} ===")

        t_layout = _median_wall_seconds(
            lambda g=graph: _compute_layout(g, dimensions=2, seed=0),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        positions: NodePositions = _compute_layout(graph, dimensions=2, seed=0)
        print(f"{prefix}  _compute_layout(2d):      {t_layout * 1000.0:.3f} ms")

        def axis_job(g: _GraphData = graph, pos: NodePositions = positions) -> None:
            groups = _group_contractions(g)
            if use_pad and resolve_pad is not None:
                scale, _pad = resolve_pad(g, pos, groups)
            else:
                scale = _resolve_draw_scale(g, pos)
            _compute_axis_directions(
                g,
                pos,
                dimensions=2,
                draw_scale=scale,
                contraction_groups=groups,
            )

        t_axis = _median_wall_seconds(axis_job, repeats=args.repeats, warmup=args.warmup)
        print(f"{prefix}  axis directions + scale: {t_axis * 1000.0:.3f} ms")

        def plot_job(g: _GraphData = graph) -> None:
            fig, _ax = _plot_graph(
                g,
                dimensions=2,
                config=plot_config,
                renderer_name="bench_compare",
            )
            plt.close(fig)

        t_plot = _median_wall_seconds(plot_job, repeats=args.repeats, warmup=args.warmup)
        print(f"{prefix}  _plot_graph (global):     {t_plot * 1000.0:.3f} ms")
        print()


if __name__ == "__main__":
    main()
