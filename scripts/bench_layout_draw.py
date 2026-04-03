#!/usr/bin/env python3
"""Benchmark layout, axis directions, and full plot path (median ``perf_counter``, with warm-up).

Usage (from repo root):
  python scripts/bench_layout_draw.py
  python scripts/bench_layout_draw.py --repeats 11 --warmup 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensor_network_viz._core.contractions import _group_contractions
from tensor_network_viz._core.graph import (
    _EdgeData,
    _EdgeEndpoint,
    _GraphData,
    _make_contraction_edge,
    _make_node,
)
from tensor_network_viz._core.layout import NodePositions, _compute_axis_directions, _compute_layout
from tensor_network_viz._core.renderer import _plot_graph, _resolve_draw_scale_and_bond_curve_pad
from tensor_network_viz.config import PlotConfig


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


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warm-up runs (default 1)")
    args = parser.parse_args()

    # Sizes chosen to exercise layout vs. Matplotlib without multi-minute runs on typical laptops.
    suites: list[tuple[str, _GraphData]] = [
        ("chain_n24", _chain_graph(24)),
        ("chain_n48", _chain_graph(48)),
        ("chain_n72", _chain_graph(72)),
    ]

    plot_config = PlotConfig(figsize=(6, 4), tensor_label_refinement="never")

    print(f"warmup={args.warmup} repeats={args.repeats} backend=Agg\n")

    for label, graph in suites:
        print(f"=== {label} |V|={len(graph.nodes)} |E|={len(graph.edges)} ===")

        t_layout = _median_wall_seconds(
            lambda g=graph: _compute_layout(g, dimensions=2, seed=0),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        positions: NodePositions = _compute_layout(graph, dimensions=2, seed=0)
        print(f"  _compute_layout(2d):      {_fmt_ms(t_layout)}")

        def axis_job(g: _GraphData = graph, pos: NodePositions = positions) -> None:
            groups = _group_contractions(g)
            scale, _pad = _resolve_draw_scale_and_bond_curve_pad(g, pos, groups)
            _compute_axis_directions(
                g,
                pos,
                dimensions=2,
                draw_scale=scale,
                contraction_groups=groups,
            )

        t_axis = _median_wall_seconds(axis_job, repeats=args.repeats, warmup=args.warmup)
        print(f"  axis directions + scale: {_fmt_ms(t_axis)}")

        def plot_job(g: _GraphData = graph) -> None:
            fig, _ax = _plot_graph(
                g,
                dimensions=2,
                config=plot_config,
                renderer_name="bench",
            )
            plt.close(fig)

        t_plot = _median_wall_seconds(plot_job, repeats=args.repeats, warmup=args.warmup)
        print(f"  _plot_graph (global):    {_fmt_ms(t_plot)}")
        print()


if __name__ == "__main__":
    main()
