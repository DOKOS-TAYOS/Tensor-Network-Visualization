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
from itertools import combinations
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


def _generic_subdivided_k5_graph(
    *,
    subdivisions: int = 2,
    pendant_length: int = 3,
) -> _GraphData:
    axes_by_node: dict[int, list[str]] = {node_id: [] for node_id in range(5)}
    edge_specs: list[tuple[int, int, str]] = []
    next_node_id = 5

    def add_edge(left_id: int, right_id: int, name: str) -> None:
        axes_by_node.setdefault(left_id, []).append(name)
        axes_by_node.setdefault(right_id, []).append(name)
        edge_specs.append((left_id, right_id, name))

    for left_id, right_id in combinations(range(5), 2):
        previous_id = left_id
        for step in range(subdivisions):
            current_id = next_node_id
            next_node_id += 1
            axes_by_node[current_id] = []
            add_edge(previous_id, current_id, f"k5_{left_id}_{right_id}_{step}")
            previous_id = current_id
        add_edge(previous_id, right_id, f"k5_{left_id}_{right_id}_end")

    for core_id in range(5):
        previous_id = core_id
        for step in range(pendant_length):
            current_id = next_node_id
            next_node_id += 1
            axes_by_node[current_id] = []
            add_edge(previous_id, current_id, f"tail_{core_id}_{step}")
            previous_id = current_id

    nodes = {
        node_id: _make_node(f"G_{node_id}", tuple(axis_names))
        for node_id, axis_names in axes_by_node.items()
    }
    axis_lookup = {
        node_id: {name: index for index, name in enumerate(node.axes_names)}
        for node_id, node in nodes.items()
    }
    edges = tuple(
        _make_contraction_edge(
            _EdgeEndpoint(left_id, axis_lookup[left_id][name], name),
            _EdgeEndpoint(right_id, axis_lookup[right_id][name], name),
            name=name,
            label=None,
        )
        for left_id, right_id, name in edge_specs
    )
    return _GraphData(nodes=nodes, edges=edges)


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
    parser.add_argument(
        "--suite",
        choices=("all", "chain", "generic"),
        default="all",
        help="Benchmark suite to run",
    )
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

    chain_suites: list[tuple[str, _GraphData]] = [
        ("chain_n24", _chain_graph(24)),
        ("chain_n48", _chain_graph(48)),
        ("chain_n72", _chain_graph(72)),
    ]
    generic_suites: list[tuple[str, _GraphData]] = [
        ("generic_k5_subdivided_tails", _generic_subdivided_k5_graph()),
    ]
    if args.suite == "chain":
        suites = chain_suites
    elif args.suite == "generic":
        suites = generic_suites
    else:
        suites = [*chain_suites, *generic_suites]
    plot_config = PlotConfig(figsize=(6, 4), tensor_label_refinement="never")

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

        t_layout_3d = _median_wall_seconds(
            lambda g=graph: _compute_layout(g, dimensions=3, seed=0),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        print(f"{prefix}  _compute_layout(3d):      {t_layout_3d * 1000.0:.3f} ms")

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
