#!/usr/bin/env python3
"""Benchmark public render workflows for cache reuse and widget overhead.

Usage (from repo root):
  python scripts/bench_render_workflows.py
  python scripts/bench_render_workflows.py --nodes 180 --warmup 1 --repeats 5

This script compares:
- first render vs repeated render
- network object input vs list-of-tensors input
- static render vs interactive controls
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensor_network_viz import PlotConfig, clear_tensor_network_graph_cache, show_tensor_network


def _build_quimb_chain(n_nodes: int) -> object:
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - local benchmark helper
        raise SystemExit(
            "This benchmark needs quimb. Install the project dev requirements or "
            '`pip install "tensor-network-visualization[quimb]"` first.'
        ) from exc

    tensors: list[object] = []
    for index in range(n_nodes):
        if index == 0:
            tensors.append(
                qtn.Tensor(
                    data=np.ones((2, 3)),
                    inds=(f"i{index}", f"b{index}"),
                    tags={f"T{index}"},
                )
            )
            continue
        if index == n_nodes - 1:
            tensors.append(
                qtn.Tensor(
                    data=np.ones((3, 2)),
                    inds=(f"b{index - 1}", f"o{index}"),
                    tags={f"T{index}"},
                )
            )
            continue
        tensors.append(
            qtn.Tensor(
                data=np.ones((3, 3, 3)),
                inds=(f"b{index - 1}", f"b{index}", f"aux{index}"),
                tags={f"T{index}"},
            )
        )
    return qtn.TensorNetwork(tensors)


def _measure_render_seconds(
    network: object,
    *,
    show_controls: bool,
) -> float:
    config = PlotConfig(figsize=(9, 5), tensor_label_refinement="never")
    started = time.perf_counter()
    fig, _ax = show_tensor_network(
        network,
        config=config,
        show_controls=show_controls,
        show=False,
    )
    elapsed = time.perf_counter() - started
    plt.close(fig)
    return elapsed


def _median_repeated_render_seconds(
    network: object,
    *,
    show_controls: bool,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        _measure_render_seconds(network, show_controls=show_controls)
    samples: list[float] = []
    for _ in range(repeats):
        samples.append(_measure_render_seconds(network, show_controls=show_controls))
    return statistics.median(samples)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=160, help="Tensor count in the chain.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed repeated-render warmup runs per scenario.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeated-render runs per scenario.",
    )
    args = parser.parse_args()

    network = _build_quimb_chain(args.nodes)
    tensor_list_input = list(cast(Any, network).tensors)
    scenarios: list[tuple[str, object, bool]] = [
        ("network-object | static-render", network, False),
        ("network-object | interactive-controls", network, True),
        ("tensor-list | static-render", tensor_list_input, False),
        ("tensor-list | interactive-controls", tensor_list_input, True),
    ]

    print(
        f"backend=Agg nodes={args.nodes} warmup={args.warmup} repeats={args.repeats}\n"
        "Comparing first render vs repeated render for public show_tensor_network().\n"
    )

    for label, input_network, show_controls in scenarios:
        clear_tensor_network_graph_cache(input_network)
        first_render = _measure_render_seconds(input_network, show_controls=show_controls)
        repeated_render = _median_repeated_render_seconds(
            input_network,
            show_controls=show_controls,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        speedup = first_render / max(repeated_render, 1e-12)
        print(f"{label}")
        print(f"  first render:    {_format_ms(first_render)}")
        print(f"  repeated render: {_format_ms(repeated_render)}")
        print(f"  speedup:         {speedup:.2f}x")
        print()


if __name__ == "__main__":
    main()
