from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorkrowch as tk

try:
    from tensor_visualization import plot_tensor_network_2d, plot_tensor_network_3d
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from tensor_visualization import plot_tensor_network_2d, plot_tensor_network_3d


def _make_node(
    network: tk.TensorNetwork,
    name: str,
    axes_names: tuple[str, ...],
) -> tk.Node:
    shape = tuple(2 if axis in {"phys", "in", "out"} else 3 for axis in axes_names)
    return tk.Node(shape=shape, axes_names=axes_names, name=name, network=network)


def build_mps_example(length: int = 5) -> tk.TensorNetwork:
    network = tk.TensorNetwork(name="mps")
    nodes: list[tk.Node] = []

    for index in range(length):
        axes_names = []
        if index > 0:
            axes_names.append("left")
        axes_names.append("phys")
        if index < length - 1:
            axes_names.append("right")
        nodes.append(_make_node(network, f"A{index}", tuple(axes_names)))

    for index in range(length - 1):
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return network


def build_mpo_example(length: int = 4) -> tk.TensorNetwork:
    network = tk.TensorNetwork(name="mpo")
    nodes: list[tk.Node] = []

    for index in range(length):
        axes_names = []
        if index > 0:
            axes_names.append("left")
        axes_names.extend(["in", "out"])
        if index < length - 1:
            axes_names.append("right")
        nodes.append(_make_node(network, f"W{index}", tuple(axes_names)))

    for index in range(length - 1):
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return network


def build_peps_example(rows: int = 2, cols: int = 3) -> tk.TensorNetwork:
    network = tk.TensorNetwork(name="peps")
    grid: list[list[tk.Node]] = []

    for row in range(rows):
        row_nodes: list[tk.Node] = []
        for col in range(cols):
            axes_names = []
            if row > 0:
                axes_names.append("up")
            if col > 0:
                axes_names.append("left")
            axes_names.append("phys")
            if col < cols - 1:
                axes_names.append("right")
            if row < rows - 1:
                axes_names.append("down")
            row_nodes.append(_make_node(network, f"P{row}{col}", tuple(axes_names)))
        grid.append(row_nodes)

    for row in range(rows):
        for col in range(cols):
            if col < cols - 1:
                grid[row][col]["right"] ^ grid[row][col + 1]["left"]
            if row < rows - 1:
                grid[row][col]["down"] ^ grid[row + 1][col]["up"]

    return network


def build_weird_example() -> tk.TensorNetwork:
    network = tk.TensorNetwork(name="weird")

    center = _make_node(
        network,
        "center",
        ("north", "east", "south", "west", "phys"),
    )
    north = _make_node(network, "north", ("center", "east", "phys"))
    east = _make_node(network, "east", ("center", "north", "south", "phys"))
    south = _make_node(network, "south", ("center", "east", "west_a", "west_b", "phys"))
    west = _make_node(network, "west", ("center", "south_a", "south_b", "phys"))

    center["north"] ^ north["center"]
    center["east"] ^ east["center"]
    center["south"] ^ south["center"]
    center["west"] ^ west["center"]
    north["east"] ^ east["north"]
    east["south"] ^ south["east"]
    south["west_a"] ^ west["south_a"]
    south["west_b"] ^ west["south_b"]

    return network


BUILDERS = {
    "mps": build_mps_example,
    "mpo": build_mpo_example,
    "peps": build_peps_example,
    "weird": build_weird_example,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a TensorKrowch example network with tensor_visualization."
    )
    parser.add_argument(
        "network",
        choices=sorted(BUILDERS),
        help="Example tensor network to build.",
    )
    parser.add_argument(
        "view",
        choices=("2d", "3d"),
        help="Visualization mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network = BUILDERS[args.network]()

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    fig.suptitle(f"{args.network.upper()} ({args.view.upper()})", fontsize=16)

    if args.view == "3d":
        ax = fig.add_subplot(111, projection="3d")
        plot_tensor_network_3d(
            network,
            ax=ax,
            show_tensor_labels=False,
            show_index_labels=False,
        )
    else:
        ax = fig.add_subplot(111)
        plot_tensor_network_2d(network, ax=ax)

    ax.set_title(args.network.upper())

    plt.show()


if __name__ == "__main__":
    main()
