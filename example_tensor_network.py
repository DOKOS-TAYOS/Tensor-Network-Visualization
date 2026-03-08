from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorkrowch as tk

try:
    from tensor_network_viz import PlotConfig, show_tensor_network
except ImportError:
    # Allow running the example directly from the repo root without installing the package.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from tensor_network_viz import PlotConfig, show_tensor_network

DESCRIPTION = """\
Small demo for the plotting dispatcher.

It builds one example TensorKrowch network and shows it with the selected view.
Available network examples:
  - mps
  - mpo
  - peps
  - weird

Examples:
  python example_tensor_network.py mps 2d
  python example_tensor_network.py weird 3d
"""


def _make_node(
    network: tk.TensorNetwork,
    name: str,
    axes_names: tuple[str, ...],
) -> tk.Node:
    # Use small dummy dimensions so the example focuses on topology, not tensor values.
    shape = tuple(2 if axis in {"phys", "down", "up"} else 3 for axis in axes_names)
    return tk.Node(shape=shape, axes_names=axes_names, name=name, network=network)


def build_mps_example(length: int = 5) -> tk.TensorNetwork:
    # Build a simple 1D chain where each tensor has one physical leg and bond legs to neighbors.
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
        # Connect the right bond of each tensor with the left bond of the next one.
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return network


def build_mpo_example(length: int = 4) -> tk.TensorNetwork:
    # Build a 1D operator network: each tensor has input/output physical legs plus bond legs.
    network = tk.TensorNetwork(name="mpo")
    nodes: list[tk.Node] = []

    for index in range(length):
        axes_names = []
        if index > 0:
            axes_names.append("left")
        axes_names.extend(["down", "up"])
        if index < length - 1:
            axes_names.append("right")
        nodes.append(_make_node(network, f"W{index}", tuple(axes_names)))

    for index in range(length - 1):
        # Neighboring tensors are linked through the virtual MPO bonds.
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return network


def build_peps_example(rows: int = 2, cols: int = 3) -> tk.TensorNetwork:
    # Build a small 2D grid with horizontal and vertical bonds plus one physical leg per tensor.
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
                # Horizontal PEPS bond.
                grid[row][col]["right"] ^ grid[row][col + 1]["left"]
            if row < rows - 1:
                # Vertical PEPS bond.
                grid[row][col]["down"] ^ grid[row + 1][col]["up"]

    return network


def build_weird_example() -> tk.TensorNetwork:
    # Build a less regular graph to show that the visualizer is not limited to chain/grid layouts.
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

    # Mix star-like and lateral connections, including a double edge on the south-west side.
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
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
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
    # Pick one example builder from the CLI and create the TensorKrowch network.
    network = BUILDERS[args.network]()
    print(f"Building example tensor network: {args.network}")
    print(f"Selected visualization: {args.view}")
    print("Rendering window...")

    # Centralized plotting options shared by the generic plotting dispatcher.
    config = PlotConfig(
        figsize=(10, 6),
        show_tensor_labels=args.view == "2d",
        show_index_labels=args.view == "2d",
    )

    # The dispatcher chooses the proper rendering function from the selected engine and view.
    fig, ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view=args.view,
        config=config,
        show=False,
    )
    fig.suptitle(f"{args.network.upper()} ({args.view.upper()})", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
