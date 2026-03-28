from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tensornetwork as tn

try:
    from tensor_network_viz import show_tensor_network
except ImportError:
    # Allow running the example directly from the repo without installing the package.
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import show_tensor_network

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from demo_cli import add_hover_labels_argument, demo_plot_config

DESCRIPTION = """\
Small demo for the TensorNetwork backend.

It builds one example TensorNetwork node collection and shows it with the selected view.
Available network examples:
  - mps
  - mpo
  - peps
  - weird
  - disconnected

Examples:
  python examples/tensornetwork_demo.py mps 2d
  python examples/tensornetwork_demo.py weird 3d
  python examples/tensornetwork_demo.py disconnected 2d
  python examples/tensornetwork_demo.py mps 2d --save mps.png --no-show
  python examples/tensornetwork_demo.py mps 2d --hover-labels
"""


def _make_node(name: str, axis_names: tuple[str, ...]) -> tn.Node:
    # Use small dummy dimensions so the example focuses on topology, not tensor values.
    shape = tuple(2 if axis in {"phys", "down", "up"} else 3 for axis in axis_names)
    return tn.Node(np.ones(shape, dtype=float), name=name, axis_names=axis_names)


def build_mps_example(length: int = 5) -> list[tn.Node]:
    # Build a simple 1D chain where each tensor has one physical leg and bond legs to neighbors.
    nodes: list[tn.Node] = []

    for index in range(length):
        axis_names = []
        if index > 0:
            axis_names.append("left")
        axis_names.append("phys")
        if index < length - 1:
            axis_names.append("right")
        nodes.append(_make_node(f"A{index}", tuple(axis_names)))

    for index in range(length - 1):
        # Connect the right bond of each tensor with the left bond of the next one.
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return nodes


def build_mpo_example(length: int = 4) -> list[tn.Node]:
    # Build a 1D operator network: each tensor has input/output physical legs plus bond legs.
    nodes: list[tn.Node] = []

    for index in range(length):
        axis_names = []
        if index > 0:
            axis_names.append("left")
        axis_names.extend(["down", "up"])
        if index < length - 1:
            axis_names.append("right")
        nodes.append(_make_node(f"W{index}", tuple(axis_names)))

    for index in range(length - 1):
        # Neighboring tensors are linked through the virtual MPO bonds.
        nodes[index]["right"] ^ nodes[index + 1]["left"]

    return nodes


def build_peps_example(rows: int = 2, cols: int = 3) -> list[tn.Node]:
    # Build a small 2D grid with horizontal and vertical bonds plus one physical leg per tensor.
    grid: list[list[tn.Node]] = []

    for row in range(rows):
        row_nodes: list[tn.Node] = []
        for col in range(cols):
            axis_names = []
            if row > 0:
                axis_names.append("up")
            if col > 0:
                axis_names.append("left")
            axis_names.append("phys")
            if col < cols - 1:
                axis_names.append("right")
            if row < rows - 1:
                axis_names.append("down")
            row_nodes.append(_make_node(f"P{row}{col}", tuple(axis_names)))
        grid.append(row_nodes)

    for row in range(rows):
        for col in range(cols):
            if col < cols - 1:
                # Horizontal PEPS bond.
                grid[row][col]["right"] ^ grid[row][col + 1]["left"]
            if row < rows - 1:
                # Vertical PEPS bond.
                grid[row][col]["down"] ^ grid[row + 1][col]["up"]

    return [node for row_nodes in grid for node in row_nodes]


def build_weird_example() -> list[tn.Node]:
    # Build a less regular graph to show that the visualizer is not limited to chain/grid layouts.
    center = _make_node("center", ("north", "east", "south", "west", "phys"))
    north = _make_node("north", ("center", "east", "phys"))
    east = _make_node("east", ("center", "north", "south", "phys"))
    south = _make_node("south", ("center", "east", "west_a", "west_b", "phys"))
    west = _make_node("west", ("center", "south_a", "south_b", "phys"))

    # Mix star-like and lateral connections, including a double edge on the south-west side.
    center["north"] ^ north["center"]
    center["east"] ^ east["center"]
    center["south"] ^ south["center"]
    center["west"] ^ west["center"]
    north["east"] ^ east["north"]
    east["south"] ^ south["east"]
    south["west_a"] ^ west["south_a"]
    south["west_b"] ^ west["south_b"]

    return [center, north, east, south, west]


def build_disconnected_example() -> list[tn.Node]:
    # Two separate components: a dimer A-B and a triangle C-D-E.
    a = _make_node("A", ("bond", "phys"))
    b = _make_node("B", ("bond", "phys"))
    a["bond"] ^ b["bond"]

    c = _make_node("C", ("left", "right", "phys"))
    d = _make_node("D", ("left", "right", "phys"))
    e = _make_node("E", ("left", "right", "phys"))
    c["left"] ^ d["right"]
    d["left"] ^ e["right"]
    e["left"] ^ c["right"]

    return [a, b, c, d, e]


BUILDERS = {
    "disconnected": build_disconnected_example,
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
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the rendered figure to this path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render without opening an interactive Matplotlib window.",
    )
    add_hover_labels_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = BUILDERS[args.network]()
    print(f"Building example TensorNetwork nodes: {args.network}")
    print(f"Selected visualization: {args.view}")
    print("Passing as: list of TensorNetwork nodes")
    print("Rendering figure...")

    fig, ax = show_tensor_network(
        nodes,
        engine="tensornetwork",
        view=args.view,
        config=demo_plot_config(args),
        show=False,
    )
    fig.suptitle(f"{args.network.upper()} ({args.view.upper()})", fontsize=16)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to: {args.save}")
    if args.no_show:
        plt.close(fig)
        return
    plt.show()


if __name__ == "__main__":
    main()
