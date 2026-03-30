from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import tensorkrowch as tk

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
from demo_cli import (
    add_compact_argument,
    add_hover_labels_argument,
    apply_demo_caption,
    demo_plot_config,
)

DESCRIPTION = """\
TensorKrowch: native ``TensorNetwork`` / ``Node`` graphs — same ``show_tensor_network`` entry point.

Available network examples:
  - mps, mpo, peps, ladder, weird, disconnected

Examples:
  python examples/tensorkrowch_demo.py mps 2d
  python examples/tensorkrowch_demo.py ladder 3d
  python examples/tensorkrowch_demo.py peps 2d --from-list
  python examples/tensorkrowch_demo.py mps 2d --save tk.png --no-show
  python examples/tensorkrowch_demo.py mps 2d --hover-labels
"""


def _make_node(
    network: tk.TensorNetwork,
    name: str,
    axes_names: tuple[str, ...],
) -> tk.Node:
    # Use small dummy dimensions so the example focuses on topology, not tensor values.
    shape = tuple(2 if axis in {"phys", "down", "up"} else 3 for axis in axes_names)
    return tk.Node(shape=shape, axes_names=axes_names, name=name, network=network)


def build_mps_example(length: int = 11) -> tk.TensorNetwork:
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


def build_mpo_example(length: int = 7) -> tk.TensorNetwork:
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


def build_peps_example(rows: int = 4, cols: int = 5) -> tk.TensorNetwork:
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


def build_ladder_example(length: int = 8) -> tk.TensorNetwork:
    if length < 2:
        raise ValueError("ladder length must be >= 2")
    network = tk.TensorNetwork(name="ladder")
    top: list[tk.Node] = []
    bot: list[tk.Node] = []
    for i in range(length):
        if i == 0:
            top.append(_make_node(network, f"T{i}", ("phys", "right", "down")))
            bot.append(_make_node(network, f"B{i}", ("phys", "right", "up")))
        elif i == length - 1:
            top.append(_make_node(network, f"T{i}", ("left", "phys", "down")))
            bot.append(_make_node(network, f"B{i}", ("left", "phys", "up")))
        else:
            top.append(_make_node(network, f"T{i}", ("left", "phys", "right", "down")))
            bot.append(_make_node(network, f"B{i}", ("left", "phys", "right", "up")))
    for i in range(length - 1):
        top[i]["right"] ^ top[i + 1]["left"]
        bot[i]["right"] ^ bot[i + 1]["left"]
    for i in range(length):
        top[i]["down"] ^ bot[i]["up"]
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


def build_disconnected_example() -> tk.TensorNetwork:
    # Two separate components in the same network: a dimer A-B and a triangle C-D-E.
    # Demonstrates visualization of disconnected subgraphs (e.g. from --from-list with a subset).
    network = tk.TensorNetwork(name="disconnected")

    # Component 1: simple dimer
    a = _make_node(network, "A", ("bond", "phys"))
    b = _make_node(network, "B", ("bond", "phys"))
    a["bond"] ^ b["bond"]

    # Component 2: triangle
    c = _make_node(network, "C", ("left", "right", "phys"))
    d = _make_node(network, "D", ("left", "right", "phys"))
    e = _make_node(network, "E", ("left", "right", "phys"))
    c["left"] ^ d["right"]
    d["left"] ^ e["right"]
    e["left"] ^ c["right"]

    return network


BUILDERS = {
    "disconnected": build_disconnected_example,
    "ladder": build_ladder_example,
    "mps": build_mps_example,
    "mpo": build_mpo_example,
    "peps": build_peps_example,
    "weird": build_weird_example,
}


TK_TAGLINES: dict[str, str] = {
    "mps": "Finite MPS chain — contraction order and topology preserved in the node graph.",
    "mpo": "MPO string — mirrors how algorithms attach operators to the chain.",
    "peps": "Grid PEPS — 2D adjacency from local ``^`` contractions only.",
    "ladder": "Rails + rungs — useful for PEPS doubling layers or coupled wires.",
    "weird": "Messy topology — shows robustness of normalization + layout.",
    "disconnected": "Several motifs at once — still one call to the visualizer.",
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
        "--from-list",
        action="store_true",
        help="Pass the network as a list of nodes instead of a TensorNetwork object.",
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
    add_compact_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    network = BUILDERS[args.network]()
    print(f"Building example tensor network: {args.network}")
    print(f"Selected visualization: {args.view}")
    print(f"Passing as: {'list of nodes' if args.from_list else 'TensorNetwork'}")
    print("Rendering figure...")

    if args.from_list:
        raw = network.nodes
        show_input: tk.TensorNetwork | list[tk.Node] = (
            list(raw.values()) if isinstance(raw, dict) else list(raw)
        )
    else:
        show_input = network

    fig, _ax = show_tensor_network(
        show_input,
        engine="tensorkrowch",
        view=args.view,
        config=demo_plot_config(args),
        show=False,
    )
    apply_demo_caption(
        fig,
        title=f"TensorKrowch · {args.network.upper()} · {args.view.upper()}",
        subtitle=TK_TAGLINES.get(args.network),
        footer="engine='tensorkrowch' — integrate with custom optimizers and slicing",
    )
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
