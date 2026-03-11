from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import quimb.tensor as qtn

try:
    from tensor_network_viz import PlotConfig, show_tensor_network
except ImportError:
    # Allow running the example directly from the repo without installing the package.
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import PlotConfig, show_tensor_network

DESCRIPTION = """\
Small demo for the Quimb backend.

It builds one example Quimb tensor network and shows it with the selected view.
Available network examples:
  - mps
  - mpo
  - peps
  - weird
  - disconnected

Examples:
  python examples/quimb_demo.py mps 2d
  python examples/quimb_demo.py weird 3d
  python examples/quimb_demo.py disconnected 2d
  python examples/quimb_demo.py mps 2d --from-list --save quimb.png --no-show
"""


def _make_tensor(name: str, inds: tuple[str, ...]) -> qtn.Tensor:
    shape = tuple(2 if "phys" in ind or "up" in ind or "down" in ind else 3 for ind in inds)
    return qtn.Tensor(data=np.ones(shape, dtype=float), inds=inds, tags={name})


def build_mps_example(length: int = 5) -> qtn.TensorNetwork:
    tensors: list[qtn.Tensor] = []
    for index in range(length):
        inds = []
        if index > 0:
            inds.append(f"bond_{index - 1}_{index}")
        inds.append(f"phys_{index}")
        if index < length - 1:
            inds.append(f"bond_{index}_{index + 1}")
        tensors.append(_make_tensor(f"A{index}", tuple(inds)))
    return qtn.TensorNetwork(tensors)


def build_mpo_example(length: int = 4) -> qtn.TensorNetwork:
    tensors: list[qtn.Tensor] = []
    for index in range(length):
        inds = []
        if index > 0:
            inds.append(f"bond_{index - 1}_{index}")
        inds.extend((f"down_{index}", f"up_{index}"))
        if index < length - 1:
            inds.append(f"bond_{index}_{index + 1}")
        tensors.append(_make_tensor(f"W{index}", tuple(inds)))
    return qtn.TensorNetwork(tensors)


def build_peps_example(rows: int = 2, cols: int = 3) -> qtn.TensorNetwork:
    tensors: list[qtn.Tensor] = []
    for row in range(rows):
        for col in range(cols):
            inds = []
            if row > 0:
                inds.append(f"v_{row - 1}_{col}")
            if col > 0:
                inds.append(f"h_{row}_{col - 1}")
            inds.append(f"phys_{row}_{col}")
            if col < cols - 1:
                inds.append(f"h_{row}_{col}")
            if row < rows - 1:
                inds.append(f"v_{row}_{col}")
            tensors.append(_make_tensor(f"P{row}{col}", tuple(inds)))
    return qtn.TensorNetwork(tensors)


def build_weird_example() -> qtn.TensorNetwork:
    tensors = [
        _make_tensor("center", ("north", "east", "south", "west", "phys_center")),
        _make_tensor("north", ("north", "north_east", "phys_north")),
        _make_tensor("east", ("east", "north_east", "east_south", "phys_east")),
        _make_tensor(
            "south",
            ("south", "east_south", "south_west_a", "south_west_b", "phys_south"),
        ),
        _make_tensor("west", ("west", "south_west_a", "south_west_b", "phys_west")),
    ]
    return qtn.TensorNetwork(tensors)


def build_disconnected_example() -> qtn.TensorNetwork:
    tensors = [
        _make_tensor("A", ("ab", "phys_a")),
        _make_tensor("B", ("ab", "phys_b")),
        _make_tensor("C", ("cd", "ec", "phys_c")),
        _make_tensor("D", ("cd", "de", "phys_d")),
        _make_tensor("E", ("de", "ec", "phys_e")),
    ]
    return qtn.TensorNetwork(tensors)


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
        "--from-list",
        action="store_true",
        help="Pass the network as a list of tensors instead of a TensorNetwork object.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    network = BUILDERS[args.network]()
    print(f"Building example Quimb tensor network: {args.network}")
    print(f"Selected visualization: {args.view}")
    print(f"Passing as: {'list of tensors' if args.from_list else 'TensorNetwork'}")
    print("Rendering figure...")

    config = PlotConfig(
        figsize=(10, 6),
        show_tensor_labels=args.view == "2d",
        show_index_labels=args.view == "2d",
    )

    show_input: qtn.TensorNetwork | list[qtn.Tensor]
    show_input = list(network.tensors) if args.from_list else network

    fig, ax = show_tensor_network(
        show_input,
        engine="quimb",
        view=args.view,
        config=config,
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
