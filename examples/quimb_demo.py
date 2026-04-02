from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import quimb.tensor as qtn

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from demo_cli import (
    add_compact_argument,
    add_contraction_scheme_argument,
    add_hover_labels_argument,
    apply_demo_caption,
    demo_runs_headless,
    demo_scheme_tensor_names_for_network,
    finalize_demo_plot_config,
    render_demo_tensor_network,
)

DESCRIPTION = """\
Quimb (quimb.tensor): ``TensorNetwork`` objects or raw tensor lists — \
same renderer as other engines.

Available network examples:
  - hyper      (single hub — star / multi-body index)
  - mps, mpo, peps, ladder (structured 1D / 2D / coupled chains)
  - weird, disconnected

Examples:
  python examples/quimb_demo.py mps 2d
  python examples/quimb_demo.py ladder 3d
  python examples/quimb_demo.py hyper 2d --from-list --save quimb.png --no-show
  python examples/quimb_demo.py peps 2d --hover-labels
  python examples/quimb_demo.py peps 2d --contraction-scheme
"""


def _make_tensor(name: str, inds: tuple[str, ...]) -> qtn.Tensor:
    shape = tuple(2 if "phys" in ind or "up" in ind or "down" in ind else 3 for ind in inds)
    return qtn.Tensor(data=np.ones(shape, dtype=float), inds=inds, tags={name})


def build_mps_example(length: int = 11) -> qtn.TensorNetwork:
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


def build_hyper_example() -> qtn.TensorNetwork:
    tensors = [
        _make_tensor("A", ("hub", "phys_a")),
        _make_tensor("B", ("hub", "phys_b")),
        _make_tensor("C", ("hub", "phys_c")),
    ]
    return qtn.TensorNetwork(tensors)


def build_mpo_example(length: int = 7) -> qtn.TensorNetwork:
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


def build_peps_example(rows: int = 4, cols: int = 5) -> qtn.TensorNetwork:
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


def build_ladder_example(length: int = 8) -> qtn.TensorNetwork:
    """Two MPS-like chains with shared rung indices (Quimb connects matching ``inds`` names)."""
    if length < 2:
        raise ValueError("ladder length must be >= 2")
    tensors: list[qtn.Tensor] = []
    for i in range(length):
        if i == 0:
            t_inds = (f"phys_t{i}", f"t_{i}_{i + 1}", f"rung_{i}")
            b_inds = (f"phys_b{i}", f"b_{i}_{i + 1}", f"rung_{i}")
        elif i == length - 1:
            t_inds = (f"t_{i - 1}_{i}", f"phys_t{i}", f"rung_{i}")
            b_inds = (f"b_{i - 1}_{i}", f"phys_b{i}", f"rung_{i}")
        else:
            t_inds = (f"t_{i - 1}_{i}", f"phys_t{i}", f"t_{i}_{i + 1}", f"rung_{i}")
            b_inds = (f"b_{i - 1}_{i}", f"phys_b{i}", f"b_{i}_{i + 1}", f"rung_{i}")
        tensors.append(_make_tensor(f"T{i}", t_inds))
        tensors.append(_make_tensor(f"B{i}", b_inds))
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
    "hyper": build_hyper_example,
    "ladder": build_ladder_example,
    "mps": build_mps_example,
    "mpo": build_mpo_example,
    "peps": build_peps_example,
    "weird": build_weird_example,
}


QUIMB_TAGLINES: dict[str, str] = {
    "hyper": "Star topology — one multi-body index shared by three tensors.",
    "mps": "1D tensor train / MPS — nearest-neighbor bonds from consistent index naming.",
    "mpo": "Operator string — up/down legs plus bond dimensions.",
    "peps": "2D PEPS — bulk horizontal/vertical indices auto-contract in the network.",
    "ladder": "Two coupled chains — rungs reuse the same index label on both rails.",
    "weird": "Irregular graph — layout falls back to force-directed placement.",
    "disconnected": "Multiple components — visual separation without manual coordinates.",
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
    add_hover_labels_argument(parser)
    add_contraction_scheme_argument(parser)
    add_compact_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    network = BUILDERS[args.network]()
    print(f"Building example Quimb tensor network: {args.network}")
    print(f"Selected visualization: {args.view}")
    print(f"Passing as: {'list of tensors' if args.from_list else 'TensorNetwork'}")
    print("Rendering figure...")

    show_input: qtn.TensorNetwork | list[qtn.Tensor]
    show_input = list(network.tensors) if args.from_list else network

    fig, ax = render_demo_tensor_network(
        show_input,
        args=args,
        engine="quimb",
        view=args.view,
        config=finalize_demo_plot_config(
            args,
            network=args.network,
            engine="quimb",
            scheme_tensor_names=demo_scheme_tensor_names_for_network(args.network),
        ),
    )
    apply_demo_caption(
        fig,
        title=f"Quimb · {args.network.upper()} · {args.view.upper()}",
        subtitle=QUIMB_TAGLINES.get(args.network),
        footer="Pass TensorNetwork or list[qtn.Tensor] — engine='quimb'",
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
