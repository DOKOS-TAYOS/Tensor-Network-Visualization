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
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import show_tensor_network

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from demo_cli import add_hover_labels_argument, demo_plot_config

DESCRIPTION = """\
3D tensor network topology: cubic PEPS (a PEPS on an Lx * Ly * Lz grid). Each bulk tensor has a
physical leg plus up to six face bonds to neighbors. Intended to be viewed in 3D so the graph can
unfold in depth; all node positions come from the library layout (no manual coordinates).

If you pass `2d`, the same graph is shown in the plane (useful for comparisons).

Examples:
  python examples/cubic_peps_demo.py
  python examples/cubic_peps_demo.py 3d --lx 3 --ly 3 --lz 4
  python examples/cubic_peps_demo.py 2d --save cubic_peps.png --no-show
  python examples/cubic_peps_demo.py 2d --hover-labels
"""


def _bond_dim(axis_name: str) -> int:
    if axis_name == "phys" or "phys" in axis_name:
        return 2
    return 3


def _make_node(name: str, axis_names: tuple[str, ...]) -> tn.Node:
    shape = tuple(_bond_dim(ax) for ax in axis_names)
    return tn.Node(np.ones(shape, dtype=float), name=name, axis_names=axis_names)


def build_cubic_peps(lx: int, ly: int, lz: int) -> list[tn.Node]:
    """Cubic PEPS: one tensor per grid site, nearest-neighbor face bonds only."""
    if min(lx, ly, lz) < 1:
        raise ValueError("lx, ly, lz must be >= 1")

    keys: list[tuple[int, int, int]] = [
        (i, j, k) for i in range(lx) for j in range(ly) for k in range(lz)
    ]
    axes_by_key: dict[tuple[int, int, int], list[str]] = {}

    for i, j, k in keys:
        axes: list[str] = ["phys"]
        if i > 0:
            axes.append("xm")
        if i < lx - 1:
            axes.append("xp")
        if j > 0:
            axes.append("ym")
        if j < ly - 1:
            axes.append("yp")
        if k > 0:
            axes.append("zm")
        if k < lz - 1:
            axes.append("zp")
        axes_by_key[(i, j, k)] = axes

    nodes: dict[tuple[int, int, int], tn.Node] = {}
    for i, j, k in keys:
        name = f"P{i}_{j}_{k}"
        nodes[(i, j, k)] = _make_node(name, tuple(axes_by_key[(i, j, k)]))

    for i, j, k in keys:
        n = nodes[(i, j, k)]
        if i < lx - 1:
            n["xp"] ^ nodes[(i + 1, j, k)]["xm"]
        if j < ly - 1:
            n["yp"] ^ nodes[(i, j + 1, k)]["ym"]
        if k < lz - 1:
            n["zp"] ^ nodes[(i, j, k + 1)]["zm"]

    return [nodes[key] for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "view",
        nargs="?",
        default="3d",
        choices=("2d", "3d"),
        help="Visualization mode (default: 3d).",
    )
    parser.add_argument(
        "--lx",
        type=int,
        default=3,
        help="Grid extent in x (default: 3).",
    )
    parser.add_argument(
        "--ly",
        type=int,
        default=3,
        help="Grid extent in y (default: 3).",
    )
    parser.add_argument(
        "--lz",
        type=int,
        default=4,
        help="Grid extent in z (default: 4).",
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

    nodes = build_cubic_peps(args.lx, args.ly, args.lz)
    n_sites = args.lx * args.ly * args.lz
    print("Building cubic PEPS (3D grid PEPS; layout is automatic, no manual coordinates).")
    print(f"  Grid: {args.lx} x {args.ly} x {args.lz} = {n_sites} tensors")
    print(f"  View: {args.view}")
    print("Rendering figure...")

    fig, _ax = show_tensor_network(
        nodes,
        engine="tensornetwork",
        view=args.view,
        config=demo_plot_config(args),
        show=False,
    )
    fig.suptitle(
        f"Cubic PEPS {args.lx}x{args.ly}x{args.lz} ({args.view.upper()})",
        fontsize=14,
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
