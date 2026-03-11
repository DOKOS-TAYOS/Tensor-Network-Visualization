from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import torch

try:
    from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network

DESCRIPTION = """\
Small demo for traced binary torch.einsum contractions.

It records sequential einsums with pair_tensor and renders the reconstructed
underlying tensor network.
Available examples:
  - disconnected
  - mps
  - peps

Examples:
  python examples/einsum_demo.py mps 2d
  python examples/einsum_demo.py peps 3d
  python examples/einsum_demo.py disconnected 3d
  python examples/einsum_demo.py mps 2d --save einsum.png --no-show
"""


def build_mps_trace() -> tuple[list[pair_tensor], tuple[int, ...]]:
    p_dim = 3
    a_dim = 2
    b_dim = 4

    a0 = torch.ones((p_dim, a_dim))
    x0 = torch.ones((p_dim,))
    a1 = torch.ones((a_dim, p_dim, b_dim))
    x1 = torch.ones((p_dim,))
    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
    ]

    r0 = torch.einsum(trace[0], a0, x0)
    r1 = torch.einsum(trace[1], r0, a1)
    result = torch.einsum(trace[2], r1, x1)
    return trace, tuple(result.shape)


def build_disconnected_trace() -> tuple[list[pair_tensor], tuple[tuple[int, ...], tuple[int, ...]]]:
    b_dim = 3
    d_dim = 2

    a = torch.ones((5, b_dim))
    x = torch.ones((b_dim,))
    b = torch.ones((7, d_dim))
    y = torch.ones((d_dim,))
    trace = [
        pair_tensor("A", "x", "r0", "ab,b->a"),
        pair_tensor("B", "y", "r1", "cd,d->c"),
    ]

    r0 = torch.einsum(trace[0], a, x)
    r1 = torch.einsum(trace[1], b, y)
    return trace, (tuple(r0.shape), tuple(r1.shape))


def build_peps_trace() -> tuple[list[pair_tensor], tuple[int, ...]]:
    bond_dim = 2
    phys_dim = 3

    p00 = torch.ones((phys_dim, bond_dim, bond_dim))
    x00 = torch.ones((phys_dim,))
    p01 = torch.ones((bond_dim, phys_dim, bond_dim, bond_dim))
    x01 = torch.ones((phys_dim,))
    p02 = torch.ones((bond_dim, phys_dim, bond_dim))
    x02 = torch.ones((phys_dim,))
    p10 = torch.ones((bond_dim, phys_dim, bond_dim))
    x10 = torch.ones((phys_dim,))
    p11 = torch.ones((bond_dim, bond_dim, phys_dim, bond_dim))
    x11 = torch.ones((phys_dim,))
    p12 = torch.ones((bond_dim, bond_dim, phys_dim))
    x12 = torch.ones((phys_dim,))
    trace = [
        pair_tensor("P00", "x00", "r0", "sad,s->ad"),
        pair_tensor("r0", "P01", "r1", "ad,atbe->dtbe"),
        pair_tensor("r1", "x01", "r2", "dtbe,t->dbe"),
        pair_tensor("r2", "P02", "r3", "dbe,buf->defu"),
        pair_tensor("r3", "x02", "r4", "defu,u->def"),
        pair_tensor("r4", "P10", "r5", "def,dvg->efvg"),
        pair_tensor("r5", "x10", "r6", "efvg,v->efg"),
        pair_tensor("r6", "P11", "r7", "efg,egwh->fwh"),
        pair_tensor("r7", "x11", "r8", "fwh,w->fh"),
        pair_tensor("r8", "P12", "r9", "fh,fhz->z"),
        pair_tensor("r9", "x12", "r10", "z,z->"),
    ]

    r0 = torch.einsum(trace[0], p00, x00)
    r1 = torch.einsum(trace[1], r0, p01)
    r2 = torch.einsum(trace[2], r1, x01)
    r3 = torch.einsum(trace[3], r2, p02)
    r4 = torch.einsum(trace[4], r3, x02)
    r5 = torch.einsum(trace[5], r4, p10)
    r6 = torch.einsum(trace[6], r5, x10)
    r7 = torch.einsum(trace[7], r6, p11)
    r8 = torch.einsum(trace[8], r7, x11)
    r9 = torch.einsum(trace[9], r8, p12)
    result = torch.einsum(trace[10], r9, x12)
    return trace, tuple(result.shape)


BUILDERS = {
    "disconnected": build_disconnected_trace,
    "mps": build_mps_trace,
    "peps": build_peps_trace,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "network",
        choices=sorted(BUILDERS),
        help="Example einsum trace to build.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace, result_shape = BUILDERS[args.network]()
    print(f"Building traced einsum example: {args.network}")
    print(f"Selected visualization: {args.view}")
    print(f"Final result shape: {result_shape}")
    print("Passing as: ordered list of pair_tensor entries")
    print("Rendering figure...")

    config = PlotConfig(
        figsize=(12, 8) if args.network == "peps" else (10, 6),
        show_tensor_labels=args.view == "2d",
        show_index_labels=args.view == "2d",
    )
    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
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
