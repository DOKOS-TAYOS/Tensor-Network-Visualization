from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib
import torch

try:
    from tensor_network_viz import EinsumTrace, einsum, pair_tensor, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import EinsumTrace, einsum, pair_tensor, show_tensor_network

TraceMode = Literal["auto", "manual"]
TraceInput = EinsumTrace | list[pair_tensor]

DESCRIPTION = """\
Small demo for traced binary torch.einsum contractions.

It can build the same tensor network through either:
  - EinsumTrace plus tensor_network_viz.einsum(...)
  - explicit pair_tensor entries plus torch.einsum(...)

Both modes render the reconstructed underlying tensor network.
Available examples:
  - disconnected
  - mps
  - peps

Examples:
  python examples/einsum_demo.py mps 2d
  python examples/einsum_demo.py mps 2d --mode manual
  python examples/einsum_demo.py peps 3d
  python examples/einsum_demo.py disconnected 3d
  python examples/einsum_demo.py mps 2d --save einsum.png --no-show
"""


def build_mps_trace(*, mode: TraceMode = "auto") -> tuple[TraceInput, tuple[int, ...]]:
    p_dim = 3
    a_dim = 2
    b_dim = 4

    a0 = torch.ones((p_dim, a_dim))
    x0 = torch.ones((p_dim,))
    a1 = torch.ones((a_dim, p_dim, b_dim))
    x1 = torch.ones((p_dim,))
    if mode == "auto":
        trace = EinsumTrace()
        trace.bind("A0", a0)
        trace.bind("x0", x0)
        trace.bind("A1", a1)
        trace.bind("x1", x1)

        r0 = einsum("pa,p->a", a0, x0, trace=trace, backend="torch")
        r1 = einsum("a,apb->pb", r0, a1, trace=trace, backend="torch")
        result = einsum("pb,p->b", r1, x1, trace=trace, backend="torch")
        return trace, tuple(result.shape)

    trace = [
        pair_tensor("A0", "x0", "r0", "pa,p->a"),
        pair_tensor("r0", "A1", "r1", "a,apb->pb"),
        pair_tensor("r1", "x1", "r2", "pb,p->b"),
    ]
    r0 = torch.einsum(trace[0], a0, x0)
    r1 = torch.einsum(trace[1], r0, a1)
    result = torch.einsum(trace[2], r1, x1)
    return trace, tuple(result.shape)


def build_disconnected_trace(
    *,
    mode: TraceMode = "auto",
) -> tuple[TraceInput, tuple[tuple[int, ...], tuple[int, ...]]]:
    b_dim = 3
    d_dim = 2

    a = torch.ones((5, b_dim))
    x = torch.ones((b_dim,))
    b = torch.ones((7, d_dim))
    y = torch.ones((d_dim,))
    if mode == "auto":
        trace = EinsumTrace()
        trace.bind("A", a)
        trace.bind("x", x)
        trace.bind("B", b)
        trace.bind("y", y)

        r0 = einsum("ab,b->a", a, x, trace=trace, backend="torch")
        r1 = einsum("cd,d->c", b, y, trace=trace, backend="torch")
        return trace, (tuple(r0.shape), tuple(r1.shape))

    trace = [
        pair_tensor("A", "x", "r0", "ab,b->a"),
        pair_tensor("B", "y", "r1", "cd,d->c"),
    ]
    r0 = torch.einsum(trace[0], a, x)
    r1 = torch.einsum(trace[1], b, y)
    return trace, (tuple(r0.shape), tuple(r1.shape))


def build_peps_trace(*, mode: TraceMode = "auto") -> tuple[TraceInput, tuple[int, ...]]:
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
    if mode == "auto":
        trace = EinsumTrace()
        for name, tensor in (
            ("P00", p00),
            ("x00", x00),
            ("P01", p01),
            ("x01", x01),
            ("P02", p02),
            ("x02", x02),
            ("P10", p10),
            ("x10", x10),
            ("P11", p11),
            ("x11", x11),
            ("P12", p12),
            ("x12", x12),
        ):
            trace.bind(name, tensor)

        r0 = einsum("sad,s->ad", p00, x00, trace=trace, backend="torch")
        r1 = einsum("ad,atbe->dtbe", r0, p01, trace=trace, backend="torch")
        r2 = einsum("dtbe,t->dbe", r1, x01, trace=trace, backend="torch")
        r3 = einsum("dbe,buf->defu", r2, p02, trace=trace, backend="torch")
        r4 = einsum("defu,u->def", r3, x02, trace=trace, backend="torch")
        r5 = einsum("def,dvg->efvg", r4, p10, trace=trace, backend="torch")
        r6 = einsum("efvg,v->efg", r5, x10, trace=trace, backend="torch")
        r7 = einsum("efg,egwh->fwh", r6, p11, trace=trace, backend="torch")
        r8 = einsum("fwh,w->fh", r7, x11, trace=trace, backend="torch")
        r9 = einsum("fh,fhz->z", r8, p12, trace=trace, backend="torch")
        result = einsum("z,z->", r9, x12, trace=trace, backend="torch")
        return trace, tuple(result.shape)

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
        "--mode",
        choices=("auto", "manual"),
        default="auto",
        help=(
            "Trace construction mode: auto uses tv.einsum, manual uses pair_tensor + torch.einsum."
        ),
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

    trace, result_shape = BUILDERS[args.network](mode=args.mode)
    print(f"Building traced einsum example: {args.network}")
    print(f"Trace mode: {args.mode}")
    print(f"Selected visualization: {args.view}")
    print(f"Final result shape: {result_shape}")
    if args.mode == "auto":
        print("Passing as: EinsumTrace with auto-recorded pair_tensor entries")
    else:
        print("Passing as: ordered list of pair_tensor entries")
    print("Rendering figure...")

    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        view=args.view,
        show=False,
    )
    fig.suptitle(f"{args.network.upper()} ({args.view.upper()}, {args.mode.upper()})", fontsize=16)
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
