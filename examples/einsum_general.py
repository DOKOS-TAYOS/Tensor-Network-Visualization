from __future__ import annotations

"""Significant einsum patterns beyond the minimal MVP (ellipsis, batch bonds, traces).

How this is drawn (implementation sketch, all logic lives in ``tensor_network_viz.einsum_module``):

1. **Equation normalization** (``_equation.py``): the trace string is validated with NumPy
   ``einsum`` on zero arrays matching operand ranks. Ellipsis ``...`` is expanded to concrete
   single-character labels shared across operands and the output.
2. **Fundamental graph** (``graph.py``): an ordinary **pairwise** contraction (exactly one leg per
   operand, index not in the output) is drawn as a single bond between the two tensors. A **virtual
   hub** is used when the index appears **more than twice** on the left, when **both** legs lie on
   the same tensor (trace / diagonal), or when the index is **kept in the output** so several legs
   merge at a hyperedge (batch / ``ab,ab->ab``-style), matching the Quimb hub pattern.
3. **Open indices preserved in the output**: if such an index also appears in the output, the hub
   gains an extra axis; the logical result axis is tracked on that hub so the final ``dangling``
   legs match the surviving tensor-network bond structure without duplicating stubs.

**Automatic trace (recommended):** use ``EinsumTrace`` + ``tensor_network_viz.einsum`` exactly like
``einsum_demo.py``. Each committed step stores ``left_shape`` / ``right_shape`` in
``pair_tensor.metadata``, which the graph builder needs to expand ``...`` when visualizing.

**Manual ``pair_tensor`` lists:** for equations with ellipsis you must supply
``metadata=dict(left_shape=..., right_shape=...)`` on each entry; otherwise the ranks cannot be
resolved.

Examples::

  python examples/einsum_general.py ellipsis 2d
  python examples/einsum_general.py batch 3d
  python examples/einsum_general.py nway 3d
  python examples/einsum_general.py trace 2d
  python examples/einsum_general.py mps_short 2d --save einsum_general.png --no-show
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib
import torch

try:
    from tensor_network_viz import EinsumTrace, einsum, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import EinsumTrace, einsum, show_tensor_network

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from demo_cli import (
    add_compact_argument,
    add_hover_labels_argument,
    apply_demo_caption,
    demo_plot_config,
)

ExampleName = Literal["ellipsis", "batch", "nway", "trace", "mps_short"]


def build_ellipsis_example() -> EinsumTrace:
    """Batched matmul ``...ij,...jk->...ik`` (two shared batch axes here)."""
    trace = EinsumTrace()
    a = torch.ones((2, 3, 4))
    b = torch.ones((2, 4, 5))
    trace.bind("A", a)
    trace.bind("B", b)
    _ = einsum("...ij,...jk->...ik", a, b, trace=trace, backend="torch")
    return trace


def build_batch_example() -> EinsumTrace:
    """Hadamard-style batch ``ab,ab->ab`` (hyperedges on *a* and *b*)."""
    trace = EinsumTrace()
    a = torch.ones((3, 4))
    b = torch.ones((3, 4))
    trace.bind("U", a)
    trace.bind("V", b)
    _ = einsum("ab,ab->ab", a, b, trace=trace, backend="torch")
    return trace


def build_nway_example() -> EinsumTrace:
    """Three tensors merged via two traced binary einsums (pairwise tracing only)."""
    trace = EinsumTrace()
    t = torch.ones((3, 4, 5))
    u = torch.ones((3, 4, 6))
    v = torch.ones((5, 6, 7))
    trace.bind("T", t)
    trace.bind("U", u)
    trace.bind("V", v)
    r1 = einsum("abc,abd->cd", t, u, trace=trace, backend="torch")
    _ = einsum("cd,cde->e", r1, v, trace=trace, backend="torch")
    return trace


def build_trace_example() -> EinsumTrace:
    """Diagonal / restrict with a vector ``ii,i->i``."""
    trace = EinsumTrace()
    m = torch.ones((4, 4))
    v = torch.ones((4,))
    trace.bind("M", m)
    trace.bind("x", v)
    _ = einsum("ii,i->i", m, v, trace=trace, backend="torch")
    return trace


def build_mps_short_example() -> EinsumTrace:
    """Two-site MPS-like contractions (same equations as the small MPS demo)."""
    trace = EinsumTrace()
    p_dim, a_dim, b_dim = 3, 2, 4
    a0 = torch.ones((p_dim, a_dim))
    x0 = torch.ones((p_dim,))
    a1 = torch.ones((a_dim, p_dim, b_dim))
    x1 = torch.ones((p_dim,))
    trace.bind("A0", a0)
    trace.bind("x0", x0)
    trace.bind("A1", a1)
    trace.bind("x1", x1)
    r0 = einsum("pa,p->a", a0, x0, trace=trace, backend="torch")
    _ = einsum("a,apb->pb", r0, a1, trace=trace, backend="torch")
    return trace


BUILDERS: dict[ExampleName, Callable[[], EinsumTrace]] = {
    "ellipsis": build_ellipsis_example,
    "batch": build_batch_example,
    "nway": build_nway_example,
    "trace": build_trace_example,
    "mps_short": build_mps_short_example,
}


GENERAL_TAGLINES: dict[str, str] = {
    "ellipsis": "Ellipsis + batch matmul — internal ranks expanded from tensor shapes.",
    "batch": "Elementwise / broadcasting hub — kept indices merge at hyperedges.",
    "nway": (
        "Two-step fusion of three operands — shared batch legs in step one, "
        "contraction into a vector in step two."
    ),
    "trace": "Diagonal-style selective trace — same-tensor legs and vector masking.",
    "mps_short": "Two-site MPS matvec — minimal chain showing traced intermediates.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Examples for generalized traced einsum visualizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "example",
        choices=sorted(BUILDERS),
        help="Which pattern to build and render.",
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
    add_compact_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = BUILDERS[args.example]()
    print(f"einsum_general: example={args.example!r}, view={args.view!r}, len(trace)={len(trace)}")

    fig, ax = show_tensor_network(
        trace,
        engine="einsum",
        view=args.view,
        config=demo_plot_config(args),
        show=False,
    )
    apply_demo_caption(
        fig,
        title=f"General einsum · {args.example} · {args.view.upper()}",
        subtitle=GENERAL_TAGLINES.get(args.example),
        footer="Virtual hubs encode traces, batches, and multi-use edges — see module docstring",
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
