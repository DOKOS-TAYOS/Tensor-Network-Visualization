"""TeNPy explicit tensor networks: ``TenPyTensorNetwork`` + ``make_tenpy_tensor_network``.

TeNPy has no generic graph container in ``tenpy.networks``; this package type lists named
``npc.Array`` tensors and which legs share a bond. Requires only **TeNPy** (no Quimb / other viz
backends).

Install: ``pip install -e ".[tenpy]"``

Examples:
  python examples/tenpy_explicit_tn_demo.py chain 2d
  python examples/tenpy_explicit_tn_demo.py hub 3d --save explicit.png --no-show
  python examples/tenpy_explicit_tn_demo.py chain 2d --contraction-scheme
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

try:
    from tensor_network_viz import make_tenpy_tensor_network, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import make_tenpy_tensor_network, show_tensor_network

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from demo_cli import (
    add_compact_argument,
    add_contraction_scheme_argument,
    add_hover_labels_argument,
    apply_demo_caption,
    finalize_demo_plot_config,
)


def _npc_matrix(labels: tuple[str, str]) -> Any:
    from tenpy.linalg import np_conserved as npc

    leg = npc.LegCharge.from_trivial(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return npc.Array.from_ndarray(np.zeros((2, 2)), [leg, leg], labels=list(labels))


def build_chain_network() -> Any:
    """Open chain T0 -- T1 -- T2 (binary bonds only)."""
    t0 = _npc_matrix(("e", "m"))
    t1 = _npc_matrix(("m", "n"))
    t2 = _npc_matrix(("n", "f"))
    return make_tenpy_tensor_network(
        nodes=[("T0", t0), ("T1", t1), ("T2", t2)],
        bonds=[
            (("T0", "m"), ("T1", "m")),
            (("T1", "n"), ("T2", "n")),
        ],
    )


def build_hub_network() -> Any:
    """Three tensors meeting on one logical index (virtual hub in the graph)."""
    leg = ("j", "d")
    t0 = _npc_matrix(leg)
    t1 = _npc_matrix(leg)
    t2 = _npc_matrix(leg)
    return make_tenpy_tensor_network(
        nodes=[("A", t0), ("B", t1), ("C", t2)],
        bonds=[(("A", "j"), ("B", "j"), ("C", "j"))],
    )


BUILDERS: dict[str, Callable[[], Any]] = {
    "chain": build_chain_network,
    "hub": build_hub_network,
}

TAGLINES: dict[str, str] = {
    "chain": "Hand-made open chain: bonds (T0.m–T1.m) and (T1.n–T2.n); ends e,f dangling.",
    "hub": "Three-way bond on leg j; one virtual hub node; d legs dangle per tensor.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "network",
        choices=sorted(BUILDERS),
        help="Explicit TeNPy topology to build.",
    )
    parser.add_argument("view", choices=("2d", "3d"), help="Visualization mode.")
    parser.add_argument("--save", type=Path, help="Save the rendered figure to this path.")
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
    if args.no_show or args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    network = BUILDERS[args.network]()
    print(f"Building explicit TeNPy network: {args.network}")
    print(f"View: {args.view}")
    print("Rendering figure...")

    fig, ax = show_tensor_network(
        network,
        engine="tenpy",
        view=args.view,
        config=finalize_demo_plot_config(
            args, network=args.network, engine="tenpy_explicit"
        ),
        show=False,
    )
    apply_demo_caption(
        fig,
        title=f"TeNPy explicit · {args.network.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(args.network),
        footer="make_tenpy_tensor_network + engine='tenpy' (physics-tenpy only)",
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
