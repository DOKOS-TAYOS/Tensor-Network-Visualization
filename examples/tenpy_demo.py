from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

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
TeNPy: visualize real ``MPS`` / ``MPO`` objects (finite and infinite BC) \
with the same Matplotlib path.

Available network examples:
  - mps, mpo (finite TFI-chain Hamiltonian / product states)
  - imps, impo (infinite MPS/MPO from TeNPy’s constructors)

Examples:
  python examples/tenpy_demo.py mps 2d
  python examples/tenpy_demo.py mpo 3d
  python examples/tenpy_demo.py imps 2d --save tenpy-imps.png --no-show
  python examples/tenpy_demo.py mps 2d --hover-labels
"""


def build_mps_example(length: int = 11):
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(length)]
    states = ["up" if index % 2 == 0 else "down" for index in range(length)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        return MPS.from_product_state(sites, states, bc="finite")


def build_mpo_example(length: int = 8):
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": length, "J": 1.0, "g": 1.0, "bc_MPS": "finite"})
    return model.calc_H_MPO()


def build_imps_example(length: int = 3):
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(length)]
    states = ["up" if index % 2 == 0 else "down" for index in range(length)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        return MPS.from_product_state(sites, states, bc="infinite")


def build_impo_example(length: int = 3):
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": length, "J": 1.0, "g": 1.0, "bc_MPS": "infinite"})
    return model.calc_H_MPO()


BUILDERS = {
    "impo": build_impo_example,
    "imps": build_imps_example,
    "mps": build_mps_example,
    "mpo": build_mpo_example,
}

TENPY_TAGLINES: dict[str, str] = {
    "mps": "Finite MPS from product state — sites, legs, and charges mapped into one graph.",
    "mpo": "Exact H_MPO for TFI chain — dense-ish bond structure made readable by layout.",
    "imps": "Infinite MPS (iMPS) unit cell — periodic boundaries as a compact ring-like motif.",
    "impo": "Infinite MPO Hamiltonian — same API as finite, different boundary semantics.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "network",
        choices=sorted(BUILDERS),
        help="Example TeNPy network to build.",
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

    network = BUILDERS[args.network]()
    print(f"Building example TeNPy network: {args.network}")
    print(f"Selected visualization: {args.view}")
    print("Rendering figure...")

    fig, ax = show_tensor_network(
        network,
        engine="tenpy",
        view=args.view,
        config=demo_plot_config(args),
        show=False,
    )
    apply_demo_caption(
        fig,
        title=f"TeNPy · {args.network.upper()} · {args.view.upper()}",
        subtitle=TENPY_TAGLINES.get(args.network),
        footer="engine='tenpy' — drop-in diagrams for DMRG / TEBD workflows",
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
