from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tensornetwork as tn

try:
    from tensor_network_viz import PlotConfig, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import PlotConfig, show_tensor_network

DESCRIPTION = """\
Large topology demo: a binary MERA stack merged at the top into a hierarchical binary tensor tree
(TTN). Useful to stress-test layout and rendering on deep, wide graphs.

The physical chain has 2^L sites (default L=4 → 16). On top of the MERA apex, a full binary tree
with depth D (default 4 → 16 open legs at the leaves) is attached.

Examples:
  python examples/mera_tree_demo.py 2d
  python examples/mera_tree_demo.py 3d --mera-log2 5 --tree-depth 4
  python examples/mera_tree_demo.py 2d --save mera_tree.png --no-show
"""


def _bond_dim(axis_name: str) -> int:
    if axis_name == "phys" or "phys" in axis_name:
        return 2
    return 3


def _make_node(name: str, axis_names: tuple[str, ...]) -> tn.Node:
    shape = tuple(_bond_dim(ax) for ax in axis_names)
    return tn.Node(np.ones(shape, dtype=float), name=name, axis_names=axis_names)


def build_binary_mera(num_sites_log2: int) -> tuple[list[tn.Node], tn.Node, str]:
    """Binary MERA: sites, then alternating disentangler / isometry layers.

    Returns all nodes, the top isometry node, and the name of its open upward axis.
    """
    n = 2**num_sites_log2
    if n < 2:
        raise ValueError("num_sites_log2 must be >= 1")

    nodes: list[tn.Node] = []
    sites: list[tn.Node] = []
    for j in range(n):
        s_j = _make_node(f"S{j}", ("phys", "virt"))
        sites.append(s_j)
        nodes.append(s_j)

    layer = 0
    current = sites

    while len(current) > 1:
        next_isos: list[tn.Node] = []
        for k in range(len(current) // 2):
            d_k = _make_node(f"D{layer}_{k}", ("inL", "inR", "outL", "outR"))
            u_k = _make_node(f"U{layer}_{k}", ("legL", "legR", "virt"))
            nodes.extend((d_k, u_k))

            current[2 * k]["virt"] ^ d_k["inL"]
            current[2 * k + 1]["virt"] ^ d_k["inR"]
            d_k["outL"] ^ u_k["legL"]
            d_k["outR"] ^ u_k["legR"]
            next_isos.append(u_k)

        current = next_isos
        layer += 1

    return nodes, current[0], "virt"


def build_tree_below(
    apex_node: tn.Node,
    apex_axis: str,
    depth: int,
) -> list[tn.Node]:
    """Full binary tensor tree (TTN) rooted at ``apex_node[apex_axis]``."""

    if depth < 1:
        return []

    root = _make_node("TRoot", ("top", "to_L", "to_R"))
    apex_node[apex_axis] ^ root["top"]
    out: list[tn.Node] = [root]
    out.extend(_ttn_branch(root, "to_L", depth - 1, "L"))
    out.extend(_ttn_branch(root, "to_R", depth - 1, "R"))
    return out


def _ttn_branch(
    parent: tn.Node,
    parent_axis: str,
    rem_depth: int,
    path: str,
) -> list[tn.Node]:
    if rem_depth == 0:
        leaf = _make_node(f"TL{path}", ("from_parent", "phys"))
        parent[parent_axis] ^ leaf["from_parent"]
        return [leaf]

    node = _make_node(f"TI{path}", ("from_parent", "to_L", "to_R"))
    parent[parent_axis] ^ node["from_parent"]
    acc: list[tn.Node] = [node]
    acc.extend(_ttn_branch(node, "to_L", rem_depth - 1, path + "L"))
    acc.extend(_ttn_branch(node, "to_R", rem_depth - 1, path + "R"))
    return acc


def build_mera_tree_network(
    *,
    mera_log2: int,
    tree_depth: int,
) -> list[tn.Node]:
    mera_nodes, apex, apex_axis = build_binary_mera(mera_log2)
    tree_nodes = build_tree_below(apex, apex_axis, tree_depth)
    return mera_nodes + tree_nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "view",
        choices=("2d", "3d"),
        help="Visualization mode.",
    )
    parser.add_argument(
        "--mera-log2",
        type=int,
        default=4,
        help="Binary MERA width: number of physical sites is 2**this (default: 4 → 16 sites).",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Depth of the TTN above the MERA apex (default: 4 → 16 leaf tensors).",
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

    nodes = build_mera_tree_network(mera_log2=args.mera_log2, tree_depth=args.tree_depth)
    n_phys = 2**args.mera_log2
    n_leaf = 2**args.tree_depth if args.tree_depth > 0 else 0
    print("Building MERA + hierarchical tensor tree")
    print(f"  MERA: 2**{args.mera_log2} = {n_phys} physical sites")
    print(f"  TTN depth: {args.tree_depth} (~{n_leaf} leaf tensors)")
    print(f"  Total TensorNetwork nodes: {len(nodes)}")
    print(f"  View: {args.view}")
    print("Rendering figure...")

    config = PlotConfig(
        figsize=(18, 12),
        show_tensor_labels=args.view == "2d",
        show_index_labels=args.view == "2d",
    )
    fig, _ax = show_tensor_network(
        nodes,
        engine="tensornetwork",
        view=args.view,
        config=config,
        show=False,
    )
    fig.suptitle(
        f"MERA (2^{args.mera_log2} sites) + TTN (depth {args.tree_depth}) — {args.view.upper()}",
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
