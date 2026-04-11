from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from demo_cli import (
    BuiltExample,
    ExampleCliArgs,
    ExampleDefinition,
    apply_demo_caption,
    axis_dimension,
    cumulative_prefix_contraction_scheme,
    demo_runs_headless,
    ensure_minimum,
    finalize_demo_plot_config,
    pairwise_merge_contraction_scheme,
    render_demo_tensor_network,
    resolve_example_definition,
)
from demo_tensors import build_demo_torch_tensor

NodeSpec: TypeAlias = tuple[str, tuple[str, ...]]
BondSpec: TypeAlias = tuple[tuple[str, str], ...]

TAGLINES: dict[str, str] = {
    "cubic_peps": "Cubic lattice created directly with TensorKrowch nodes.",
    "disconnected": "Several disconnected components in one TensorNetwork.",
    "ladder": "Coupled chains with rung bonds.",
    "mera": "Binary MERA hierarchy assembled node by node.",
    "mera_ttn": "Binary MERA connected to a tree network.",
    "mps": "Finite tensor-train / MPS chain.",
    "mpo": "Finite MPO chain.",
    "peps": "2D PEPS grid.",
    "weird": "Irregular topology for layout fallback.",
}
_SMALL_CONTRACTED_EXAMPLES = frozenset({"mps", "mpo"})


def _build_tensorkrowch_network(
    node_specs: tuple[NodeSpec, ...],
    bond_specs: tuple[BondSpec, ...],
) -> tuple[Any, list[Any]]:
    import tensorkrowch as tk
    import torch

    network = tk.TensorNetwork(name="demo")
    nodes: dict[str, Any] = {}
    ordered_nodes: list[Any] = []
    for name, axes in node_specs:
        shape = tuple(axis_dimension(axis) for axis in axes)
        node = tk.Node(
            tensor=build_demo_torch_tensor(name=name, shape=shape, dtype=torch.float32),
            axes_names=axes,
            name=name,
            network=network,
        )
        nodes[name] = node
        ordered_nodes.append(node)

    for bond in bond_specs:
        if len(bond) != 2:
            raise ValueError("TensorKrowch examples only support pairwise bonds.")
        (left_name, left_axis), (right_name, right_axis) = bond
        nodes[left_name][left_axis] ^ nodes[right_name][right_axis]
    return network, ordered_nodes


def _mps_specs(n_sites: int) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("n_sites", n_sites)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    for index in range(n_sites):
        name = f"A{index}"
        axes: list[str] = []
        if index > 0:
            axes.append("left")
        axes.append("phys")
        if index < n_sites - 1:
            axes.append("right")
        node_specs.append((name, tuple(axes)))
        if index > 0:
            bond_specs.append(((f"A{index - 1}", "right"), (name, "left")))
    return tuple(node_specs), tuple(bond_specs)


def _mpo_specs(n_sites: int) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("n_sites", n_sites)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    for index in range(n_sites):
        name = f"W{index}"
        axes: list[str] = []
        if index > 0:
            axes.append("left")
        axes.extend(("down", "up"))
        if index < n_sites - 1:
            axes.append("right")
        node_specs.append((name, tuple(axes)))
        if index > 0:
            bond_specs.append(((f"W{index - 1}", "right"), (name, "left")))
    return tuple(node_specs), tuple(bond_specs)


def _ladder_specs(n_sites: int) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("n_sites", n_sites, minimum=2)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    for index in range(n_sites):
        top = f"T{index}"
        bottom = f"B{index}"
        top_axes: list[str] = []
        bottom_axes: list[str] = []
        if index > 0:
            top_axes.append("left")
            bottom_axes.append("left")
        top_axes.extend(("phys", "down"))
        bottom_axes.extend(("phys", "up"))
        if index < n_sites - 1:
            top_axes.append("right")
            bottom_axes.append("right")
        node_specs.append((top, tuple(top_axes)))
        node_specs.append((bottom, tuple(bottom_axes)))
        bond_specs.append(((top, "down"), (bottom, "up")))
        if index > 0:
            bond_specs.append(((f"T{index - 1}", "right"), (top, "left")))
            bond_specs.append(((f"B{index - 1}", "right"), (bottom, "left")))
    return tuple(node_specs), tuple(bond_specs)


def _peps_specs(lx: int, ly: int) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("lx", lx)
    ensure_minimum("ly", ly)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    for i in range(lx):
        for j in range(ly):
            name = f"P{i}_{j}"
            axes: list[str] = []
            if i > 0:
                axes.append("up")
            if j > 0:
                axes.append("left")
            axes.append("phys")
            if j < ly - 1:
                axes.append("right")
            if i < lx - 1:
                axes.append("down")
            node_specs.append((name, tuple(axes)))
            if i > 0:
                bond_specs.append(((f"P{i - 1}_{j}", "down"), (name, "up")))
            if j > 0:
                bond_specs.append(((f"P{i}_{j - 1}", "right"), (name, "left")))
    return tuple(node_specs), tuple(bond_specs)


def _cubic_peps_specs(
    lx: int,
    ly: int,
    lz: int,
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("lx", lx)
    ensure_minimum("ly", ly)
    ensure_minimum("lz", lz)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                name = f"P{i}_{j}_{k}"
                axes = ["phys"]
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
                node_specs.append((name, tuple(axes)))
                if i > 0:
                    bond_specs.append(((f"P{i - 1}_{j}_{k}", "xp"), (name, "xm")))
                if j > 0:
                    bond_specs.append(((f"P{i}_{j - 1}_{k}", "yp"), (name, "ym")))
                if k > 0:
                    bond_specs.append(((f"P{i}_{j}_{k - 1}", "zp"), (name, "zm")))
    return tuple(node_specs), tuple(bond_specs)


def _pairwise_specs(
    node_names: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    axes_by_node: dict[str, list[str]] = {name: ["phys"] for name in node_names}
    for edge_index, (left, right) in enumerate(edges):
        axis_name = f"b{edge_index}"
        axes_by_node[left].append(axis_name)
        axes_by_node[right].append(axis_name)

    node_specs = tuple((name, tuple(axes_by_node[name])) for name in node_names)
    bond_specs = tuple(
        ((left, f"b{edge_index}"), (right, f"b{edge_index}"))
        for edge_index, (left, right) in enumerate(edges)
    )
    return node_specs, bond_specs


def _weird_specs() -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    node_names = tuple(f"V{index}" for index in range(23))
    edges = (
        ("V0", "V1"),
        ("V1", "V2"),
        ("V2", "V5"),
        ("V5", "V8"),
        ("V8", "V7"),
        ("V7", "V6"),
        ("V6", "V3"),
        ("V3", "V0"),
        ("V3", "V4"),
        ("V4", "V5"),
        ("V1", "V4"),
        ("V4", "V7"),
        ("V0", "V4"),
        ("V2", "V4"),
        ("V6", "V4"),
        ("V8", "V4"),
        ("V1", "V9"),
        ("V9", "V10"),
        ("V10", "V11"),
        ("V11", "V5"),
        ("V3", "V12"),
        ("V12", "V13"),
        ("V13", "V14"),
        ("V14", "V7"),
        ("V0", "V15"),
        ("V15", "V16"),
        ("V16", "V8"),
        ("V10", "V17"),
        ("V17", "V18"),
        ("V18", "V19"),
        ("V19", "V13"),
        ("V10", "V13"),
        ("V11", "V14"),
        ("V16", "V19"),
        ("V2", "V20"),
        ("V18", "V21"),
        ("V12", "V22"),
    )
    return _pairwise_specs(node_names, edges)


def _disconnected_specs() -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    chain = tuple(f"A{index}" for index in range(8))
    ring = tuple(f"B{index}" for index in range(6))
    grid = tuple(f"C{i}_{j}" for i in range(3) for j in range(3))
    star = tuple(f"D{index}" for index in range(6))
    node_names = (*chain, *ring, *grid, *star)
    edges: list[tuple[str, str]] = []
    edges.extend((f"A{index}", f"A{index + 1}") for index in range(7))
    edges.extend((f"B{index}", f"B{(index + 1) % 6}") for index in range(6))
    edges.extend((f"B{index}", f"B{(index + 3) % 6}") for index in range(3))
    for i in range(3):
        for j in range(3):
            if i < 2:
                edges.append((f"C{i}_{j}", f"C{i + 1}_{j}"))
            if j < 2:
                edges.append((f"C{i}_{j}", f"C{i}_{j + 1}"))
    edges.extend(("D0", f"D{index}") for index in range(1, 6))
    edges.extend((f"D{index}", f"D{index + 1}") for index in range(1, 5))
    return _pairwise_specs(node_names, tuple(edges))


def _mera_specs(mera_log2: int) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("mera_log2", mera_log2)
    node_specs: list[NodeSpec] = []
    bond_specs: list[BondSpec] = []
    current_names: list[str] = []
    for index in range(2**mera_log2):
        name = f"S{index}"
        node_specs.append((name, ("phys", "virt")))
        current_names.append(name)

    layer = 0
    while len(current_names) > 1:
        next_names: list[str] = []
        for index in range(0, len(current_names), 2):
            left = current_names[index]
            right = current_names[index + 1]
            disentangler = f"D{layer}_{index // 2}"
            isometry = f"U{layer}_{index // 2}"
            node_specs.append((disentangler, ("inL", "inR", "outL", "outR")))
            node_specs.append((isometry, ("legL", "legR", "virt")))
            bond_specs.append(((left, "virt"), (disentangler, "inL")))
            bond_specs.append(((right, "virt"), (disentangler, "inR")))
            bond_specs.append(((disentangler, "outL"), (isometry, "legL")))
            bond_specs.append(((disentangler, "outR"), (isometry, "legR")))
            next_names.append(isometry)
        current_names = next_names
        layer += 1
    return tuple(node_specs), tuple(bond_specs)


def _mera_ttn_specs(
    mera_log2: int,
    tree_depth: int,
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    ensure_minimum("tree_depth", tree_depth)
    node_specs, bond_specs = _mera_specs(mera_log2)
    nodes = list(node_specs)
    bonds = list(bond_specs)
    apex = nodes[-1][0]
    root_name = "TRoot"
    nodes.append((root_name, ("top", "to_L", "to_R")))
    bonds.append(((apex, "virt"), (root_name, "top")))

    def add_branch(parent: str, parent_axis: str, depth: int, label: str) -> None:
        if depth == 0:
            leaf_name = f"TL{label}"
            nodes.append((leaf_name, ("from_parent", "phys")))
            bonds.append(((parent, parent_axis), (leaf_name, "from_parent")))
            return
        node_name = f"TI{label}"
        nodes.append((node_name, ("from_parent", "to_L", "to_R")))
        bonds.append(((parent, parent_axis), (node_name, "from_parent")))
        add_branch(node_name, "to_L", depth - 1, label + "L")
        add_branch(node_name, "to_R", depth - 1, label + "R")

    add_branch(root_name, "to_L", tree_depth - 1, "L")
    add_branch(root_name, "to_R", tree_depth - 1, "R")
    return tuple(nodes), tuple(bonds)


def _specs_for_example(
    example: str,
    args: ExampleCliArgs,
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    if example == "mps":
        return _mps_specs(args.n_sites)
    if example == "mpo":
        return _mpo_specs(args.n_sites)
    if example == "ladder":
        return _ladder_specs(args.n_sites)
    if example == "peps":
        return _peps_specs(args.lx, args.ly)
    if example == "cubic_peps":
        return _cubic_peps_specs(args.lx, args.ly, args.lz)
    if example == "mera":
        return _mera_specs(args.mera_log2)
    if example == "mera_ttn":
        return _mera_ttn_specs(args.mera_log2, args.tree_depth)
    if example == "weird":
        return _weird_specs()
    if example == "disconnected":
        return _disconnected_specs()
    raise ValueError(f"Unsupported TensorKrowch example: {example}")


def _contract_small_chain(nodes: list[Any]) -> Any:
    if len(nodes) < 2:
        raise ValueError("Small contracted demos need at least two tensors.")
    active_nodes = list(nodes)
    while len(active_nodes) > 1:
        next_nodes: list[Any] = []
        index = 0
        while index < len(active_nodes):
            left_node = active_nodes[index]
            if index + 1 >= len(active_nodes):
                next_nodes.append(left_node)
                index += 1
                continue
            right_node = active_nodes[index + 1]
            next_nodes.append(left_node @ right_node)
            index += 2
        active_nodes = next_nodes
    return active_nodes[0]


def _scheme_steps(
    example: str,
    tensor_names: tuple[str, ...],
) -> tuple[tuple[str, ...], ...] | None:
    if example in {"mps", "mpo"}:
        return pairwise_merge_contraction_scheme(tensor_names)
    if example in {"ladder", "peps", "cubic_peps"}:
        return cumulative_prefix_contraction_scheme(tensor_names)
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    node_specs, bond_specs = _specs_for_example(definition.name, args)
    tensor_names = tuple(name for name, _axes in node_specs)
    use_auto_contracted_demo = args.contracted and definition.name in _SMALL_CONTRACTED_EXAMPLES
    network, nodes = _build_tensorkrowch_network(node_specs, bond_specs)
    if use_auto_contracted_demo:
        _contract_small_chain(nodes)
        network_input: Any = network
        footer = (
            "Small native TensorKrowch demo contracted in advance to expose "
            "auto-recovered contraction history."
        )
        scheme_steps_by_name = None
    else:
        network_input = nodes if args.from_list else network
        footer = "Render the native TensorNetwork or a list of TensorKrowch nodes."
        scheme_steps_by_name = _scheme_steps(definition.name, tensor_names)
    return BuiltExample(
        network=network_input,
        plot_engine="tensorkrowch",
        title=f"TensorKrowch - {definition.name.upper()} - {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer=footer,
        scheme_steps_by_name=scheme_steps_by_name,
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        name="mps",
        aliases=("tt",),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Finite MPS / tensor-train chain.",
    ),
    ExampleDefinition(
        name="mpo",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Finite MPO chain.",
    ),
    ExampleDefinition(
        name="ladder",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Two coupled chains.",
    ),
    ExampleDefinition(
        name="peps",
        aliases=(),
        size_knobs=frozenset({"lx", "ly"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="2D PEPS grid.",
    ),
    ExampleDefinition(
        name="cubic_peps",
        aliases=(),
        size_knobs=frozenset({"lx", "ly", "lz"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="3D PEPS lattice.",
    ),
    ExampleDefinition(
        name="mera",
        aliases=(),
        size_knobs=frozenset({"mera_log2"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Binary MERA hierarchy.",
    ),
    ExampleDefinition(
        name="mera_ttn",
        aliases=(),
        size_knobs=frozenset({"mera_log2", "tree_depth"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Binary MERA connected to a TTN.",
    ),
    ExampleDefinition(
        name="weird",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Irregular non-grid network.",
    ),
    ExampleDefinition(
        name="disconnected",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Multiple disconnected components.",
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported TensorKrowch example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args,
        engine="tensorkrowch",
        scheme_tensor_names=built.scheme_steps_by_name,
    )
    fig, _ax = render_demo_tensor_network(
        built.network,
        args=args,
        engine="tensorkrowch",
        view=args.view,
        config=config,
    )
    apply_demo_caption(fig, title=built.title, subtitle=built.subtitle, footer=built.footer)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return fig, args.save


__all__ = ["EXAMPLES", "run_example"]
