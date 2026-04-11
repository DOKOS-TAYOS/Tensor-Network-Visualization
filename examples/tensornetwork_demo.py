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
from demo_tensors import build_demo_numpy_tensor

NodeSpec: TypeAlias = tuple[str, tuple[str, ...]]
BondSpec: TypeAlias = tuple[tuple[str, str], ...]

TAGLINES: dict[str, str] = {
    "mps": "Finite tensor-train / MPS chain.",
    "peps": "Rectangular PEPS grid with local nearest-neighbor bonds.",
    "weird": "Larger irregular topology built from backend-native Node objects.",
}


def _build_tensornetwork_nodes(
    node_specs: tuple[NodeSpec, ...],
    bond_specs: tuple[BondSpec, ...],
) -> list[Any]:
    import tensornetwork as tn

    nodes: dict[str, Any] = {}
    ordered_nodes: list[Any] = []
    for name, axes in node_specs:
        shape = tuple(axis_dimension(axis) for axis in axes)
        node = tn.Node(
            build_demo_numpy_tensor(name=name, shape=shape, dtype=float),
            name=name,
            axis_names=axes,
        )
        nodes[name] = node
        ordered_nodes.append(node)

    for bond in bond_specs:
        if len(bond) != 2:
            raise ValueError("TensorNetwork examples only support pairwise bonds.")
        (left_name, left_axis), (right_name, right_axis) = bond
        nodes[left_name][left_axis] ^ nodes[right_name][right_axis]
    return ordered_nodes


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


def _specs_for_example(
    example: str,
    args: ExampleCliArgs,
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    if example == "mps":
        return _mps_specs(args.n_sites)
    if example == "peps":
        return _peps_specs(args.lx, args.ly)
    if example == "weird":
        return _weird_specs()
    raise ValueError(f"Unsupported TensorNetwork example: {example}")


def _scheme_steps(
    example: str,
    tensor_names: tuple[str, ...],
) -> tuple[tuple[str, ...], ...] | None:
    if example == "mps":
        return pairwise_merge_contraction_scheme(tensor_names)
    if example == "peps":
        return cumulative_prefix_contraction_scheme(tensor_names)
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    node_specs, bond_specs = _specs_for_example(definition.name, args)
    nodes = _build_tensornetwork_nodes(node_specs, bond_specs)
    tensor_names = tuple(name for name, _axes in node_specs)
    return BuiltExample(
        network=nodes,
        plot_engine="tensornetwork",
        title=f"TensorNetwork - {definition.name.upper()} - {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Backend-native tensornetwork.Node objects passed into show_tensor_network.",
        scheme_steps_by_name=_scheme_steps(definition.name, tensor_names),
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        name="mps",
        aliases=("tt",),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Finite MPS / tensor-train chain.",
    ),
    ExampleDefinition(
        name="peps",
        aliases=(),
        size_knobs=frozenset({"lx", "ly"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="2D PEPS grid.",
    ),
    ExampleDefinition(
        name="weird",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Irregular non-grid network.",
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported TensorNetwork example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args,
        engine="tensornetwork",
        scheme_tensor_names=built.scheme_steps_by_name,
    )
    fig, _ax = render_demo_tensor_network(
        built.network,
        args=args,
        engine="tensornetwork",
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
