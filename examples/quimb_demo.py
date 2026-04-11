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
    "hyper": "Multi-index hypergraph with pairwise bonds around the shared-index hubs.",
    "mps": "Finite tensor-train / MPS chain.",
    "peps": "2D PEPS grid.",
}


def _build_quimb_network(
    node_specs: tuple[NodeSpec, ...],
    bond_specs: tuple[BondSpec, ...],
) -> tuple[Any, list[Any]]:
    import quimb.tensor as qtn

    bonded_axes: dict[tuple[str, str], str] = {}
    for bond_index, bond in enumerate(bond_specs):
        index_name = f"bond_{bond_index}"
        for node_name, axis_name in bond:
            bonded_axes[node_name, axis_name] = index_name

    tensors: list[Any] = []
    for name, axes in node_specs:
        inds = tuple(bonded_axes.get((name, axis), f"{name}_{axis}") for axis in axes)
        shape = tuple(axis_dimension(axis) for axis in axes)
        tensors.append(
            qtn.Tensor(
                data=build_demo_numpy_tensor(name=name, shape=shape, dtype=float),
                inds=inds,
                tags={name},
            )
        )
    return qtn.TensorNetwork(tensors), tensors


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


def _hyper_specs() -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    axes_by_node: dict[str, list[str]] = {f"H{index}": ["phys"] for index in range(12)}
    hyper_bonds: tuple[BondSpec, ...] = (
        (("H0", "alpha"), ("H1", "alpha"), ("H2", "alpha"), ("H3", "alpha")),
        (("H3", "beta"), ("H4", "beta"), ("H5", "beta")),
        (
            ("H6", "gamma"),
            ("H7", "gamma"),
            ("H8", "gamma"),
            ("H9", "gamma"),
            ("H10", "gamma"),
        ),
    )
    pair_edges = (
        ("H0", "H4"),
        ("H1", "H6"),
        ("H2", "H7"),
        ("H5", "H8"),
        ("H8", "H11"),
        ("H11", "H9"),
        ("H9", "H4"),
        ("H10", "H2"),
        ("H11", "H3"),
    )
    for bond in hyper_bonds:
        for node_name, axis_name in bond:
            axes_by_node[node_name].append(axis_name)

    pair_bonds: list[BondSpec] = []
    for edge_index, (left, right) in enumerate(pair_edges):
        axis_name = f"ring_{edge_index}"
        axes_by_node[left].append(axis_name)
        axes_by_node[right].append(axis_name)
        pair_bonds.append(((left, axis_name), (right, axis_name)))

    node_specs = tuple((name, tuple(axes)) for name, axes in axes_by_node.items())
    return node_specs, (*hyper_bonds, *tuple(pair_bonds))


def _specs_for_example(
    example: str,
    args: ExampleCliArgs,
) -> tuple[tuple[NodeSpec, ...], tuple[BondSpec, ...]]:
    if example == "mps":
        return _mps_specs(args.n_sites)
    if example == "peps":
        return _peps_specs(args.lx, args.ly)
    if example == "hyper":
        return _hyper_specs()
    raise ValueError(f"Unsupported Quimb example: {example}")


def _scheme_steps(
    example: str,
    tensor_names: tuple[str, ...],
) -> tuple[tuple[str, ...], ...] | None:
    if example == "mps":
        return pairwise_merge_contraction_scheme(tensor_names)
    if example == "peps":
        return cumulative_prefix_contraction_scheme(tensor_names)
    if example == "hyper":
        return (
            ("H0", "H1", "H2", "H3"),
            ("H3", "H4", "H5"),
            ("H6", "H7", "H8", "H9", "H10"),
        )
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    node_specs, bond_specs = _specs_for_example(definition.name, args)
    network, tensors = _build_quimb_network(node_specs, bond_specs)
    network_input: Any = tensors if args.from_list else network
    tensor_names = tuple(name for name, _axes in node_specs)
    return BuiltExample(
        network=network_input,
        plot_engine="quimb",
        title=f"Quimb - {definition.name.upper()} - {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Render the native quimb TensorNetwork or a plain tensor list.",
        scheme_steps_by_name=_scheme_steps(definition.name, tensor_names),
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
        name="hyper",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Hypergraph / multi-body shared index.",
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported Quimb example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args,
        engine="quimb",
        scheme_tensor_names=built.scheme_steps_by_name,
    )
    fig, _ax = render_demo_tensor_network(
        built.network,
        args=args,
        engine="quimb",
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
