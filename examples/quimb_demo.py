from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from demo_cli import (
    BuiltExample,
    ExampleCliArgs,
    ExampleDefinition,
    GraphBlueprint,
    apply_demo_caption,
    axis_dimension,
    build_cubic_peps_blueprint,
    build_disconnected_blueprint,
    build_hyper_blueprint,
    build_ladder_blueprint,
    build_mera_blueprint,
    build_mera_ttn_blueprint,
    build_mpo_blueprint,
    build_mps_blueprint,
    build_peps_blueprint,
    build_weird_blueprint,
    cumulative_prefix_contraction_scheme,
    demo_runs_headless,
    finalize_demo_plot_config,
    graph_tensor_names,
    pairwise_merge_contraction_scheme,
    render_demo_tensor_network,
    resolve_example_definition,
)
from demo_tensors import build_demo_numpy_tensor

TAGLINES: dict[str, str] = {
    "cubic_peps": "Cubic lattice encoded through shared index names.",
    "disconnected": "Disconnected components in a single TensorNetwork.",
    "hyper": "Three tensors share one common hyper-index.",
    "ladder": "Two coupled chains linked by rungs.",
    "mera": "Binary MERA hierarchy expressed with tensor indices.",
    "mera_ttn": "Binary MERA connected to a TTN.",
    "mps": "Finite tensor-train / MPS chain.",
    "mpo": "Finite MPO chain.",
    "peps": "2D PEPS grid.",
    "weird": "Irregular topology for layout fallback.",
}


def _build_blueprint(example: str, args: ExampleCliArgs) -> GraphBlueprint:
    if example == "mps":
        return build_mps_blueprint(args.n_sites)
    if example == "mpo":
        return build_mpo_blueprint(args.n_sites)
    if example == "ladder":
        return build_ladder_blueprint(args.n_sites)
    if example == "peps":
        return build_peps_blueprint(args.lx, args.ly)
    if example == "cubic_peps":
        return build_cubic_peps_blueprint(args.lx, args.ly, args.lz)
    if example == "mera":
        return build_mera_blueprint(args.mera_log2)
    if example == "mera_ttn":
        return build_mera_ttn_blueprint(args.mera_log2, args.tree_depth)
    if example == "weird":
        return build_weird_blueprint()
    if example == "disconnected":
        return build_disconnected_blueprint()
    if example == "hyper":
        return build_hyper_blueprint()
    raise ValueError(f"Unsupported Quimb example: {example}")


def _build_quimb_network(blueprint: GraphBlueprint) -> tuple[Any, list[Any]]:
    import quimb.tensor as qtn

    bonded_axes: dict[tuple[str, str], str] = {}
    for index, bond in enumerate(blueprint.bonds):
        ind_name = f"bond_{index}"
        for leg in bond:
            bonded_axes[leg] = ind_name

    tensors: list[Any] = []
    for node in blueprint.nodes:
        inds = tuple(
            bonded_axes.get((node.name, axis), f"{node.name}_{axis}") for axis in node.axes
        )
        shape = tuple(axis_dimension(axis) for axis in node.axes)
        tensors.append(
            qtn.Tensor(
                data=build_demo_numpy_tensor(name=node.name, shape=shape, dtype=float),
                inds=inds,
                tags={node.name},
            )
        )
    return qtn.TensorNetwork(tensors), tensors


def _scheme_steps(example: str, blueprint: GraphBlueprint) -> tuple[tuple[str, ...], ...] | None:
    if example in {"mps", "mpo"}:
        return pairwise_merge_contraction_scheme(graph_tensor_names(blueprint))
    if example in {"ladder", "peps", "cubic_peps"}:
        return cumulative_prefix_contraction_scheme(graph_tensor_names(blueprint))
    if example == "hyper":
        return (("A", "B", "C"),)
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    blueprint = _build_blueprint(definition.name, args)
    network, tensors = _build_quimb_network(blueprint)
    network_input: Any = tensors if args.from_list else network
    return BuiltExample(
        network=network_input,
        plot_engine="quimb",
        title=f"Quimb · {definition.name.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Render the native quimb TensorNetwork or a plain tensor list.",
        scheme_steps_by_name=_scheme_steps(definition.name, blueprint),
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
