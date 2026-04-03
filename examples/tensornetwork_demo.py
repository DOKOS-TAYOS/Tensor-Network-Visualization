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
    render_demo_tensor_network,
    resolve_example_definition,
)

TAGLINES: dict[str, str] = {
    "cubic_peps": "Cubic lattice with six-neighbor bulk tensors.",
    "disconnected": "Several disconnected components in a single render.",
    "ladder": "Two coupled chains linked by rungs.",
    "mera": "Binary MERA hierarchy built node by node.",
    "mera_ttn": "Binary MERA topped by a binary TTN.",
    "mps": "Finite tensor-train / MPS chain.",
    "mpo": "Finite matrix-product operator chain.",
    "peps": "Rectangular PEPS grid with local nearest-neighbor bonds.",
    "weird": "Irregular topology for force-directed placement.",
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
    raise ValueError(f"Unsupported TensorNetwork example: {example}")


def _build_tensornetwork_nodes(blueprint: GraphBlueprint) -> list[Any]:
    import numpy as np
    import tensornetwork as tn

    nodes: dict[str, Any] = {}
    ordered_nodes: list[Any] = []
    for node in blueprint.nodes:
        shape = tuple(axis_dimension(axis) for axis in node.axes)
        created = tn.Node(
            np.ones(shape, dtype=float),
            name=node.name,
            axis_names=node.axes,
        )
        nodes[node.name] = created
        ordered_nodes.append(created)
    for bond in blueprint.bonds:
        if len(bond) != 2:
            raise ValueError("TensorNetwork examples only support pairwise bonds.")
        (left_name, left_axis), (right_name, right_axis) = bond
        nodes[left_name][left_axis] ^ nodes[right_name][right_axis]
    return ordered_nodes


def _scheme_steps(example: str, blueprint: GraphBlueprint) -> tuple[tuple[str, ...], ...] | None:
    if example in {"mps", "mpo", "ladder", "peps", "cubic_peps"}:
        return cumulative_prefix_contraction_scheme(graph_tensor_names(blueprint))
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    blueprint = _build_blueprint(definition.name, args)
    nodes = _build_tensornetwork_nodes(blueprint)
    return BuiltExample(
        network=nodes,
        plot_engine="tensornetwork",
        title=f"TensorNetwork · {definition.name.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Backend-native tensornetwork.Node objects passed into show_tensor_network.",
        scheme_steps_by_name=_scheme_steps(definition.name, blueprint),
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
        name="mpo",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Finite MPO chain.",
    ),
    ExampleDefinition(
        name="ladder",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Two coupled chains.",
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
        name="cubic_peps",
        aliases=(),
        size_knobs=frozenset({"lx", "ly", "lz"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="3D PEPS lattice.",
    ),
    ExampleDefinition(
        name="mera",
        aliases=(),
        size_knobs=frozenset({"mera_log2"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Binary MERA hierarchy.",
    ),
    ExampleDefinition(
        name="mera_ttn",
        aliases=(),
        size_knobs=frozenset({"mera_log2", "tree_depth"}),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Binary MERA connected to a TTN.",
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
    ExampleDefinition(
        name="disconnected",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=False,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Multiple disconnected components.",
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
