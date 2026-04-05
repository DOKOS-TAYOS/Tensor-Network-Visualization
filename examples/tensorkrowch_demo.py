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
from demo_tensors import build_demo_torch_tensor

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
    raise ValueError(f"Unsupported TensorKrowch example: {example}")


def _build_tensorkrowch_network(
    blueprint: GraphBlueprint,
    *,
    materialize_tensors: bool = True,
) -> tuple[Any, list[Any]]:
    import tensorkrowch as tk

    torch = None
    if materialize_tensors:
        import torch as _torch

        torch = _torch

    network = tk.TensorNetwork(name="demo")
    nodes: dict[str, Any] = {}
    ordered_nodes: list[Any] = []
    for node in blueprint.nodes:
        shape = tuple(axis_dimension(axis) for axis in node.axes)
        if torch is None:
            created = tk.Node(shape=shape, axes_names=node.axes, name=node.name, network=network)
        else:
            created = tk.Node(
                tensor=build_demo_torch_tensor(name=node.name, shape=shape, dtype=torch.float32),
                axes_names=node.axes,
                name=node.name,
                network=network,
            )
        nodes[node.name] = created
        ordered_nodes.append(created)
    for bond in blueprint.bonds:
        if len(bond) != 2:
            raise ValueError("TensorKrowch examples only support pairwise bonds.")
        (left_name, left_axis), (right_name, right_axis) = bond
        nodes[left_name][left_axis] ^ nodes[right_name][right_axis]
    return network, ordered_nodes


def _contract_small_chain(nodes: list[Any]) -> Any:
    if len(nodes) < 2:
        raise ValueError("Small contracted demos need at least two tensors.")
    current = nodes[0]
    for next_node in nodes[1:]:
        current = current @ next_node
    return current


def _scheme_steps(example: str, blueprint: GraphBlueprint) -> tuple[tuple[str, ...], ...] | None:
    if example in {"mps", "mpo", "ladder", "peps", "cubic_peps"}:
        return cumulative_prefix_contraction_scheme(graph_tensor_names(blueprint))
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    blueprint = _build_blueprint(definition.name, args)
    use_auto_contracted_demo = args.contracted and definition.name in _SMALL_CONTRACTED_EXAMPLES
    network, nodes = _build_tensorkrowch_network(
        blueprint,
        materialize_tensors=True,
    )
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
        scheme_steps_by_name = _scheme_steps(definition.name, blueprint)
    return BuiltExample(
        network=network_input,
        plot_engine="tensorkrowch",
        title=f"TensorKrowch · {definition.name.upper()} · {args.view.upper()}",
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
