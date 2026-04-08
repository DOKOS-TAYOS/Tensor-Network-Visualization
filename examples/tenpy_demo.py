from __future__ import annotations

import sys
import warnings
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
    build_chain_blueprint,
    build_hub_blueprint,
    build_hyper_blueprint,
    demo_runs_headless,
    finalize_demo_plot_config,
    render_demo_tensor_network,
    resolve_example_definition,
)
from demo_tensors import build_demo_numpy_tensor

TAGLINES: dict[str, str] = {
    "chain": "Explicit TenPyTensorNetwork open chain.",
    "excitation": "Momentum-style excitation chain on top of a UniformMPS.",
    "hub": "Explicit TenPyTensorNetwork star topology.",
    "hyper": "Explicit TenPyTensorNetwork hyperedge / shared bond.",
    "impo": "Infinite MPO unit cell.",
    "imps": "Infinite MPS unit cell.",
    "mps": "Finite MPS from a product state.",
    "mpo": "Finite MPO from the transverse-field Ising model.",
    "purification": "Finite-temperature purification MPS.",
    "uniform": "UniformMPS built from an iMPS unit cell.",
}


class _ExcitationLikeChain:
    def __init__(self, uniform: Any) -> None:
        self.uMPS_GS = uniform

    def get_X(self, i: int, copy: bool = False) -> Any:
        return self.uMPS_GS.get_AR(i, copy=copy)


def _build_native_mps(n_sites: int) -> Any:
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(n_sites)]
    states = ["up" if index % 2 == 0 else "down" for index in range(n_sites)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        return MPS.from_product_state(sites, states, bc="finite")


def _build_native_mpo(n_sites: int) -> Any:
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": n_sites, "J": 1.0, "g": 1.0, "bc_MPS": "finite"})
    return model.calc_H_MPO()


def _build_native_imps(n_sites: int) -> Any:
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(n_sites)]
    states = ["up" if index % 2 == 0 else "down" for index in range(n_sites)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        return MPS.from_product_state(sites, states, bc="infinite")


def _build_native_impo(n_sites: int) -> Any:
    from tenpy.models.tf_ising import TFIChain

    model = TFIChain({"L": n_sites, "J": 1.0, "g": 1.0, "bc_MPS": "infinite"})
    return model.calc_H_MPO()


def _build_native_purification(n_sites: int) -> Any:
    from tenpy.networks.purification_mps import PurificationMPS
    from tenpy.networks.site import SpinHalfSite

    sites = [SpinHalfSite() for _ in range(n_sites)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        return PurificationMPS.from_infiniteT(sites, bc="finite")


def _build_native_uniform(n_sites: int) -> Any:
    from tenpy.tools.misc import BetaWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BetaWarning)
        warnings.filterwarnings("ignore", message="unit_cell_width.*", category=UserWarning)
        unit_cell = _build_native_imps(n_sites)
        from tenpy.networks.uniform_mps import UniformMPS

        return UniformMPS.from_MPS(unit_cell)


def _build_native_excitation(n_sites: int) -> Any:
    return _ExcitationLikeChain(_build_native_uniform(n_sites))


def _npc_tensor(name: str, axes: tuple[str, ...]) -> Any:
    from tenpy.linalg import np_conserved as npc

    shape = tuple(axis_dimension(axis) for axis in axes)
    legs = [npc.LegCharge.from_trivial(dim) for dim in shape]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return npc.Array.from_ndarray(
            build_demo_numpy_tensor(name=name, shape=shape, dtype=float),
            legs,
            labels=list(axes),
        )


def _build_explicit_network(blueprint: GraphBlueprint) -> Any:
    from tensor_network_viz import make_tenpy_tensor_network

    nodes = [(node.name, _npc_tensor(node.name, node.axes)) for node in blueprint.nodes]
    return make_tenpy_tensor_network(nodes=nodes, bonds=blueprint.bonds)


def _build_native_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    if definition.name == "mps":
        network = _build_native_mps(args.n_sites)
    elif definition.name == "mpo":
        network = _build_native_mpo(args.n_sites)
    elif definition.name == "imps":
        network = _build_native_imps(args.n_sites)
    elif definition.name == "impo":
        network = _build_native_impo(args.n_sites)
    elif definition.name == "purification":
        network = _build_native_purification(args.n_sites)
    elif definition.name == "uniform":
        network = _build_native_uniform(args.n_sites)
    elif definition.name == "excitation":
        network = _build_native_excitation(args.n_sites)
    else:
        raise ValueError(f"Unsupported native TeNPy example: {definition.name}")
    return BuiltExample(
        network=network,
        plot_engine="tenpy",
        title=f"TeNPy · {definition.name.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Render the native TeNPy object directly with engine='tenpy'.",
        scheme_steps_by_name=None,
    )


def _build_explicit_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    if definition.name == "chain":
        blueprint = build_chain_blueprint(args.n_sites)
    elif definition.name == "hub":
        blueprint = build_hub_blueprint(args.n_sites)
    elif definition.name == "hyper":
        blueprint = build_hyper_blueprint()
    else:
        raise ValueError(f"Unsupported explicit TeNPy example: {definition.name}")
    network = _build_explicit_network(blueprint)
    scheme_steps = None
    if definition.name == "chain":
        names = tuple(node.name for node in blueprint.nodes)
        scheme_steps = tuple(tuple(names[:size]) for size in range(2, len(names) + 1))
    elif definition.name == "hub":
        scheme_steps = (("H0", "L1", "L2"),)
    elif definition.name == "hyper":
        scheme_steps = (("A", "B", "C"),)
    return BuiltExample(
        network=network,
        plot_engine="tenpy",
        title=f"TeNPy · {definition.name.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Render an explicit TenPyTensorNetwork built from npc.Array tensors.",
        scheme_steps_by_name=scheme_steps,
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        name="mps",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="Finite TeNPy MPS.",
    ),
    ExampleDefinition(
        name="mpo",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="Finite TeNPy MPO.",
    ),
    ExampleDefinition(
        name="imps",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="Infinite TeNPy MPS.",
    ),
    ExampleDefinition(
        name="impo",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="Infinite TeNPy MPO.",
    ),
    ExampleDefinition(
        name="purification",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="TeNPy PurificationMPS.",
    ),
    ExampleDefinition(
        name="uniform",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="TeNPy UniformMPS.",
    ),
    ExampleDefinition(
        name="excitation",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_native_example,
        description="TeNPy excitation-like chain.",
    ),
    ExampleDefinition(
        name="chain",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=False,
        builder=_build_explicit_example,
        description="Explicit TenPyTensorNetwork chain.",
    ),
    ExampleDefinition(
        name="hub",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=False,
        builder=_build_explicit_example,
        description="Explicit TenPyTensorNetwork hub.",
    ),
    ExampleDefinition(
        name="hyper",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=False,
        builder=_build_explicit_example,
        description="Explicit TenPyTensorNetwork hyperedge.",
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported TeNPy example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args, engine="tenpy", scheme_tensor_names=built.scheme_steps_by_name
    )
    fig, _ax = render_demo_tensor_network(
        built.network,
        args=args,
        engine="tenpy",
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
