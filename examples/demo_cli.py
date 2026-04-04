from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.config import EngineName, ViewName

RenderedAxes: TypeAlias = Axes | Axes3D
SizeKnob: TypeAlias = Literal["n_sites", "lx", "ly", "lz", "mera_log2", "tree_depth"]
SchemeByName: TypeAlias = tuple[tuple[str, ...], ...]

AUTO_SAVE_SENTINEL = Path("__AUTO_SAVE__")


@dataclass(frozen=True)
class ExampleCliArgs:
    engine: str
    example: str
    view: ViewName
    labels_nodes: bool
    labels_edges: bool
    labels: bool | None
    hover_labels: bool
    scheme: bool
    playback: bool
    hover_cost: bool
    from_scratch: bool
    from_list: bool
    save: Path | None
    no_show: bool
    n_sites: int
    lx: int
    ly: int
    lz: int
    mera_log2: int
    tree_depth: int


@dataclass(frozen=True)
class NodeBlueprint:
    name: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class GraphBlueprint:
    nodes: tuple[NodeBlueprint, ...]
    bonds: tuple[tuple[tuple[str, str], ...], ...]


@dataclass(frozen=True)
class BuiltExample:
    network: Any
    plot_engine: EngineName
    title: str
    subtitle: str | None = None
    footer: str | None = None
    scheme_steps_by_name: SchemeByName | None = None


ExampleBuilder: TypeAlias = Callable[[ExampleCliArgs, "ExampleDefinition"], BuiltExample]


@dataclass(frozen=True)
class ExampleDefinition:
    name: str
    aliases: tuple[str, ...]
    size_knobs: frozenset[SizeKnob]
    supports_native_object: bool
    supports_from_scratch: bool
    supports_list: bool
    builder: ExampleBuilder
    description: str = ""


class _GraphBlueprintBuilder:
    def __init__(self) -> None:
        self._nodes: list[NodeBlueprint] = []
        self._bonds: list[tuple[tuple[str, str], ...]] = []

    def add_node(self, name: str, axes: tuple[str, ...]) -> None:
        self._nodes.append(NodeBlueprint(name=name, axes=axes))

    def add_bond(self, *legs: tuple[str, str]) -> None:
        if len(legs) < 2:
            raise ValueError("Each bond must connect at least two legs.")
        self._bonds.append(tuple(legs))

    def build(self) -> GraphBlueprint:
        return GraphBlueprint(nodes=tuple(self._nodes), bonds=tuple(self._bonds))


def add_bool_flag(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    default: bool | None,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def build_run_demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tensor-network-visualization demos from the repository examples folder.",
    )
    parser.add_argument("engine", help="Backend engine to demo.")
    parser.add_argument("example", help="Example name for the selected engine.")
    parser.add_argument(
        "--view",
        choices=("2d", "3d"),
        default="2d",
        help="Visualization mode (default: 2d).",
    )
    add_bool_flag(
        parser,
        name="labels-nodes",
        default=True,
        help_text="Draw static tensor-name labels (default: true).",
    )
    add_bool_flag(
        parser,
        name="labels-edges",
        default=False,
        help_text="Draw static edge/index labels (default: false).",
    )
    add_bool_flag(
        parser,
        name="labels",
        default=None,
        help_text="Override both static label flags at once.",
    )
    add_bool_flag(
        parser,
        name="hover-labels",
        default=True,
        help_text="Enable hover tooltips for labels (default: true).",
    )
    add_bool_flag(
        parser,
        name="scheme",
        default=False,
        help_text="Draw contraction-scheme overlays when the example provides them.",
    )
    add_bool_flag(
        parser,
        name="playback",
        default=False,
        help_text="Enable contraction playback widgets; also enables scheme rendering.",
    )
    add_bool_flag(
        parser,
        name="hover-cost",
        default=False,
        help_text="Show contraction-cost details during playback; also enables scheme rendering.",
    )
    add_bool_flag(
        parser,
        name="from-scratch",
        default=False,
        help_text="Force the manual construction path when the example supports it.",
    )
    add_bool_flag(
        parser,
        name="from-list",
        default=False,
        help_text="Pass an iterable/list form to show_tensor_network when supported.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        type=Path,
        const=AUTO_SAVE_SENTINEL,
        default=None,
        help="Save the figure. Without a path, uses .tmp/examples/<engine>/<example>.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render without opening the Matplotlib window.",
    )
    parser.add_argument(
        "--n-sites",
        type=int,
        default=6,
        help="Number of tensors for 1D examples such as MPS, MPO, ladder, chain, hub (default: 6).",
    )
    parser.add_argument(
        "--lx", type=int, default=3, help="Grid width/depth parameter x (default: 3)."
    )
    parser.add_argument(
        "--ly", type=int, default=4, help="Grid width/depth parameter y (default: 4)."
    )
    parser.add_argument(
        "--lz", type=int, default=3, help="Grid width/depth parameter z (default: 3)."
    )
    parser.add_argument(
        "--mera-log2",
        type=int,
        default=3,
        help="Binary MERA width; physical sites are 2**mera_log2 (default: 3).",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Binary TTN depth attached above MERA (default: 4).",
    )
    return parser


def namespace_to_cli_args(namespace: argparse.Namespace) -> ExampleCliArgs:
    return ExampleCliArgs(
        engine=str(namespace.engine),
        example=str(namespace.example),
        view=namespace.view,
        labels_nodes=bool(namespace.labels_nodes),
        labels_edges=bool(namespace.labels_edges),
        labels=namespace.labels,
        hover_labels=bool(namespace.hover_labels),
        scheme=bool(namespace.scheme),
        playback=bool(namespace.playback),
        hover_cost=bool(namespace.hover_cost),
        from_scratch=bool(namespace.from_scratch),
        from_list=bool(namespace.from_list),
        save=namespace.save,
        no_show=bool(namespace.no_show),
        n_sites=int(namespace.n_sites),
        lx=int(namespace.lx),
        ly=int(namespace.ly),
        lz=int(namespace.lz),
        mera_log2=int(namespace.mera_log2),
        tree_depth=int(namespace.tree_depth),
    )


def available_examples(definitions: tuple[ExampleDefinition, ...]) -> tuple[str, ...]:
    return tuple(definition.name for definition in definitions)


def resolve_example_definition(
    definitions: tuple[ExampleDefinition, ...],
    requested: str,
) -> ExampleDefinition | None:
    lowered = requested.lower()
    for definition in definitions:
        if lowered == definition.name or lowered in definition.aliases:
            return definition
    return None


def format_joined_names(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def auto_save_path(*, engine: str, example: str) -> Path:
    return Path(".tmp") / "examples" / engine / f"{example}.png"


def demo_runs_headless(args: ExampleCliArgs | argparse.Namespace) -> bool:
    return bool(getattr(args, "no_show", False) or getattr(args, "save", None) is not None)


def cumulative_prefix_contraction_scheme(names: tuple[str, ...]) -> SchemeByName:
    if not names:
        return ()
    if len(names) == 1:
        return (names,)
    return tuple(tuple(names[:size]) for size in range(2, len(names) + 1))


def cubic_peps_tensor_names(lx: int, ly: int, lz: int) -> tuple[str, ...]:
    if min(lx, ly, lz) < 1:
        raise ValueError("lx, ly, lz must be >= 1")
    return tuple(f"P{i}_{j}_{k}" for i in range(lx) for j in range(ly) for k in range(lz))


def graph_tensor_names(graph: GraphBlueprint) -> tuple[str, ...]:
    return tuple(node.name for node in graph.nodes)


def finalize_demo_plot_config(
    args: ExampleCliArgs | argparse.Namespace,
    *,
    engine: str,
    scheme_tensor_names: SchemeByName | None,
) -> PlotConfig:
    labels_nodes = bool(getattr(args, "labels_nodes", True))
    labels_edges = bool(getattr(args, "labels_edges", False))
    labels_override = getattr(args, "labels", None)
    if labels_override is not None:
        labels_nodes = bool(labels_override)
        labels_edges = bool(labels_override)
    scheme_enabled = bool(
        getattr(args, "scheme", False)
        or getattr(args, "playback", False)
        or getattr(args, "hover_cost", False)
    )
    return PlotConfig(
        show_tensor_labels=labels_nodes,
        show_index_labels=labels_edges,
        hover_labels=bool(getattr(args, "hover_labels", True)),
        show_contraction_scheme=scheme_enabled,
        contraction_scheme_by_name=scheme_tensor_names if scheme_enabled else None,
        contraction_playback=bool(getattr(args, "playback", False)),
        contraction_scheme_cost_hover=bool(getattr(args, "hover_cost", False)),
    )


def render_demo_tensor_network(
    network: Any,
    *,
    args: ExampleCliArgs | argparse.Namespace,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig,
) -> tuple[Figure, RenderedAxes]:
    return show_tensor_network(
        network,
        engine=engine,
        view=view,
        config=config,
        show_controls=not demo_runs_headless(args),
        show=False,
    )


def apply_demo_caption(
    fig: Figure,
    *,
    title: str,
    subtitle: str | None = None,
    footer: str | None = None,
) -> None:
    fig.suptitle(title, fontsize=15, fontweight="semibold", color="#0F172A", y=0.965)
    if subtitle:
        fig.text(
            0.5,
            0.918,
            subtitle,
            ha="center",
            va="top",
            fontsize=11,
            color="#475569",
            transform=fig.transFigure,
        )
    if footer:
        fig.text(
            0.5,
            0.02,
            footer,
            ha="center",
            fontsize=9.5,
            color="#64748B",
            style="italic",
            transform=fig.transFigure,
        )


def ensure_minimum(name: str, value: int, *, minimum: int = 1) -> int:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def axis_dimension(axis_name: str) -> int:
    if axis_name in {"phys", "up", "down", "p", "q"}:
        return 2
    if "phys" in axis_name or axis_name.endswith("_phys"):
        return 2
    return 3


def build_mps_blueprint(n_sites: int) -> GraphBlueprint:
    ensure_minimum("n_sites", n_sites)
    builder = _GraphBlueprintBuilder()
    for index in range(n_sites):
        name = f"A{index}"
        axes: list[str] = []
        if index > 0:
            axes.append("left")
        axes.append("phys")
        if index < n_sites - 1:
            axes.append("right")
        builder.add_node(name, tuple(axes))
        if index > 0:
            builder.add_bond((f"A{index - 1}", "right"), (name, "left"))
    return builder.build()


def build_mpo_blueprint(n_sites: int) -> GraphBlueprint:
    ensure_minimum("n_sites", n_sites)
    builder = _GraphBlueprintBuilder()
    for index in range(n_sites):
        name = f"W{index}"
        axes: list[str] = []
        if index > 0:
            axes.append("left")
        axes.extend(("down", "up"))
        if index < n_sites - 1:
            axes.append("right")
        builder.add_node(name, tuple(axes))
        if index > 0:
            builder.add_bond((f"W{index - 1}", "right"), (name, "left"))
    return builder.build()


def build_ladder_blueprint(n_sites: int) -> GraphBlueprint:
    ensure_minimum("n_sites", n_sites, minimum=2)
    builder = _GraphBlueprintBuilder()
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
        builder.add_node(top, tuple(top_axes))
        builder.add_node(bottom, tuple(bottom_axes))
        builder.add_bond((top, "down"), (bottom, "up"))
        if index > 0:
            builder.add_bond((f"T{index - 1}", "right"), (top, "left"))
            builder.add_bond((f"B{index - 1}", "right"), (bottom, "left"))
    return builder.build()


def build_peps_blueprint(lx: int, ly: int) -> GraphBlueprint:
    ensure_minimum("lx", lx)
    ensure_minimum("ly", ly)
    builder = _GraphBlueprintBuilder()
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
            builder.add_node(name, tuple(axes))
            if i > 0:
                builder.add_bond((f"P{i - 1}_{j}", "down"), (name, "up"))
            if j > 0:
                builder.add_bond((f"P{i}_{j - 1}", "right"), (name, "left"))
    return builder.build()


def build_cubic_peps_blueprint(lx: int, ly: int, lz: int) -> GraphBlueprint:
    ensure_minimum("lx", lx)
    ensure_minimum("ly", ly)
    ensure_minimum("lz", lz)
    builder = _GraphBlueprintBuilder()
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                name = f"P{i}_{j}_{k}"
                axes: list[str] = ["phys"]
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
                builder.add_node(name, tuple(axes))
                if i > 0:
                    builder.add_bond((f"P{i - 1}_{j}_{k}", "xp"), (name, "xm"))
                if j > 0:
                    builder.add_bond((f"P{i}_{j - 1}_{k}", "yp"), (name, "ym"))
                if k > 0:
                    builder.add_bond((f"P{i}_{j}_{k - 1}", "zp"), (name, "zm"))
    return builder.build()


def build_weird_blueprint() -> GraphBlueprint:
    builder = _GraphBlueprintBuilder()
    builder.add_node("center", ("north", "east", "south", "west", "phys"))
    builder.add_node("north", ("center", "east", "phys"))
    builder.add_node("east", ("center", "north", "south", "phys"))
    builder.add_node("south", ("center", "east", "west_a", "west_b", "phys"))
    builder.add_node("west", ("center", "south_a", "south_b", "phys"))
    builder.add_bond(("center", "north"), ("north", "center"))
    builder.add_bond(("center", "east"), ("east", "center"))
    builder.add_bond(("center", "south"), ("south", "center"))
    builder.add_bond(("center", "west"), ("west", "center"))
    builder.add_bond(("north", "east"), ("east", "north"))
    builder.add_bond(("east", "south"), ("south", "east"))
    builder.add_bond(("south", "west_a"), ("west", "south_a"))
    builder.add_bond(("south", "west_b"), ("west", "south_b"))
    return builder.build()


def build_disconnected_blueprint() -> GraphBlueprint:
    builder = _GraphBlueprintBuilder()
    builder.add_node("A", ("bond", "phys"))
    builder.add_node("B", ("bond", "phys"))
    builder.add_node("C", ("left", "right", "phys"))
    builder.add_node("D", ("left", "right", "phys"))
    builder.add_node("E", ("left", "right", "phys"))
    builder.add_bond(("A", "bond"), ("B", "bond"))
    builder.add_bond(("C", "left"), ("D", "right"))
    builder.add_bond(("D", "left"), ("E", "right"))
    builder.add_bond(("E", "left"), ("C", "right"))
    return builder.build()


def build_hyper_blueprint() -> GraphBlueprint:
    builder = _GraphBlueprintBuilder()
    builder.add_node("A", ("hub", "phys"))
    builder.add_node("B", ("hub", "phys"))
    builder.add_node("C", ("hub", "phys"))
    builder.add_bond(("A", "hub"), ("B", "hub"), ("C", "hub"))
    return builder.build()


def build_chain_blueprint(n_sites: int) -> GraphBlueprint:
    ensure_minimum("n_sites", n_sites, minimum=2)
    builder = _GraphBlueprintBuilder()
    for index in range(n_sites):
        name = f"T{index}"
        axes: list[str] = []
        if index > 0:
            axes.append("left")
        axes.append("phys")
        if index < n_sites - 1:
            axes.append("right")
        builder.add_node(name, tuple(axes))
        if index > 0:
            builder.add_bond((f"T{index - 1}", "right"), (name, "left"))
    return builder.build()


def build_hub_blueprint(n_sites: int) -> GraphBlueprint:
    ensure_minimum("n_sites", n_sites, minimum=3)
    builder = _GraphBlueprintBuilder()
    center_axes = ["phys"]
    for index in range(1, n_sites):
        center_axes.append(f"leaf_{index}")
    builder.add_node("H0", tuple(center_axes))
    for index in range(1, n_sites):
        leaf = f"L{index}"
        builder.add_node(leaf, ("center", "phys"))
        builder.add_bond(("H0", f"leaf_{index}"), (leaf, "center"))
    return builder.build()


def build_mera_blueprint(mera_log2: int) -> GraphBlueprint:
    ensure_minimum("mera_log2", mera_log2)
    builder = _GraphBlueprintBuilder()
    current_names: list[str] = []
    for index in range(2**mera_log2):
        name = f"S{index}"
        builder.add_node(name, ("phys", "virt"))
        current_names.append(name)
    layer = 0
    while len(current_names) > 1:
        next_names: list[str] = []
        for index in range(0, len(current_names), 2):
            left = current_names[index]
            right = current_names[index + 1]
            disentangler = f"D{layer}_{index // 2}"
            isometry = f"U{layer}_{index // 2}"
            builder.add_node(disentangler, ("inL", "inR", "outL", "outR"))
            builder.add_node(isometry, ("legL", "legR", "virt"))
            builder.add_bond((left, "virt"), (disentangler, "inL"))
            builder.add_bond((right, "virt"), (disentangler, "inR"))
            builder.add_bond((disentangler, "outL"), (isometry, "legL"))
            builder.add_bond((disentangler, "outR"), (isometry, "legR"))
            next_names.append(isometry)
        current_names = next_names
        layer += 1
    return builder.build()


def build_mera_ttn_blueprint(mera_log2: int, tree_depth: int) -> GraphBlueprint:
    ensure_minimum("mera_log2", mera_log2)
    ensure_minimum("tree_depth", tree_depth)
    builder = _GraphBlueprintBuilder()
    current_names: list[str] = []
    for index in range(2**mera_log2):
        name = f"S{index}"
        builder.add_node(name, ("phys", "virt"))
        current_names.append(name)
    layer = 0
    while len(current_names) > 1:
        next_names = []
        for index in range(0, len(current_names), 2):
            left = current_names[index]
            right = current_names[index + 1]
            disentangler = f"D{layer}_{index // 2}"
            isometry = f"U{layer}_{index // 2}"
            builder.add_node(disentangler, ("inL", "inR", "outL", "outR"))
            builder.add_node(isometry, ("legL", "legR", "virt"))
            builder.add_bond((left, "virt"), (disentangler, "inL"))
            builder.add_bond((right, "virt"), (disentangler, "inR"))
            builder.add_bond((disentangler, "outL"), (isometry, "legL"))
            builder.add_bond((disentangler, "outR"), (isometry, "legR"))
            next_names.append(isometry)
        current_names = next_names
        layer += 1

    apex = current_names[0]
    root_name = "TRoot"
    builder.add_node(root_name, ("top", "to_L", "to_R"))
    builder.add_bond((apex, "virt"), (root_name, "top"))

    def add_branch(parent: str, parent_axis: str, depth: int, label: str) -> None:
        if depth == 0:
            leaf_name = f"TL{label}"
            builder.add_node(leaf_name, ("from_parent", "phys"))
            builder.add_bond((parent, parent_axis), (leaf_name, "from_parent"))
            return
        node_name = f"TI{label}"
        builder.add_node(node_name, ("from_parent", "to_L", "to_R"))
        builder.add_bond((parent, parent_axis), (node_name, "from_parent"))
        add_branch(node_name, "to_L", depth - 1, label + "L")
        add_branch(node_name, "to_R", depth - 1, label + "R")

    add_branch(root_name, "to_L", tree_depth - 1, "L")
    add_branch(root_name, "to_R", tree_depth - 1, "R")
    return builder.build()


__all__ = [
    "AUTO_SAVE_SENTINEL",
    "BuiltExample",
    "ExampleCliArgs",
    "ExampleDefinition",
    "GraphBlueprint",
    "NodeBlueprint",
    "RenderedAxes",
    "SchemeByName",
    "SizeKnob",
    "add_bool_flag",
    "apply_demo_caption",
    "auto_save_path",
    "available_examples",
    "axis_dimension",
    "build_chain_blueprint",
    "build_cubic_peps_blueprint",
    "build_disconnected_blueprint",
    "build_hub_blueprint",
    "build_hyper_blueprint",
    "build_ladder_blueprint",
    "build_mera_blueprint",
    "build_mera_ttn_blueprint",
    "build_mpo_blueprint",
    "build_mps_blueprint",
    "build_peps_blueprint",
    "build_run_demo_parser",
    "build_weird_blueprint",
    "cubic_peps_tensor_names",
    "cumulative_prefix_contraction_scheme",
    "demo_runs_headless",
    "ensure_minimum",
    "finalize_demo_plot_config",
    "format_joined_names",
    "graph_tensor_names",
    "namespace_to_cli_args",
    "render_demo_tensor_network",
    "resolve_example_definition",
]
