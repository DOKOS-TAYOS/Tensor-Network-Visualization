from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, get_args

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.config import EngineName, PlotTheme, ViewName

RenderedAxes: TypeAlias = Axes | Axes3D
SizeKnob: TypeAlias = Literal["n_sites", "lx", "ly", "lz", "mera_log2", "tree_depth"]
SchemeByName: TypeAlias = tuple[tuple[str, ...], ...]
_PLOT_THEME_CHOICES: tuple[PlotTheme, ...] = cast(tuple[PlotTheme, ...], get_args(PlotTheme))

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
    hover_cost: bool
    tensor_inspector: bool
    contracted: bool | None
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
    theme: PlotTheme = "default"


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
    parser.add_argument(
        "--theme",
        choices=_PLOT_THEME_CHOICES,
        default="default",
        help="Visual theme preset (default: default).",
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
        name="hover-cost",
        default=False,
        help_text="Show contraction-cost details during playback; also enables scheme rendering.",
    )
    add_bool_flag(
        parser,
        name="tensor-inspector",
        default=False,
        help_text=(
            "Open the linked tensor inspector for EinsumTrace playback examples and for "
            "contracted TensorKrowch playback when the result tensors can be recovered."
        ),
    )
    add_bool_flag(
        parser,
        name="contracted",
        default=None,
        help_text=(
            "For small TensorKrowch demos, contract the native network first so the "
            "auto-recovered contraction history can be visualized. Supported small "
            "TensorKrowch demos enable this by default; use --no-contracted to disable it."
        ),
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
        theme=namespace.theme,
        labels_nodes=bool(namespace.labels_nodes),
        labels_edges=bool(namespace.labels_edges),
        labels=namespace.labels,
        hover_labels=bool(namespace.hover_labels),
        scheme=bool(namespace.scheme),
        hover_cost=bool(namespace.hover_cost),
        tensor_inspector=bool(namespace.tensor_inspector),
        contracted=namespace.contracted,
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


def pairwise_merge_group_contraction_scheme(groups: SchemeByName) -> SchemeByName:
    active_groups = [tuple(group) for group in groups if group]
    if not active_groups:
        return ()
    if len(active_groups) == 1:
        return (active_groups[0],)

    steps: list[tuple[str, ...]] = []
    while len(active_groups) > 1:
        next_groups: list[tuple[str, ...]] = []
        index = 0
        while index < len(active_groups):
            left_group = active_groups[index]
            if index + 1 >= len(active_groups):
                next_groups.append(left_group)
                index += 1
                continue
            right_group = active_groups[index + 1]
            merged_group = (*left_group, *right_group)
            steps.append(tuple(merged_group))
            next_groups.append(tuple(merged_group))
            index += 2
        active_groups = next_groups
    return tuple(steps)


def pairwise_merge_contraction_scheme(names: tuple[str, ...]) -> SchemeByName:
    return pairwise_merge_group_contraction_scheme(
        tuple((name,) for name in names),
    )


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
    scheme_enabled = bool(getattr(args, "scheme", False) or getattr(args, "hover_cost", False))
    return PlotConfig(
        show_tensor_labels=labels_nodes,
        show_index_labels=labels_edges,
        hover_labels=bool(getattr(args, "hover_labels", True)),
        theme=getattr(args, "theme", "default"),
        show_contraction_scheme=scheme_enabled,
        contraction_scheme_cost_hover=bool(getattr(args, "hover_cost", False)),
        contraction_tensor_inspector=bool(getattr(args, "tensor_inspector", False)),
        contraction_scheme_by_name=scheme_tensor_names if scheme_enabled else None,
    )


def render_demo_tensor_network(
    network: Any,
    *,
    args: ExampleCliArgs | argparse.Namespace,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig,
) -> tuple[Figure, RenderedAxes]:
    if demo_runs_headless(args) and config.show_contraction_scheme:
        raise ValueError(
            "The contraction scheme is dynamic-only. Remove --scheme, or run without "
            "--no-show/--save so the slider controls are available."
        )
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


def _build_pairwise_blueprint(
    node_names: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> GraphBlueprint:
    axes_by_node: dict[str, list[str]] = {name: ["phys"] for name in node_names}
    for edge_index, (left, right) in enumerate(edges):
        axis_name = f"b{edge_index}"
        axes_by_node[left].append(axis_name)
        axes_by_node[right].append(axis_name)

    builder = _GraphBlueprintBuilder()
    for node_name in node_names:
        builder.add_node(node_name, tuple(axes_by_node[node_name]))
    for edge_index, (left, right) in enumerate(edges):
        axis_name = f"b{edge_index}"
        builder.add_bond((left, axis_name), (right, axis_name))
    return builder.build()


def build_weird_blueprint() -> GraphBlueprint:
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
    return _build_pairwise_blueprint(node_names, edges)


def build_disconnected_blueprint() -> GraphBlueprint:
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
    return _build_pairwise_blueprint(node_names, tuple(edges))


def build_hyper_blueprint() -> GraphBlueprint:
    builder = _GraphBlueprintBuilder()
    node_axes: dict[str, list[str]] = {f"H{index}": ["phys"] for index in range(12)}
    hyper_bonds = (
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
    pair_bonds = (
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
            node_axes[node_name].append(axis_name)
    for edge_index, (left, right) in enumerate(pair_bonds):
        axis_name = f"ring_{edge_index}"
        node_axes[left].append(axis_name)
        node_axes[right].append(axis_name)

    for node_name in node_axes:
        builder.add_node(node_name, tuple(node_axes[node_name]))
    for bond in hyper_bonds:
        builder.add_bond(*bond)
    for edge_index, (left, right) in enumerate(pair_bonds):
        axis_name = f"ring_{edge_index}"
        builder.add_bond((left, axis_name), (right, axis_name))
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
    "pairwise_merge_contraction_scheme",
    "pairwise_merge_group_contraction_scheme",
    "render_demo_tensor_network",
    "resolve_example_definition",
]
