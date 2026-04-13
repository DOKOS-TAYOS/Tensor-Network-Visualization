from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, cast, get_args

import matplotlib

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from demo_cli import (
    BuiltExample,
    ExampleCliArgs,
    ExampleDefinition,
    apply_demo_caption,
    demo_runs_headless,
    resolve_example_definition,
)
from demo_tensors import build_demo_numpy_tensor

from tensor_network_viz import PlotConfig, show_tensor_network
from tensor_network_viz.config import PlotTheme

_THEMES: tuple[PlotTheme, ...] = cast(tuple[PlotTheme, ...], get_args(PlotTheme))


def _theme_grid() -> tuple[int, int]:
    theme_count = len(_THEMES)
    columns = min(4, max(theme_count, 1))
    rows = math.ceil(theme_count / columns)
    return rows, columns


def _theme_figure_size(view: str) -> tuple[float, float]:
    rows, columns = _theme_grid()
    width = max(11.0, 4.6 * columns)
    height = max(5.2, (4.2 if view == "2d" else 4.8) * rows)
    return width, height


def _theme_footer() -> str:
    theme_names = ", ".join(_THEMES[:-1])
    return f"Compare the available PlotConfig theme presets: {theme_names} and {_THEMES[-1]}."


def _dim_for_index(index_name: str) -> int:
    if "latent" in index_name or "bond" in index_name or index_name == "hyper_shared":
        return 3
    return 2


def _theme_tensor(name: str, inds: tuple[str, ...]) -> Any:
    import quimb.tensor as qtn

    shape = tuple(_dim_for_index(ind) for ind in inds)
    return qtn.Tensor(
        data=build_demo_numpy_tensor(name=f"theme_{name}", shape=shape, dtype=float),
        inds=inds,
        tags={name},
    )


def _build_theme_network() -> Any:
    import quimb.tensor as qtn

    tensors = [
        _theme_tensor("Input", ("input_sample", "bond_input_core")),
        _theme_tensor(
            "Core",
            ("bond_input_core", "bond_core_memory", "hyper_shared", "core_probe"),
        ),
        _theme_tensor("Memory", ("bond_core_memory", "bond_memory_readout", "memory_state")),
        _theme_tensor(
            "Readout",
            ("bond_memory_readout", "bond_bias", "hyper_shared", "score_index"),
        ),
        _theme_tensor("Bias", ("bond_bias", "bias_feature")),
        _theme_tensor("Probe", ("hyper_shared", "probe_reading")),
    ]
    return qtn.TensorNetwork(tensors)


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    del args, definition
    return BuiltExample(
        network=_build_theme_network(),
        plot_engine="quimb",
        title="Visual themes overview",
        subtitle=(
            "Normal tensors, degree-one tensors, free indices, labels, bonds and a hyper-index."
        ),
        footer=_theme_footer(),
        scheme_steps_by_name=None,
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        name="overview",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=False,
        builder=_build_example,
        description="Compare the three PlotConfig visual themes on one representative network.",
    ),
)


def _subplot_for_view(fig: Any, *, index: int, view: str, rows: int, columns: int) -> Any:
    if view == "3d":
        return fig.add_subplot(rows, columns, index, projection="3d")
    return fig.add_subplot(rows, columns, index)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported themes example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    rows, columns = _theme_grid()
    fig = plt.figure(figsize=_theme_figure_size(args.view))
    for index, theme in enumerate(_THEMES, start=1):
        ax = _subplot_for_view(
            fig,
            index=index,
            view=args.view,
            rows=rows,
            columns=columns,
        )
        show_tensor_network(
            built.network,
            engine="quimb",
            view=args.view,
            ax=ax,
            config=PlotConfig(
                theme=theme,
                show_index_labels=True,
                hover_labels=False,
                tensor_label_fontsize=9,
                edge_label_fontsize=7,
                layout_iterations=260,
            ),
            show_controls=False,
            show=False,
        )
        ax.set_title(theme, fontsize=12, fontweight="semibold")
    apply_demo_caption(fig, title=built.title, subtitle=built.subtitle, footer=built.footer)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.9))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return fig, args.save


__all__ = ["EXAMPLES", "run_example"]
