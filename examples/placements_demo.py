from __future__ import annotations

import sys
from dataclasses import replace
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
    demo_runs_headless,
    finalize_demo_plot_config,
    resolve_example_definition,
)
from demo_tensors import build_demo_numpy_tensor

from tensor_network_viz import show_tensor_network

Coord2D: TypeAlias = tuple[int, int]
Coord3D: TypeAlias = tuple[int, int, int]

_BASE_SCHEME = (
    ("Prior", "Transition"),
    ("Emission", "Calibrator", "Readout"),
    ("Readout", "Loss"),
)


def _dim_for_index(index_name: str) -> int:
    if "latent" in index_name or "grid" in index_name or "bond" in index_name:
        return 3
    return 2


def _tensor(name: str, inds: tuple[str, ...]) -> Any:
    import quimb.tensor as qtn

    shape = tuple(_dim_for_index(ind) for ind in inds)
    return qtn.Tensor(
        data=build_demo_numpy_tensor(name=f"placement_{name}", shape=shape, dtype=float),
        inds=inds,
        tags={name},
    )


def _base_tensors() -> list[Any]:
    return [
        _tensor("Prior", ("sample_state", "latent_prev")),
        _tensor("Transition", ("latent_prev", "latent_now", "control_signal")),
        _tensor("Emission", ("latent_now", "sensor_reading", "readout_latent")),
        _tensor("Calibrator", ("sensor_reading", "calibration_feature")),
        _tensor("Readout", ("readout_latent", "calibration_feature", "class_score")),
        _tensor("Loss", ("class_score", "target_label")),
    ]


def _base_network() -> Any:
    import quimb.tensor as qtn

    return qtn.TensorNetwork(_base_tensors())


def _tensor_by_tag(network: Any, tag: str) -> Any:
    for tensor in network.tensors:
        if tag in tensor.tags:
            return tensor
    raise ValueError(f"Tensor tag {tag!r} was not found.")


def _manual_positions(network: Any) -> dict[int, tuple[float, float]]:
    return {
        id(_tensor_by_tag(network, "Prior")): (-2.7, 0.4),
        id(_tensor_by_tag(network, "Transition")): (-1.4, 0.4),
        id(_tensor_by_tag(network, "Emission")): (0.0, 0.25),
        id(_tensor_by_tag(network, "Calibrator")): (0.0, -1.0),
        id(_tensor_by_tag(network, "Readout")): (1.45, 0.2),
        id(_tensor_by_tag(network, "Loss")): (2.75, 0.15),
    }


def _grid_tensors_2d(active: set[Coord2D], *, rows: int, cols: int) -> list[list[Any | None]]:
    axes_by_coord: dict[Coord2D, list[str]] = {
        coord: [f"site_{coord[0]}_{coord[1]}"] for coord in active
    }
    for i, j in sorted(active):
        for di, dj, label in ((1, 0, "down"), (0, 1, "right")):
            neighbor = (i + di, j + dj)
            if neighbor not in active:
                continue
            index_name = f"grid2_{i}_{j}_{label}"
            axes_by_coord[(i, j)].append(index_name)
            axes_by_coord[neighbor].append(index_name)

    tensors = {
        coord: _tensor(f"G{coord[0]}_{coord[1]}", tuple(axes_by_coord[coord]))
        for coord in sorted(active)
    }
    return [[tensors.get((i, j)) for j in range(cols)] for i in range(rows)]


def _grid_tensors_3d(
    active: set[Coord3D],
    *,
    layers: int,
    rows: int,
    cols: int,
) -> list[list[list[Any | None]]]:
    axes_by_coord: dict[Coord3D, list[str]] = {
        coord: [f"site_{coord[0]}_{coord[1]}_{coord[2]}"] for coord in active
    }
    for k, i, j in sorted(active):
        for dk, di, dj, label in ((1, 0, 0, "front"), (0, 1, 0, "down"), (0, 0, 1, "right")):
            neighbor = (k + dk, i + di, j + dj)
            if neighbor not in active:
                continue
            index_name = f"grid3_{k}_{i}_{j}_{label}"
            axes_by_coord[(k, i, j)].append(index_name)
            axes_by_coord[neighbor].append(index_name)

    tensors = {
        coord: _tensor(f"Q{coord[0]}_{coord[1]}_{coord[2]}", tuple(axes_by_coord[coord]))
        for coord in sorted(active)
    }
    return [
        [[tensors.get((k, i, j)) for j in range(cols)] for i in range(rows)] for k in range(layers)
    ]


def _grid2d_input() -> list[list[Any | None]]:
    active = {
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
    }
    return _grid_tensors_2d(active, rows=3, cols=4)


def _grid3d_input() -> list[list[list[Any | None]]]:
    active = {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (2, 1, 2),
    }
    return _grid_tensors_3d(active, layers=3, rows=3, cols=3)


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    if definition.name == "object":
        network: Any = _base_network()
        subtitle = "Pass a native quimb TensorNetwork object directly."
        footer = (
            "This mirrors the most common user path: build the object, then call "
            "show_tensor_network(network)."
        )
        scheme = None
    elif definition.name == "list":
        network = _base_tensors()
        subtitle = "Pass a flat Python list of quimb Tensor objects."
        footer = (
            "Useful when tensors are produced by a pipeline before being wrapped into a network."
        )
        scheme = None
    elif definition.name == "grid2d":
        network = _grid2d_input()
        subtitle = "Pass a nested list with None holes to fix a 2D placement."
        footer = "The nested-list shape controls positions; None leaves empty grid cells."
        scheme = None
    elif definition.name == "grid3d":
        network = _grid3d_input()
        subtitle = "Pass a layer-row-column nested list to fix a 3D placement."
        footer = "The 3D grid form keeps coordinates meaningful in both 3D and projected 2D views."
        scheme = None
    elif definition.name == "manual_positions":
        network = _base_network()
        subtitle = "Pass manual coordinates through PlotConfig(positions=...)."
        footer = (
            "Manual coordinates override the force layout while preserving the same "
            "tensor-network object."
        )
        scheme = None
    elif definition.name == "manual_scheme":
        network = _base_network()
        subtitle = "Pass an explicit contraction scheme using visible tensor names."
        footer = (
            "The scheme is intentionally small so the slider highlights the groups without clutter."
        )
        scheme = _BASE_SCHEME
    elif definition.name == "named_indices":
        network = _base_network()
        subtitle = "Use descriptive shared-index names and static index labels."
        footer = (
            "The labels come from the tensor index names, not from a separate drawing annotation."
        )
        scheme = None
    else:
        raise ValueError(f"Unsupported placements example: {definition.name}")
    return BuiltExample(
        network=network,
        plot_engine="quimb",
        title=f"Placements - {definition.name.replace('_', ' ')} - {args.view.upper()}",
        subtitle=subtitle,
        footer=footer,
        scheme_steps_by_name=scheme,
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition("object", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("list", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("grid2d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("grid3d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("manual_positions", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("manual_scheme", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("named_indices", (), frozenset(), True, False, False, _build_example),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported placements example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args,
        engine="quimb",
        scheme_tensor_names=built.scheme_steps_by_name,
    )
    show_controls = not demo_runs_headless(args)
    if definition.name == "manual_positions":
        config = replace(
            config,
            positions=_manual_positions(built.network),
            validate_positions=True,
        )
    if definition.name == "named_indices":
        config = replace(config, show_index_labels=True)
    if definition.name == "manual_scheme":
        config = replace(
            config,
            show_contraction_scheme=True,
            contraction_scheme_by_name=built.scheme_steps_by_name,
        )
        show_controls = True

    fig, _ax = show_tensor_network(
        built.network,
        engine="quimb",
        view=args.view,
        config=config,
        show_controls=show_controls,
        show=False,
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
