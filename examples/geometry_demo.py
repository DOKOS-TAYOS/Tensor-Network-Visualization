from __future__ import annotations

import random
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

TAGLINES: dict[str, str] = {
    "partial_grid2d": "A 2D grid with holes and an uneven boundary.",
    "decorated_sparse_grid2d": (
        "A sparse 2D grid with triangular holes and observed boundary leaves."
    ),
    "upper_triangle2d": "Only the upper-triangular part of a regular 2D grid.",
    "partial_grid3d": "A 3D lattice with missing cells across several layers.",
    "upper_pyramid3d": "A stacked 3D pyramid with fewer nodes on each upper layer.",
    "random_irregular": "A deterministic irregular graph with many short and long links.",
    "circular_ring": "A pure ring where only the endpoints wrap around.",
    "circular_chords": "A circular network with ring bonds and internal chords.",
    "tubular_grid": "A 2D grid wrapped around one periodic direction.",
    "disconnected_irregular": "Several unrelated irregular components in one figure.",
}


def _dim_for_index(index_name: str) -> int:
    if index_name.startswith(("edge", "grid")):
        return 3
    return 2


def _tensor(name: str, inds: tuple[str, ...]) -> Any:
    import quimb.tensor as qtn

    shape = tuple(_dim_for_index(ind) for ind in inds)
    return qtn.Tensor(
        data=build_demo_numpy_tensor(name=f"geometry_{name}", shape=shape, dtype=float),
        inds=inds,
        tags={name},
    )


def _network_from_edges(
    node_names: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> list[Any]:
    inds_by_node: dict[str, list[str]] = {
        node_name: [f"obs_{node_name}"] for node_name in node_names
    }
    for edge_index, (left, right) in enumerate(edges):
        ind_name = f"edge_{edge_index}"
        inds_by_node[left].append(ind_name)
        inds_by_node[right].append(ind_name)
    return [_tensor(node_name, tuple(inds_by_node[node_name])) for node_name in node_names]


def _flatten_tensor_grid(grid: Any) -> list[Any]:
    if grid is None:
        return []
    if isinstance(grid, list):
        flattened: list[Any] = []
        for item in grid:
            flattened.extend(_flatten_tensor_grid(item))
        return flattened
    return [grid]


def _grid_tensors_2d(
    active: set[Coord2D],
    *,
    rows: int,
    cols: int,
    prefix: str,
) -> list[list[Any | None]]:
    axes_by_coord: dict[Coord2D, list[str]] = {
        coord: [f"obs_{prefix}_{coord[0]}_{coord[1]}"] for coord in active
    }
    for i, j in sorted(active):
        for di, dj, label in ((1, 0, "down"), (0, 1, "right")):
            neighbor = (i + di, j + dj)
            if neighbor not in active:
                continue
            index_name = f"grid_{prefix}_{i}_{j}_{label}"
            axes_by_coord[(i, j)].append(index_name)
            axes_by_coord[neighbor].append(index_name)

    tensors = {
        coord: _tensor(f"{prefix}{coord[0]}_{coord[1]}", tuple(axes_by_coord[coord]))
        for coord in sorted(active)
    }
    return [[tensors.get((i, j)) for j in range(cols)] for i in range(rows)]


def _grid_tensors_3d(
    active: set[Coord3D],
    *,
    layers: int,
    rows: int,
    cols: int,
    prefix: str,
) -> list[list[list[Any | None]]]:
    axes_by_coord: dict[Coord3D, list[str]] = {
        coord: [f"obs_{prefix}_{coord[0]}_{coord[1]}_{coord[2]}"] for coord in active
    }
    for k, i, j in sorted(active):
        for dk, di, dj, label in ((1, 0, 0, "front"), (0, 1, 0, "down"), (0, 0, 1, "right")):
            neighbor = (k + dk, i + di, j + dj)
            if neighbor not in active:
                continue
            index_name = f"grid_{prefix}_{k}_{i}_{j}_{label}"
            axes_by_coord[(k, i, j)].append(index_name)
            axes_by_coord[neighbor].append(index_name)

    tensors = {
        coord: _tensor(
            f"{prefix}{coord[0]}_{coord[1]}_{coord[2]}",
            tuple(axes_by_coord[coord]),
        )
        for coord in sorted(active)
    }
    return [
        [[tensors.get((k, i, j)) for j in range(cols)] for i in range(rows)] for k in range(layers)
    ]


def _partial_grid2d() -> list[Any]:
    rows, cols = 6, 8
    active = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if (i + 2 * j) % 7 != 0 and not (i in {0, rows - 1} and j in {0, cols - 1})
    }
    return _flatten_tensor_grid(_grid_tensors_2d(active, rows=rows, cols=cols, prefix="PG"))


def _decorated_sparse_grid2d_active(
    *,
    length: int = 15,
    thickness: int = 5,
) -> set[Coord2D]:
    if length < 3:
        raise ValueError("length must be >= 3")
    if thickness < 2:
        raise ValueError("thickness must be >= 2")
    if length < thickness:
        raise ValueError("length must be >= thickness")

    active: set[Coord2D] = set()
    for diagonal_index in range(length):
        for offset in range(thickness):
            active.add((diagonal_index + offset, diagonal_index))
    return active


def _decorated_sparse_grid2d() -> list[Any]:
    active = _decorated_sparse_grid2d_active()
    prefix = "DSG"
    axes_by_tensor: dict[str, list[str]] = {
        f"{prefix}_{row}_{col}": [] for row, col in sorted(active)
    }
    tensors: list[Any] = []
    rightmost_by_row = {
        row: max(col for candidate_row, col in active if candidate_row == row)
        for row, _col in active
    }
    right_leaf_rows = tuple(
        row for row in sorted(rightmost_by_row) if row % 2 == 1 and row < max(rightmost_by_row)
    )
    leaf_specs: tuple[tuple[str, Coord2D, str], ...] = (
        ("top", (0, 0), "up"),
        *(("right", (row, rightmost_by_row[row]), "right") for row in right_leaf_rows),
    )

    for row, col in sorted(active):
        tensor_name = f"{prefix}_{row}_{col}"
        for dr, dc, label in ((1, 0, "down"), (0, 1, "right")):
            neighbor = (row + dr, col + dc)
            if neighbor not in active:
                continue
            neighbor_name = f"{prefix}_{neighbor[0]}_{neighbor[1]}"
            index_name = f"grid_{prefix}_{row}_{col}_{label}"
            axes_by_tensor[tensor_name].append(index_name)
            axes_by_tensor[neighbor_name].append(index_name)

    for side, (row, col), direction in leaf_specs:
        tensor_name = f"{prefix}_{row}_{col}"
        bond_name = f"grid_{prefix}_leaf_{side}_{row}_{col}"
        axes_by_tensor[tensor_name].append(bond_name)
        tensors.append(
            _tensor(
                f"{prefix}_leaf_{side}_{row}_{col}",
                (bond_name, f"obs_{prefix}_{direction}_{row}_{col}"),
            )
        )

    tensors.extend(
        _tensor(tensor_name, tuple(axes_by_tensor[tensor_name]))
        for tensor_name in sorted(axes_by_tensor)
    )
    return tensors


def _upper_triangle2d() -> list[Any]:
    size = 8
    active = {(i, j) for i in range(size) for j in range(i, size)}
    return _flatten_tensor_grid(_grid_tensors_2d(active, rows=size, cols=size, prefix="UT"))


def _partial_grid3d() -> list[Any]:
    layers, rows, cols = 3, 4, 5
    active = {
        (k, i, j)
        for k in range(layers)
        for i in range(rows)
        for j in range(cols)
        if (k + i + 2 * j) % 4 != 0
    }
    return _flatten_tensor_grid(
        _grid_tensors_3d(active, layers=layers, rows=rows, cols=cols, prefix="P3")
    )


def _upper_pyramid3d() -> list[Any]:
    layers, rows, cols = 4, 7, 7
    center = rows // 2
    active = {
        (k, i, j)
        for k in range(layers)
        for i in range(rows)
        for j in range(cols)
        if abs(i - center) + abs(j - center) <= center - k
    }
    return _flatten_tensor_grid(
        _grid_tensors_3d(active, layers=layers, rows=rows, cols=cols, prefix="PY")
    )


def _random_irregular() -> Any:
    rng = random.Random(1729)
    node_names = tuple(f"R{index:02d}" for index in range(42))
    edges: set[tuple[str, str]] = {
        (f"R{index:02d}", f"R{index + 1:02d}") for index in range(len(node_names) - 1)
    }
    degree = dict.fromkeys(node_names, 0)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    while len(edges) < 72:
        left_index = rng.randrange(len(node_names))
        right_index = rng.randrange(len(node_names))
        if left_index == right_index:
            continue
        left, right = sorted((node_names[left_index], node_names[right_index]))
        if (left, right) in edges or degree[left] >= 5 or degree[right] >= 5:
            continue
        edges.add((left, right))
        degree[left] += 1
        degree[right] += 1
    return _network_from_edges(node_names, tuple(sorted(edges)))


def _circular_chords() -> Any:
    node_names = tuple(f"C{index:02d}" for index in range(36))
    edges: set[tuple[str, str]] = {
        tuple(sorted((f"C{index:02d}", f"C{(index + 1) % 36:02d}"))) for index in range(36)
    }
    edges.update(
        tuple(sorted((f"C{index:02d}", f"C{(index + 9) % 36:02d}"))) for index in range(0, 36, 3)
    )
    edges.update(
        tuple(sorted((f"C{index:02d}", f"C{(index + 17) % 36:02d}"))) for index in range(0, 36, 6)
    )
    return _network_from_edges(node_names, tuple(sorted(edges)))


def _circular_ring() -> list[Any]:
    node_names = tuple(f"O{index:02d}" for index in range(28))
    edges = tuple(
        sorted((f"O{index:02d}", f"O{(index + 1) % len(node_names):02d}"))
        for index in range(len(node_names))
    )
    return _network_from_edges(node_names, tuple(edges))


def _tubular_grid() -> list[Any]:
    periodic, length = 12, 5
    node_names = tuple(
        f"TB_{theta}_{z_index}" for theta in range(periodic) for z_index in range(length)
    )
    edges: list[tuple[str, str]] = []
    for theta in range(periodic):
        for z_index in range(length):
            edges.append(
                (
                    f"TB_{theta}_{z_index}",
                    f"TB_{(theta + 1) % periodic}_{z_index}",
                )
            )
            if z_index < length - 1:
                edges.append((f"TB_{theta}_{z_index}", f"TB_{theta}_{z_index + 1}"))
    return _network_from_edges(node_names, tuple(sorted(tuple(sorted(edge)) for edge in edges)))


def _disconnected_irregular() -> Any:
    component_a = tuple(f"A{index:02d}" for index in range(14))
    component_b = tuple(f"B{index:02d}" for index in range(10))
    component_c = tuple(f"C{index:02d}" for index in range(12))
    component_d = tuple(f"D{index:02d}" for index in range(6))
    edges: list[tuple[str, str]] = []
    edges.extend((f"A{index:02d}", f"A{(index + 1) % 14:02d}") for index in range(14))
    edges.extend((f"A{index:02d}", f"A{(index + 5) % 14:02d}") for index in range(0, 14, 2))
    edges.extend(("B00", f"B{index:02d}") for index in range(1, 10))
    edges.extend((f"B{index:02d}", f"B{index + 1:02d}") for index in range(1, 9))
    edges.extend((f"C{index:02d}", f"C{index + 1:02d}") for index in range(11))
    edges.extend((f"C{index:02d}", f"C{index + 3:02d}") for index in range(0, 9, 3))
    edges.extend((("D00", "D01"), ("D01", "D02"), ("D02", "D00")))
    edges.extend(("D02", f"D{index:02d}") for index in range(3, 6))
    return _network_from_edges(
        (*component_a, *component_b, *component_c, *component_d),
        tuple(edges),
    )


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    if definition.name == "partial_grid2d":
        network: Any = _partial_grid2d()
    elif definition.name == "decorated_sparse_grid2d":
        network = _decorated_sparse_grid2d()
    elif definition.name == "upper_triangle2d":
        network = _upper_triangle2d()
    elif definition.name == "partial_grid3d":
        network = _partial_grid3d()
    elif definition.name == "upper_pyramid3d":
        network = _upper_pyramid3d()
    elif definition.name == "random_irregular":
        network = _random_irregular()
    elif definition.name == "circular_ring":
        network = _circular_ring()
    elif definition.name == "circular_chords":
        network = _circular_chords()
    elif definition.name == "tubular_grid":
        network = _tubular_grid()
    elif definition.name == "disconnected_irregular":
        network = _disconnected_irregular()
    else:
        raise ValueError(f"Unsupported geometry example: {definition.name}")
    return BuiltExample(
        network=network,
        plot_engine="quimb",
        title=f"Geometry - {definition.name.replace('_', ' ')} - {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer="Large enough to stress layout behavior while staying quick for interactive checks.",
        scheme_steps_by_name=None,
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition("partial_grid2d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition(
        "decorated_sparse_grid2d",
        (),
        frozenset(),
        True,
        False,
        False,
        _build_example,
    ),
    ExampleDefinition("upper_triangle2d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("partial_grid3d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("upper_pyramid3d", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("random_irregular", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("circular_ring", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("circular_chords", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition("tubular_grid", (), frozenset(), True, False, False, _build_example),
    ExampleDefinition(
        "disconnected_irregular",
        (),
        frozenset(),
        True,
        False,
        False,
        _build_example,
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported geometry example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(args, engine="quimb", scheme_tensor_names=None)
    config = replace(config, tensor_label_fontsize=7.0, layout_iterations=360)
    if definition.name == "decorated_sparse_grid2d":
        config = replace(
            config,
            show_nodes=True,
            show_tensor_labels=False,
            node_color="#8CA2B0",
            node_edge_color="#111827",
            node_color_degree_one="#D86A76",
            node_edge_color_degree_one="#111827",
        )
    fig, _ax = show_tensor_network(
        built.network,
        engine="quimb",
        view=args.view,
        config=config,
        show_controls=not demo_runs_headless(args),
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
