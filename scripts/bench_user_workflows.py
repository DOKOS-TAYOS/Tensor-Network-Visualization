#!/usr/bin/env python3
"""Benchmark user-visible tensor workflows with GUI-aware subprocess isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
for import_path in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

SurfaceName: TypeAlias = Literal["tensor-elements", "tensor-network"]
SizeLevel: TypeAlias = Literal["small", "medium", "large"]
TemperatureName: TypeAlias = Literal["cold", "hot"]
GuiBackendPreference: TypeAlias = Literal["auto", "tkagg", "agg"]

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ScenarioSpec:
    surface: SurfaceName
    backend: str
    case: str
    size_level: SizeLevel
    action: str
    structure_kind: str
    parameters: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> ScenarioSpec:
        return cls(
            surface=cast(SurfaceName, str(payload["surface"])),
            backend=str(payload["backend"]),
            case=str(payload["case"]),
            size_level=cast(SizeLevel, str(payload["size_level"])),
            action=str(payload["action"]),
            structure_kind=str(payload["structure_kind"]),
            parameters=dict(cast(dict[str, object], payload.get("parameters", {}))),
        )


@dataclass(frozen=True)
class MeasurementResult:
    surface: SurfaceName
    backend: str
    case: str
    size_level: SizeLevel
    temperature: TemperatureName
    action: str
    wall_ms: float
    cpu_ms: float
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float
    approximate_gui: bool
    notes: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_row(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "backend": self.backend,
            "case": self.case,
            "size_level": self.size_level,
            "temperature": self.temperature,
            "action": self.action,
            "wall_ms": self.wall_ms,
            "cpu_ms": self.cpu_ms,
            "rss_before_mb": self.rss_before_mb,
            "rss_after_mb": self.rss_after_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "approximate_gui": self.approximate_gui,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class _TensorElementsCaseDefinition:
    case: str
    structure_kind: str
    shape_by_size: dict[SizeLevel, tuple[int, ...]]
    complex_values: bool = False


@dataclass(frozen=True)
class _TensorNetworkCaseDefinition:
    case: str
    backend: str
    structure_kind: str
    size_parameters: dict[SizeLevel, dict[str, int]]


@dataclass(frozen=True)
class _PreparedNetworkCase:
    network: object
    engine: str
    scheme_steps: tuple[tuple[str, ...], ...] | None


@dataclass
class _PeakRssSampler:
    process: Any
    interval_s: float = 0.01

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb: float = _rss_mb(self.process)

    def start(self) -> None:
        def _run() -> None:
            while not self._stop_event.wait(self.interval_s):
                self.peak_rss_mb = max(self.peak_rss_mb, _rss_mb(self.process))

        self._thread = threading.Thread(target=_run, name="peak-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_mb = max(self.peak_rss_mb, _rss_mb(self.process))
        return self.peak_rss_mb


_TENSOR_ELEMENTS_CASES: tuple[_TensorElementsCaseDefinition, ...] = (
    _TensorElementsCaseDefinition(
        case="real_dense",
        structure_kind="matrix",
        shape_by_size={
            "small": (64, 64),
            "medium": (256, 256),
            "large": (384, 384),
        },
    ),
    _TensorElementsCaseDefinition(
        case="complex_square",
        structure_kind="matrix",
        shape_by_size={
            "small": (48, 48),
            "medium": (128, 128),
            "large": (256, 256),
        },
        complex_values=True,
    ),
    _TensorElementsCaseDefinition(
        case="rank3_analysis",
        structure_kind="tensor3",
        shape_by_size={
            "small": (12, 16, 20),
            "medium": (24, 32, 48),
            "large": (56, 72, 96),
        },
    ),
)

_TENSOR_NETWORK_CASES: tuple[_TensorNetworkCaseDefinition, ...] = (
    _TensorNetworkCaseDefinition(
        case="linear_einsum_mps",
        backend="einsum",
        structure_kind="linear",
        size_parameters={
            "small": {"n_sites": 6},
            "medium": {"n_sites": 12},
            "large": {"n_sites": 24},
        },
    ),
    _TensorNetworkCaseDefinition(
        case="circular_quimb_ring",
        backend="quimb",
        structure_kind="circular",
        size_parameters={
            "small": {"n_nodes": 16},
            "medium": {"n_nodes": 28},
            "large": {"n_nodes": 40},
        },
    ),
    _TensorNetworkCaseDefinition(
        case="planar_quimb_grid",
        backend="quimb",
        structure_kind="planar",
        size_parameters={
            "small": {"rows": 3, "cols": 4},
            "medium": {"rows": 4, "cols": 6},
            "large": {"rows": 6, "cols": 8},
        },
    ),
    _TensorNetworkCaseDefinition(
        case="generic_quimb_irregular",
        backend="quimb",
        structure_kind="generic",
        size_parameters={
            "small": {"n_nodes": 18, "target_edges": 28},
            "medium": {"n_nodes": 30, "target_edges": 50},
            "large": {"n_nodes": 42, "target_edges": 72},
        },
    ),
)

_RESULT_FIELDNAMES: tuple[str, ...] = (
    "surface",
    "backend",
    "case",
    "size_level",
    "temperature",
    "action",
    "wall_ms",
    "cpu_ms",
    "rss_before_mb",
    "rss_after_mb",
    "peak_rss_mb",
    "approximate_gui",
    "notes",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark user-visible tensor workflows with isolated subprocesses.",
    )
    parser.add_argument(
        "--surface",
        choices=("tensor-elements", "tensor-network", "all"),
        default="all",
        help="Benchmark surface to run. Defaults to all.",
    )
    parser.add_argument(
        "--size",
        choices=("small", "medium", "large", "all"),
        default="all",
        help="Scenario size level to run. Defaults to all.",
    )
    parser.add_argument(
        "--temperature",
        choices=("cold", "hot", "all"),
        default="all",
        help="Run cold, hot, or both measurement styles. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/profiling"),
        help="Directory where results.json and results.csv are written.",
    )
    parser.add_argument(
        "--gui-backend",
        choices=("auto", "tkagg", "agg"),
        default="auto",
        help="GUI backend preference for worker subprocesses.",
    )
    parser.add_argument(
        "--cold-samples",
        type=int,
        default=3,
        help="Fresh subprocess samples used for cold medians.",
    )
    parser.add_argument(
        "--hot-repeats",
        type=int,
        default=5,
        help="Timed repeats used after one warm-up in a hot worker.",
    )
    parser.add_argument("--case-filter", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--action-filter", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-temperature",
        choices=("hot",),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-gui-backend",
        choices=("auto", "tkagg", "agg"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-repeats", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmup", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _size_levels(size: str) -> tuple[SizeLevel, ...]:
    if size == "all":
        return ("small", "medium", "large")
    return (cast(SizeLevel, size),)


def _temperatures(temperature: str) -> tuple[TemperatureName, ...]:
    if temperature == "all":
        return ("cold", "hot")
    return (cast(TemperatureName, temperature),)


def _seed_from_name_and_shape(name: str, shape: tuple[int, ...]) -> int:
    digest = hashlib.sha256(f"{name}|{shape}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _demo_array(
    *,
    name: str,
    shape: tuple[int, ...],
    complex_values: bool = False,
) -> Any:
    import numpy as np

    rng = np.random.default_rng(_seed_from_name_and_shape(name, shape))
    real_part = rng.standard_normal(shape)
    if not complex_values:
        return real_part.astype(np.float64)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def _bundle_arrays(
    *,
    name_prefix: str,
    shape: tuple[int, ...],
    complex_values: bool,
) -> list[Any]:
    return [
        _demo_array(name=f"{name_prefix}_{index}", shape=shape, complex_values=complex_values)
        for index in range(3)
    ]


def _tensor_elements_case_definition(case: str) -> _TensorElementsCaseDefinition:
    for definition in _TENSOR_ELEMENTS_CASES:
        if definition.case == case:
            return definition
    raise ValueError(f"Unsupported tensor-elements case: {case!r}.")


def _network_case_definition(case: str) -> _TensorNetworkCaseDefinition:
    for definition in _TENSOR_NETWORK_CASES:
        if definition.case == case:
            return definition
    raise ValueError(f"Unsupported tensor-network case: {case!r}.")


def _tensor_elements_case_data(case: str, size_level: SizeLevel) -> tuple[Any, Any]:
    from tensor_network_viz import TensorElementsConfig

    definition = _tensor_elements_case_definition(case)
    shape = definition.shape_by_size[size_level]
    bundle = _bundle_arrays(
        name_prefix=f"{case}_{size_level}",
        shape=shape,
        complex_values=definition.complex_values,
    )
    config = TensorElementsConfig(
        mode="auto",
        figsize=(7.4, 6.4),
        max_matrix_shape=(384, 384),
        shared_color_scale=True,
        robust_percentiles=(1.0, 99.0),
        topk_count=10,
    )
    return bundle, config


def _tensor_elements_actions_for_case(case: str, size_level: SizeLevel) -> tuple[str, ...]:
    from tensor_network_viz._tensor_elements_support import (
        _extract_tensor_records,
        _valid_group_modes_for_record,
    )

    data, config = _tensor_elements_case_data(case, size_level)
    _engine, records = _extract_tensor_records(data, engine=None)
    record = records[0]
    actions: list[str] = ["initial_render"]
    for group in ("basic", "complex", "diagnostic", "analysis"):
        modes = _valid_group_modes_for_record(record, group, config=config)
        for index in range(len(modes) - 1):
            actions.append(f"mode:{modes[index]}->{modes[index + 1]}")
    analysis_modes = set(_valid_group_modes_for_record(record, "analysis", config=config))
    if "slice" in analysis_modes:
        actions.append("analysis:slice_index+1")
    if "reduce" in analysis_modes:
        actions.append("analysis:reduce_mean->norm")
    if "profiles" in analysis_modes:
        actions.append("analysis:profiles_mean->norm")
    if len(records) > 1:
        actions.append("tensor_slider:0->1")
    return tuple(actions)


def _enumerate_scenarios(
    *,
    surface: str,
    size: str,
    case_filter: str | None = None,
    action_filter: str | None = None,
) -> tuple[ScenarioSpec, ...]:
    scenarios: list[ScenarioSpec] = []
    include_tensor_elements = surface in {"tensor-elements", "all"}
    include_tensor_network = surface in {"tensor-network", "all"}

    if include_tensor_elements:
        for definition in _TENSOR_ELEMENTS_CASES:
            if case_filter is not None and definition.case != case_filter:
                continue
            for size_level in _size_levels(size):
                for action in _tensor_elements_actions_for_case(definition.case, size_level):
                    if action_filter is not None and action != action_filter:
                        continue
                    scenarios.append(
                        ScenarioSpec(
                            surface="tensor-elements",
                            backend="numpy",
                            case=definition.case,
                            size_level=size_level,
                            action=action,
                            structure_kind=definition.structure_kind,
                            parameters={},
                        )
                    )

    if include_tensor_network:
        base_actions = (
            "static_clean",
            "interactive_baseline",
            "tensor_labels",
            "edge_labels",
            "diagnostics",
            "view_3d_initial",
            "toggle:hover",
            "toggle:nodes",
            "toggle:tensor_labels",
            "toggle:edge_labels",
            "toggle:diagnostics",
            "view:2d->3d",
            "view:3d->2d",
        )
        einsum_extra_actions = (
            "scheme",
            "scheme+costs",
            "scheme:off->on",
            "costs:off->on",
            "inspector_open",
        )
        for definition in _TENSOR_NETWORK_CASES:
            if case_filter is not None and definition.case != case_filter:
                continue
            for size_level in _size_levels(size):
                action_names = base_actions
                if definition.backend == "einsum":
                    action_names = (*action_names, *einsum_extra_actions)
                for action in action_names:
                    if action_filter is not None and action != action_filter:
                        continue
                    scenarios.append(
                        ScenarioSpec(
                            surface="tensor-network",
                            backend=definition.backend,
                            case=definition.case,
                            size_level=size_level,
                            action=action,
                            structure_kind=definition.structure_kind,
                            parameters=dict(definition.size_parameters[size_level]),
                        )
                    )

    return tuple(scenarios)


def _linear_scheme_steps(n_sites: int) -> tuple[tuple[str, ...], ...]:
    steps: list[tuple[str, ...]] = []
    running: list[str] = []
    for index in range(n_sites):
        running.extend((f"A{index}", f"x{index}"))
        steps.append(tuple(running))
    return tuple(steps)


def _build_linear_einsum_trace(n_sites: int) -> tuple[Any, tuple[tuple[str, ...], ...]]:
    from tensor_network_viz import EinsumTrace, einsum

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]
    numpy: Any | None = None
    if torch is None:
        import numpy as np

        numpy = np

    if n_sites < 1:
        raise ValueError("n_sites must be >= 1.")
    trace = EinsumTrace()
    bond_dims = [2 + (index % 3) for index in range(max(n_sites - 1, 1))]
    backend = "numpy" if torch is None else "torch"

    def _tensor(name: str, shape: tuple[int, ...]) -> Any:
        array = _demo_array(name=name, shape=shape, complex_values=False)
        if torch is None:
            return array
        return torch.tensor(array, dtype=torch.float32)

    def _scalar_out(reference: Any) -> Any | None:
        if numpy is None:
            return None
        return numpy.empty((), dtype=getattr(reference, "dtype", numpy.float64))

    def _einsum_with_trace(
        expression: str,
        *operands: Any,
        out: Any | None = None,
    ) -> Any:
        if out is None:
            return einsum(expression, *operands, trace=trace, backend=backend)
        return einsum(expression, *operands, trace=trace, backend=backend, out=out)

    if n_sites == 1:
        tensor = _tensor("A0", (2,))
        vector = _tensor("x0", (2,))
        trace.bind("A0", tensor)
        trace.bind("x0", vector)
        result = _einsum_with_trace("p,p->", tensor, vector, out=_scalar_out(tensor))
        trace._bench_keepalive = [tensor, vector, result]  # type: ignore[attr-defined]
        return trace, (("A0", "x0"),)

    first_tensor = _tensor("A0", (2, bond_dims[0]))
    first_vector = _tensor("x0", (2,))
    trace.bind("A0", first_tensor)
    trace.bind("x0", first_vector)
    keepalive: list[Any] = [first_tensor, first_vector]
    current = _einsum_with_trace("pa,p->a", first_tensor, first_vector)
    keepalive.append(current)

    for index in range(1, n_sites - 1):
        tensor = _tensor(
            f"A{index}",
            (bond_dims[index - 1], 2, bond_dims[index]),
        )
        vector = _tensor(f"x{index}", (2,))
        trace.bind(f"A{index}", tensor)
        trace.bind(f"x{index}", vector)
        keepalive.extend((tensor, vector))
        current = _einsum_with_trace("a,apb,p->b", current, tensor, vector)
        keepalive.append(current)

    last_tensor = _tensor(f"A{n_sites - 1}", (bond_dims[n_sites - 2], 2))
    last_vector = _tensor(f"x{n_sites - 1}", (2,))
    trace.bind(f"A{n_sites - 1}", last_tensor)
    trace.bind(f"x{n_sites - 1}", last_vector)
    keepalive.extend((last_tensor, last_vector))
    result = _einsum_with_trace(
        "a,ap,p->",
        current,
        last_tensor,
        last_vector,
        out=_scalar_out(current),
    )
    keepalive.append(result)
    trace._bench_keepalive = keepalive  # type: ignore[attr-defined]
    return trace, _linear_scheme_steps(n_sites)


def _quimb_tensor_network_from_edges(
    *,
    node_names: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> object:
    import quimb.tensor as qtn

    indices_by_node: dict[str, list[str]] = {name: [f"obs_{name}"] for name in node_names}
    for edge_index, (left, right) in enumerate(edges):
        index_name = f"bond_{edge_index}"
        indices_by_node[left].append(index_name)
        indices_by_node[right].append(index_name)

    tensors: list[object] = []
    for name in node_names:
        indices = tuple(indices_by_node[name])
        shape = tuple(2 if index_name.startswith("obs_") else 3 for index_name in indices)
        tensors.append(
            qtn.Tensor(
                data=_demo_array(name=f"quimb_{name}", shape=shape, complex_values=False),
                inds=indices,
                tags={name},
            )
        )
    return qtn.TensorNetwork(tensors)


def _sorted_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _build_circular_quimb_network(n_nodes: int) -> object:
    node_names = tuple(f"C{index:02d}" for index in range(n_nodes))
    edges = tuple(
        _sorted_edge(node_names[index], node_names[(index + 1) % n_nodes])
        for index in range(n_nodes)
    )
    return _quimb_tensor_network_from_edges(node_names=node_names, edges=edges)


def _build_planar_quimb_network(*, rows: int, cols: int) -> object:
    node_names = tuple(f"P{row}_{col}" for row in range(rows) for col in range(cols))
    edges: list[tuple[str, str]] = []
    for row in range(rows):
        for col in range(cols):
            current = f"P{row}_{col}"
            if row + 1 < rows:
                edges.append((current, f"P{row + 1}_{col}"))
            if col + 1 < cols:
                edges.append((current, f"P{row}_{col + 1}"))
    return _quimb_tensor_network_from_edges(
        node_names=node_names,
        edges=tuple(_sorted_edge(left, right) for left, right in edges),
    )


def _build_irregular_quimb_network(*, n_nodes: int, target_edges: int) -> object:
    rng = random.Random(_seed_from_name_and_shape("irregular", (n_nodes, target_edges)))
    node_names = tuple(f"R{index:02d}" for index in range(n_nodes))
    edge_set: set[tuple[str, str]] = {
        _sorted_edge(node_names[index], node_names[index + 1]) for index in range(n_nodes - 1)
    }
    degrees = dict.fromkeys(node_names, 0)
    for left, right in edge_set:
        degrees[left] += 1
        degrees[right] += 1

    while len(edge_set) < target_edges:
        left_index = rng.randrange(n_nodes)
        right_index = rng.randrange(n_nodes)
        if left_index == right_index:
            continue
        left, right = _sorted_edge(node_names[left_index], node_names[right_index])
        if (left, right) in edge_set:
            continue
        if degrees[left] >= 5 or degrees[right] >= 5:
            continue
        edge_set.add((left, right))
        degrees[left] += 1
        degrees[right] += 1
    return _quimb_tensor_network_from_edges(node_names=node_names, edges=tuple(sorted(edge_set)))


def _network_case_data(case: str, size_level: SizeLevel) -> _PreparedNetworkCase:
    definition = _network_case_definition(case)
    parameters = definition.size_parameters[size_level]
    if case == "linear_einsum_mps":
        trace, scheme_steps = _build_linear_einsum_trace(int(parameters["n_sites"]))
        return _PreparedNetworkCase(network=trace, engine="einsum", scheme_steps=scheme_steps)
    if case == "circular_quimb_ring":
        return _PreparedNetworkCase(
            network=_build_circular_quimb_network(int(parameters["n_nodes"])),
            engine="quimb",
            scheme_steps=None,
        )
    if case == "planar_quimb_grid":
        return _PreparedNetworkCase(
            network=_build_planar_quimb_network(
                rows=int(parameters["rows"]),
                cols=int(parameters["cols"]),
            ),
            engine="quimb",
            scheme_steps=None,
        )
    if case == "generic_quimb_irregular":
        return _PreparedNetworkCase(
            network=_build_irregular_quimb_network(
                n_nodes=int(parameters["n_nodes"]),
                target_edges=int(parameters["target_edges"]),
            ),
            engine="quimb",
            scheme_steps=None,
        )
    raise ValueError(f"Unsupported tensor-network case: {case!r}.")


def _local_tkagg_available() -> bool:
    if sys.platform != "win32" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    try:
        import tkinter  # noqa: F401

        import matplotlib.backends.backend_tkagg  # noqa: F401
    except Exception:
        return False
    return True


def _tkagg_available() -> bool:
    return _local_tkagg_available()


def _resolve_gui_backend(preference: GuiBackendPreference) -> tuple[str, bool]:
    if preference == "agg":
        return "Agg", True
    if preference == "tkagg":
        return ("TkAgg", False) if _local_tkagg_available() else ("Agg", True)
    if _local_tkagg_available():
        return "TkAgg", False
    return "Agg", True


def _rss_mb(process: Any) -> float:
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _cpu_ms(process: Any) -> float:
    cpu_times = process.cpu_times()
    return float(cpu_times.user + cpu_times.system) * 1000.0


def _measure_callable(action: Any) -> tuple[float, float, float, float, float]:
    import psutil

    process = psutil.Process(os.getpid())
    rss_before_mb = _rss_mb(process)
    cpu_before_ms = _cpu_ms(process)
    sampler = _PeakRssSampler(process)
    sampler.start()
    started = time.perf_counter()
    action()
    wall_ms = (time.perf_counter() - started) * 1000.0
    peak_rss_mb = sampler.stop()
    cpu_after_ms = _cpu_ms(process)
    rss_after_mb = _rss_mb(process)
    return (
        wall_ms,
        cpu_after_ms - cpu_before_ms,
        rss_before_mb,
        rss_after_mb,
        peak_rss_mb,
    )


def _flush_canvas(figure: Figure, *, resolved_backend: str, reveal: bool) -> None:
    draw = getattr(figure.canvas, "draw", None)
    if callable(draw):
        draw()
    manager = getattr(figure.canvas, "manager", None)
    show = getattr(manager, "show", None)
    if reveal and resolved_backend == "TkAgg" and callable(show):
        show()
    flush_events = getattr(figure.canvas, "flush_events", None)
    if callable(flush_events):
        flush_events()


def _dispose_figure(figure: Figure | None, *, resolved_backend: str) -> None:
    if figure is None:
        return
    if resolved_backend == "TkAgg":
        from matplotlib._pylab_helpers import Gcf

        manager = getattr(figure.canvas, "manager", None)
        if manager is not None:
            window = getattr(manager, "window", None)
            if window is not None:
                window.destroy()
            manager_num = getattr(manager, "num", None)
            if isinstance(manager_num, int):
                Gcf.figs.pop(manager_num, None)
        return
    import matplotlib.pyplot as plt

    plt.close(figure)


def _median_result(
    results: list[MeasurementResult],
    *,
    temperature: TemperatureName,
) -> MeasurementResult:
    if not results:
        raise ValueError("Cannot aggregate an empty result list.")
    first = results[0]
    return MeasurementResult(
        surface=first.surface,
        backend=first.backend,
        case=first.case,
        size_level=first.size_level,
        temperature=temperature,
        action=first.action,
        wall_ms=float(statistics.median(result.wall_ms for result in results)),
        cpu_ms=float(statistics.median(result.cpu_ms for result in results)),
        rss_before_mb=float(statistics.median(result.rss_before_mb for result in results)),
        rss_after_mb=float(statistics.median(result.rss_after_mb for result in results)),
        peak_rss_mb=float(statistics.median(result.peak_rss_mb for result in results)),
        approximate_gui=all(result.approximate_gui for result in results),
        notes=first.notes,
    )


def _tensor_elements_mode_transition(controller: object, transition: str) -> Any:
    source_mode, dest_mode = transition.split("->", maxsplit=1)

    def _timed_action() -> None:
        cast(Any, controller).set_mode(dest_mode)

    cast(Any, controller).set_mode(source_mode, redraw=False)
    return _timed_action


def _execute_tensor_elements_scenario(
    scenario: ScenarioSpec,
    *,
    resolved_backend: str,
    approximate_gui: bool,
) -> MeasurementResult:
    from tensor_network_viz import show_tensor_elements

    figures: list[Figure] = []

    def _initial_render_action() -> None:
        data, config = _tensor_elements_case_data(scenario.case, scenario.size_level)
        figure, _ax = show_tensor_elements(
            data,
            config=config,
            show_controls=True,
            show=False,
        )
        fig = cast(Figure, figure)
        figures.append(fig)
        _flush_canvas(fig, resolved_backend=resolved_backend, reveal=True)

    def _setup_controller() -> tuple[Figure, Any]:
        data, config = _tensor_elements_case_data(scenario.case, scenario.size_level)
        figure, _ax = show_tensor_elements(
            data,
            config=config,
            show_controls=True,
            show=False,
        )
        fig = cast(Figure, figure)
        figures.append(fig)
        _flush_canvas(fig, resolved_backend=resolved_backend, reveal=True)
        controller = getattr(fig, "_tensor_network_viz_tensor_elements_controls", None)
        if controller is None:
            raise RuntimeError("Tensor-elements controls were not attached to the figure.")
        return fig, controller

    try:
        if scenario.action == "initial_render":
            wall_ms, cpu_ms, rss_before_mb, rss_after_mb, peak_rss_mb = _measure_callable(
                _initial_render_action
            )
        else:
            figure, controller = _setup_controller()
            if scenario.action.startswith("mode:"):
                timed_action = _tensor_elements_mode_transition(
                    controller,
                    scenario.action[len("mode:") :],
                )
            elif scenario.action == "analysis:slice_index+1":
                cast(Any, controller).set_mode("slice", redraw=False)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    slider = getattr(controller, "_analysis_slider", None)
                    if slider is None:
                        raise RuntimeError("Slice slider is unavailable for this tensor.")
                    slider.set_val(float(slider.val) + 1.0)

            elif scenario.action == "analysis:reduce_mean->norm":
                cast(Any, controller).set_mode("reduce", redraw=False)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    radio = getattr(controller, "_analysis_method_radio", None)
                    if radio is None:
                        raise RuntimeError("Reduce method radio is unavailable for this tensor.")
                    radio.set_active(1)

            elif scenario.action == "analysis:profiles_mean->norm":
                cast(Any, controller).set_mode("profiles", redraw=False)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    radio = getattr(controller, "_analysis_method_radio", None)
                    if radio is None:
                        raise RuntimeError("Profiles method radio is unavailable for this tensor.")
                    radio.set_active(1)

            elif scenario.action == "tensor_slider:0->1":
                cast(Any, controller).set_tensor_index(0, redraw=False)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    cast(Any, controller).set_tensor_index(1)

            else:
                raise ValueError(f"Unsupported tensor-elements action: {scenario.action!r}.")

            wall_ms, cpu_ms, rss_before_mb, rss_after_mb, peak_rss_mb = _measure_callable(
                timed_action
            )
        return MeasurementResult(
            surface="tensor-elements",
            backend=scenario.backend,
            case=scenario.case,
            size_level=scenario.size_level,
            temperature="hot",
            action=scenario.action,
            wall_ms=wall_ms,
            cpu_ms=cpu_ms,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_mb,
            peak_rss_mb=peak_rss_mb,
            approximate_gui=approximate_gui,
            notes=f"resolved_backend={resolved_backend}",
        )
    finally:
        while figures:
            _dispose_figure(figures.pop(), resolved_backend=resolved_backend)


def _baseline_network_config(
    *,
    scheme_steps: tuple[tuple[str, ...], ...] | None,
) -> Any:
    from tensor_network_viz import PlotConfig, TensorNetworkDiagnosticsConfig

    return PlotConfig(
        show_nodes=True,
        show_tensor_labels=False,
        show_index_labels=False,
        hover_labels=False,
        diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=False),
        contraction_scheme_by_name=scheme_steps,
    )


def _initial_network_config(
    *,
    action: str,
    scheme_steps: tuple[tuple[str, ...], ...] | None,
) -> Any:
    from tensor_network_viz import PlotConfig, TensorNetworkDiagnosticsConfig

    return PlotConfig(
        show_nodes=True,
        show_tensor_labels=action == "tensor_labels",
        show_index_labels=action == "edge_labels",
        hover_labels=False,
        diagnostics=TensorNetworkDiagnosticsConfig(show_overlay=action == "diagnostics"),
        show_contraction_scheme=action in {"scheme", "scheme+costs"},
        contraction_scheme_cost_hover=action == "scheme+costs",
        contraction_scheme_by_name=scheme_steps,
    )


def _apply_scene_property(controller: Any, property_name: str, value: bool) -> None:
    setattr(controller, property_name, value)
    controller._apply_scene_state(controller.current_scene)


def _execute_tensor_network_scenario(
    scenario: ScenarioSpec,
    *,
    resolved_backend: str,
    approximate_gui: bool,
) -> MeasurementResult:
    from tensor_network_viz import show_tensor_network

    figures: list[Figure] = []

    def _initial_render_action() -> None:
        prepared = _network_case_data(scenario.case, scenario.size_level)
        if scenario.action == "static_clean":
            figure, _ax = show_tensor_network(
                prepared.network,
                engine=cast(Any, prepared.engine),
                config=_initial_network_config(
                    action=scenario.action,
                    scheme_steps=prepared.scheme_steps,
                ),
                show_controls=False,
                show=False,
            )
        else:
            initial_view = "3d" if scenario.action == "view_3d_initial" else "2d"
            figure, _ax = show_tensor_network(
                prepared.network,
                engine=cast(Any, prepared.engine),
                view=cast(Any, initial_view),
                config=_initial_network_config(
                    action=scenario.action,
                    scheme_steps=prepared.scheme_steps,
                ),
                show_controls=True,
                show=False,
            )
        fig = cast(Figure, figure)
        figures.append(fig)
        _flush_canvas(fig, resolved_backend=resolved_backend, reveal=True)

    def _setup_interactive_figure(
        *,
        initial_view: str = "2d",
    ) -> tuple[Figure, Any]:
        prepared = _network_case_data(scenario.case, scenario.size_level)
        figure, _ax = show_tensor_network(
            prepared.network,
            engine=cast(Any, prepared.engine),
            view=cast(Any, initial_view),
            config=_baseline_network_config(scheme_steps=prepared.scheme_steps),
            show_controls=True,
            show=False,
        )
        fig = cast(Figure, figure)
        figures.append(fig)
        _flush_canvas(fig, resolved_backend=resolved_backend, reveal=True)
        controller = getattr(fig, "_tensor_network_viz_interactive_controls", None)
        if controller is None:
            raise RuntimeError("Interactive controls were not attached to the network figure.")
        return fig, controller

    try:
        if scenario.action in {
            "static_clean",
            "interactive_baseline",
            "tensor_labels",
            "edge_labels",
            "diagnostics",
            "view_3d_initial",
            "scheme",
            "scheme+costs",
        }:
            wall_ms, cpu_ms, rss_before_mb, rss_after_mb, peak_rss_mb = _measure_callable(
                _initial_render_action
            )
        else:
            initial_view = "3d" if scenario.action == "view:3d->2d" else "2d"
            figure, controller = _setup_interactive_figure(initial_view=initial_view)
            if scenario.action == "toggle:hover":

                def timed_action() -> None:
                    controller.set_hover_enabled(True)

            elif scenario.action == "toggle:nodes":

                def timed_action() -> None:
                    controller.set_nodes_enabled(False)

            elif scenario.action == "toggle:tensor_labels":

                def timed_action() -> None:
                    controller.set_tensor_labels_enabled(True)

            elif scenario.action == "toggle:edge_labels":

                def timed_action() -> None:
                    controller.set_edge_labels_enabled(True)

            elif scenario.action == "toggle:diagnostics":

                def timed_action() -> None:
                    _apply_scene_property(controller, "diagnostics_on", True)

            elif scenario.action == "view:2d->3d":

                def timed_action() -> None:
                    controller.set_view("3d")

            elif scenario.action == "view:3d->2d":

                def timed_action() -> None:
                    controller.set_view("2d")

            elif scenario.action == "scheme:off->on":

                def timed_action() -> None:
                    _apply_scene_property(controller, "scheme_on", True)

            elif scenario.action == "costs:off->on":
                _apply_scene_property(controller, "scheme_on", True)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    _apply_scene_property(controller, "cost_hover_on", True)

            elif scenario.action == "inspector_open":
                _apply_scene_property(controller, "scheme_on", True)
                scene_controls = controller.current_scene.contraction_controls
                if scene_controls is None:
                    raise RuntimeError("Contraction controls are unavailable for inspector setup.")
                scene_controls.ensure_viewer()
                viewer = scene_controls._viewer
                if viewer is None:
                    raise RuntimeError("Contraction viewer is unavailable for inspector setup.")
                viewer.set_step(1)
                _flush_canvas(figure, resolved_backend=resolved_backend, reveal=False)

                def timed_action() -> None:
                    _apply_scene_property(controller, "tensor_inspector_on", True)
                    inspector = getattr(figure, "_tensor_network_viz_tensor_inspector", None)
                    if inspector is not None and inspector._figure is not None:
                        inspector_figure = cast(Figure, inspector._figure)
                        if inspector_figure not in figures:
                            figures.append(inspector_figure)
                        _flush_canvas(
                            inspector_figure,
                            resolved_backend=resolved_backend,
                            reveal=True,
                        )

            else:
                raise ValueError(f"Unsupported tensor-network action: {scenario.action!r}.")

            wall_ms, cpu_ms, rss_before_mb, rss_after_mb, peak_rss_mb = _measure_callable(
                timed_action
            )

        return MeasurementResult(
            surface="tensor-network",
            backend=scenario.backend,
            case=scenario.case,
            size_level=scenario.size_level,
            temperature="hot",
            action=scenario.action,
            wall_ms=wall_ms,
            cpu_ms=cpu_ms,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_mb,
            peak_rss_mb=peak_rss_mb,
            approximate_gui=approximate_gui,
            notes=f"resolved_backend={resolved_backend}; structure={scenario.structure_kind}",
        )
    finally:
        while figures:
            _dispose_figure(figures.pop(), resolved_backend=resolved_backend)


def _run_scenario_once(
    scenario: ScenarioSpec,
    *,
    resolved_backend: str,
    approximate_gui: bool,
) -> MeasurementResult:
    if scenario.surface == "tensor-elements":
        return _execute_tensor_elements_scenario(
            scenario,
            resolved_backend=resolved_backend,
            approximate_gui=approximate_gui,
        )
    return _execute_tensor_network_scenario(
        scenario,
        resolved_backend=resolved_backend,
        approximate_gui=approximate_gui,
    )


def _run_worker_samples(
    *,
    scenario: ScenarioSpec,
    gui_backend: GuiBackendPreference,
    warmup: int,
    repeats: int,
) -> MeasurementResult:
    resolved_backend, approximate_gui = _resolve_gui_backend(gui_backend)
    import matplotlib

    matplotlib.use(resolved_backend, force=True)

    for _ in range(warmup):
        _run_scenario_once(
            scenario,
            resolved_backend=resolved_backend,
            approximate_gui=approximate_gui,
        )

    sample_results = [
        _run_scenario_once(
            scenario,
            resolved_backend=resolved_backend,
            approximate_gui=approximate_gui,
        )
        for _ in range(repeats)
    ]
    return _median_result(sample_results, temperature="hot")


def _run_worker_subprocess(
    scenario: ScenarioSpec,
    *,
    temperature: TemperatureName,
    gui_backend: GuiBackendPreference,
    cold_samples: int,
    hot_repeats: int,
) -> MeasurementResult:
    if temperature == "cold":
        fresh_results = [
            _run_worker_subprocess(
                scenario,
                temperature="hot",
                gui_backend=gui_backend,
                cold_samples=1,
                hot_repeats=1,
            )
            for _ in range(cold_samples)
        ]
        return _median_result(fresh_results, temperature="cold")

    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-scenario-json",
        json.dumps(scenario.to_json_dict(), separators=(",", ":")),
        "--worker-temperature",
        "hot",
        "--worker-gui-backend",
        gui_backend,
        "--worker-warmup",
        "1",
        "--worker-repeats",
        str(hot_repeats),
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    json_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not json_lines:
        raise RuntimeError("Worker subprocess did not emit any JSON result.")
    payload = json.loads(json_lines[-1])
    return MeasurementResult(
        surface=cast(SurfaceName, str(payload["surface"])),
        backend=str(payload["backend"]),
        case=str(payload["case"]),
        size_level=cast(SizeLevel, str(payload["size_level"])),
        temperature=cast(TemperatureName, str(payload["temperature"])),
        action=str(payload["action"]),
        wall_ms=float(payload["wall_ms"]),
        cpu_ms=float(payload["cpu_ms"]),
        rss_before_mb=float(payload["rss_before_mb"]),
        rss_after_mb=float(payload["rss_after_mb"]),
        peak_rss_mb=float(payload["peak_rss_mb"]),
        approximate_gui=bool(payload["approximate_gui"]),
        notes=str(payload["notes"]),
    )


def _write_results(
    results: list[MeasurementResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(
        json.dumps([result.to_json_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RESULT_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    return json_path, csv_path


def _run_parent(args: argparse.Namespace) -> int:
    scenarios = _enumerate_scenarios(
        surface=str(args.surface),
        size=str(args.size),
        case_filter=cast(str | None, args.case_filter),
        action_filter=cast(str | None, args.action_filter),
    )
    if not scenarios:
        print("[bench] No scenarios matched the selected filters.")
        return 0

    results: list[MeasurementResult] = []
    total = len(scenarios) * len(_temperatures(str(args.temperature)))
    completed_count = 0
    for scenario in scenarios:
        for temperature in _temperatures(str(args.temperature)):
            completed_count += 1
            print(
                f"[bench] {completed_count}/{total} "
                f"{scenario.surface} {scenario.case} {scenario.size_level} "
                f"{scenario.action} ({temperature})"
            )
            results.append(
                _run_worker_subprocess(
                    scenario,
                    temperature=temperature,
                    gui_backend=cast(GuiBackendPreference, str(args.gui_backend)),
                    cold_samples=int(args.cold_samples),
                    hot_repeats=int(args.hot_repeats),
                )
            )

    json_path, csv_path = _write_results(results, Path(args.output_dir))
    print(f"[bench] wrote {json_path}")
    print(f"[bench] wrote {csv_path}")
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    payload = json.loads(str(args.worker_scenario_json))
    scenario = ScenarioSpec.from_json_dict(cast(dict[str, object], payload))
    result = _run_worker_samples(
        scenario=scenario,
        gui_backend=cast(GuiBackendPreference, str(args.worker_gui_backend)),
        warmup=int(args.worker_warmup),
        repeats=int(args.worker_repeats),
    )
    print(json.dumps(result.to_json_dict(), separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.worker_scenario_json is not None:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
