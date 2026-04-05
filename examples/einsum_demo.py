from __future__ import annotations

import string
import sys
from dataclasses import dataclass
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
    apply_demo_caption,
    cumulative_prefix_contraction_scheme,
    demo_runs_headless,
    ensure_minimum,
    finalize_demo_plot_config,
    render_demo_tensor_network,
    resolve_example_definition,
)
from demo_tensors import build_demo_torch_tensor

TAGLINES: dict[str, str] = {
    "batch": "Hadamard-style batch contraction with kept indices.",
    "disconnected": "One trace containing several independent components.",
    "ellipsis": "Batched matmul using ellipsis-aware tracing.",
    "implicit_out": "Implicit output plus an explicit out= buffer.",
    "mps": "Sequential MPS-style contraction trace.",
    "mpo": "Sequential MPO-style contraction trace.",
    "nway": "Multi-step fusion across several tensors.",
    "peps": "Row-major PEPS contraction trace.",
    "ternary": "Single ternary einsum step.",
    "trace": "Diagonal / trace-like contraction.",
    "unary": "Single-operand einsum trace step.",
}


@dataclass(frozen=True)
class _PepsSiteData:
    tensor_name: str
    tensor_labels: tuple[str, ...]
    tensor_shape: tuple[int, ...]
    vector_name: str
    vector_shape: tuple[int, ...]
    phys_label: str


def _torch() -> Any:
    import torch

    return torch


def _einsum_api() -> tuple[Any, Any, Any, Any]:
    from tensor_network_viz import EinsumTrace, einsum, einsum_trace_step, pair_tensor

    return EinsumTrace, einsum, einsum_trace_step, pair_tensor


def _keep_trace_tensors_alive(trace: Any, *tensors: Any) -> None:
    keepalive = list(getattr(trace, "_example_keepalive", ()))
    keepalive.extend(tensor for tensor in tensors if tensor is not None)
    trace._example_keepalive = keepalive


def _cumulative_group_contraction_scheme(
    groups: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    if not groups:
        return ()
    running_names: list[str] = []
    steps: list[tuple[str, ...]] = []
    for group in groups:
        running_names.extend(group)
        steps.append(tuple(running_names))
    return tuple(steps)


def _site_bond_dims(n_sites: int) -> list[int]:
    return [2 + (index % 3) for index in range(max(n_sites - 1, 1))]


def _build_mps_auto(n_sites: int) -> Any:
    ensure_minimum("n_sites", n_sites)
    EinsumTrace, einsum, _einsum_trace_step, _pair_tensor = _einsum_api()
    trace = EinsumTrace()
    phys_dim = 2
    bond_dims = _site_bond_dims(n_sites)

    if n_sites == 1:
        a0 = build_demo_torch_tensor(name="A0", shape=(phys_dim,))
        x0 = build_demo_torch_tensor(name="x0", shape=(phys_dim,))
        trace.bind("A0", a0)
        trace.bind("x0", x0)
        _keep_trace_tensors_alive(trace, a0, x0)
        result = einsum("p,p->", a0, x0, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace

    a0 = build_demo_torch_tensor(name="A0", shape=(phys_dim, bond_dims[0]))
    x0 = build_demo_torch_tensor(name="x0", shape=(phys_dim,))
    trace.bind("A0", a0)
    trace.bind("x0", x0)
    _keep_trace_tensors_alive(trace, a0, x0)
    current = einsum("pa,p->a", a0, x0, trace=trace, backend="torch")
    _keep_trace_tensors_alive(trace, current)
    for index in range(1, n_sites - 1):
        tensor = build_demo_torch_tensor(
            name=f"A{index}",
            shape=(bond_dims[index - 1], phys_dim, bond_dims[index]),
        )
        vector = build_demo_torch_tensor(name=f"x{index}", shape=(phys_dim,))
        trace.bind(f"A{index}", tensor)
        trace.bind(f"x{index}", vector)
        _keep_trace_tensors_alive(trace, tensor, vector)
        current = einsum(
            "a,apb,p->b",
            current,
            tensor,
            vector,
            trace=trace,
            backend="torch",
        )
        _keep_trace_tensors_alive(trace, current)
    last = build_demo_torch_tensor(name=f"A{n_sites - 1}", shape=(bond_dims[n_sites - 2], phys_dim))
    last_vec = build_demo_torch_tensor(name=f"x{n_sites - 1}", shape=(phys_dim,))
    trace.bind(f"A{n_sites - 1}", last)
    trace.bind(f"x{n_sites - 1}", last_vec)
    _keep_trace_tensors_alive(trace, last, last_vec)
    result = einsum(
        "a,ap,p->",
        current,
        last,
        last_vec,
        trace=trace,
        backend="torch",
    )
    _keep_trace_tensors_alive(trace, result)
    return trace


def _build_mps_manual(n_sites: int) -> list[Any]:
    ensure_minimum("n_sites", n_sites)
    _EinsumTrace, _einsum, einsum_trace_step, pair_tensor = _einsum_api()
    phys_dim = 2
    bond_dims = _site_bond_dims(n_sites)

    if n_sites == 1:
        return [pair_tensor("A0", "x0", "r0", "p,p->")]

    steps: list[Any] = [pair_tensor("A0", "x0", "r0", "pa,p->a")]
    current_name = "r0"
    current_shape = (bond_dims[0],)
    for index in range(1, n_sites - 1):
        tensor_shape = (bond_dims[index - 1], phys_dim, bond_dims[index])
        vector_shape = (phys_dim,)
        next_name = f"r{index}"
        steps.append(
            einsum_trace_step(
                operand_names=(current_name, f"A{index}", f"x{index}"),
                result_name=next_name,
                equation="a,apb,p->b",
                metadata={"operand_shapes": (current_shape, tensor_shape, vector_shape)},
            )
        )
        current_name = next_name
        current_shape = (bond_dims[index],)
    steps.append(
        einsum_trace_step(
            operand_names=(current_name, f"A{n_sites - 1}", f"x{n_sites - 1}"),
            result_name=f"r{n_sites - 1}",
            equation="a,ap,p->",
            metadata={
                "operand_shapes": (current_shape, (bond_dims[n_sites - 2], phys_dim), (phys_dim,))
            },
        )
    )
    return steps


def _build_mpo_auto(n_sites: int) -> Any:
    ensure_minimum("n_sites", n_sites)
    EinsumTrace, einsum, _einsum_trace_step, _pair_tensor = _einsum_api()
    trace = EinsumTrace()
    phys_dim = 2
    bond_dims = _site_bond_dims(n_sites)

    if n_sites == 1:
        w0 = build_demo_torch_tensor(name="W0", shape=(phys_dim, phys_dim))
        d0 = build_demo_torch_tensor(name="d0", shape=(phys_dim,))
        u0 = build_demo_torch_tensor(name="u0", shape=(phys_dim,))
        trace.bind("W0", w0)
        trace.bind("d0", d0)
        trace.bind("u0", u0)
        _keep_trace_tensors_alive(trace, w0, d0, u0)
        result = einsum("du,d,u->", w0, d0, u0, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace

    w0 = build_demo_torch_tensor(name="W0", shape=(phys_dim, phys_dim, bond_dims[0]))
    d0 = build_demo_torch_tensor(name="d0", shape=(phys_dim,))
    u0 = build_demo_torch_tensor(name="u0", shape=(phys_dim,))
    trace.bind("W0", w0)
    trace.bind("d0", d0)
    trace.bind("u0", u0)
    _keep_trace_tensors_alive(trace, w0, d0, u0)
    current = einsum("dub,d,u->b", w0, d0, u0, trace=trace, backend="torch")
    _keep_trace_tensors_alive(trace, current)
    for index in range(1, n_sites - 1):
        tensor = build_demo_torch_tensor(
            name=f"W{index}",
            shape=(bond_dims[index - 1], phys_dim, phys_dim, bond_dims[index]),
        )
        d_vec = build_demo_torch_tensor(name=f"d{index}", shape=(phys_dim,))
        u_vec = build_demo_torch_tensor(name=f"u{index}", shape=(phys_dim,))
        trace.bind(f"W{index}", tensor)
        trace.bind(f"d{index}", d_vec)
        trace.bind(f"u{index}", u_vec)
        _keep_trace_tensors_alive(trace, tensor, d_vec, u_vec)
        current = einsum(
            "a,adub,d,u->b",
            current,
            tensor,
            d_vec,
            u_vec,
            trace=trace,
            backend="torch",
        )
        _keep_trace_tensors_alive(trace, current)
    last = build_demo_torch_tensor(
        name=f"W{n_sites - 1}",
        shape=(bond_dims[n_sites - 2], phys_dim, phys_dim),
    )
    last_d = build_demo_torch_tensor(name=f"d{n_sites - 1}", shape=(phys_dim,))
    last_u = build_demo_torch_tensor(name=f"u{n_sites - 1}", shape=(phys_dim,))
    trace.bind(f"W{n_sites - 1}", last)
    trace.bind(f"d{n_sites - 1}", last_d)
    trace.bind(f"u{n_sites - 1}", last_u)
    _keep_trace_tensors_alive(trace, last, last_d, last_u)
    result = einsum(
        "a,adu,d,u->",
        current,
        last,
        last_d,
        last_u,
        trace=trace,
        backend="torch",
    )
    _keep_trace_tensors_alive(trace, result)
    return trace


def _build_mpo_manual(n_sites: int) -> list[Any]:
    ensure_minimum("n_sites", n_sites)
    _EinsumTrace, _einsum, einsum_trace_step, pair_tensor = _einsum_api()
    phys_dim = 2
    bond_dims = _site_bond_dims(n_sites)

    if n_sites == 1:
        return [
            einsum_trace_step(
                ("W0", "d0", "u0"),
                "r0",
                "du,d,u->",
                metadata={"operand_shapes": ((phys_dim, phys_dim), (phys_dim,), (phys_dim,))},
            )
        ]

    steps: list[Any] = [
        einsum_trace_step(
            operand_names=("W0", "d0", "u0"),
            result_name="r0",
            equation="dub,d,u->b",
            metadata={
                "operand_shapes": ((phys_dim, phys_dim, bond_dims[0]), (phys_dim,), (phys_dim,))
            },
        )
    ]
    current_name = "r0"
    current_shape = (bond_dims[0],)
    for index in range(1, n_sites - 1):
        tensor_shape = (bond_dims[index - 1], phys_dim, phys_dim, bond_dims[index])
        vector_shape = (phys_dim,)
        next_name = f"r{index}"
        steps.append(
            einsum_trace_step(
                operand_names=(current_name, f"W{index}", f"d{index}", f"u{index}"),
                result_name=next_name,
                equation="a,adub,d,u->b",
                metadata={
                    "operand_shapes": (current_shape, tensor_shape, vector_shape, vector_shape)
                },
            )
        )
        current_name = next_name
        current_shape = (bond_dims[index],)
    steps.append(
        einsum_trace_step(
            operand_names=(current_name, f"W{n_sites - 1}", f"d{n_sites - 1}", f"u{n_sites - 1}"),
            result_name=f"r{n_sites - 1}",
            equation="a,adu,d,u->",
            metadata={
                "operand_shapes": (
                    current_shape,
                    (bond_dims[n_sites - 2], phys_dim, phys_dim),
                    (phys_dim,),
                    (phys_dim,),
                )
            },
        )
    )
    return steps


def _build_disconnected_auto() -> Any:
    EinsumTrace, einsum, _einsum_trace_step, _pair_tensor = _einsum_api()
    trace = EinsumTrace()
    a = build_demo_torch_tensor(name="A", shape=(5, 3))
    x = build_demo_torch_tensor(name="x", shape=(3,))
    b = build_demo_torch_tensor(name="B", shape=(7, 2))
    y = build_demo_torch_tensor(name="y", shape=(2,))
    trace.bind("A", a)
    trace.bind("x", x)
    trace.bind("B", b)
    trace.bind("y", y)
    _keep_trace_tensors_alive(trace, a, x, b, y)
    left_result = einsum("ab,b->a", a, x, trace=trace, backend="torch")
    right_result = einsum("cd,d->c", b, y, trace=trace, backend="torch")
    _keep_trace_tensors_alive(trace, left_result, right_result)
    return trace


def _build_disconnected_manual() -> list[Any]:
    _EinsumTrace, _einsum, _einsum_trace_step, pair_tensor = _einsum_api()
    return [
        pair_tensor("A", "x", "r0", "ab,b->a"),
        pair_tensor("B", "y", "r1", "cd,d->c"),
    ]


def _einsum_symbol_pool(required: int) -> list[str]:
    pool = list(string.ascii_lowercase + string.ascii_uppercase)
    if required > len(pool):
        raise ValueError("PEPS example needs more einsum symbols than the demo supports.")
    return pool


def _build_peps_site_data(lx: int, ly: int) -> list[_PepsSiteData]:
    ensure_minimum("lx", lx)
    ensure_minimum("ly", ly)
    horizontal = lx * max(ly - 1, 0)
    vertical = max(lx - 1, 0) * ly
    physical = lx * ly
    symbols = iter(_einsum_symbol_pool(horizontal + vertical + physical))
    right_labels: dict[tuple[int, int], str] = {}
    down_labels: dict[tuple[int, int], str] = {}
    site_data: list[_PepsSiteData] = []

    for i in range(lx):
        for j in range(ly):
            labels: list[str] = []
            shape: list[int] = []
            if i > 0:
                labels.append(down_labels[(i - 1, j)])
                shape.append(3)
            if j > 0:
                labels.append(right_labels[(i, j - 1)])
                shape.append(3)
            phys_label = next(symbols)
            labels.append(phys_label)
            shape.append(2)
            if j < ly - 1:
                right = next(symbols)
                right_labels[(i, j)] = right
                labels.append(right)
                shape.append(3)
            if i < lx - 1:
                down = next(symbols)
                down_labels[(i, j)] = down
                labels.append(down)
                shape.append(3)
            site_data.append(
                _PepsSiteData(
                    tensor_name=f"P{i}_{j}",
                    tensor_labels=tuple(labels),
                    tensor_shape=tuple(shape),
                    vector_name=f"x{i}_{j}",
                    vector_shape=(2,),
                    phys_label=phys_label,
                )
            )
    return site_data


def _build_peps_auto(lx: int, ly: int) -> Any:
    EinsumTrace, einsum, _einsum_trace_step, _pair_tensor = _einsum_api()
    trace = EinsumTrace()
    sites = _build_peps_site_data(lx, ly)
    tensors: dict[str, Any] = {}
    for site in sites:
        tensor = build_demo_torch_tensor(name=site.tensor_name, shape=site.tensor_shape)
        vector = build_demo_torch_tensor(name=site.vector_name, shape=site.vector_shape)
        tensors[site.tensor_name] = tensor
        tensors[site.vector_name] = vector
        trace.bind(site.tensor_name, tensor)
        trace.bind(site.vector_name, vector)
        _keep_trace_tensors_alive(trace, tensor, vector)

    current: Any | None = None
    current_labels: tuple[str, ...] = ()
    for site in sites:
        tensor = tensors[site.tensor_name]
        vector = tensors[site.vector_name]
        if current is None:
            output_labels = tuple(label for label in site.tensor_labels if label != site.phys_label)
            einsum_equation = (
                f"{''.join(site.tensor_labels)},{site.phys_label}->{''.join(output_labels)}"
            )
            current = einsum(einsum_equation, tensor, vector, trace=trace, backend="torch")
            _keep_trace_tensors_alive(trace, current)
            current_labels = output_labels
            continue
        shared_labels = tuple(
            label
            for label in site.tensor_labels
            if label != site.phys_label and label in current_labels
        )
        output_labels = tuple(
            label for label in current_labels if label not in shared_labels
        ) + tuple(
            label
            for label in site.tensor_labels
            if label != site.phys_label and label not in current_labels
        )
        einsum_equation = (
            f"{''.join(current_labels)},{''.join(site.tensor_labels)},{site.phys_label}->"
            f"{''.join(output_labels)}"
        )
        current = einsum(einsum_equation, current, tensor, vector, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, current)
        current_labels = output_labels
    return trace


def _build_peps_manual(lx: int, ly: int) -> list[Any]:
    _EinsumTrace, _einsum, einsum_trace_step, pair_tensor = _einsum_api()
    sites = _build_peps_site_data(lx, ly)
    label_dims: dict[str, int] = {}
    for site in sites:
        for label, dim in zip(site.tensor_labels, site.tensor_shape, strict=True):
            label_dims[label] = dim

    steps: list[Any] = []
    current_name: str | None = None
    current_shape: tuple[int, ...] = ()
    current_labels: tuple[str, ...] = ()
    for index, site in enumerate(sites):
        if current_name is None:
            output_labels = tuple(label for label in site.tensor_labels if label != site.phys_label)
            equation = f"{''.join(site.tensor_labels)},{site.phys_label}->{''.join(output_labels)}"
            steps.append(pair_tensor(site.tensor_name, site.vector_name, "r0", equation))
            current_name = "r0"
            current_labels = output_labels
            current_shape = tuple(label_dims[label] for label in output_labels)
            continue
        shared_labels = tuple(
            label
            for label in site.tensor_labels
            if label != site.phys_label and label in current_labels
        )
        output_labels = tuple(
            label for label in current_labels if label not in shared_labels
        ) + tuple(
            label
            for label in site.tensor_labels
            if label != site.phys_label and label not in current_labels
        )
        equation = (
            f"{''.join(current_labels)},{''.join(site.tensor_labels)},{site.phys_label}->"
            f"{''.join(output_labels)}"
        )
        next_name = f"r{index}"
        steps.append(
            einsum_trace_step(
                operand_names=(current_name, site.tensor_name, site.vector_name),
                result_name=next_name,
                equation=equation,
                metadata={"operand_shapes": (current_shape, site.tensor_shape, site.vector_shape)},
            )
        )
        current_name = next_name
        current_labels = output_labels
        current_shape = tuple(label_dims[label] for label in output_labels)
    return steps


def _build_pattern_trace(example: str) -> Any:
    torch = _torch()
    EinsumTrace, einsum, _einsum_trace_step, _pair_tensor = _einsum_api()
    trace = EinsumTrace()
    if example == "ellipsis":
        a = build_demo_torch_tensor(name="A", shape=(2, 3, 4))
        b = build_demo_torch_tensor(name="B", shape=(2, 4, 5))
        trace.bind("A", a)
        trace.bind("B", b)
        _keep_trace_tensors_alive(trace, a, b)
        result = einsum("...ij,...jk->...ik", a, b, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "batch":
        u = build_demo_torch_tensor(name="U", shape=(3, 4))
        v = build_demo_torch_tensor(name="V", shape=(3, 4))
        trace.bind("U", u)
        trace.bind("V", v)
        _keep_trace_tensors_alive(trace, u, v)
        result = einsum("ab,ab->ab", u, v, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "trace":
        matrix = build_demo_torch_tensor(name="M", shape=(4, 4))
        vector = build_demo_torch_tensor(name="x", shape=(4,))
        trace.bind("M", matrix)
        trace.bind("x", vector)
        _keep_trace_tensors_alive(trace, matrix, vector)
        result = einsum("ii,i->i", matrix, vector, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "ternary":
        a = build_demo_torch_tensor(name="A", shape=(2, 3))
        b = build_demo_torch_tensor(name="B", shape=(3, 4))
        c = build_demo_torch_tensor(name="C", shape=(4, 5))
        trace.bind("A", a)
        trace.bind("B", b)
        trace.bind("C", c)
        _keep_trace_tensors_alive(trace, a, b, c)
        result = einsum("ab,bc,cd->ad", a, b, c, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "unary":
        matrix = build_demo_torch_tensor(name="M", shape=(4, 4))
        trace.bind("M", matrix)
        _keep_trace_tensors_alive(trace, matrix)
        result = einsum("ii->i", matrix, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "nway":
        t = build_demo_torch_tensor(name="T", shape=(3, 4, 5))
        u = build_demo_torch_tensor(name="U", shape=(3, 4, 6))
        v = build_demo_torch_tensor(name="V", shape=(5, 6, 7))
        trace.bind("T", t)
        trace.bind("U", u)
        trace.bind("V", v)
        _keep_trace_tensors_alive(trace, t, u, v)
        reduced = einsum("abc,abd->cd", t, u, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, reduced)
        result = einsum("cd,cde->e", reduced, v, trace=trace, backend="torch")
        _keep_trace_tensors_alive(trace, result)
        return trace
    if example == "implicit_out":
        a = build_demo_torch_tensor(name="A", shape=(2, 3))
        b = build_demo_torch_tensor(name="b", shape=(3,))
        out = torch.empty((2,))
        trace.bind("A", a)
        trace.bind("b", b)
        _keep_trace_tensors_alive(trace, a, b, out)
        result = einsum("ij,j", a, b, trace=trace, backend="torch", out=out)
        _keep_trace_tensors_alive(trace, result)
        return trace
    raise ValueError(f"Unsupported einsum pattern example: {example}")


def _trace_steps_for(name: str, args: ExampleCliArgs) -> Any:
    if name == "mps":
        return (
            _build_mps_manual(args.n_sites) if args.from_scratch else _build_mps_auto(args.n_sites)
        )
    if name == "mpo":
        return (
            _build_mpo_manual(args.n_sites) if args.from_scratch else _build_mpo_auto(args.n_sites)
        )
    if name == "peps":
        return (
            _build_peps_manual(args.lx, args.ly)
            if args.from_scratch
            else _build_peps_auto(args.lx, args.ly)
        )
    if name == "disconnected":
        return _build_disconnected_manual() if args.from_scratch else _build_disconnected_auto()
    return _build_pattern_trace(name)


def _renderable_trace(trace: Any, args: ExampleCliArgs) -> Any:
    if args.from_scratch:
        return trace
    if args.from_list:
        return list(trace)
    return trace


def _scheme_steps(name: str, args: ExampleCliArgs) -> tuple[tuple[str, ...], ...] | None:
    if name == "mps":
        return _cumulative_group_contraction_scheme(
            tuple((f"A{i}", f"x{i}") for i in range(args.n_sites))
        )
    if name == "mpo":
        return _cumulative_group_contraction_scheme(
            tuple((f"W{i}", f"d{i}", f"u{i}") for i in range(args.n_sites))
        )
    if name == "peps":
        names = tuple(f"P{i}_{j}" for i in range(args.lx) for j in range(args.ly))
        return cumulative_prefix_contraction_scheme(names)
    if name == "disconnected":
        return (("A", "x"), ("B", "y"))
    return None


def _build_example(args: ExampleCliArgs, definition: ExampleDefinition) -> BuiltExample:
    trace = _trace_steps_for(definition.name, args)
    footer = "Render an EinsumTrace or an explicit ordered list of trace steps."
    if not args.from_scratch and not args.from_list:
        footer = (
            "Auto-traced EinsumTrace example. Try --tensor-inspector to inspect tensors during "
            "playback."
        )
    return BuiltExample(
        network=_renderable_trace(trace, args),
        plot_engine="einsum",
        title=f"Einsum · {definition.name.upper()} · {args.view.upper()}",
        subtitle=TAGLINES.get(definition.name),
        footer=footer,
        scheme_steps_by_name=_scheme_steps(definition.name, args),
    )


EXAMPLES: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        name="mps",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Sequential MPS contraction trace.",
    ),
    ExampleDefinition(
        name="mpo",
        aliases=(),
        size_knobs=frozenset({"n_sites"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Sequential MPO contraction trace.",
    ),
    ExampleDefinition(
        name="peps",
        aliases=(),
        size_knobs=frozenset({"lx", "ly"}),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="PEPS contraction trace.",
    ),
    ExampleDefinition(
        name="disconnected",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=True,
        supports_list=True,
        builder=_build_example,
        description="Disconnected contraction trace.",
    ),
    ExampleDefinition(
        name="ellipsis",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Ellipsis-aware trace.",
    ),
    ExampleDefinition(
        name="batch",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Hadamard-style batch contraction.",
    ),
    ExampleDefinition(
        name="trace",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Trace / diagonal example.",
    ),
    ExampleDefinition(
        name="ternary",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Single ternary trace step.",
    ),
    ExampleDefinition(
        name="unary",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Single unary trace step.",
    ),
    ExampleDefinition(
        name="nway",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Multi-step n-way contraction.",
    ),
    ExampleDefinition(
        name="implicit_out",
        aliases=(),
        size_knobs=frozenset(),
        supports_native_object=True,
        supports_from_scratch=False,
        supports_list=True,
        builder=_build_example,
        description="Implicit output with out=.",
    ),
)


def run_example(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    definition = resolve_example_definition(EXAMPLES, args.example)
    if definition is None:
        raise ValueError(f"Unsupported einsum example: {args.example}")
    if demo_runs_headless(args):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    built = definition.builder(args, definition)
    config = finalize_demo_plot_config(
        args, engine="einsum", scheme_tensor_names=built.scheme_steps_by_name
    )
    fig, _ax = render_demo_tensor_network(
        built.network,
        args=args,
        engine="einsum",
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
