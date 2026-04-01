from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._equation import parse_einsum_equation, parse_equation_for_shapes
    from .renderer import plot_einsum_network_2d, plot_einsum_network_3d
    from .trace import EinsumTrace, einsum, einsum_trace_step, pair_tensor


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EinsumTrace": (".trace", "EinsumTrace"),
    "einsum": (".trace", "einsum"),
    "einsum_trace_step": (".trace", "einsum_trace_step"),
    "pair_tensor": (".trace", "pair_tensor"),
    "parse_einsum_equation": ("._equation", "parse_einsum_equation"),
    "parse_equation_for_shapes": ("._equation", "parse_equation_for_shapes"),
    "plot_einsum_network_2d": (".renderer", "plot_einsum_network_2d"),
    "plot_einsum_network_3d": (".renderer", "plot_einsum_network_3d"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "EinsumTrace",
    "einsum",
    "einsum_trace_step",
    "pair_tensor",
    "parse_einsum_equation",
    "parse_equation_for_shapes",
    "plot_einsum_network_2d",
    "plot_einsum_network_3d",
]
