from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "plot_tensorkrowch_network_2d": (".renderer", "plot_tensorkrowch_network_2d"),
    "plot_tensorkrowch_network_3d": (".renderer", "plot_tensorkrowch_network_3d"),
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
    "plot_tensorkrowch_network_2d",
    "plot_tensorkrowch_network_3d",
]
