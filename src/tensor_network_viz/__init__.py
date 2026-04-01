from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._core.graph_cache import clear_tensor_network_graph_cache
from .config import EngineName, PlotConfig, ViewName

if TYPE_CHECKING:
    from .contraction_viewer import ContractionViewer2D, ContractionViewer3D
    from .einsum_module.trace import EinsumTrace, einsum, einsum_trace_step, pair_tensor
    from .tenpy.explicit import TenPyTensorNetwork, make_tenpy_tensor_network
    from .viewer import show_tensor_network
else:

    def show_tensor_network(*args: Any, **kwargs: Any) -> Any:
        from .viewer import show_tensor_network as _show_tensor_network

        return _show_tensor_network(*args, **kwargs)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ContractionViewer2D": (".contraction_viewer", "ContractionViewer2D"),
    "ContractionViewer3D": (".contraction_viewer", "ContractionViewer3D"),
    "EinsumTrace": (".einsum_module.trace", "EinsumTrace"),
    "TenPyTensorNetwork": (".tenpy.explicit", "TenPyTensorNetwork"),
    "einsum": (".einsum_module.trace", "einsum"),
    "einsum_trace_step": (".einsum_module.trace", "einsum_trace_step"),
    "make_tenpy_tensor_network": (".tenpy.explicit", "make_tenpy_tensor_network"),
    "pair_tensor": (".einsum_module.trace", "pair_tensor"),
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
    "ContractionViewer2D",
    "ContractionViewer3D",
    "EngineName",
    "EinsumTrace",
    "PlotConfig",
    "TenPyTensorNetwork",
    "ViewName",
    "clear_tensor_network_graph_cache",
    "einsum",
    "einsum_trace_step",
    "make_tenpy_tensor_network",
    "pair_tensor",
    "show_tensor_network",
]
