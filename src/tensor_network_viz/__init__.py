from typing import TYPE_CHECKING, Any

from ._core.graph_cache import clear_tensor_network_graph_cache
from .config import EngineName, PlotConfig, ViewName
from .einsum_module.trace import EinsumTrace, einsum, einsum_trace_step, pair_tensor
from .tenpy.explicit import TenPyTensorNetwork, make_tenpy_tensor_network

if TYPE_CHECKING:
    from .viewer import show_tensor_network
else:

    def show_tensor_network(*args: Any, **kwargs: Any) -> Any:
        from .viewer import show_tensor_network as _show_tensor_network

        return _show_tensor_network(*args, **kwargs)


__all__ = [
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
