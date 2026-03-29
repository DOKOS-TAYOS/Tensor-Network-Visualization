from typing import TYPE_CHECKING, Any

from .config import EngineName, PlotConfig, ViewName
from ._core.graph_cache import clear_tensor_network_graph_cache
from .einsum_module.trace import EinsumTrace, einsum, pair_tensor

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
    "ViewName",
    "clear_tensor_network_graph_cache",
    "einsum",
    "pair_tensor",
    "show_tensor_network",
]
