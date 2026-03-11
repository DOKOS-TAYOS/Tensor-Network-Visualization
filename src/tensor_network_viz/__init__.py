from .config import EngineName, PlotConfig, ViewName
from .einsum_module.trace import EinsumTrace, einsum, pair_tensor


def show_tensor_network(*args, **kwargs):
    from .viewer import show_tensor_network as _show_tensor_network

    return _show_tensor_network(*args, **kwargs)


__all__ = [
    "EngineName",
    "EinsumTrace",
    "PlotConfig",
    "ViewName",
    "einsum",
    "pair_tensor",
    "show_tensor_network",
]
