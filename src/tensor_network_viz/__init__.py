from .config import EngineName, PlotConfig, ViewName


def show_tensor_network(*args, **kwargs):
    from .viewer import show_tensor_network as _show_tensor_network

    return _show_tensor_network(*args, **kwargs)


__all__ = ["EngineName", "PlotConfig", "ViewName", "show_tensor_network"]
