from .config import PlotConfig


def show_tensor_network(*args, **kwargs):
    from .viewer import show_tensor_network as _show_tensor_network

    return _show_tensor_network(*args, **kwargs)


__all__ = ["PlotConfig", "show_tensor_network"]
