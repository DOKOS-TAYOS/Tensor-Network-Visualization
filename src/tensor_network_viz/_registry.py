"""Engine registry for lazy-loaded tensor network plotters."""

from __future__ import annotations

import importlib

from .config import EngineName

_ENGINE_CONFIG: dict[str, tuple[str, str, str]] = {
    "tensorkrowch": (
        "tensor_network_viz.tensorkrowch",
        "plot_tensorkrowch_network_2d",
        "plot_tensorkrowch_network_3d",
    ),
    "tensornetwork": (
        "tensor_network_viz.tensornetwork",
        "plot_tensornetwork_network_2d",
        "plot_tensornetwork_network_3d",
    ),
    "quimb": (
        "tensor_network_viz.quimb",
        "plot_quimb_network_2d",
        "plot_quimb_network_3d",
    ),
    "tenpy": (
        "tensor_network_viz.tenpy",
        "plot_tenpy_network_2d",
        "plot_tenpy_network_3d",
    ),
    "einsum": (
        "tensor_network_viz.einsum_module",
        "plot_einsum_network_2d",
        "plot_einsum_network_3d",
    ),
}


def _get_plotters(engine: EngineName) -> tuple[object, object]:
    """Get plot_2d and plot_3d for an engine, loading the module if needed."""
    config = _ENGINE_CONFIG.get(engine)
    if config is None:
        raise ValueError(f"Unsupported tensor network engine: {engine}")
    module_path, name_2d, name_3d = config
    mod = importlib.import_module(module_path)
    return (getattr(mod, name_2d), getattr(mod, name_3d))
