"""Single source of truth for tensor-network engine module paths and plotter names."""

from __future__ import annotations

from typing import Literal, TypeAlias

EngineName: TypeAlias = Literal["tensorkrowch", "tensornetwork", "quimb", "tenpy", "einsum"]

ENGINE_MODULE_MAP: dict[EngineName, tuple[str, str, str]] = {
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

ENGINE_GRAPH_BUILDER_MAP: dict[EngineName, tuple[str, str]] = {
    "tensorkrowch": ("tensor_network_viz.tensorkrowch.graph", "_build_graph"),
    "tensornetwork": ("tensor_network_viz.tensornetwork.graph", "_build_graph"),
    "quimb": ("tensor_network_viz.quimb.graph", "_build_graph"),
    "tenpy": ("tensor_network_viz.tenpy.graph", "_build_graph"),
    "einsum": ("tensor_network_viz.einsum_module.graph", "_build_graph"),
}
