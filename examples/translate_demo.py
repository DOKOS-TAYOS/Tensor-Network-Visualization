from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, TypeAlias

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import numpy as np

from tensor_network_viz import (
    PlotConfig,
    export_tensor_network_snapshot,
    normalize_tensor_network,
    show_tensor_network,
    translate_tensor_network,
)

TranslationExampleName: TypeAlias = Literal["simple", "mps", "peps", "weird", "disconnected"]
_EXAMPLE_CHOICES: tuple[TranslationExampleName, ...] = (
    "simple",
    "mps",
    "peps",
    "weird",
    "disconnected",
)


class BuiltTranslationDemo:
    def __init__(self, network: Any, title: str) -> None:
        self.network = network
        self.title = title


def _build_tensornetwork_simple() -> BuiltTranslationDemo:
    import tensornetwork as tn

    left = tn.Node(np.arange(6, dtype=float).reshape(2, 3), name="L", axis_names=("a", "b"))
    right = tn.Node(np.arange(12, dtype=float).reshape(3, 4), name="R", axis_names=("b", "c"))
    left["b"] ^ right["b"]
    return BuiltTranslationDemo(network=[left, right], title="TensorNetwork simple pair")


def _build_quimb_simple() -> BuiltTranslationDemo:
    import quimb.tensor as qtn

    tensors = [
        qtn.Tensor(np.arange(6, dtype=float).reshape(2, 3), inds=("a", "b"), tags={"L"}),
        qtn.Tensor(np.arange(12, dtype=float).reshape(3, 4), inds=("b", "c"), tags={"R"}),
    ]
    return BuiltTranslationDemo(
        network=qtn.TensorNetwork(tensors),
        title="Quimb simple pair",
    )


def _build_einsum_simple() -> BuiltTranslationDemo:
    from tensor_network_viz import EinsumTrace, einsum

    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(12, dtype=float).reshape(3, 4)
    trace.bind("L", left)
    trace.bind("R", right)
    _ = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")
    return BuiltTranslationDemo(network=trace, title="Einsum simple pair")


def _build_tensorkrowch_simple() -> BuiltTranslationDemo:
    import tensorkrowch as tk
    import torch

    network = tk.TensorNetwork(name="demo")
    left = tk.Node(
        shape=(2, 3),
        axes_names=("a", "b"),
        name="L",
        network=network,
        tensor=torch.tensor(np.arange(6, dtype=float).reshape(2, 3), dtype=torch.float64),
    )
    right = tk.Node(
        shape=(3, 4),
        axes_names=("b", "c"),
        name="R",
        network=network,
        tensor=torch.tensor(np.arange(12, dtype=float).reshape(3, 4), dtype=torch.float64),
    )
    left["b"] ^ right["b"]
    return BuiltTranslationDemo(network=network, title="TensorKrowch simple pair")


def _build_tensornetwork_source(
    example: TranslationExampleName,
    *,
    n_sites: int,
    lx: int,
    ly: int,
) -> BuiltTranslationDemo:
    import tensornetwork_demo as tn_demo

    if example == "simple":
        return _build_tensornetwork_simple()
    if example == "mps":
        node_specs, bond_specs = tn_demo._mps_specs(n_sites)
        return BuiltTranslationDemo(
            network=tn_demo._build_tensornetwork_nodes(node_specs, bond_specs),
            title=f"TensorNetwork MPS ({n_sites} sites)",
        )
    if example == "peps":
        node_specs, bond_specs = tn_demo._peps_specs(lx, ly)
        return BuiltTranslationDemo(
            network=tn_demo._build_tensornetwork_nodes(node_specs, bond_specs),
            title=f"TensorNetwork PEPS ({lx}x{ly})",
        )
    if example == "weird":
        node_specs, bond_specs = tn_demo._weird_specs()
        return BuiltTranslationDemo(
            network=tn_demo._build_tensornetwork_nodes(node_specs, bond_specs),
            title="TensorNetwork weird graph",
        )
    raise ValueError("TensorNetwork translation demo supports examples: simple, mps, peps, weird.")


def _build_quimb_source(
    example: TranslationExampleName,
    *,
    n_sites: int,
    lx: int,
    ly: int,
) -> BuiltTranslationDemo:
    import quimb_demo as quimb_demo_module

    if example == "simple":
        return _build_quimb_simple()
    if example == "mps":
        node_specs, bond_specs = quimb_demo_module._mps_specs(n_sites)
        network, _tensors = quimb_demo_module._build_quimb_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title=f"Quimb MPS ({n_sites} sites)")
    if example == "peps":
        node_specs, bond_specs = quimb_demo_module._peps_specs(lx, ly)
        network, _tensors = quimb_demo_module._build_quimb_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title=f"Quimb PEPS ({lx}x{ly})")
    raise ValueError("Quimb translation demo supports examples: simple, mps, peps.")


def _build_einsum_source(
    example: TranslationExampleName,
    *,
    n_sites: int,
    lx: int,
    ly: int,
) -> BuiltTranslationDemo:
    import einsum_demo as einsum_demo_module

    if example == "simple":
        return _build_einsum_simple()
    if example == "mps":
        return BuiltTranslationDemo(
            network=einsum_demo_module._build_mps_auto(n_sites),
            title=f"Einsum MPS ({n_sites} sites)",
        )
    if example == "peps":
        return BuiltTranslationDemo(
            network=einsum_demo_module._build_peps_auto(lx, ly),
            title=f"Einsum PEPS ({lx}x{ly})",
        )
    if example == "disconnected":
        return BuiltTranslationDemo(
            network=einsum_demo_module._build_disconnected_auto(),
            title="Einsum disconnected components",
        )
    raise ValueError("Einsum translation demo supports examples: simple, mps, peps, disconnected.")


def _build_tensorkrowch_source(
    example: TranslationExampleName,
    *,
    n_sites: int,
    lx: int,
    ly: int,
) -> BuiltTranslationDemo:
    import tensorkrowch_demo as tk_demo

    if example == "simple":
        return _build_tensorkrowch_simple()
    if example == "mps":
        node_specs, bond_specs = tk_demo._mps_specs(n_sites)
        network, _nodes = tk_demo._build_tensorkrowch_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title=f"TensorKrowch MPS ({n_sites} sites)")
    if example == "peps":
        node_specs, bond_specs = tk_demo._peps_specs(lx, ly)
        network, _nodes = tk_demo._build_tensorkrowch_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title=f"TensorKrowch PEPS ({lx}x{ly})")
    if example == "weird":
        node_specs, bond_specs = tk_demo._weird_specs()
        network, _nodes = tk_demo._build_tensorkrowch_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title="TensorKrowch weird graph")
    if example == "disconnected":
        node_specs, bond_specs = tk_demo._disconnected_specs()
        network, _nodes = tk_demo._build_tensorkrowch_network(node_specs, bond_specs)
        return BuiltTranslationDemo(network=network, title="TensorKrowch disconnected components")
    raise ValueError(
        "TensorKrowch translation demo supports examples: simple, mps, peps, weird, disconnected."
    )


def _build_source_network(
    source_engine: str,
    *,
    example: TranslationExampleName,
    n_sites: int,
    lx: int,
    ly: int,
) -> BuiltTranslationDemo:
    if source_engine == "tensornetwork":
        return _build_tensornetwork_source(example, n_sites=n_sites, lx=lx, ly=ly)
    if source_engine == "quimb":
        return _build_quimb_source(example, n_sites=n_sites, lx=lx, ly=ly)
    if source_engine == "einsum":
        return _build_einsum_source(example, n_sites=n_sites, lx=lx, ly=ly)
    if source_engine == "tensorkrowch":
        return _build_tensorkrowch_source(example, n_sites=n_sites, lx=lx, ly=ly)
    raise ValueError(f"Unsupported source engine: {source_engine}")


def _load_generated_network(code: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace["network"]


def _snapshot_positions_by_name(snapshot: Any) -> dict[str, tuple[float, ...]]:
    node_name_by_id = {int(node.id): node.name for node in snapshot.graph.nodes}
    return {
        node_name_by_id[int(node_id)]: tuple(float(value) for value in coords)
        for node_id, coords in snapshot.layout.positions.items()
    }


def _build_comparison_configs(
    source_network: Any,
    *,
    source_engine: str,
    translated_network: Any,
    target_engine: str,
    view: str,
) -> tuple[PlotConfig, PlotConfig]:
    source_snapshot = export_tensor_network_snapshot(
        source_network,
        engine=source_engine,
        view=view,
        seed=0,
    )
    source_positions = {
        int(node_id): tuple(float(value) for value in coords)
        for node_id, coords in source_snapshot.layout.positions.items()
    }
    source_positions_by_name = _snapshot_positions_by_name(source_snapshot)
    translated_graph = normalize_tensor_network(translated_network, engine=target_engine)
    translated_positions = {
        int(node.id): source_positions_by_name[node.name]
        for node in translated_graph.nodes
        if node.name in source_positions_by_name
    }
    base_kwargs = {
        "hover_labels": False,
        "show_tensor_labels": True,
        "positions": source_positions,
    }
    translated_kwargs = {
        "hover_labels": False,
        "show_tensor_labels": True,
        "positions": translated_positions,
    }
    return PlotConfig(**base_kwargs), PlotConfig(**translated_kwargs)


def _subplot_for_view(fig: Any, *, index: int, view: str) -> Any:
    if view == "3d":
        return fig.add_subplot(1, 2, index, projection="3d")
    return fig.add_subplot(1, 2, index)


def _render_translation_figure(
    source_network: Any,
    *,
    source_engine: str,
    source_title: str,
    translated_network: Any,
    target_engine: str,
    view: str,
    save_figure: Path | None,
    no_show: bool,
) -> None:
    import matplotlib.pyplot as plt

    source_config, translated_config = _build_comparison_configs(
        source_network,
        source_engine=source_engine,
        translated_network=translated_network,
        target_engine=target_engine,
        view=view,
    )
    figsize = (15, 7) if view == "2d" else (16, 7)
    fig = plt.figure(figsize=figsize)
    left_ax = _subplot_for_view(fig, index=1, view=view)
    right_ax = _subplot_for_view(fig, index=2, view=view)
    show_tensor_network(
        source_network,
        engine=source_engine,
        view=view,
        ax=left_ax,
        config=source_config,
        show_controls=False,
        show=False,
    )
    show_tensor_network(
        translated_network,
        engine=target_engine,
        view=view,
        ax=right_ax,
        config=translated_config,
        show_controls=False,
        show=False,
    )
    left_ax.set_title(f"Original: {source_title}", fontsize=12, fontweight="semibold")
    right_ax.set_title(
        f"Translated: {target_engine}",
        fontsize=12,
        fontweight="semibold",
    )
    fig.suptitle(
        f"translate_tensor_network demo ({source_engine} -> {target_engine})",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    if save_figure is not None:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, bbox_inches="tight")
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a demo tensor network and optionally compare source vs translated."
    )
    parser.add_argument(
        "--source-engine",
        choices=("tensorkrowch", "tensornetwork", "quimb", "einsum"),
        default="tensornetwork",
    )
    parser.add_argument(
        "--target-engine",
        choices=("tensorkrowch", "tensornetwork", "quimb", "einsum"),
        default="quimb",
    )
    parser.add_argument(
        "--example",
        choices=_EXAMPLE_CHOICES,
        default="simple",
    )
    parser.add_argument("--view", choices=("2d", "3d"), default="2d")
    parser.add_argument("--n-sites", type=int, default=5)
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=3)
    parser.add_argument("--save-code", type=Path, default=None)
    parser.add_argument("--save-figure", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        built = _build_source_network(
            args.source_engine,
            example=args.example,
            n_sites=args.n_sites,
            lx=args.lx,
            ly=args.ly,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.save_code is not None:
        args.save_code.parent.mkdir(parents=True, exist_ok=True)
    code = translate_tensor_network(
        built.network,
        engine=args.source_engine,
        target_engine=args.target_engine,
        path=args.save_code,
    )
    translated_network = _load_generated_network(code)

    if args.save_code is None:
        print(code)

    if not args.no_show or args.save_figure is not None:
        _render_translation_figure(
            built.network,
            source_engine=args.source_engine,
            source_title=built.title,
            translated_network=translated_network,
            target_engine=args.target_engine,
            view=args.view,
            save_figure=args.save_figure,
            no_show=args.no_show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
