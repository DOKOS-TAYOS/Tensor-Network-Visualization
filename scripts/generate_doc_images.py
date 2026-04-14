from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

RenderedAxes: TypeAlias = Axes | Axes3D

DOC_GALLERY_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("images/gallery/cubic_peps_3d.png"),
    Path("images/gallery/mera_2d.png"),
    Path("images/gallery/tubular_grid_3d.png"),
    Path("images/gallery/network_controls.png"),
    Path("images/gallery/tensor_elements_phase.png"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_import_paths(repo_root: Path) -> None:
    for path in (repo_root / "examples", repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _make_axes_background_transparent(ax: RenderedAxes) -> None:
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_alpha(0.0)
    if isinstance(ax, Axes3D):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def _save_figure(fig: Figure, output_path: Path, *, transparent: bool) -> Path:
    if transparent:
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            if isinstance(ax, Axes | Axes3D):
                _make_axes_background_transparent(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    return output_path


def _render_cubic_peps(output_root: Path) -> Path:
    import tensorkrowch_demo

    from tensor_network_viz import PlotConfig, show_tensor_network

    node_specs, bond_specs = tensorkrowch_demo._cubic_peps_specs(3, 4, 3)
    network, _nodes = tensorkrowch_demo._build_tensorkrowch_network(node_specs, bond_specs)
    fig, ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="3d",
        config=PlotConfig(
            theme="midnight",
            show_tensor_labels=False,
            hover_labels=False,
            figsize=(9.0, 7.0),
            layout_iterations=320,
        ),
        show_controls=False,
        show=False,
    )
    if isinstance(ax, Axes3D):
        ax.view_init(elev=22.0, azim=35.0)
    return _save_figure(fig, output_root / DOC_GALLERY_RELATIVE_PATHS[0], transparent=True)


def _render_mera(output_root: Path) -> Path:
    import tensorkrowch_demo

    from tensor_network_viz import PlotConfig, show_tensor_network

    node_specs, bond_specs = tensorkrowch_demo._mera_specs(4)
    network, _nodes = tensorkrowch_demo._build_tensorkrowch_network(node_specs, bond_specs)
    fig, _ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="2d",
        config=PlotConfig(
            theme="slate",
            show_tensor_labels=True,
            hover_labels=False,
            figsize=(10.5, 5.8),
            layout_iterations=260,
            tensor_label_fontsize=7.5,
        ),
        show_controls=False,
        show=False,
    )
    return _save_figure(fig, output_root / DOC_GALLERY_RELATIVE_PATHS[1], transparent=True)


def _render_tubular_grid(output_root: Path) -> Path:
    import geometry_demo

    from tensor_network_viz import PlotConfig, show_tensor_network

    fig, ax = show_tensor_network(
        geometry_demo._tubular_grid(),
        engine="quimb",
        view="3d",
        config=PlotConfig(
            theme="dark",
            show_tensor_labels=False,
            hover_labels=False,
            figsize=(9.0, 7.0),
            layout_iterations=360,
            tensor_label_fontsize=7.0,
        ),
        show_controls=False,
        show=False,
    )
    if isinstance(ax, Axes3D):
        ax.view_init(elev=24.0, azim=42.0)
    return _save_figure(fig, output_root / DOC_GALLERY_RELATIVE_PATHS[2], transparent=True)


def _render_network_controls(output_root: Path) -> Path:
    import tensorkrowch_demo

    from tensor_network_viz import PlotConfig, show_tensor_network

    node_specs, bond_specs = tensorkrowch_demo._mps_specs(6)
    network, _nodes = tensorkrowch_demo._build_tensorkrowch_network(node_specs, bond_specs)
    fig, _ax = show_tensor_network(
        network,
        engine="tensorkrowch",
        view="2d",
        config=PlotConfig(
            theme="paper",
            show_tensor_labels=True,
            show_index_labels=False,
            hover_labels=True,
            figsize=(9.0, 6.8),
            tensor_label_fontsize=9.0,
        ),
        show_controls=True,
        show=False,
    )
    return _save_figure(fig, output_root / DOC_GALLERY_RELATIVE_PATHS[3], transparent=False)


def _render_tensor_elements_phase(output_root: Path) -> Path:
    import tensor_elements_demo

    from tensor_network_viz import TensorElementsConfig, show_tensor_elements

    nodes = tensor_elements_demo.build_structured_network()
    psi = next(node for node in nodes if node.name == "Psi")
    fig, ax = show_tensor_elements(
        psi.tensor,
        config=TensorElementsConfig(
            mode="phase",
            row_axes=(0,),
            col_axes=(1, 2),
            figsize=(8.6, 6.8),
            max_matrix_shape=(384, 384),
            shared_color_scale=True,
        ),
        show_controls=False,
        show=False,
    )
    ax.set_title("Tensor Elements - phase")
    return _save_figure(fig, output_root / DOC_GALLERY_RELATIVE_PATHS[4], transparent=True)


def generate_doc_images(output_root: Path | None = None) -> tuple[Path, ...]:
    repo_root = _repo_root()
    _bootstrap_import_paths(repo_root)
    target_root = repo_root if output_root is None else Path(output_root)
    return (
        _render_cubic_peps(target_root),
        _render_mera(target_root),
        _render_tubular_grid(target_root),
        _render_network_controls(target_root),
        _render_tensor_elements_phase(target_root),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the documentation gallery images used by README and docs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory where the images/ tree should be written. "
            "Defaults to the repository root."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generated = generate_doc_images(output_root=args.output_root)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
