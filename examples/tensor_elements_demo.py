from __future__ import annotations

import argparse
from typing import Any, Literal, TypeAlias

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

RenderedTensorAxes: TypeAlias = Axes

# Large tensors: stats reflect the full tensor; only the displayed heatmap is
# downsampled according to TensorElementsConfig.max_matrix_shape (384x384 here).
_MATVEC_ROWS: int = 240
_MATVEC_COLS: int = 256
_BATCH: int = 24
_BATCH_INNER: int = 80
_RANK3: tuple[int, int, int] = (56, 72, 96)
_GRID_SIDE: int = 320


def _demo_config() -> Any:
    from tensor_network_viz import TensorElementsConfig

    return TensorElementsConfig(
        mode="auto",
        figsize=(7.4, 6.4),
        max_matrix_shape=(384, 384),
    )


def build_matvec_trace() -> tuple[Any, tuple[np.ndarray[Any, Any], ...]]:
    from tensor_network_viz import EinsumTrace, einsum

    trace = EinsumTrace()
    row_angles = np.linspace(0.0, 4.0 * np.pi, _MATVEC_ROWS, dtype=np.float64)[:, np.newaxis]
    col_angles = np.linspace(0.0, 4.0 * np.pi, _MATVEC_COLS, dtype=np.float64)[np.newaxis, :]
    left = np.sin(row_angles) * np.cos(col_angles) + 0.35 * np.sin(
        2.0 * row_angles * col_angles / float(_MATVEC_COLS)
    )
    jj = np.arange(_MATVEC_COLS, dtype=np.float64)
    right = np.exp(-0.5 * ((jj - 0.5 * float(_MATVEC_COLS)) / (float(_MATVEC_COLS) / 6.0)) ** 2)

    trace.bind("A", left)
    trace.bind("x", right)
    result = einsum("ab,b->a", left, right, trace=trace, backend="numpy")
    keepalive = (left, right, np.asarray(result))
    return trace, keepalive


def build_batch_matmul_trace() -> tuple[Any, tuple[np.ndarray[Any, Any], ...]]:
    from tensor_network_viz import EinsumTrace, einsum

    rng = np.random.default_rng(42)
    trace = EinsumTrace()
    t = np.linspace(0.0, 1.0, _BATCH, dtype=np.float64)[:, np.newaxis, np.newaxis]
    i = np.arange(_BATCH_INNER, dtype=np.float64)[np.newaxis, :, np.newaxis]
    j = np.arange(_BATCH_INNER, dtype=np.float64)[np.newaxis, np.newaxis, :]
    envelope = 0.35 + 0.65 * np.sin(np.pi * t.squeeze())[:, np.newaxis, np.newaxis]
    a_core = np.sin(i * 0.11 + j * 0.09) * np.cos((i - j) * 0.05)
    b_core = np.cos(i * 0.08 - j * 0.12) * np.sin((i + j) * 0.04)
    left = envelope * a_core + 0.08 * rng.standard_normal((_BATCH, _BATCH_INNER, _BATCH_INNER))
    right = envelope * b_core + 0.08 * rng.standard_normal((_BATCH, _BATCH_INNER, _BATCH_INNER))

    trace.bind("A", left)
    trace.bind("B", right)
    result = einsum("bij,bjk->bik", left, right, trace=trace, backend="numpy")
    keepalive = (left, right, np.asarray(result))
    return trace, keepalive


class _DemoTensorNetworkNode:
    """Minimal node compatible with the TensorNetwork path of ``show_tensor_elements``."""

    def __init__(
        self,
        name: str,
        axis_names: tuple[str, ...],
        tensor: np.ndarray[Any, Any],
    ) -> None:
        self.name = name
        self.axis_names = axis_names
        self.tensor = tensor


def build_structured_network() -> list[_DemoTensorNetworkNode]:
    c, h, w = _RANK3
    zc = np.linspace(-1.0, 1.0, c, dtype=np.float64)[:, np.newaxis, np.newaxis]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float64)[np.newaxis, :, np.newaxis]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float64)[np.newaxis, np.newaxis, :]
    core = np.sin(5.0 * x * y) * np.cos(4.0 * zc + 2.0 * (x**2 + y**2))
    phase = np.exp(1j * np.pi * (0.25 * zc + 0.4 * x - 0.3 * y))
    psi = (core * phase.real + 0.15j * np.sin(3.0 * zc * y)).astype(np.complex128)

    g = np.linspace(-3.0, 3.0, _GRID_SIDE, dtype=np.float64)
    gx = g[:, np.newaxis]
    gy = g[np.newaxis, :]
    lattice = np.sin(gx * 2.1) * np.sin(gy * 2.3) + 0.45 * np.sin((gx + gy) * 1.7)

    return [
        _DemoTensorNetworkNode("Psi", ("channel", "y", "x"), psi),
        _DemoTensorNetworkNode("Lattice", ("row", "col"), lattice),
    ]


def run_matvec_demo(*, show: bool = True) -> tuple[Figure, RenderedTensorAxes]:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    from tensor_network_viz import show_tensor_elements

    trace, keepalive = build_matvec_trace()
    trace._tensor_elements_demo_keepalive = keepalive  # type: ignore[attr-defined]
    return show_tensor_elements(
        trace,
        config=_demo_config(),
        show_controls=True,
        show=show,
    )


def run_batch_matmul_demo(*, show: bool = True) -> tuple[Figure, RenderedTensorAxes]:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    from tensor_network_viz import show_tensor_elements

    trace, keepalive = build_batch_matmul_trace()
    trace._tensor_elements_demo_keepalive = keepalive  # type: ignore[attr-defined]
    return show_tensor_elements(
        trace,
        config=_demo_config(),
        show_controls=True,
        show=show,
    )


def run_structured_demo(*, show: bool = True) -> tuple[Figure, RenderedTensorAxes]:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    from tensor_network_viz import show_tensor_elements

    nodes = build_structured_network()
    return show_tensor_elements(
        nodes,
        config=_demo_config(),
        show_controls=True,
        show=show,
    )


DemoName: TypeAlias = Literal["matvec", "batch", "structured"]


def main(
    *,
    show: bool = True,
    demo: DemoName = "matvec",
) -> tuple[Figure, RenderedTensorAxes]:
    if demo == "matvec":
        return run_matvec_demo(show=show)
    if demo == "batch":
        return run_batch_matmul_demo(show=show)
    return run_structured_demo(show=show)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small show_tensor_elements demos with grouped Matplotlib controls.",
    )
    parser.add_argument(
        "--demo",
        choices=("matvec", "batch", "structured"),
        default="matvec",
        help=(
            "matvec: traced matrix-vector; batch: traced batched matmul; "
            "structured: complex 3D tensor plus lattice."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(show=True, demo=args.demo)
