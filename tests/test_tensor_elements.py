from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent

from tensor_network_viz import TensorElementsConfig, pair_tensor, show_tensor_elements
from tensor_network_viz._tensor_elements_support import (
    _HeatmapPayload,
    _HistogramPayload,
    _matrixize_tensor,
    _prepare_mode_payload,
    _resolve_matrix_axes,
    _TensorRecord,
    _TextSummaryPayload,
)


class DummyTensorNetworkNode:
    def __init__(self, tensor: np.ndarray, *, name: str, axis_names: tuple[str, ...]) -> None:
        self.tensor = tensor
        self.name = name
        self.axis_names = list(axis_names)
        self.shape = tensor.shape
        self.edges: list[object | None] = [None] * len(self.axis_names)


class DummyTensorKrowchNode:
    def __init__(
        self,
        *,
        name: str,
        axes_names: tuple[str, ...],
        tensor: Any | None,
        shape: tuple[int, ...],
    ) -> None:
        self.name = name
        self.axes_names = list(axes_names)
        self.tensor = tensor
        self.shape = shape
        self.edges: list[object | None] = [None] * len(self.axes_names)


class DummyTensorKrowchNetwork:
    def __init__(self, nodes: list[DummyTensorKrowchNode]) -> None:
        self.nodes = nodes


class DummyQuimbTensor:
    def __init__(self, data: np.ndarray, *, inds: tuple[str, ...], tags: tuple[str, ...]) -> None:
        self.data = data
        self.inds = inds
        self.tags = set(tags)
        self.shape = data.shape


def _widget_center_event(fig: matplotlib.figure.Figure, artist: object) -> MouseEvent:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer)  # type: ignore[attr-defined]
    x = int(round((bbox.x0 + bbox.x1) / 2.0))
    y = int(round((bbox.y0 + bbox.y1) / 2.0))
    return MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT)


def _click_radio_label(radio: object, label_index: int) -> None:
    radio_widget = radio  # keep helper simple for private-widget smoke tests
    event = _widget_center_event(radio_widget.ax.figure, radio_widget.labels[label_index])
    radio_widget._clicked(event)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_tensor_elements_config_has_expected_defaults() -> None:
    config = TensorElementsConfig()

    assert config.mode == "auto"
    assert config.figsize == (7.2, 6.4)
    assert config.row_axes is None
    assert config.col_axes is None
    assert config.max_matrix_shape == (256, 256)
    assert config.histogram_bins == 40
    assert config.histogram_max_samples == 100_000


def test_tensor_elements_config_supports_grouped_modes() -> None:
    config = TensorElementsConfig(mode="phase")

    assert config.mode == "phase"


def test_show_tensor_elements_public_signature_matches_config_centric_style() -> None:
    signature = inspect.signature(show_tensor_elements)

    assert tuple(signature.parameters) == (
        "data",
        "engine",
        "config",
        "ax",
        "show_controls",
        "show",
    )


def test_show_tensor_elements_returns_fig_ax_for_single_tensor_with_autodetect() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="Left",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False)

    assert fig is ax.figure
    assert ax.images
    assert "Left" in ax.get_title()


def test_show_tensor_elements_supports_single_quimb_like_tensor() -> None:
    tensor = DummyQuimbTensor(
        np.arange(6, dtype=float).reshape(2, 3),
        inds=("a", "b"),
        tags=("Q0",),
    )

    fig, ax = show_tensor_elements(tensor, show=False)

    assert fig is ax.figure
    assert ax.images
    assert "quimb" in ax.get_title().lower()


def test_prepare_mode_payload_returns_typed_heatmap_payload() -> None:
    record = _TensorRecord(
        array=np.arange(6, dtype=float).reshape(2, 3),
        name="Heatmap",
        axis_names=("row", "col"),
        engine="tensornetwork",
    )

    resolved_mode, payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="elements"),
        mode="elements",
    )

    assert resolved_mode == "elements"
    assert isinstance(payload, _HeatmapPayload)
    assert tuple(payload.matrix.shape) == (2, 3)
    assert payload.mode_label == "elements"
    assert payload.colorbar_label == "value"


def test_prepare_mode_payload_returns_typed_non_heatmap_payloads() -> None:
    record = _TensorRecord(
        array=np.arange(6, dtype=float).reshape(2, 3),
        name="PayloadKinds",
        axis_names=("row", "col"),
        engine="tensornetwork",
    )

    distribution_mode, distribution_payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="distribution"),
        mode="distribution",
    )
    data_mode, data_payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="data"),
        mode="data",
    )

    assert distribution_mode == "distribution"
    assert isinstance(distribution_payload, _HistogramPayload)
    assert distribution_payload.xlabel == "value"
    assert distribution_payload.values.ndim == 1

    assert data_mode == "data"
    assert isinstance(data_payload, _TextSummaryPayload)
    assert "shape:" in data_payload.text.lower()


def test_show_tensor_elements_multiple_tensors_use_slider_and_single_axes() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(6, dtype=float).reshape(2, 3),
            name="A",
            axis_names=("x", "y"),
        ),
        DummyTensorNetworkNode(
            np.arange(12, dtype=float).reshape(3, 4),
            name="B",
            axis_names=("u", "v"),
        ),
    ]

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert isinstance(ax, Axes)
    assert ax.figure is fig
    assert controller._slider is not None
    assert "A" in ax.get_title()

    controller._slider.set_val(1.0)

    assert "B" in ax.get_title()


def test_show_tensor_elements_reuses_prepared_payloads_for_revisited_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz.tensor_elements as tensor_elements_module

    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="CachedModes",
        axis_names=("row", "col"),
    )
    counts: dict[str, int] = {}
    original_prepare_mode_payload = tensor_elements_module._prepare_mode_payload

    def counting_prepare_mode_payload(
        record: Any,
        *,
        config: TensorElementsConfig,
        mode: str,
    ) -> tuple[str, Any]:
        counts[mode] = counts.get(mode, 0) + 1
        return original_prepare_mode_payload(record, config=config, mode=mode)

    monkeypatch.setattr(
        tensor_elements_module,
        "_prepare_mode_payload",
        counting_prepare_mode_payload,
    )

    fig, _ = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    controller.set_mode("distribution")
    controller.set_mode("elements")
    controller.set_mode("distribution")

    assert counts["elements"] == 1
    assert counts["distribution"] == 1


def test_show_tensor_elements_rejects_multi_tensor_with_explicit_ax() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(4, dtype=float).reshape(2, 2),
            name="A",
            axis_names=("a", "b"),
        ),
        DummyTensorNetworkNode(
            np.arange(4, dtype=float).reshape(2, 2),
            name="B",
            axis_names=("a", "b"),
        ),
    ]
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="single tensor"):
        show_tensor_elements(tensors, ax=ax, show=False)

    plt.close(fig)


def test_show_tensor_elements_rejects_controls_on_external_multi_axes_figure() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(4, dtype=float).reshape(2, 2),
        name="A",
        axis_names=("a", "b"),
    )
    fig, axes = plt.subplots(1, 2)

    with pytest.raises(ValueError, match="external ax"):
        show_tensor_elements(tensor, ax=axes[0], show_controls=True, show=False)

    plt.close(fig)


def test_show_tensor_elements_uses_magnitude_by_default_for_complex_tensors() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128),
        name="Psi",
        axis_names=("batch", "state"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=False)

    assert fig is ax.figure
    assert "magnitude" in ax.get_title().lower()


def test_show_tensor_elements_real_mode_renders_real_component() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128),
        name="Psi",
        axis_names=("batch", "state"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="real"),
        show=False,
        show_controls=False,
    )

    assert fig is ax.figure
    assert "real" in ax.get_title().lower()


def test_show_tensor_elements_imag_mode_renders_imag_component() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128),
        name="PsiImag",
        axis_names=("batch", "state"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="imag"),
        show=False,
        show_controls=False,
    )

    assert fig is ax.figure
    assert "imag" in ax.get_title().lower()


def test_show_tensor_elements_phase_mode_produces_heatmap() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 0.0j, 1.0j]], dtype=np.complex128),
        name="Phase",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="phase"),
        show=False,
        show_controls=False,
    )

    assert fig is ax.figure
    assert ax.images
    assert "phase" in ax.get_title().lower()


def test_show_tensor_elements_real_tensor_rejects_phase_mode_cleanly() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="A",
        axis_names=("x", "y"),
    )

    with pytest.raises(ValueError, match="phase"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="phase"),
            show=False,
            show_controls=False,
        )


def test_show_tensor_elements_real_tensor_rejects_phase_mode_even_with_controls() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="A-controls",
        axis_names=("x", "y"),
    )

    with pytest.raises(ValueError, match="phase"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="phase"),
            show=False,
            show_controls=True,
        )


def test_show_tensor_elements_sign_mode_is_discrete() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[-3.0, 0.0, 2.0]], dtype=float),
        name="Sign",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="sign"),
        show=False,
        show_controls=False,
    )
    image = ax.images[0].get_array()
    image_array = np.asarray(image, dtype=float)

    assert fig is ax.figure
    assert "sign" in ax.get_title().lower()
    assert set(np.unique(image_array)) <= {-1.0, 0.0, 1.0}


def test_show_tensor_elements_signed_value_mode_is_continuous() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[-1.0, 0.0, 2.0]], dtype=float),
        name="SignedValue",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="signed_value"),
        show=False,
        show_controls=False,
    )

    assert fig is ax.figure
    assert "signed value" in ax.get_title().lower()


def test_show_tensor_elements_data_mode_uses_main_axis_for_textual_summary() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="Info",
        axis_names=("left", "right"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="data"),
        show=False,
        show_controls=False,
    )
    all_text = "\n".join(text.get_text() for text in ax.texts)

    assert fig is ax.figure
    assert "data" in ax.get_title().lower()
    assert not ax.images
    assert "shape:" in all_text.lower()
    assert "dtype:" in all_text.lower()


def test_show_tensor_elements_downsamples_large_heatmaps() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(512 * 384, dtype=float).reshape(512, 384),
        name="Big",
        axis_names=("rows", "cols"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(max_matrix_shape=(64, 32)),
        show=False,
        show_controls=False,
    )

    image = ax.images[0].get_array()
    assert image is not None
    image_array = np.asarray(image, dtype=float)
    assert tuple(image_array.shape) == (64, 32)
    assert fig is ax.figure


def test_show_tensor_elements_heatmap_adds_colorbar_axis_on_the_right() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="Color",
        axis_names=("left", "right"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=False)

    assert len(fig.axes) == 2
    colorbar_ax = next(axis for axis in fig.axes if axis is not ax)
    assert colorbar_ax.get_position().x0 >= ax.get_position().x1


def test_show_tensor_elements_phase_mode_labels_colorbar() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 0.0j, 1.0j]], dtype=np.complex128),
        name="PhaseColorbar",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="phase"),
        show=False,
        show_controls=False,
    )

    colorbar_ax = next(axis for axis in fig.axes if axis is not ax)

    assert "phase" in colorbar_ax.get_ylabel().lower()


def test_show_tensor_elements_rejects_shape_only_tensorkrowch_nodes() -> None:
    node = DummyTensorKrowchNode(
        name="shape-only",
        axes_names=("a", "b"),
        tensor=None,
        shape=(2, 3),
    )
    network = DummyTensorKrowchNetwork([node])

    with pytest.raises(ValueError, match="materialized"):
        show_tensor_elements(network, show=False)


def test_show_tensor_elements_rejects_manual_pair_tensor_iterables() -> None:
    with pytest.raises(TypeError, match="tensor values"):
        show_tensor_elements([pair_tensor("A", "x", "r0", "ab,b->a")], show=False)


def test_resolve_matrix_axes_balances_high_rank_tensors() -> None:
    row_axes, col_axes = _resolve_matrix_axes(
        shape=(2, 3, 5, 7),
        row_axes=None,
        col_axes=None,
        axis_names=("a", "b", "c", "d"),
    )

    assert not set(row_axes).intersection(col_axes)
    assert set(row_axes) | set(col_axes) == {0, 1, 2, 3}
    dims = (2, 3, 5, 7)
    row_product = int(np.prod([dims[index] for index in row_axes], dtype=int))
    col_product = int(np.prod([dims[index] for index in col_axes], dtype=int))
    assert {row_product, col_product} == {14, 15}


def test_matrixize_tensor_honors_explicit_row_and_column_axes() -> None:
    matrix, metadata = _matrixize_tensor(
        np.arange(2 * 3 * 5 * 7, dtype=float).reshape(2, 3, 5, 7),
        axis_names=("a", "b", "c", "d"),
        row_axes=(0, 2),
        col_axes=(1, 3),
    )

    assert matrix.shape == (10, 21)
    assert metadata.row_axes == (0, 2)
    assert metadata.col_axes == (1, 3)


def test_show_tensor_elements_widgets_switch_modes() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="Widget",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    _click_radio_label(controller._mode_radio, 2)

    assert fig is ax.figure
    assert "distribution" in ax.get_title().lower()


def test_show_tensor_elements_widgets_offer_data_mode() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="WidgetData",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    _click_radio_label(controller._mode_radio, 3)
    text_blob = "\n".join(text.get_text() for text in ax.texts)

    assert fig is ax.figure
    assert "data" in ax.get_title().lower()
    assert "shape:" in text_blob.lower()


def test_show_tensor_elements_widgets_switch_group_then_mode() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128),
        name="WidgetComplex",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    _click_radio_label(controller._group_radio, 1)
    _click_radio_label(controller._mode_radio, 1)

    assert fig is ax.figure
    assert "imag" in ax.get_title().lower()


def test_show_tensor_elements_slider_keeps_group_and_falls_back_to_valid_mode() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.array([[1.0 + 1.0j]], dtype=np.complex128),
            name="C",
            axis_names=("a", "b"),
        ),
        DummyTensorNetworkNode(
            np.array([[1.0]], dtype=float),
            name="R",
            axis_names=("a", "b"),
        ),
    ]

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    _click_radio_label(controller._group_radio, 1)
    _click_radio_label(controller._mode_radio, 2)
    controller._slider.set_val(1.0)

    assert fig is ax.figure
    assert "real" in ax.get_title().lower()
