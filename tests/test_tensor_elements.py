from __future__ import annotations

import gc
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

from plotting_helpers import assert_rendered_figure
from tensor_network_viz import (
    EinsumTrace,
    TensorElementsConfig,
    einsum,
    pair_tensor,
    show_tensor_elements,
)
from tensor_network_viz._tensor_elements_data import (
    _extract_einsum_playback_step_records,
    _extract_playback_step_records,
)
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


class DummyTensorKrowchContractedNetwork:
    def __init__(
        self,
        *,
        nodes: list[DummyTensorKrowchNode],
        leaf_nodes: list[DummyTensorKrowchNode],
        resultant_nodes: list[DummyTensorKrowchNode],
    ) -> None:
        self.nodes = nodes
        self.leaf_nodes = leaf_nodes
        self.resultant_nodes = resultant_nodes


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
    assert config.topk_count == 8
    assert config.zero_threshold == pytest.approx(1e-12)
    assert config.log_magnitude_floor == pytest.approx(1e-12)
    assert config.robust_percentiles is None
    assert config.shared_color_scale is False
    assert config.highlight_outliers is False
    assert config.outlier_zscore == pytest.approx(3.5)


def test_tensor_elements_config_supports_grouped_modes() -> None:
    config = TensorElementsConfig(mode="log_magnitude")

    assert config.mode == "log_magnitude"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"topk_count": 0}, "topk_count"),
        ({"zero_threshold": 0.0}, "zero_threshold"),
        ({"log_magnitude_floor": 0.0}, "log_magnitude_floor"),
        ({"outlier_zscore": 0.0}, "outlier_zscore"),
        ({"robust_percentiles": (-1.0, 90.0)}, "robust_percentiles"),
        ({"robust_percentiles": (90.0, 90.0)}, "robust_percentiles"),
        ({"robust_percentiles": (20.0, 101.0)}, "robust_percentiles"),
    ],
)
def test_tensor_elements_config_validates_numeric_inputs(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TensorElementsConfig(**kwargs)


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

    assert_rendered_figure(fig, ax)
    assert ax.images
    assert "Left" in ax.get_title()


def test_show_tensor_elements_supports_single_quimb_like_tensor() -> None:
    tensor = DummyQuimbTensor(
        np.arange(6, dtype=float).reshape(2, 3),
        inds=("a", "b"),
        tags=("Q0",),
    )

    fig, ax = show_tensor_elements(tensor, show=False)

    assert_rendered_figure(fig, ax)
    assert ax.images
    assert "quimb" in ax.get_title().lower()


def test_extract_einsum_playback_step_records_follow_trace_order_and_output_axes() -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    mid = np.arange(12, dtype=float).reshape(3, 4)
    right = np.arange(8, dtype=float).reshape(4, 2)

    trace.bind("Left", left)
    trace.bind("Mid", mid)
    trace.bind("Right", right)
    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, r0, r1]  # type: ignore[attr-defined]

    step_records = _extract_einsum_playback_step_records(trace)

    assert [step.result_name for step in step_records] == ["r0", "r1"]
    assert [step.record.name if step.record is not None else None for step in step_records] == [
        "r0",
        "r1",
    ]
    assert step_records[0].record is not None
    assert step_records[1].record is not None
    assert step_records[0].record.axis_names == ("a", "c")
    assert step_records[1].record.axis_names == ("a", "d")
    assert tuple(step_records[0].record.array.shape) == (2, 4)
    assert tuple(step_records[1].record.array.shape) == (2, 2)


def test_extract_einsum_playback_step_records_marks_missing_intermediate_tensors() -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    mid = np.arange(12, dtype=float).reshape(3, 4)
    right = np.arange(8, dtype=float).reshape(4, 2)

    r0 = einsum("ab,bc->ac", left, mid, trace=trace, backend="numpy")
    r1 = einsum("ac,cd->ad", r0, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, mid, right, r1]  # type: ignore[attr-defined]
    del r0
    gc.collect()

    step_records = _extract_einsum_playback_step_records(trace)

    assert [step.result_name for step in step_records] == ["r0", "r1"]
    assert step_records[0].record is None
    assert step_records[1].record is not None
    assert step_records[1].record.name == "r1"


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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
    assert "signed value" in ax.get_title().lower()


def test_show_tensor_elements_log_magnitude_mode_uses_log_scaled_magnitude() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, 100.0]], dtype=float),
        name="LogMag",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="log_magnitude"),
        show=False,
        show_controls=False,
    )
    image_array = np.asarray(ax.images[0].get_array(), dtype=float)

    assert_rendered_figure(fig, ax)
    assert "log" in ax.get_title().lower()
    np.testing.assert_allclose(image_array, np.array([[0.0, 2.0]]))


def test_show_tensor_elements_distribution_mode_filters_nonfinite_values() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, np.nan, np.inf, -np.inf, 2.0]], dtype=float),
        name="SpecialDistribution",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="distribution"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "distribution" in ax.get_title().lower()
    assert ax.patches


def test_show_tensor_elements_sparsity_mode_marks_near_zero_entries() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, 1e-14, 1e-3]], dtype=float),
        name="Sparse",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="sparsity", zero_threshold=1e-12),
        show=False,
        show_controls=False,
    )
    image_array = np.asarray(ax.images[0].get_array(), dtype=float)

    assert_rendered_figure(fig, ax)
    assert "sparsity" in ax.get_title().lower()
    np.testing.assert_array_equal(image_array, np.array([[1.0, 1.0, 0.0]]))


def test_show_tensor_elements_nan_inf_mode_marks_special_values() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, np.nan, np.inf, -np.inf]], dtype=float),
        name="Specials",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="nan_inf"),
        show=False,
        show_controls=False,
    )
    image_array = np.asarray(ax.images[0].get_array(), dtype=float)

    assert_rendered_figure(fig, ax)
    np.testing.assert_array_equal(image_array, np.array([[0.0, 1.0, 2.0, 3.0]]))


def test_show_tensor_elements_nan_inf_mode_marks_complex_nonfinite_components() -> None:
    tensor = DummyTensorNetworkNode(
        np.array(
            [
                [
                    complex(1.0, 0.0),
                    complex(np.nan, 0.0),
                    complex(1.0, np.inf),
                    complex(-np.inf, 0.0),
                ]
            ],
            dtype=np.complex128,
        ),
        name="ComplexSpecials",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="nan_inf"),
        show=False,
        show_controls=False,
    )
    image_array = np.asarray(ax.images[0].get_array(), dtype=float)

    assert_rendered_figure(fig, ax)
    np.testing.assert_array_equal(image_array, np.array([[0.0, 1.0, 2.0, 3.0]]))


def test_show_tensor_elements_singular_values_mode_renders_ordered_spectrum() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),
        name="Spectrum",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "singular values" in ax.get_title().lower()
    assert ax.get_yscale() == "linear"
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([3.0, 1.0]))


def test_show_tensor_elements_singular_values_mode_switches_to_log_scale_for_wide_ranges() -> None:
    tensor = DummyTensorNetworkNode(
        np.diag(np.array([1_000_000.0, 1.0], dtype=float)),
        name="WideSpectrum",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.get_yscale() == "log"


def test_show_tensor_elements_singular_values_mode_supports_complex_tensors() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[3.0 + 4.0j, 0.0], [0.0, 2.0j]], dtype=np.complex128),
        name="ComplexSpectrum",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([5.0, 2.0]))


def test_show_tensor_elements_singular_values_mode_uses_rank3_matrixization() -> None:
    tensor_data = np.arange(24, dtype=float).reshape(2, 3, 4)
    tensor = DummyTensorNetworkNode(
        tensor_data,
        name="Rank3Spectrum",
        axis_names=("a", "b", "c"),
    )
    expected_matrix, _ = _matrixize_tensor(
        tensor_data,
        axis_names=("a", "b", "c"),
        row_axes=None,
        col_axes=None,
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        np.linalg.svd(expected_matrix, compute_uv=False),
    )


def test_show_tensor_elements_singular_values_mode_supports_scalar_tensors() -> None:
    tensor = DummyTensorNetworkNode(
        np.asarray(3.0, dtype=float),
        name="ScalarSpectrum",
        axis_names=(),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([3.0]))


def test_show_tensor_elements_eigen_real_mode_renders_ranked_real_parts() -> None:
    tensor = DummyTensorNetworkNode(
        np.diag(np.array([3.0 + 4.0j, -1.0 + 2.0j], dtype=np.complex128)),
        name="EigenReal",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="eigen_real"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "eigenvalues (real)" in ax.get_title().lower()
    assert ax.get_yscale() == "linear"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([3.0, -1.0]))


def test_show_tensor_elements_eigen_imag_mode_renders_ranked_imag_parts() -> None:
    tensor = DummyTensorNetworkNode(
        np.diag(np.array([3.0 + 4.0j, -1.0 + 2.0j], dtype=np.complex128)),
        name="EigenImag",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="eigen_imag"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "eigenvalues (imag)" in ax.get_title().lower()
    assert ax.get_yscale() == "linear"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([4.0, 2.0]))


def test_show_tensor_elements_nonfinite_tensor_rejects_singular_values_mode() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, np.nan], [0.0, 1.0]], dtype=float),
        name="NonFiniteSpectrum",
        axis_names=("row", "col"),
    )

    with pytest.raises(ValueError, match="singular_values"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="singular_values"),
            show=False,
            show_controls=False,
        )


def test_show_tensor_elements_non_square_tensor_rejects_eigen_modes() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="RectangularSpectrum",
        axis_names=("row", "col"),
    )

    with pytest.raises(ValueError, match="eigen_real"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="eigen_real"),
            show=False,
            show_controls=False,
        )

    with pytest.raises(ValueError, match="eigen_imag"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="eigen_imag"),
            show=False,
            show_controls=False,
        )


def test_show_tensor_elements_nonfinite_tensor_rejects_eigen_modes() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, np.nan], [0.0, 1.0]], dtype=float),
        name="NonFiniteEigen",
        axis_names=("row", "col"),
    )

    with pytest.raises(ValueError, match="eigen_real"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="eigen_real"),
            show=False,
            show_controls=False,
        )

    with pytest.raises(ValueError, match="eigen_imag"):
        show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="eigen_imag"),
            show=False,
            show_controls=False,
        )


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

    assert_rendered_figure(fig, ax)
    assert "data" in ax.get_title().lower()
    assert not ax.images
    assert "shape:" in all_text.lower()
    assert "dtype:" in all_text.lower()


def test_show_tensor_elements_data_mode_includes_axis_summary_and_topk() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, 9.0, 3.0], [7.0, 2.0, 8.0]], dtype=float),
        name="DetailedInfo",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="data", topk_count=3),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts)

    assert_rendered_figure(fig, ax)
    assert "axis summary:" in text_blob.lower()
    assert "top 3 by magnitude:" in text_blob.lower()
    assert "row (size=2)" in text_blob
    assert "col (size=3)" in text_blob
    assert "row=0, col=1" in text_blob
    assert "row=1, col=2" in text_blob


def test_show_tensor_elements_data_mode_uses_magnitude_for_complex_topk() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[3.0 + 4.0j, 6.0 + 0.0j]], dtype=np.complex128),
        name="ComplexInfo",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="data", topk_count=1),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts)

    assert_rendered_figure(fig, ax)
    assert "mean|x|" in text_blob
    assert "row=0, col=1" in text_blob


def test_show_tensor_elements_data_mode_excludes_spectral_analysis_details() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),
        name="SquareSpectrum",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="data", topk_count=2),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts)

    assert_rendered_figure(fig, ax)
    assert "spectral analysis:" not in text_blob.lower()
    assert "matrixized shape:" not in text_blob.lower()
    assert "stable rank:" not in text_blob.lower()
    assert "condition number:" not in text_blob.lower()
    assert "spectral radius:" not in text_blob.lower()


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
    image_array = np.asarray(image, dtype=float)
    assert tuple(image_array.shape) == (64, 32)
    assert_rendered_figure(fig, ax)


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


def test_show_tensor_elements_nan_inf_mode_labels_colorbar_states() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, np.nan, np.inf, -np.inf]], dtype=float),
        name="SpecialColorbar",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="nan_inf"),
        show=False,
        show_controls=False,
    )
    colorbar_ax = next(axis for axis in fig.axes if axis is not ax)
    labels = [tick.get_text() for tick in colorbar_ax.get_yticklabels()]

    assert labels == ["finite", "NaN", "+Inf", "-Inf"]


def test_show_tensor_elements_robust_scaling_ignores_outliers_and_nonfinite_values() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1000.0, np.nan, np.inf]]),
        name="Robust",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements", robust_percentiles=(5.0, 95.0)),
        show=False,
        show_controls=False,
    )
    low, high = ax.images[0].get_clim()

    assert_rendered_figure(fig, ax)
    assert np.isfinite(low)
    assert np.isfinite(high)
    assert high < 1000.0


def test_show_tensor_elements_shared_color_scale_reuses_limits_across_slider() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.array([[0.0, 1.0]], dtype=float),
            name="Small",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            np.array([[0.0, 100.0]], dtype=float),
            name="Large",
            axis_names=("row", "col"),
        ),
    ]

    fig, ax = show_tensor_elements(
        tensors,
        config=TensorElementsConfig(mode="elements", shared_color_scale=True),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    first_clim = ax.images[0].get_clim()

    controller._slider.set_val(1.0)
    second_clim = ax.images[0].get_clim()

    assert first_clim == pytest.approx(second_clim)


def test_show_tensor_elements_signed_value_robust_scaling_stays_symmetric() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[-1.0, 0.0, 2.0, 100.0]], dtype=float),
        name="SymmetricRobust",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="signed_value", robust_percentiles=(0.0, 75.0)),
        show=False,
        show_controls=False,
    )
    low, high = ax.images[0].get_clim()

    assert_rendered_figure(fig, ax)
    assert high < 100.0
    assert low == pytest.approx(-high)


def test_show_tensor_elements_outlier_overlay_appears_for_continuous_heatmaps() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, 0.0, 1.0, 10.0]], dtype=float),
        name="OutlierOverlay",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="elements",
            highlight_outliers=True,
            outlier_zscore=3.5,
        ),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert len(ax.collections) == 1


def test_show_tensor_elements_outlier_overlay_skips_discrete_heatmaps() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[0.0, 0.0, 1.0, 10.0]], dtype=float),
        name="DiscreteOverlay",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="sparsity",
            highlight_outliers=True,
            outlier_zscore=3.5,
        ),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert not ax.collections


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


def test_extract_playback_step_records_keeps_einsum_behavior() -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(15, dtype=float).reshape(3, 5)
    trace.bind("A", left)
    trace.bind("B", right)
    result = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, right, result]  # type: ignore[attr-defined]

    records = _extract_playback_step_records(trace)

    assert records is not None
    assert len(records) == 1
    assert records[0].record is not None
    assert records[0].result_name == "r0"


def test_extract_playback_step_records_returns_none_for_tensorless_tensorkrowch_history() -> None:
    left = DummyTensorKrowchNode(
        name="A",
        axes_names=("a", "b"),
        tensor=np.ones((2, 3)),
        shape=(2, 3),
    )
    right = DummyTensorKrowchNode(
        name="B",
        axes_names=("b", "c"),
        tensor=np.ones((3, 5)),
        shape=(3, 5),
    )
    result = DummyTensorKrowchNode(
        name="contract_edges",
        axes_names=("a", "c"),
        tensor=None,
        shape=(2, 5),
    )
    left.successors = {  # type: ignore[attr-defined]
        "contract_edges": {
            (left, right): type("S", (), {"node_ref": (left, right), "child": result})()
        }
    }
    right.successors = {}  # type: ignore[attr-defined]
    network = DummyTensorKrowchContractedNetwork(
        nodes=[left, right, result],
        leaf_nodes=[left, right],
        resultant_nodes=[result],
    )

    records = _extract_playback_step_records(network)

    assert records is None


def test_extract_playback_step_records_supports_contracted_tensorkrowch_network() -> None:
    tk = pytest.importorskip("tensorkrowch")
    torch = pytest.importorskip("torch")

    network = tk.TensorNetwork(name="demo")
    left = tk.Node(
        tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        axes_names=("input", "bond"),
        name="left",
        network=network,
    )
    right = tk.Node(
        tensor=torch.arange(15, dtype=torch.float32).reshape(3, 5),
        axes_names=("bond", "output"),
        name="right",
        network=network,
    )
    left["bond"] ^ right["bond"]
    _ = left @ right

    records = _extract_playback_step_records(network)

    assert records is not None
    assert len(records) == 1
    assert records[0].record is not None
    assert records[0].record.engine == "tensorkrowch"
    assert records[0].record.axis_names == ("input", "output")


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
    _click_radio_label(controller._mode_radio, 3)

    assert_rendered_figure(fig, ax)
    assert "distribution" in ax.get_title().lower()


def test_show_tensor_elements_widgets_offer_data_mode() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="WidgetData",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    _click_radio_label(controller._mode_radio, 4)
    text_blob = "\n".join(text.get_text() for text in ax.texts)

    assert_rendered_figure(fig, ax)
    assert "data" in ax.get_title().lower()
    assert "shape:" in text_blob.lower()


def test_show_tensor_elements_widgets_offer_new_basic_and_diagnostic_modes() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(4, dtype=float).reshape(2, 2),
        name="WidgetModes",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    basic_modes = tuple(text.get_text() for text in controller._mode_radio.labels)

    _click_radio_label(controller._group_radio, 2)
    diagnostic_modes = tuple(text.get_text() for text in controller._mode_radio.labels)

    assert_rendered_figure(fig, ax)
    assert basic_modes == ("elements", "magnitude", "log_magnitude", "distribution", "data")
    assert diagnostic_modes == (
        "sign",
        "signed_value",
        "sparsity",
        "nan_inf",
        "singular_values",
        "eigen_real",
        "eigen_imag",
    )


def test_show_tensor_elements_widgets_hide_eigen_modes_for_non_square_tensor() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),
            name="FiniteSpectrum",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            np.arange(6, dtype=float).reshape(2, 3),
            name="RectangularSpectrum",
            axis_names=("row", "col"),
        ),
    ]

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 2)
    square_modes = tuple(text.get_text() for text in controller._mode_radio.labels)
    controller._slider.set_val(1.0)
    rectangular_modes = tuple(text.get_text() for text in controller._mode_radio.labels)

    assert_rendered_figure(fig, ax)
    assert "singular_values" in square_modes
    assert "eigen_real" in square_modes
    assert "eigen_imag" in square_modes
    assert "singular_values" in rectangular_modes
    assert "eigen_real" not in rectangular_modes
    assert "eigen_imag" not in rectangular_modes


def test_show_tensor_elements_widgets_hide_spectral_modes_for_nonfinite_tensor() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),
            name="FiniteSpectrum",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            np.array([[1.0, np.nan], [0.0, 1.0]], dtype=float),
            name="NonFiniteSpectrum",
            axis_names=("row", "col"),
        ),
    ]

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 2)
    finite_modes = tuple(text.get_text() for text in controller._mode_radio.labels)
    controller._slider.set_val(1.0)
    nonfinite_modes = tuple(text.get_text() for text in controller._mode_radio.labels)

    assert_rendered_figure(fig, ax)
    assert "singular_values" in finite_modes
    assert "eigen_real" in finite_modes
    assert "eigen_imag" in finite_modes
    assert "singular_values" not in nonfinite_modes
    assert "eigen_real" not in nonfinite_modes
    assert "eigen_imag" not in nonfinite_modes


def test_show_tensor_elements_group_selector_fits_diagnostic_label() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="WidgetModes",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    assert controller._group_radio is not None

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    diagnostic_label = controller._group_radio.labels[2]
    label_bbox = diagnostic_label.get_window_extent(renderer)
    axis_bbox = controller._group_radio.ax.get_window_extent(renderer)

    assert_rendered_figure(fig, ax)
    assert label_bbox.x0 >= axis_bbox.x0
    assert label_bbox.x1 <= axis_bbox.x1


def test_show_tensor_elements_controls_use_same_tray_style_as_viewer_controls() -> None:
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

    assert_rendered_figure(fig, ax)
    assert controller._group_radio_ax is not None
    assert controller._mode_radio_ax is not None
    assert controller._slider_ax is not None

    for control_ax in (
        controller._group_radio_ax,
        controller._mode_radio_ax,
        controller._slider_ax,
    ):
        assert control_ax.patch.get_facecolor() == pytest.approx((0.97, 0.97, 0.99, 0.88))
        assert control_ax.patch.get_linewidth() == pytest.approx(0.6)
        for spine in control_ax.spines.values():
            assert spine.get_visible()
            assert spine.get_linewidth() == pytest.approx(0.6)


def test_show_tensor_elements_mode_selector_sits_lower_than_plot_area() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="WidgetModes",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert_rendered_figure(fig, ax)
    assert controller._mode_radio_ax is not None
    mode_bounds = controller._mode_radio_ax.get_position().bounds
    mode_top = mode_bounds[1] + mode_bounds[3]

    assert float(mode_top) <= 0.19


def test_show_tensor_elements_slider_sits_farther_right_of_mode_selector() -> None:
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

    assert_rendered_figure(fig, ax)
    assert controller._mode_radio_ax is not None
    assert controller._slider_ax is not None

    mode_bounds = controller._mode_radio_ax.get_position().bounds
    slider_bounds = controller._slider_ax.get_position().bounds
    mode_right = mode_bounds[0] + mode_bounds[2]
    slider_left = slider_bounds[0]

    assert float(slider_left - mode_right) >= 0.075


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

    assert_rendered_figure(fig, ax)
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

    assert_rendered_figure(fig, ax)
    assert "real" in ax.get_title().lower()
