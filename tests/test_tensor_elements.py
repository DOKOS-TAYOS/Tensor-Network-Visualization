from __future__ import annotations

import gc
import inspect
import warnings
from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent

import tensor_network_viz._tensor_elements_data as tensor_elements_data_module
import tensor_network_viz._tensor_elements_inputs as tensor_elements_inputs_module
import tensor_network_viz._tensor_elements_payloads as tensor_elements_payloads_module
from plotting_helpers import assert_rendered_figure
from tensor_network_viz import (
    EinsumTrace,
    TenPyTensorNetwork,
    TensorAnalysisConfig,
    TensorElementsConfig,
    einsum,
    make_tenpy_tensor_network,
    pair_tensor,
    show_tensor_elements,
)
from tensor_network_viz._tensor_elements_data import (
    _build_topk_lines,
    _extract_einsum_playback_step_records,
    _extract_playback_step_records,
)
from tensor_network_viz._tensor_elements_support import (
    _HeatmapPayload,
    _HistogramPayload,
    _matrixize_tensor,
    _prepare_mode_payload,
    _resolve_matrix_axes,
    _SeriesPayload,
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
    def __init__(self, nodes: Iterable[DummyTensorKrowchNode]) -> None:
        self.nodes = nodes


class DummyTensorKrowchContractedNetwork:
    def __init__(
        self,
        *,
        nodes: Iterable[DummyTensorKrowchNode],
        leaf_nodes: Iterable[DummyTensorKrowchNode],
        resultant_nodes: Iterable[DummyTensorKrowchNode],
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


class DummyQuimbNetwork:
    def __init__(self, tensors: Iterable[DummyQuimbTensor]) -> None:
        self.tensors = tensors


class DummyTenPyTensor:
    def __init__(
        self, array: np.ndarray[Any, np.dtype[np.float64]], labels: tuple[str, ...]
    ) -> None:
        self._array = array
        self._labels = labels

    def to_ndarray(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._array

    def get_leg_labels(self) -> tuple[str, ...]:
        return self._labels


def _line_ydata_as_float(ax: Axes) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(ax.lines[0].get_ydata(), dtype=float)


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


def _dispatch_motion_event_at_data(
    ax: Axes,
    *,
    x: float,
    y: float,
) -> MouseEvent:
    ax.figure.canvas.draw()
    x_display, y_display = ax.transData.transform((x, y))
    event = MouseEvent(
        "motion_notify_event",
        ax.figure.canvas,
        int(round(x_display)),
        int(round(y_display)),
    )
    ax.figure.canvas.callbacks.process("motion_notify_event", event)
    return event


def _dispatch_motion_event_outside_figure(fig: matplotlib.figure.Figure) -> MouseEvent:
    event = MouseEvent("motion_notify_event", fig.canvas, -10, -10)
    fig.canvas.callbacks.process("motion_notify_event", event)
    return event


def _visible_axis_ticks(
    ax: Axes, *, axis: Literal["x", "y"]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ticks = np.asarray(ax.get_xticks() if axis == "x" else ax.get_yticks(), dtype=float)
    low, high = ax.get_xlim() if axis == "x" else ax.get_ylim()
    lower = min(float(low), float(high))
    upper = max(float(low), float(high))
    return ticks[(ticks >= lower) & (ticks <= upper)]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_tensor_elements_config_has_expected_defaults() -> None:
    config = TensorElementsConfig()

    assert config.mode == "auto"
    assert config.row_axes is None
    assert config.col_axes is None
    assert config.analysis is None
    assert config.theme == "default"
    assert config.continuous_cmap == "viridis"
    assert config.log_magnitude_cmap == "magma"
    assert config.phase_cmap == "twilight"
    assert config.diverging_cmap == "RdBu_r"
    assert config.sign_colors == ("#B91C1C", "#E2E8F0", "#0369A1")
    assert config.sparsity_colors == ("#0F172A", "#F8FAFC")
    assert config.nan_inf_colors == ("#0F766E", "#D97706", "#7C3AED", "#B91C1C")
    assert config.series_color == "#0369A1"
    assert config.histogram_color == "#0369A1"
    assert config.histogram_edge_color == "#0F172A"
    assert config.zero_marker_color == "#7F1D1D"
    assert config.hover_facecolor == "#F8FAFC"
    assert config.hover_edgecolor == "#CBD5E1"
    assert config.summary_facecolor == "#F8FAFC"
    assert config.summary_edgecolor == "#CBD5E1"
    assert config.figsize == (7.2, 6.4)
    assert config.max_matrix_shape == (256, 256)
    assert config.shared_color_scale is False
    assert config.robust_percentiles is None
    assert config.highlight_outliers is False
    assert config.outlier_zscore == pytest.approx(3.5)
    assert config.histogram_bins == 40
    assert config.histogram_max_samples == 100_000
    assert config.zero_threshold == pytest.approx(1e-12)
    assert config.log_magnitude_floor == pytest.approx(1e-12)
    assert config.topk_count == 8


def test_tensor_elements_config_public_signature_orders_mode_before_detail() -> None:
    signature = inspect.signature(TensorElementsConfig)

    assert tuple(signature.parameters) == (
        "mode",
        "row_axes",
        "col_axes",
        "analysis",
        "theme",
        "continuous_cmap",
        "log_magnitude_cmap",
        "phase_cmap",
        "diverging_cmap",
        "sign_colors",
        "sparsity_colors",
        "nan_inf_colors",
        "series_color",
        "histogram_color",
        "histogram_edge_color",
        "zero_marker_color",
        "hover_facecolor",
        "hover_edgecolor",
        "summary_facecolor",
        "summary_edgecolor",
        "figsize",
        "max_matrix_shape",
        "shared_color_scale",
        "robust_percentiles",
        "highlight_outliers",
        "outlier_zscore",
        "zero_threshold",
        "log_magnitude_floor",
        "histogram_bins",
        "histogram_max_samples",
        "topk_count",
    )


def test_tensor_elements_config_supports_grouped_modes() -> None:
    config = TensorElementsConfig(mode="slice")

    assert config.mode == "slice"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_matrix_shape": (0, 256)}, "max_matrix_shape"),
        ({"topk_count": 0}, "topk_count"),
        ({"zero_threshold": 0.0}, "zero_threshold"),
        ({"log_magnitude_floor": 0.0}, "log_magnitude_floor"),
        ({"outlier_zscore": 0.0}, "outlier_zscore"),
        ({"histogram_bins": 0}, "histogram_bins"),
        ({"histogram_max_samples": 0}, "histogram_max_samples"),
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


def test_show_tensor_elements_heatmap_uses_compact_title_without_engine_suffix() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="CompactTitle",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.get_title() == "CompactTitle - elements"
    assert "[" not in ax.get_title()


def test_show_tensor_elements_supports_single_quimb_like_tensor() -> None:
    tensor = DummyQuimbTensor(
        np.arange(6, dtype=float).reshape(2, 3),
        inds=("a", "b"),
        tags=("Q0",),
    )

    fig, ax = show_tensor_elements(tensor, show=False)

    assert_rendered_figure(fig, ax)
    assert ax.images
    assert ax.get_title() == "Q0 - elements"


def test_show_tensor_elements_supports_direct_numpy_array_input() -> None:
    tensor = np.arange(6, dtype=float).reshape(2, 3)

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=False)

    assert_rendered_figure(fig, ax)
    assert ax.images
    assert "tensor" in ax.get_title().lower()


def test_show_tensor_elements_accepts_einsum_trace_subclass() -> None:
    class DerivedEinsumTrace(EinsumTrace):
        pass

    trace = DerivedEinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(12, dtype=float).reshape(3, 4)

    trace.bind("Left", left)
    trace.bind("Right", right)
    result = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, right, result]  # type: ignore[attr-defined]

    fig, ax = show_tensor_elements(trace, show=False, show_controls=False)

    assert_rendered_figure(fig, ax)
    assert ax.get_title()


def test_show_tensor_elements_accepts_tenpy_network_subclass() -> None:
    class DerivedTenPyTensorNetwork(TenPyTensorNetwork):
        pass

    base = make_tenpy_tensor_network(
        [
            (
                "A",
                DummyTenPyTensor(
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                    ("left", "right"),
                ),
            )
        ],
        [],
    )
    network = DerivedTenPyTensorNetwork(nodes=base.nodes, bonds=base.bonds)

    fig, ax = show_tensor_elements(network, show=False, show_controls=False)
    assert_rendered_figure(fig, ax)
    assert ax.get_title()

    fig, ax = show_tensor_elements(
        network,
        show=False,
        show_controls=False,
        engine="tenpy",
    )
    assert_rendered_figure(fig, ax)
    assert ax.get_title()


def test_show_tensor_elements_slice_mode_renders_requested_plane() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="SliceTensor",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="slice",
            analysis=TensorAnalysisConfig(slice_axis="mid", slice_index=1),
        ),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "slice" in ax.get_title().lower()
    assert np.allclose(np.asarray(ax.images[0].get_array(), dtype=float), tensor.tensor[:, 1, :])


def test_show_tensor_elements_reduce_mode_renders_axis_mean_heatmap() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="ReduceTensor",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="reduce",
            analysis=TensorAnalysisConfig(reduce_axes=("col",), reduce_method="mean"),
        ),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert "reduce" in ax.get_title().lower()
    assert np.allclose(
        np.asarray(ax.images[0].get_array(), dtype=float),
        np.mean(tensor.tensor, axis=2),
    )


def test_show_tensor_elements_profiles_mode_renders_axis_norm_profile() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="ProfileTensor",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="profiles",
            analysis=TensorAnalysisConfig(profile_axis="mid", profile_method="norm"),
        ),
        show=False,
        show_controls=False,
    )

    expected = np.sqrt(np.sum(np.square(tensor.tensor), axis=(0, 2)))

    assert_rendered_figure(fig, ax)
    assert "profiles" in ax.get_title().lower()
    assert np.allclose(_line_ydata_as_float(ax), expected)


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


def test_extract_playback_step_records_accepts_einsum_trace_subclass() -> None:
    class DerivedEinsumTrace(EinsumTrace):
        pass

    trace = DerivedEinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(12, dtype=float).reshape(3, 4)

    trace.bind("Left", left)
    trace.bind("Right", right)
    result = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, right, result]  # type: ignore[attr-defined]

    step_records = _extract_playback_step_records(trace)

    assert step_records is not None
    assert [step.result_name for step in step_records] == ["r0"]
    assert step_records[0].record is not None
    assert step_records[0].record.axis_names == ("a", "c")


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


def test_extract_einsum_playback_step_records_surfaces_unexpected_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = EinsumTrace()
    left = np.arange(6, dtype=float).reshape(2, 3)
    right = np.arange(12, dtype=float).reshape(3, 4)

    trace.bind("Left", left)
    trace.bind("Right", right)
    result = einsum("ab,bc->ac", left, right, trace=trace, backend="numpy")
    trace._test_keepalive = [left, right, result]  # type: ignore[attr-defined]

    def _raise_unexpected_error(
        _equation: str,
        _operand_shapes: tuple[tuple[int, ...], ...],
    ) -> Any:
        raise RuntimeError("unexpected equation parser failure")

    monkeypatch.setattr(
        tensor_elements_inputs_module,
        "parse_einsum_equation",
        _raise_unexpected_error,
    )

    with pytest.raises(RuntimeError, match="unexpected equation parser failure"):
        _extract_einsum_playback_step_records(trace)


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


def test_prepare_mode_payload_empty_heatmap_returns_text_summary_payload() -> None:
    record = _TensorRecord(
        array=np.empty((0, 3), dtype=float),
        name="EmptyHeatmap",
        axis_names=("row", "col"),
        engine="tensornetwork",
    )

    resolved_mode, payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="elements"),
        mode="elements",
    )

    assert resolved_mode == "elements"
    assert isinstance(payload, _TextSummaryPayload)
    assert "empty tensor" in payload.text.lower()


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


def test_prepare_mode_payload_supports_slice_reduce_and_profiles() -> None:
    record = _TensorRecord(
        array=np.arange(24, dtype=float).reshape(2, 3, 4),
        name="AnalysisPayloads",
        axis_names=("row", "mid", "col"),
        engine="tensornetwork",
    )
    analysis = TensorAnalysisConfig(
        slice_axis="mid",
        slice_index=1,
        reduce_axes=("col",),
        reduce_method="mean",
        profile_axis="col",
        profile_method="norm",
    )

    slice_mode, slice_payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="slice", analysis=analysis),
        mode="slice",
    )
    reduce_mode, reduce_payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="reduce", analysis=analysis),
        mode="reduce",
    )
    profiles_mode, profiles_payload = _prepare_mode_payload(
        record,
        config=TensorElementsConfig(mode="profiles", analysis=analysis),
        mode="profiles",
    )

    assert slice_mode == "slice"
    assert isinstance(slice_payload, _HeatmapPayload)
    assert tuple(slice_payload.matrix.shape) == (2, 4)
    assert "slice" in slice_payload.mode_label

    assert reduce_mode == "reduce"
    assert isinstance(reduce_payload, _HeatmapPayload)
    assert tuple(reduce_payload.matrix.shape) == (2, 1)
    assert "reduce" in reduce_payload.mode_label

    assert profiles_mode == "profiles"
    assert isinstance(profiles_payload, _SeriesPayload)
    assert "profiles" in profiles_payload.mode_label


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


def test_show_tensor_elements_direct_iterables_preserve_duplicate_tensors() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="Repeat",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements([tensor, tensor], show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert isinstance(ax, Axes)
    assert controller._slider is not None
    assert len(controller._records) == 2
    assert "Repeat" in ax.get_title()


def test_show_tensor_elements_auto_detects_one_shot_backend_node_generator() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(6, dtype=float).reshape(2, 3),
            name="A",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            np.arange(12, dtype=float).reshape(3, 4),
            name="B",
            axis_names=("u", "v"),
        ),
    ]

    fig, ax = show_tensor_elements((tensor for tensor in tensors), show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert isinstance(ax, Axes)
    assert controller._slider is not None
    assert len(controller._records) == 2
    assert "A" in ax.get_title()


def test_show_tensor_elements_keeps_first_node_for_single_pass_nodes_attribute() -> None:
    tensors = [
        DummyTensorKrowchNode(
            name="A",
            axes_names=("x", "y"),
            tensor=np.arange(6, dtype=float).reshape(2, 3),
            shape=(2, 3),
        ),
        DummyTensorKrowchNode(
            name="B",
            axes_names=("u", "v"),
            tensor=np.arange(12, dtype=float).reshape(3, 4),
            shape=(3, 4),
        ),
    ]
    network = DummyTensorKrowchNetwork(iter(tensors))

    fig, ax = show_tensor_elements(network, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert controller._slider is not None
    assert "A" in ax.get_title()
    controller._slider.set_val(1.0)
    assert "B" in ax.get_title()


def test_show_tensor_elements_keeps_first_leaf_node_for_single_pass_leaf_nodes_attribute() -> None:
    tensors = [
        DummyTensorKrowchNode(
            name="A",
            axes_names=("x", "y"),
            tensor=np.arange(6, dtype=float).reshape(2, 3),
            shape=(2, 3),
        ),
        DummyTensorKrowchNode(
            name="B",
            axes_names=("u", "v"),
            tensor=np.arange(12, dtype=float).reshape(3, 4),
            shape=(3, 4),
        ),
    ]
    network = DummyTensorKrowchContractedNetwork(
        nodes=tensors,
        leaf_nodes=iter(tensors),
        resultant_nodes=(),
    )

    fig, ax = show_tensor_elements(network, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert controller._slider is not None
    assert "A" in ax.get_title()
    controller._slider.set_val(1.0)
    assert "B" in ax.get_title()


def test_show_tensor_elements_keeps_first_tensor_for_single_pass_tensors_attribute() -> None:
    tensors = [
        DummyQuimbTensor(
            np.arange(6, dtype=float).reshape(2, 3),
            inds=("x", "y"),
            tags=("Q0",),
        ),
        DummyQuimbTensor(
            np.arange(12, dtype=float).reshape(3, 4),
            inds=("u", "v"),
            tags=("Q1",),
        ),
    ]
    network = DummyQuimbNetwork(iter(tensors))

    fig, ax = show_tensor_elements(network, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert controller._slider is not None
    assert "Q0" in ax.get_title()
    controller._slider.set_val(1.0)
    assert "Q1" in ax.get_title()


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


def test_show_tensor_elements_reuses_prepared_payloads_for_revisited_data_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensor_network_viz.tensor_elements as tensor_elements_module

    tensor = DummyTensorNetworkNode(
        np.arange(12, dtype=float).reshape(3, 4),
        name="CachedDataMode",
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
        config=TensorElementsConfig(mode="magnitude"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    controller.set_mode("data", redraw=False)
    controller.set_mode("magnitude", redraw=False)
    controller.set_mode("data", redraw=False)

    assert counts["magnitude"] == 1
    assert counts["data"] == 1


def test_show_tensor_elements_mode_switch_with_same_options_reuses_mode_radio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="ModeRadioReuse",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]
    mode_radio_before = controller._mode_radio
    rebuild_calls = {"count": 0}
    original_rebuild_mode_radio = controller._rebuild_mode_radio

    def counting_rebuild_mode_radio() -> None:
        rebuild_calls["count"] += 1
        original_rebuild_mode_radio()

    monkeypatch.setattr(controller, "_rebuild_mode_radio", counting_rebuild_mode_radio)

    controller.set_mode("magnitude", redraw=False)

    assert_rendered_figure(fig, ax)
    assert rebuild_calls["count"] == 0
    assert controller._mode_radio is mode_radio_before


def test_show_tensor_elements_reuses_spectral_mode_flags_for_seen_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(4, dtype=float).reshape(2, 2),
            name="SpectralA",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            (np.arange(4, dtype=float).reshape(2, 2) + 10.0),
            name="SpectralB",
            axis_names=("row", "col"),
        ),
    ]
    calls = {"count": 0}
    original_spectral_mode_flags = tensor_elements_payloads_module._spectral_mode_flags

    def counting_spectral_mode_flags(
        record: _TensorRecord,
        *,
        config: TensorElementsConfig,
    ) -> tuple[bool, bool]:
        calls["count"] += 1
        return original_spectral_mode_flags(record, config=config)

    monkeypatch.setattr(
        tensor_elements_payloads_module,
        "_spectral_mode_flags",
        counting_spectral_mode_flags,
    )

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    controller.set_group("diagnostic", redraw=False)
    calls_after_first_tensor = calls["count"]
    controller.set_mode("sign", redraw=False)
    controller.set_tensor_index(1, redraw=False)
    calls_after_second_tensor = calls["count"]
    controller.set_group("basic", redraw=False)
    controller.set_tensor_index(0, redraw=False)
    controller.set_group("diagnostic", redraw=False)

    assert_rendered_figure(fig, ax)
    assert calls_after_first_tensor > 0
    assert calls_after_second_tensor > calls_after_first_tensor
    assert calls["count"] == calls_after_second_tensor


def test_show_tensor_elements_reuses_cached_spectral_analysis_across_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(16, dtype=float).reshape(4, 4),
        name="SpectralModes",
        axis_names=("row", "col"),
    )
    calls = {"count": 0}
    original_spectral_analysis = tensor_elements_data_module._spectral_analysis_for_record

    def counting_spectral_analysis(
        record: _TensorRecord,
        *,
        config: TensorElementsConfig,
    ) -> Any:
        calls["count"] += 1
        return original_spectral_analysis(record, config=config)

    monkeypatch.setattr(
        tensor_elements_data_module,
        "_spectral_analysis_for_record",
        counting_spectral_analysis,
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    controller.set_group("diagnostic", redraw=False)
    controller.set_mode("singular_values", redraw=False)
    controller.set_mode("eigen_real", redraw=False)
    controller.set_mode("eigen_imag", redraw=False)
    controller.set_mode("singular_values", redraw=False)

    assert_rendered_figure(fig, ax)
    assert calls["count"] == 1


def test_show_tensor_elements_reuses_cached_spectral_analysis_for_revisited_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(16, dtype=float).reshape(4, 4),
            name="Spectral0",
            axis_names=("row", "col"),
        ),
        DummyTensorNetworkNode(
            (np.arange(16, dtype=float).reshape(4, 4) + 100.0),
            name="Spectral1",
            axis_names=("row", "col"),
        ),
    ]
    calls = {"count": 0}
    original_spectral_analysis = tensor_elements_data_module._spectral_analysis_for_record

    def counting_spectral_analysis(
        record: _TensorRecord,
        *,
        config: TensorElementsConfig,
    ) -> Any:
        calls["count"] += 1
        return original_spectral_analysis(record, config=config)

    monkeypatch.setattr(
        tensor_elements_data_module,
        "_spectral_analysis_for_record",
        counting_spectral_analysis,
    )

    fig, ax = show_tensor_elements(
        tensors,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    controller.set_group("diagnostic", redraw=False)
    controller.set_mode("singular_values", redraw=False)
    assert calls["count"] == 1

    controller.set_mode("eigen_real", redraw=False)
    assert calls["count"] == 1

    controller.set_tensor_index(1, redraw=False)
    assert calls["count"] == 2

    controller.set_mode("eigen_imag", redraw=False)
    assert calls["count"] == 2

    controller.set_tensor_index(0, redraw=False)
    assert calls["count"] == 2

    controller.set_mode("singular_values", redraw=False)

    assert_rendered_figure(fig, ax)
    assert calls["count"] == 2


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


def test_show_tensor_elements_data_mode_handles_all_nan_without_runtime_warnings() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[np.nan, np.nan]], dtype=float),
        name="AllNaN",
        axis_names=("row", "col"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, ax = show_tensor_elements(
            tensor,
            config=TensorElementsConfig(mode="data"),
            show=False,
            show_controls=False,
        )

    assert_rendered_figure(fig, ax)
    assert "allnan" in ax.get_title().lower()
    assert ax.texts
    assert "nan" in ax.texts[0].get_text().lower()


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


def test_show_tensor_elements_distribution_mode_uses_soft_grid() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(16, dtype=float).reshape(4, 4),
        name="SoftGridDistribution",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="distribution"),
        show=False,
        show_controls=False,
    )
    fig.canvas.draw()
    visible_gridlines = [
        line for line in (*ax.get_xgridlines(), *ax.get_ygridlines()) if line.get_visible()
    ]

    assert_rendered_figure(fig, ax)
    assert visible_gridlines
    assert all(float(line.get_alpha() or 1.0) <= 0.18 for line in visible_gridlines)
    assert all(line.get_linestyle() == ":" for line in visible_gridlines)


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
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 1
    np.testing.assert_allclose(_line_ydata_as_float(ax), np.array([3.0, 1.0]))


def test_show_tensor_elements_profiles_mode_uses_soft_grid() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="SoftGridProfile",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="profiles",
            analysis=TensorAnalysisConfig(profile_axis="mid", profile_method="norm"),
        ),
        show=False,
        show_controls=False,
    )
    fig.canvas.draw()
    visible_gridlines = [
        line for line in (*ax.get_xgridlines(), *ax.get_ygridlines()) if line.get_visible()
    ]

    assert_rendered_figure(fig, ax)
    assert visible_gridlines
    assert all(float(line.get_alpha() or 1.0) <= 0.18 for line in visible_gridlines)
    assert all(line.get_linestyle() == ":" for line in visible_gridlines)


def test_show_tensor_elements_profiles_mode_uses_integer_index_ticks() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="IntegerProfileTicks",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="profiles",
            analysis=TensorAnalysisConfig(profile_axis="mid", profile_method="norm"),
        ),
        show=False,
        show_controls=False,
    )
    fig.canvas.draw()
    x_ticks = _visible_axis_ticks(ax, axis="x")

    assert_rendered_figure(fig, ax)
    assert x_ticks.size > 0
    assert np.allclose(x_ticks, np.round(x_ticks))


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


def test_show_tensor_elements_singular_values_mode_uses_integer_index_ticks() -> None:
    tensor = DummyTensorNetworkNode(
        np.diag(np.array([8.0, 5.0, 3.0, 1.0], dtype=float)),
        name="IntegerSpectrumTicks",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values"),
        show=False,
        show_controls=False,
    )
    fig.canvas.draw()
    x_ticks = _visible_axis_ticks(ax, axis="x")

    assert_rendered_figure(fig, ax)
    assert x_ticks.size > 0
    assert np.allclose(x_ticks, np.round(x_ticks))


def test_show_tensor_elements_singular_values_mode_marks_zero_values_in_blood_red() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[3.0, 0.0], [0.0, 0.0]], dtype=float),
        name="ZeroSpectrum",
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
    assert ax.collections
    zero_marker = ax.collections[0]
    np.testing.assert_allclose(
        np.asarray(zero_marker.get_facecolors()[0], dtype=float),
        matplotlib.colors.to_rgba("#7F1D1D"),
    )


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
    np.testing.assert_allclose(_line_ydata_as_float(ax), np.array([5.0, 2.0]))


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
    expected_singular_values = np.linalg.svd(expected_matrix, compute_uv=False)
    default_config = TensorElementsConfig()
    visual_floor = max(default_config.zero_threshold, default_config.log_magnitude_floor)
    np.testing.assert_allclose(
        _line_ydata_as_float(ax),
        np.maximum(expected_singular_values, visual_floor),
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
    np.testing.assert_allclose(_line_ydata_as_float(ax), np.array([3.0]))


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
    np.testing.assert_allclose(_line_ydata_as_float(ax), np.array([3.0, -1.0]))


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
    np.testing.assert_allclose(_line_ydata_as_float(ax), np.array([4.0, 2.0]))


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


def test_build_topk_lines_preserves_tie_order_and_relegates_nan() -> None:
    record = _TensorRecord(
        array=np.array([[3.0, -3.0, np.nan], [2.0, -1.0, 0.0]], dtype=float),
        axis_names=("row", "col"),
        engine="numpy",
        name="TopKNaN",
    )

    lines = _build_topk_lines(record, count=4)
    text_blob = "\n".join(lines)

    assert lines[0] == "top 4 by magnitude:"
    assert "1. |x|=3, value=3, at (row=0, col=0)" in text_blob
    assert "2. |x|=3, value=-3, at (row=0, col=1)" in text_blob
    assert "3. |x|=2, value=2, at (row=1, col=0)" in text_blob
    assert "4. |x|=1, value=-1, at (row=1, col=1)" in text_blob
    assert "row=0, col=2" not in text_blob


def test_build_topk_lines_uses_partial_selection_for_large_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _TensorRecord(
        array=np.arange(1024, dtype=float).reshape(32, 32),
        axis_names=("row", "col"),
        engine="numpy",
        name="TopKLarge",
    )
    calls = {"count": 0}
    original_argpartition = tensor_elements_data_module.np.argpartition

    def counting_argpartition(
        values: np.ndarray[Any, Any],
        kth: int | np.ndarray[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, Any]:
        calls["count"] += 1
        return np.asarray(original_argpartition(values, kth, axis=axis))

    monkeypatch.setattr(tensor_elements_data_module.np, "argpartition", counting_argpartition)

    lines = _build_topk_lines(record, count=8)

    assert len(lines) == 9
    assert calls["count"] == 1


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


def test_show_tensor_elements_empty_heatmap_renders_text_fallback() -> None:
    fig, ax = show_tensor_elements(
        np.empty((0, 3), dtype=float),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts).lower()

    assert_rendered_figure(fig, ax)
    assert ax.texts
    assert not ax.images
    assert "empty tensor" in text_blob
    assert "elements" in ax.get_title().lower()


def test_show_tensor_elements_empty_heatmap_with_explicit_axes_renders_text_fallback() -> None:
    fig, ax = show_tensor_elements(
        np.empty((0, 3), dtype=float),
        config=TensorElementsConfig(row_axes=(0,), col_axes=(1,)),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts).lower()

    assert_rendered_figure(fig, ax)
    assert ax.texts
    assert not ax.images
    assert "empty tensor" in text_blob
    assert "elements" in ax.get_title().lower()


def test_show_tensor_elements_empty_data_mode_keeps_existing_summary() -> None:
    fig, ax = show_tensor_elements(
        np.empty((0, 3), dtype=float),
        config=TensorElementsConfig(mode="data"),
        show=False,
        show_controls=False,
    )
    text_blob = "\n".join(text.get_text() for text in ax.texts).lower()

    assert_rendered_figure(fig, ax)
    assert "stats: empty tensor" in text_blob
    assert "top 8 by magnitude" in text_blob


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


def test_show_tensor_elements_continuous_colorbar_uses_compact_ticks() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(100, dtype=float).reshape(10, 10),
        name="CompactColorbar",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=False,
    )
    colorbar_ax = next(axis for axis in fig.axes if axis is not ax)
    ticklabels = [tick.get_text() for tick in colorbar_ax.get_yticklabels()]

    assert_rendered_figure(fig, ax)
    assert len(colorbar_ax.get_yticks()) <= 5
    assert all("e+" not in label.lower() for label in ticklabels)


@pytest.mark.parametrize(
    ("theme", "expected_cmap"),
    [
        ("grayscale", "gray"),
        ("rainbow", "gist_rainbow"),
        ("spectral", "nipy_spectral"),
    ],
)
def test_show_tensor_elements_theme_changes_continuous_heatmap_colormap(
    theme: str,
    expected_cmap: str,
) -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(16, dtype=float).reshape(4, 4),
        name="ThemeHeatmap",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements", theme=theme),  # type: ignore[arg-type]
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.images[0].get_cmap().name == expected_cmap


def test_show_tensor_elements_theme_changes_histogram_colors() -> None:
    tensor = DummyTensorNetworkNode(
        np.linspace(0.0, 1.0, 12, dtype=float).reshape(3, 4),
        name="ThemeHistogram",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="distribution", theme="grayscale"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.patches
    first_patch = ax.patches[0]
    assert matplotlib.colors.to_hex(first_patch.get_facecolor()).lower() == "#444444"
    assert matplotlib.colors.to_hex(first_patch.get_edgecolor()).lower() == "#111111"


def test_show_tensor_elements_theme_changes_series_color() -> None:
    tensor = DummyTensorNetworkNode(
        np.diag(np.array([5.0, 3.0, 1.0], dtype=float)),
        name="ThemeSeries",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="singular_values", theme="grayscale"),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.lines
    assert matplotlib.colors.to_hex(ax.lines[0].get_color()).lower() == "#111111"


def test_show_tensor_elements_heatmap_axes_use_integer_ticks() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(70, dtype=float).reshape(7, 10),
        name="IntegerHeatmapTicks",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=False,
    )
    fig.canvas.draw()
    x_ticks = _visible_axis_ticks(ax, axis="x")
    y_ticks = _visible_axis_ticks(ax, axis="y")

    assert_rendered_figure(fig, ax)
    assert x_ticks.size > 0
    assert y_ticks.size > 0
    assert np.allclose(x_ticks, np.round(x_ticks))
    assert np.allclose(y_ticks, np.round(y_ticks))


def test_show_tensor_elements_heatmap_hover_shows_value_and_tensor_indices() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 6.0]], dtype=float),
        name="HoverValue",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _dispatch_motion_event_at_data(ax, x=1.0, y=1.0)

    assert_rendered_figure(fig, ax)
    assert controller._heatmap_hover_annotation is not None
    assert controller._heatmap_hover_annotation.get_visible() is True
    assert controller._heatmap_hover_annotation.get_text() == "7\n(1, 1)"
    hover_patch = controller._heatmap_hover_annotation.get_bbox_patch()
    assert hover_patch is not None
    assert matplotlib.colors.to_hex(hover_patch.get_facecolor()).lower() == "#f8fafc"
    assert matplotlib.colors.to_hex(hover_patch.get_edgecolor()).lower() == "#cbd5e1"


def test_show_tensor_elements_theme_changes_hover_colors() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        name="ThemeHover",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements", theme="grayscale"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _dispatch_motion_event_at_data(ax, x=0.0, y=0.0)

    assert_rendered_figure(fig, ax)
    assert controller._heatmap_hover_annotation is not None
    hover_patch = controller._heatmap_hover_annotation.get_bbox_patch()
    assert hover_patch is not None
    assert matplotlib.colors.to_hex(hover_patch.get_facecolor()).lower() == "#ffffff"
    assert matplotlib.colors.to_hex(hover_patch.get_edgecolor()).lower() == "#111111"


def test_show_tensor_elements_heatmap_hover_hides_when_pointer_leaves_axes() -> None:
    tensor = DummyTensorNetworkNode(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        name="HoverHide",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(mode="elements"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _dispatch_motion_event_at_data(ax, x=0.0, y=0.0)
    _dispatch_motion_event_outside_figure(fig)

    assert_rendered_figure(fig, ax)
    assert controller._heatmap_hover_annotation is not None
    assert controller._heatmap_hover_annotation.get_visible() is False


def test_show_tensor_elements_heatmap_hover_uses_full_tensor_indices_for_grouped_axes() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(8, dtype=float).reshape(2, 2, 2),
        name="GroupedHover",
        axis_names=("i", "j", "k"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="elements",
            row_axes=("i", "j"),
            col_axes=("k",),
        ),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _dispatch_motion_event_at_data(ax, x=1.0, y=2.0)

    assert_rendered_figure(fig, ax)
    assert controller._heatmap_hover_annotation is not None
    assert controller._heatmap_hover_annotation.get_visible() is True
    assert controller._heatmap_hover_annotation.get_text() == "5\n(1, 0, 1)"


def test_show_tensor_elements_heatmap_does_not_draw_group_boundary_lines() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(12, dtype=float).reshape(3, 2, 2),
        name="NoGroupedBoundaries",
        axis_names=("i", "j", "k"),
    )

    fig, ax = show_tensor_elements(
        tensor,
        config=TensorElementsConfig(
            mode="elements",
            row_axes=("i", "j", "k"),
            col_axes=(),
        ),
        show=False,
        show_controls=False,
    )

    assert_rendered_figure(fig, ax)
    assert not ax.lines


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
    assert ax.collections[0].get_path_effects()


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


def test_extract_playback_step_records_normalizes_parent_and_child_sequences() -> None:
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
        tensor=np.ones((2, 5)),
        shape=(2, 5),
    )
    left.successors = {  # type: ignore[attr-defined]
        "contract_edges": {
            "mixed": SimpleNamespace(node_ref=[left, None, right], child=(None, result))
        }
    }
    right.successors = {}  # type: ignore[attr-defined]
    network = DummyTensorKrowchContractedNetwork(
        nodes=[left, right, result],
        leaf_nodes=[left, right],
        resultant_nodes=[result],
    )

    records = _extract_playback_step_records(network)

    assert records is not None
    assert len(records) == 1
    assert records[0].record is not None
    assert records[0].result_name == "contract_edges"
    assert records[0].record.axis_names == ("a", "c")
    assert tuple(records[0].record.array.shape) == (2, 5)


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


def test_show_tensor_elements_accepts_direct_torch_tensor_with_requires_grad() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6.0, requires_grad=True).reshape(2, 3)

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=False)

    assert_rendered_figure(fig, ax)
    assert np.asarray(ax.images[0].get_array(), dtype=float).shape == (2, 3)


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
    assert tuple(text.get_text() for text in controller._group_radio.labels) == (
        "basic",
        "complex",
        "diagnostic",
        "analysis",
    )
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


def test_show_tensor_elements_widgets_offer_analysis_modes() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetAnalysis",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)
    analysis_modes = tuple(text.get_text() for text in controller._mode_radio.labels)

    assert_rendered_figure(fig, ax)
    assert analysis_modes == ("slice", "reduce", "profiles")


def test_show_tensor_elements_analysis_modes_build_contextual_controls() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetAnalysisControls",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)
    assert controller._analysis_axis_radio is not None
    assert controller._analysis_slider is not None

    _click_radio_label(controller._mode_radio, 1)
    assert controller._analysis_checkbuttons is not None
    assert controller._analysis_method_radio is not None

    _click_radio_label(controller._mode_radio, 2)
    assert controller._analysis_axis_radio is not None
    assert controller._analysis_method_radio is not None
    assert_rendered_figure(fig, ax)


def test_show_tensor_elements_group_and_mode_selectors_use_tighter_spacing() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(6, dtype=float).reshape(2, 3),
        name="WidgetSelectorLayout",
        axis_names=("row", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert_rendered_figure(fig, ax)
    assert controller._group_radio_ax is not None
    assert controller._mode_radio_ax is not None
    assert controller._group_radio_ax.get_position().bounds == pytest.approx(
        (0.02, 0.04, 0.15, 0.145)
    )
    assert controller._mode_radio_ax.get_position().bounds == pytest.approx(
        (0.175, 0.028, 0.21, 0.16)
    )


def test_show_tensor_elements_slice_slider_keeps_widget_while_updating() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetSliceDrag",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)
    assert controller._analysis_slider is not None
    slice_slider = controller._analysis_slider

    _click_radio_label(controller._analysis_axis_radio, 1)
    assert controller._analysis_slider is not None
    slice_slider = controller._analysis_slider
    slice_slider.set_val(2.0)

    assert_rendered_figure(fig, ax)
    assert controller._analysis_slider is slice_slider
    assert float(controller._analysis_slider.val) == pytest.approx(2.0)
    assert np.allclose(np.asarray(ax.images[0].get_array(), dtype=float), tensor.tensor[:, 2, :])


def test_show_tensor_elements_slice_slider_uses_wider_layout_and_label_gap() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetSliceLayout",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)

    assert_rendered_figure(fig, ax)
    assert controller._analysis_slider_ax is not None
    assert controller._analysis_slider is not None
    assert controller._analysis_slider_ax.get_position().bounds == pytest.approx(
        (0.66, 0.135, 0.30, 0.05)
    )
    assert controller._analysis_slider.label.get_position() == pytest.approx((-0.05, 0.5))


def test_show_tensor_elements_analysis_controls_fall_back_when_slider_changes_rank() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(24, dtype=float).reshape(2, 3, 4),
            name="Rank3",
            axis_names=("row", "mid", "col"),
        ),
        DummyTensorNetworkNode(
            np.arange(6, dtype=float).reshape(2, 3),
            name="Rank2",
            axis_names=("left", "right"),
        ),
    ]

    fig, ax = show_tensor_elements(tensors, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)
    assert controller._analysis_axis_radio is not None
    _click_radio_label(controller._analysis_axis_radio, 2)
    assert controller._analysis_slider is not None
    controller._analysis_slider.set_val(2.0)
    controller._slider.set_val(1.0)

    assert_rendered_figure(fig, ax)
    assert controller._analysis_axis_radio is not None
    assert controller._analysis_axis_radio.value_selected == "left"
    assert controller._analysis_slider is not None
    assert float(controller._analysis_slider.valmax) == pytest.approx(1.0)
    assert float(controller._analysis_slider.val) == pytest.approx(0.0)


def test_show_tensor_elements_analysis_controls_reuse_widgets_for_same_tensor_shape() -> None:
    tensors = [
        DummyTensorNetworkNode(
            np.arange(24, dtype=float).reshape(2, 3, 4),
            name="Rank3A",
            axis_names=("row", "mid", "col"),
        ),
        DummyTensorNetworkNode(
            (np.arange(24, dtype=float).reshape(2, 3, 4) + 100.0),
            name="Rank3B",
            axis_names=("row", "mid", "col"),
        ),
    ]

    fig, ax = show_tensor_elements(
        tensors,
        config=TensorElementsConfig(mode="slice"),
        show=False,
        show_controls=True,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert controller._analysis_axis_radio is not None
    assert controller._analysis_slider is not None
    _click_radio_label(controller._analysis_axis_radio, 1)
    controller._analysis_slider.set_val(2.0)
    axis_radio = controller._analysis_axis_radio
    slice_slider = controller._analysis_slider

    controller._slider.set_val(1.0)

    assert_rendered_figure(fig, ax)
    assert controller._analysis_axis_radio is axis_radio
    assert controller._analysis_slider is slice_slider
    assert controller._analysis_axis_radio.value_selected == "mid"
    assert float(controller._analysis_slider.val) == pytest.approx(2.0)


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


def test_show_tensor_elements_slider_uses_thicker_control_height() -> None:
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
    assert controller._slider_ax is not None
    assert controller._slider is not None
    slider_bounds = controller._slider_ax.get_position().bounds

    assert slider_bounds == pytest.approx((0.48, 0.01, 0.38, 0.055))
    assert controller._slider.label.get_position() == pytest.approx((-0.075, 0.5))


def test_show_tensor_elements_slider_uses_polished_progress_track() -> None:
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
    assert controller._slider is not None
    assert controller._slider.poly.get_facecolor() == pytest.approx(
        matplotlib.colors.to_rgba("#0369A1")
    )
    assert controller._slider.track.get_facecolor() == pytest.approx(
        matplotlib.colors.to_rgba("#D7EAF2")
    )
    assert controller._slider._handle.get_markersize() == pytest.approx(11.0)


def test_show_tensor_elements_analysis_axis_controls_stretch_left_and_down() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetAxisStretch",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)

    assert_rendered_figure(fig, ax)
    assert controller._mode_radio_ax is not None
    assert controller._analysis_axis_ax is not None

    mode_bounds = controller._mode_radio_ax.get_position().bounds
    axis_bounds = controller._analysis_axis_ax.get_position().bounds
    mode_right = mode_bounds[0] + mode_bounds[2]

    assert axis_bounds[0] >= mode_right - 0.005
    assert axis_bounds[0] - mode_right <= 0.01
    assert axis_bounds[2] >= 0.20
    assert axis_bounds[3] >= 0.15


def test_show_tensor_elements_reduce_axis_controls_stretch_left_and_down() -> None:
    tensor = DummyTensorNetworkNode(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        name="WidgetReduceStretch",
        axis_names=("row", "mid", "col"),
    )

    fig, ax = show_tensor_elements(tensor, show=False, show_controls=True)
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    _click_radio_label(controller._group_radio, 3)
    _click_radio_label(controller._mode_radio, 1)

    assert_rendered_figure(fig, ax)
    assert controller._mode_radio_ax is not None
    assert controller._analysis_check_ax is not None
    assert controller._slider_ax is None

    mode_bounds = controller._mode_radio_ax.get_position().bounds
    check_bounds = controller._analysis_check_ax.get_position().bounds
    mode_right = mode_bounds[0] + mode_bounds[2]

    assert check_bounds[0] >= mode_right - 0.005
    assert check_bounds[0] - mode_right <= 0.01
    assert check_bounds[2] >= 0.21
    assert check_bounds[3] >= 0.16


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
