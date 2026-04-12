"""Performance/regression checks for tensor element payload preparation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tensor_network_viz._tensor_elements_support import (
    _downsample_matrix,
    _prepare_mode_payload,
    _TensorRecord,
)
from tensor_network_viz.tensor_elements_config import TensorElementsConfig


def _complex_record(shape: tuple[int, int, int]) -> _TensorRecord:
    real_part = np.random.default_rng(0).standard_normal(shape)
    imag_part = np.random.default_rng(1).standard_normal(shape)
    array = (real_part + 1j * imag_part).astype(np.complex128)
    axis_names = ("a", "b", "c")
    return _TensorRecord(
        array=array,
        axis_names=axis_names,
        engine="tensornetwork",
        name="Complex3D",
    )


def _measure_prepare_mode(
    record: _TensorRecord,
    *,
    config: TensorElementsConfig,
    mode: str,
    repeats: int = 6,
) -> float:
    _prepare_mode_payload(record, config=config, mode=mode)
    start = time.perf_counter()
    for _ in range(repeats):
        _prepare_mode_payload(record, config=config, mode=mode)
    return time.perf_counter() - start


def test_downsample_matrix_matches_contiguous_real_view() -> None:
    complex_matrix = (
        np.random.default_rng(0).standard_normal((256, 96))
        + 1j * np.random.default_rng(1).standard_normal((256, 96))
    ).astype(np.complex128)
    real_view = np.real(complex_matrix)
    contiguous_real = np.ascontiguousarray(real_view)

    reduced_view = _downsample_matrix(real_view, max_shape=(64, 48))
    reduced_contiguous = _downsample_matrix(contiguous_real, max_shape=(64, 48))

    np.testing.assert_allclose(reduced_view, reduced_contiguous)


@pytest.mark.perf
def test_prepare_mode_payload_real_and_imag_stay_close_to_magnitude_runtime() -> None:
    record = _complex_record((128, 96, 24))
    config = TensorElementsConfig(max_matrix_shape=(64, 48))

    magnitude_s = _measure_prepare_mode(record, config=config, mode="magnitude")
    real_s = _measure_prepare_mode(record, config=config, mode="real")
    imag_s = _measure_prepare_mode(record, config=config, mode="imag")

    assert real_s < magnitude_s * 4.0, (
        f"expected real view to stay close to magnitude "
        f"(magnitude={magnitude_s:.4f}s real={real_s:.4f}s)"
    )
    assert imag_s < magnitude_s * 4.0, (
        f"expected imag view to stay close to magnitude "
        f"(magnitude={magnitude_s:.4f}s imag={imag_s:.4f}s)"
    )


@pytest.mark.perf
def test_prepare_mode_payload_singular_values_stays_bounded_for_medium_tensors() -> None:
    record = _complex_record((32, 24, 12))
    config = TensorElementsConfig(max_matrix_shape=(64, 48))

    magnitude_s = _measure_prepare_mode(record, config=config, mode="magnitude", repeats=4)
    singular_values_s = _measure_prepare_mode(
        record,
        config=config,
        mode="singular_values",
        repeats=4,
    )

    assert singular_values_s < magnitude_s * 30.0, (
        "expected singular-values view to remain within a broad runtime guard "
        f"(magnitude={magnitude_s:.4f}s singular_values={singular_values_s:.4f}s)"
    )


@pytest.mark.perf
def test_prepare_mode_payload_data_stays_bounded_for_medium_tensors() -> None:
    record = _complex_record((128, 96, 24))
    config = TensorElementsConfig(max_matrix_shape=(64, 48))

    magnitude_s = _measure_prepare_mode(record, config=config, mode="magnitude", repeats=4)
    data_s = _measure_prepare_mode(record, config=config, mode="data", repeats=4)

    assert data_s < magnitude_s * 25.0, (
        "expected data view to avoid a full global ranking pass "
        f"(magnitude={magnitude_s:.4f}s data={data_s:.4f}s)"
    )
