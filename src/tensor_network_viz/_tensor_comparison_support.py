"""Shared helpers for tensor-to-tensor comparison views."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._tensor_elements_data import _build_topk_lines
from ._tensor_elements_models import _TensorRecord
from ._tensor_elements_support import _TextSummaryPayload
from .tensor_comparison_config import TensorComparisonConfig, TensorComparisonMode


def _comparison_display_name(
    current: _TensorRecord,
    reference: _TensorRecord,
    *,
    comparison_mode: TensorComparisonMode,
) -> str:
    mode_label = str(comparison_mode).replace("_", " ")
    if comparison_mode == "reference":
        return f"{reference.name} (reference)"
    if comparison_mode == "topk_changes":
        return f"{current.name} vs {reference.name}"
    return f"{current.name} vs {reference.name} [{mode_label}]"


def _comparison_placeholder_text(
    current: _TensorRecord,
    reference: _TensorRecord,
) -> str:
    return (
        "Tensor comparison requires matching shapes.\n\n"
        f"current: {current.name}, shape={tuple(int(d) for d in current.array.shape)}\n"
        f"reference: {reference.name}, shape={tuple(int(d) for d in reference.array.shape)}"
    )


def _safe_phase_delta(
    current: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    delta = np.angle(current) - np.angle(reference)
    return np.angle(np.exp(1j * delta))


def _comparison_array(
    current: _TensorRecord,
    reference: _TensorRecord,
    *,
    comparison_config: TensorComparisonConfig,
) -> np.ndarray[Any, Any]:
    current_array = np.asarray(current.array)
    reference_array = np.asarray(reference.array)
    zero_threshold = float(comparison_config.zero_threshold)
    mode = comparison_config.mode
    if mode == "reference":
        return np.asarray(reference_array)
    if mode == "abs_diff":
        return np.abs(current_array - reference_array)
    if mode == "relative_diff":
        current_magnitude = np.abs(current_array)
        reference_magnitude = np.abs(reference_array)
        relative = np.full(current_array.shape, np.nan, dtype=float)
        valid_reference_mask = reference_magnitude > zero_threshold
        np.divide(
            np.abs(current_array - reference_array),
            reference_magnitude,
            out=relative,
            where=valid_reference_mask,
        )
        relative[np.logical_not(valid_reference_mask) & (current_magnitude <= zero_threshold)] = 0.0
        return np.asarray(relative, dtype=float)
    if mode == "ratio":
        ratio = np.full(
            current_array.shape,
            np.nan,
            dtype=np.result_type(current_array, reference_array),
        )
        valid_mask = np.abs(reference_array) > zero_threshold
        np.divide(current_array, reference_array, out=ratio, where=valid_mask)
        return ratio
    if mode == "sign_change":
        current_sign = np.sign(np.real(current_array))
        reference_sign = np.sign(np.real(reference_array))
        changed = current_sign != reference_sign
        return np.asarray(changed, dtype=float)
    if mode == "phase_change":
        return np.asarray(_safe_phase_delta(current_array, reference_array))
    raise ValueError(f"Unsupported tensor comparison mode: {mode!r}.")


def _comparison_text_payload(
    current: _TensorRecord,
    reference: _TensorRecord,
    *,
    comparison_config: TensorComparisonConfig,
) -> _TextSummaryPayload:
    delta = np.asarray(current.array) - np.asarray(reference.array)
    diff_record = _TensorRecord(
        array=delta,
        axis_names=current.axis_names,
        engine=current.engine,
        name=f"{current.name} - {reference.name}",
    )
    lines = [
        f"current: {current.name}",
        f"reference: {reference.name}",
        f"shape: {tuple(int(d) for d in diff_record.array.shape)}",
        "",
        *_build_topk_lines(diff_record, count=int(comparison_config.topk_count)),
    ]
    return _TextSummaryPayload(
        text="\n".join(lines),
        mode_label="top-k changes",
    )


def _build_comparison_record(
    current: _TensorRecord,
    reference: _TensorRecord,
    *,
    comparison_config: TensorComparisonConfig,
) -> _TensorRecord:
    return _TensorRecord(
        array=_comparison_array(
            current,
            reference,
            comparison_config=comparison_config,
        ),
        axis_names=current.axis_names,
        engine=current.engine,
        name=_comparison_display_name(
            current,
            reference,
            comparison_mode=comparison_config.mode,
        ),
    )


__all__ = [
    "_build_comparison_record",
    "_comparison_display_name",
    "_comparison_placeholder_text",
    "_comparison_text_payload",
]
