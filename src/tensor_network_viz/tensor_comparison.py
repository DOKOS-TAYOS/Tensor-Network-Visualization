"""Public tensor-to-tensor comparison rendering entry point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons

from ._matplotlib_state import set_tensor_elements_controls
from ._tensor_comparison_support import (
    _build_comparison_record,
    _comparison_display_name,
    _comparison_placeholder_text,
    _comparison_text_payload,
)
from ._tensor_elements_controller import _TensorElementsFigureController
from ._tensor_elements_data import _TensorRecord
from ._tensor_elements_rendering import _RenderedTensorPanel
from ._tensor_elements_support import _extract_tensor_records, _prepare_mode_payload
from ._typing import root_figure
from ._ui_utils import _style_control_tray_axes
from .config import EngineName
from .exceptions import AxisConfigurationError, TensorDataError
from .tensor_comparison_config import TensorComparisonConfig, TensorComparisonMode
from .tensor_elements import _build_internal_axis, _build_single_external_axis
from .tensor_elements_config import TensorElementsConfig

_COMPARE_SELECTOR_BOUNDS: tuple[float, float, float, float] = (0.48, 0.028, 0.2, 0.19)
_COMPARE_LABEL_PROPS: dict[str, Sequence[Any]] = {"fontsize": [9.2]}
_COMPARE_RADIO_PROPS: dict[str, float] = {"s": 36.0, "linewidth": 0.9}
_COMPARE_MODE_OPTIONS: tuple[TensorComparisonMode, ...] = (
    "reference",
    "abs_diff",
    "relative_diff",
    "ratio",
    "sign_change",
    "phase_change",
    "topk_changes",
)


def _comparison_tensor_mode(record: _TensorRecord, *, config: TensorElementsConfig) -> str:
    if config.mode != "auto":
        return config.mode
    return "magnitude" if np.iscomplexobj(record.array) else "elements"


def _require_single_record(
    records: list[_TensorRecord],
    *,
    role: str,
) -> _TensorRecord:
    if len(records) != 1:
        raise TensorDataError(f"{role} must resolve to exactly one tensor for comparison.")
    return records[0]


def _build_comparison_controller(
    *,
    figure: Figure,
    main_ax: Axes,
    record: _TensorRecord,
    config: TensorElementsConfig,
    initial_payload: Any | None,
    initial_mode: str | None,
) -> _TensorElementsFigureController:
    panel = _RenderedTensorPanel(
        base_position=main_ax.get_position().bounds,
        main_ax=main_ax,
    )
    initial_payload_cache: dict[int, Any] = {}
    if initial_payload is not None and initial_mode is not None:
        from ._tensor_elements_controller import _TensorPayloadCacheEntry

        initial_payload_cache[0] = _TensorPayloadCacheEntry(
            payloads={initial_mode: initial_payload}
        )
    controller = _TensorElementsFigureController(
        config=config,
        controls_layout=None,
        figure=figure,
        panel=panel,
        records=[record],
        allow_interactive_fallback=True,
        prepare_mode_payload=lambda tensor_record, tensor_config, mode: _prepare_mode_payload(
            tensor_record,
            config=tensor_config,
            mode=mode,
        ),
        initial_payload_cache=initial_payload_cache,
    )
    return controller


def _render_comparison_mode(
    controller: _TensorElementsFigureController,
    *,
    current_record: _TensorRecord,
    reference_record: _TensorRecord,
    config: TensorElementsConfig,
    comparison_config: TensorComparisonConfig,
) -> None:
    if current_record.array.shape != reference_record.array.shape:
        controller.render_placeholder(
            _comparison_placeholder_text(current_record, reference_record),
            title="Tensor comparison",
            redraw=True,
        )
        return
    if comparison_config.mode == "topk_changes":
        comparison_record = _TensorRecord(
            array=np.asarray(current_record.array),
            axis_names=current_record.axis_names,
            engine=current_record.engine,
            name=_comparison_display_name(
                current_record,
                reference_record,
                comparison_mode=comparison_config.mode,
            ),
        )
        controller.set_single_record(comparison_record, redraw=False)
        controller.render_placeholder(
            _comparison_text_payload(
                current_record,
                reference_record,
                comparison_config=comparison_config,
            ).text,
            title=f"{comparison_record.name} - top-k changes",
            redraw=True,
        )
        return
    controller.set_single_record(
        _build_comparison_record(
            current_record,
            reference_record,
            comparison_config=comparison_config,
        ),
        redraw=False,
    )
    controller.set_mode(
        _comparison_tensor_mode(controller._current_record(), config=config),
        redraw=False,
    )
    controller._figure.canvas.draw_idle()


def show_tensor_comparison(
    data: Any,
    reference: Any,
    *,
    engine: EngineName | None = None,
    config: TensorElementsConfig | None = None,
    comparison_config: TensorComparisonConfig | None = None,
    ax: Axes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Render one tensor relative to a single reference tensor."""
    style = config or TensorElementsConfig()
    compare_style = comparison_config or TensorComparisonConfig()
    _, current_records = _extract_tensor_records(data, engine=engine)
    _, reference_records = _extract_tensor_records(reference, engine=engine)
    current_record = _require_single_record(current_records, role="data")
    reference_record = _require_single_record(reference_records, role="reference")
    if ax is not None and show_controls and len(root_figure(ax.figure).axes) > 1:
        raise AxisConfigurationError(
            "show_controls=True with an external ax is only supported when the target figure "
            "contains a single axes."
        )
    if ax is None:
        figure, main_ax = _build_internal_axis(config=style)
    else:
        figure, main_ax = _build_single_external_axis(ax)
    placeholder_record = _TensorRecord(
        array=np.zeros((1, 1), dtype=float),
        axis_names=(),
        engine=current_record.engine,
        name="Tensor comparison",
    )
    initial_record = reference_record if compare_style.mode == "reference" else current_record
    resolved_mode, payload = _prepare_mode_payload(
        initial_record,
        config=replace(style, mode="elements"),
        mode="elements",
    )
    controller = _build_comparison_controller(
        figure=figure,
        main_ax=main_ax,
        record=placeholder_record if compare_style.mode == "topk_changes" else initial_record,
        config=style if compare_style.mode != "topk_changes" else replace(style, mode="data"),
        initial_payload=payload if compare_style.mode != "topk_changes" else None,
        initial_mode=resolved_mode if compare_style.mode != "topk_changes" else None,
    )
    controller.initialize(show_controls=show_controls)
    _render_comparison_mode(
        controller,
        current_record=current_record,
        reference_record=reference_record,
        config=style,
        comparison_config=compare_style,
    )
    if show_controls:
        compare_ax = figure.add_axes(_COMPARE_SELECTOR_BOUNDS)
        _style_control_tray_axes(compare_ax)
        compare_radio = RadioButtons(
            compare_ax,
            tuple(mode.replace("_", " ") for mode in _COMPARE_MODE_OPTIONS),
            active=_COMPARE_MODE_OPTIONS.index(compare_style.mode),
            label_props=_COMPARE_LABEL_PROPS,
            radio_props=_COMPARE_RADIO_PROPS,
        )

        def _on_compare_mode_clicked(label: str | None) -> None:
            if label is None:
                return
            mode = cast(
                TensorComparisonMode,
                str(label).replace(" ", "_"),
            )
            _render_comparison_mode(
                controller,
                current_record=current_record,
                reference_record=reference_record,
                config=style,
                comparison_config=replace(compare_style, mode=mode),
            )

        compare_radio.on_clicked(_on_compare_mode_clicked)
        controller._compare_radio_ax = compare_ax  # type: ignore[attr-defined]
        controller._compare_radio = compare_radio  # type: ignore[attr-defined]
        set_tensor_elements_controls(figure, controller)
    if show:
        from .viewer import _show_figure

        _show_figure(figure)
    return figure, main_ax


__all__ = ["show_tensor_comparison"]
