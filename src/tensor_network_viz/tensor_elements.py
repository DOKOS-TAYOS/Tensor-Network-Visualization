"""Public tensor-elements rendering entry point."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._logging import package_logger
from ._tensor_elements_controller import (
    _TensorElementsControlsLayout,
    _TensorElementsFigureController,
    _TensorPayloadCacheEntry,
)
from ._tensor_elements_rendering import _RenderedTensorPanel
from ._tensor_elements_support import _extract_tensor_records, _prepare_mode_payload
from ._typing import root_figure
from .config import EngineName
from .exceptions import AxisConfigurationError
from .tensor_elements_config import TensorElementsConfig


def _build_single_external_axis(ax: Axes) -> tuple[Figure, Axes]:
    figure = root_figure(ax.figure)
    return figure, ax


def _build_internal_axis(*, config: TensorElementsConfig) -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=config.figsize)
    return root_figure(figure), ax


def _show_tensor_records(
    records: list[Any],
    *,
    config: TensorElementsConfig,
    controls_layout: _TensorElementsControlsLayout | None = None,
    ax: Axes | None,
    show_controls: bool,
    show: bool,
) -> tuple[Figure, Axes, _TensorElementsFigureController]:
    initial_payload_cache: dict[int, _TensorPayloadCacheEntry] = {}
    if config.mode != "auto" and (not show_controls or len(records) == 1):
        resolved_mode, payload = _prepare_mode_payload(records[0], config=config, mode=config.mode)
        initial_payload_cache[0] = _TensorPayloadCacheEntry(payloads={resolved_mode: payload})
    if ax is not None and len(records) != 1:
        raise AxisConfigurationError(
            "An explicit ax is only supported when visualizing a single tensor."
        )
    if ax is not None and show_controls and len(root_figure(ax.figure).axes) > 1:
        raise AxisConfigurationError(
            "show_controls=True with an external ax is only supported when the target figure "
            "contains a single axes."
        )

    if ax is None:
        figure, main_ax = _build_internal_axis(config=config)
    else:
        figure, main_ax = _build_single_external_axis(ax)

    panel = _RenderedTensorPanel(
        base_position=main_ax.get_position().bounds,
        main_ax=main_ax,
    )
    controller = _TensorElementsFigureController(
        config=config,
        controls_layout=controls_layout,
        figure=figure,
        panel=panel,
        records=records,
        allow_interactive_fallback=show_controls,
        prepare_mode_payload=lambda record, config, mode: _prepare_mode_payload(
            record,
            config=config,
            mode=mode,
        ),
        initial_payload_cache=initial_payload_cache,
    )
    controller.initialize(show_controls=show_controls)
    if show:
        from .viewer import _show_figure

        _show_figure(figure)
    return figure, main_ax, controller


def show_tensor_elements(
    data: Any,
    *,
    engine: EngineName | None = None,
    config: TensorElementsConfig | None = None,
    ax: Axes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Render tensor values in a single Matplotlib figure.

    Args:
        data: Direct numeric tensor/array-like input, iterable of tensors (preserving order and
            duplicates), supported backend-native tensor container, or playback-aware inputs such
            as ``EinsumTrace``.
        engine: Optional backend override. When omitted, the backend is inferred from ``data``.
        config: Optional tensor-inspection configuration. When omitted,
            ``TensorElementsConfig()`` is used. The config is ordered from mode/axis selection
            first to scaling/detail options later.
        ax: Optional Matplotlib axes. External axes are supported only for a single tensor.
        show_controls: Whether to add the compact grouped controls and, when needed, the
            tensor slider.
        show: Whether to display the figure automatically after rendering.

    Returns:
        Tuple ``(figure, axes)`` for further customization.

    Raises:
        AxisConfigurationError: If an external ``ax`` is combined with multiple tensors or
            with a figure layout that cannot host the grouped controls.
        TensorDataError: If ``data`` does not expose supported tensor values.
        UnsupportedEngineError: If ``engine`` names an unknown backend.

    Notes:
        ``show_tensor_elements`` keeps one tensor active at a time. With multiple tensors,
        the slider switches the active tensor while the selected mode/group stays in sync
        with what is valid for that tensor.

        The ``"singular_values"`` mode always uses a logarithmic y-scale. Values at or
        below ``config.zero_threshold`` are shown at the visual floor and highlighted so
        exact or near-zero singular values remain visible.
    """
    style = config or TensorElementsConfig()
    package_logger.debug(
        "show_tensor_elements called with engine=%r show_controls=%s show=%s.",
        engine,
        show_controls,
        show,
    )
    _, records = _extract_tensor_records(data, engine=engine)
    figure, main_ax, _controller = _show_tensor_records(
        records,
        config=style,
        ax=ax,
        show_controls=show_controls,
        show=show,
    )
    return figure, main_ax
