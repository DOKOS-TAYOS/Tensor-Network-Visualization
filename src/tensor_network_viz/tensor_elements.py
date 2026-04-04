from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._tensor_elements_controller import (
    _RenderedTensorPanel,
    _TensorElementsFigureController,
    _TensorPayloadCacheEntry,
)
from ._tensor_elements_support import _extract_tensor_records, _prepare_mode_payload
from ._typing import root_figure
from .config import EngineName
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
    ax: Axes | None,
    show_controls: bool,
    show: bool,
) -> tuple[Figure, Axes, _TensorElementsFigureController]:
    initial_payload_cache: dict[int, _TensorPayloadCacheEntry] = {}
    if config.mode != "auto" and (not show_controls or len(records) == 1):
        resolved_mode, payload = _prepare_mode_payload(records[0], config=config, mode=config.mode)
        initial_payload_cache[0] = _TensorPayloadCacheEntry(payloads={resolved_mode: payload})
    if ax is not None and len(records) != 1:
        raise ValueError("An explicit ax is only supported when visualizing a single tensor.")
    if ax is not None and show_controls and len(root_figure(ax.figure).axes) > 1:
        raise ValueError(
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
    """Render tensor values in a single Matplotlib view with optional controls."""
    style = config or TensorElementsConfig()
    _, records = _extract_tensor_records(data, engine=engine)
    figure, main_ax, _controller = _show_tensor_records(
        records,
        config=style,
        ax=ax,
        show_controls=show_controls,
        show=show,
    )
    return figure, main_ax
