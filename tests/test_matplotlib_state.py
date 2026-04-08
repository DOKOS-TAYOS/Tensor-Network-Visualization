from __future__ import annotations

from typing import cast

import matplotlib
from matplotlib.figure import SubFigure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensor_network_viz import _matplotlib_state


def test_request_canvas_redraw_resolves_subfigure_to_root_figure(
    monkeypatch,
) -> None:
    fig = plt.figure()
    subfig = cast(SubFigure, fig.subfigures(1, 1))
    redraw_targets: list[object] = []
    draw_idle_calls: list[int] = []

    def fake_canvas_supports_live_redraw(figure: object) -> bool:
        redraw_targets.append(figure)
        return True

    def fake_draw_idle() -> None:
        draw_idle_calls.append(1)

    monkeypatch.setattr(
        _matplotlib_state,
        "canvas_supports_live_redraw",
        fake_canvas_supports_live_redraw,
    )
    monkeypatch.setattr(fig.canvas, "draw_idle", fake_draw_idle)

    _matplotlib_state.request_canvas_redraw(subfig)

    plt.close(fig)

    assert redraw_targets == [fig]
    assert draw_idle_calls == [1]
