from __future__ import annotations

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures_after_test() -> Generator[None, None, None]:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
