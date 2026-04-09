from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from plotting_helpers import assert_rendered_figure
from tensor_network_viz import TensorComparisonConfig, TensorElementsConfig, show_tensor_comparison


def test_show_tensor_comparison_abs_diff_renders_expected_matrix() -> None:
    current = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=float)
    reference = np.array([[1.0, 2.0], [5.0, 10.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        config=TensorElementsConfig(mode="elements"),
        comparison_config=TensorComparisonConfig(mode="abs_diff"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert np.asarray(ax.images[0].get_array(), dtype=float).tolist() == [
        [0.0, 2.0],
        [4.0, 6.0],
    ]
    assert "abs diff" in ax.get_title().lower()


def test_show_tensor_comparison_relative_diff_uses_reference_magnitude() -> None:
    current = np.array([[2.0, 0.0]], dtype=float)
    reference = np.array([[1.0, 0.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        config=TensorElementsConfig(mode="elements"),
        comparison_config=TensorComparisonConfig(mode="relative_diff"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert np.asarray(ax.images[0].get_array(), dtype=float).tolist() == [[1.0, 0.0]]
    assert "relative diff" in ax.get_title().lower()


def test_show_tensor_comparison_mismatched_shapes_render_placeholder() -> None:
    current = np.ones((2, 2), dtype=float)
    reference = np.ones((3, 3), dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        comparison_config=TensorComparisonConfig(mode="abs_diff"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.texts
    assert "shape" in ax.texts[0].get_text().lower()


def test_show_tensor_comparison_ratio_masks_near_zero_reference_entries() -> None:
    current = np.array([[2.0, 6.0]], dtype=float)
    reference = np.array([[1.0, 0.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        config=TensorElementsConfig(mode="elements"),
        comparison_config=TensorComparisonConfig(mode="ratio"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    matrix = np.asarray(ax.images[0].get_array(), dtype=float)
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == 2.0
    assert np.isnan(matrix[0, 1])


def test_show_tensor_comparison_sign_change_renders_binary_mask() -> None:
    current = np.array([[1.0, -2.0, 0.0]], dtype=float)
    reference = np.array([[-1.0, -2.0, 5.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        config=TensorElementsConfig(mode="elements"),
        comparison_config=TensorComparisonConfig(mode="sign_change"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert np.asarray(ax.images[0].get_array(), dtype=float).tolist() == [[1.0, 0.0, 1.0]]


def test_show_tensor_comparison_phase_change_uses_wrapped_phase_delta() -> None:
    current = np.array([[1j, -1.0 + 0.0j]], dtype=complex)
    reference = np.array([[1.0 + 0.0j, 1.0 + 0.0j]], dtype=complex)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        config=TensorElementsConfig(mode="elements"),
        comparison_config=TensorComparisonConfig(mode="phase_change"),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    matrix = np.asarray(ax.images[0].get_array(), dtype=float)
    assert np.allclose(matrix, np.array([[np.pi / 2.0, np.pi]], dtype=float))


def test_show_tensor_comparison_topk_changes_renders_text_summary() -> None:
    current = np.array([[1.0, 5.0], [3.0, 7.0]], dtype=float)
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        comparison_config=TensorComparisonConfig(mode="topk_changes", topk_count=2),
        show_controls=False,
        show=False,
    )

    assert_rendered_figure(fig, ax)
    assert ax.texts
    text = ax.texts[0].get_text().lower()
    assert "top 2" in text
    assert "reference" in text


def test_show_tensor_comparison_controls_switch_from_topk_summary_back_to_heatmap() -> None:
    current = np.array([[1.0, 5.0], [3.0, 7.0]], dtype=float)
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)

    fig, ax = show_tensor_comparison(
        current,
        reference,
        comparison_config=TensorComparisonConfig(mode="topk_changes", topk_count=2),
        show_controls=True,
        show=False,
    )
    controller = fig._tensor_network_viz_tensor_elements_controls  # type: ignore[attr-defined]

    assert ax.texts
    assert not ax.images

    controller._compare_radio.set_active(1)  # type: ignore[attr-defined]

    assert_rendered_figure(fig, ax)
    assert ax.images
    assert "abs diff" in ax.get_title().lower()
