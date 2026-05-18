"""Tests for crack-path extraction and tortuosity helpers."""

import numpy as np
import pytest

from frasta.core import Surface
from frasta.processing.crack_path import (
    analyze_crack_path,
    crack_opening_map,
    crack_path_curvature,
    crack_path_local_tortuosity,
    crack_path_orientation_statistics,
    crack_path_tortuosity,
    extract_crack_path,
    extract_crack_path_contour,
    sweep_crack_path_thresholds,
)


def test_extract_crack_path_returns_straight_front_for_uniform_opening():
    """A flat opening front should map to a straight polyline."""
    open_mask = np.zeros((5, 6), dtype=bool)
    open_mask[2:, :] = True

    points = extract_crack_path(open_mask, dx=2.0, dy=3.0, propagation_axis="x", front_side="min")

    expected_x = np.arange(6, dtype=float) * 2.0
    expected_y = np.full(6, 6.0, dtype=float)
    assert np.allclose(points[:, 0], expected_x)
    assert np.allclose(points[:, 1], expected_y)


def test_extract_crack_path_supports_y_axis_propagation():
    """The same extractor should work when propagation is nominally along Y."""
    open_mask = np.zeros((6, 5), dtype=bool)
    open_mask[:, 3:] = True

    points = extract_crack_path(open_mask, dx=2.0, dy=1.5, propagation_axis="y", front_side="min")

    expected_x = np.full(6, 6.0, dtype=float)
    expected_y = np.arange(6, dtype=float) * 1.5
    assert np.allclose(points[:, 0], expected_x)
    assert np.allclose(points[:, 1], expected_y)


def test_crack_path_tortuosity_is_one_for_straight_line():
    """Straight crack fronts should have unit tortuosity."""
    points = np.array([[0.0, 4.0], [2.0, 4.0], [4.0, 4.0], [6.0, 4.0]])

    result = crack_path_tortuosity(points, propagation_axis="x")

    assert result["effective_length"] == pytest.approx(6.0)
    assert result["projected_length"] == pytest.approx(6.0)
    assert result["tortuosity"] == pytest.approx(1.0)


def test_crack_path_tortuosity_exceeds_one_for_wavy_front():
    """A non-straight front must be longer than its projected span."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [4.0, 0.0],
    ])

    result = crack_path_tortuosity(points, propagation_axis="x")

    expected_length = 4.0 * np.sqrt(2.0)
    assert result["effective_length"] == pytest.approx(expected_length)
    assert result["projected_length"] == pytest.approx(4.0)
    assert result["tortuosity"] == pytest.approx(expected_length / 4.0)


def test_crack_path_tortuosity_supports_manual_propagation_angle():
    """Manual propagation angles should project a diagonal path correctly."""
    points = np.array([
        [0.0, 0.0],
        [2.0, 2.0],
        [4.0, 4.0],
    ])

    result = crack_path_tortuosity(
        points,
        propagation_axis="angle",
        propagation_angle_degrees=45.0,
    )

    assert result["effective_length"] == pytest.approx(4.0 * np.sqrt(2.0))
    assert result["projected_length"] == pytest.approx(4.0 * np.sqrt(2.0))
    assert result["tortuosity"] == pytest.approx(1.0)


def test_crack_path_curvature_is_zero_for_straight_line():
    """Straight fronts should have zero curvature everywhere."""
    points = np.array([
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [4.0, 2.0],
    ])

    result = crack_path_curvature(points)

    assert np.allclose(result["arc_length"], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(result["curvature"], 0.0)


def test_local_tortuosity_is_one_for_straight_line():
    """A straight line should have unit local tortuosity everywhere it is defined."""
    points = np.array([
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [4.0, 2.0],
        [5.0, 2.0],
    ])

    result = crack_path_local_tortuosity(points, propagation_axis="x", window_length=3.0)
    finite = np.isfinite(result["local_tortuosity"])
    assert np.allclose(result["local_tortuosity"][finite], 1.0)


def test_orientation_statistics_match_horizontal_line():
    """A horizontal path should report a dominant orientation near 0 degrees."""
    points = np.array([
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [4.0, 2.0],
    ])

    result = crack_path_orientation_statistics(points, reference_angle_degrees=0.0)

    assert result["dominant_orientation_degrees"] == pytest.approx(0.0)
    assert result["orientation_strength"] == pytest.approx(1.0)
    assert result["alignment_delta_degrees"] == pytest.approx(0.0)


def test_extract_crack_path_contour_returns_valid_polyline():
    """Contour extraction should trace the open/contact interface."""
    open_mask = np.zeros((6, 7), dtype=bool)
    open_mask[2:, :] = True

    points = extract_crack_path_contour(open_mask, dx=2.0, dy=3.0, propagation_axis="x")

    assert points.ndim == 2
    assert points.shape[1] == 2
    assert len(points) >= 2
    assert np.max(points[:, 0]) > np.min(points[:, 0])
    assert np.all(np.diff(points[:, 0]) > 0.0)


def test_extract_crack_path_contour_resamples_to_constant_axis_step():
    """Contour post-processing should use a stable propagation-axis step."""
    open_mask = np.zeros((6, 7), dtype=bool)
    open_mask[2:, :] = True

    points = extract_crack_path_contour(
        open_mask,
        dx=2.0,
        dy=3.0,
        propagation_axis="x",
        resample_step=2.0,
        smoothing_window=1,
    )

    assert np.allclose(np.diff(points[:, 0]), 2.0)


def test_extract_crack_path_supports_manual_propagation_angle():
    """The scanline extractor should support a manually defined propagation angle."""
    open_mask = np.zeros((6, 6), dtype=bool)
    for row in range(6):
        open_mask[row, row:] = True

    points = extract_crack_path(
        open_mask,
        dx=1.0,
        dy=1.0,
        propagation_axis="angle",
        propagation_angle_degrees=45.0,
        front_side="min",
    )

    assert points.ndim == 2
    assert points.shape[1] == 2
    assert len(points) >= 2
    projected = (points @ np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)]))
    assert np.all(np.diff(projected) >= 0.0)


def test_crack_opening_map_respects_separation_threshold():
    """Increasing separation should reduce the detected open region."""
    reference = np.array([[3.0, 2.0], [1.0, 0.0]])
    adjusted = np.zeros((2, 2), dtype=float)

    difference, open_mask, valid_mask = crack_opening_map(reference, adjusted, separation=1.5)

    assert np.allclose(difference, reference)
    assert np.array_equal(valid_mask, np.ones((2, 2), dtype=bool))
    assert np.array_equal(open_mask, np.array([[True, True], [False, False]]))


def test_analyze_crack_path_runs_end_to_end_for_surface_pair():
    """The high-level helper should return maps, path points, and metrics."""
    front_rows = np.array([1, 2, 1, 2, 1], dtype=int)
    reference = np.ones((5, 5), dtype=float)
    adjusted = np.ones((5, 5), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    surface_a = Surface(reference, dx=2.0, dy=3.0)
    surface_b = Surface(adjusted, dx=2.0, dy=3.0)

    result = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=1.0,
        propagation_axis="x",
        front_side="min",
    )

    assert result["path_method"] == "first_open_pixel"
    assert result["propagation_axis"] == "x"
    assert result["front_side"] == "min"
    assert result["path_points"].shape == (5, 2)
    assert np.allclose(result["path_points"][:, 0], np.arange(5, dtype=float) * 2.0)
    assert np.allclose(result["path_points"][:, 1], front_rows.astype(float) * 3.0)
    assert result["tortuosity"] > 1.0
    assert result["curvature"].shape[0] == result["path_points"].shape[0]
    assert result["local_tortuosity"].shape[0] == result["path_points"].shape[0]
    assert 0.0 <= result["dominant_orientation_degrees"] < 180.0


def test_analyze_crack_path_supports_contour_method():
    """High-level analysis should support the contour-based path extraction method."""
    front_rows = np.array([1, 2, 1, 2, 1], dtype=int)
    reference = np.ones((5, 5), dtype=float)
    adjusted = np.ones((5, 5), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    surface_a = Surface(reference, dx=2.0, dy=3.0)
    surface_b = Surface(adjusted, dx=2.0, dy=3.0)

    result = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=1.0,
        propagation_axis="x",
        front_side="min",
        method="contour",
    )

    assert result["path_method"] == "contour"
    assert result["tortuosity"] >= 1.0
    assert result["curvature"].shape[0] == result["path_points"].shape[0]


def test_analyze_crack_path_supports_manual_propagation_angle():
    """The high-level analysis should preserve a manually defined propagation angle."""
    open_mask = np.zeros((6, 6), dtype=bool)
    for row in range(6):
        open_mask[row, row:] = True

    reference = np.ones((6, 6), dtype=float)
    adjusted = np.ones((6, 6), dtype=float)
    adjusted[open_mask] = -1.0
    surface_a = Surface(reference, dx=1.0, dy=1.0)
    surface_b = Surface(adjusted, dx=1.0, dy=1.0)

    result = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=1.0,
        propagation_axis="angle",
        propagation_angle_degrees=45.0,
        front_side="min",
    )

    assert result["propagation_axis"] == "angle"
    assert result["propagation_angle_degrees"] == pytest.approx(45.0)
    assert result["tortuosity"] >= 1.0


def test_contour_smoothing_reduces_mean_absolute_curvature():
    """Transverse smoothing should suppress grid-induced curvature noise."""
    front_rows = np.array([1, 3, 1, 3, 1, 3, 1], dtype=int)
    reference = np.ones((7, 7), dtype=float)
    adjusted = np.ones((7, 7), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    surface_a = Surface(reference, dx=2.0, dy=2.0)
    surface_b = Surface(adjusted, dx=2.0, dy=2.0)

    raw = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=1.0,
        propagation_axis="x",
        method="contour",
        contour_smoothing_window=1,
    )
    smooth = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=1.0,
        propagation_axis="x",
        method="contour",
        contour_smoothing_window=7,
    )

    raw_mean = float(np.mean(np.abs(raw["curvature"])))
    smooth_mean = float(np.mean(np.abs(smooth["curvature"])))
    assert smooth_mean <= raw_mean


def test_extract_crack_path_rejects_empty_front():
    """A crack path requires at least two sampled front points."""
    with pytest.raises(ValueError):
        extract_crack_path(np.zeros((3, 3), dtype=bool))


def test_sweep_crack_path_thresholds_returns_metric_arrays():
    """Threshold sweeps should return one metric value per requested threshold."""
    front_rows = np.array([1, 2, 1, 2, 1], dtype=int)
    reference = np.ones((5, 5), dtype=float)
    adjusted = np.ones((5, 5), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    surface_a = Surface(reference, dx=2.0, dy=3.0)
    surface_b = Surface(adjusted, dx=2.0, dy=3.0)
    thresholds = np.array([0.5, 1.0, 5.0], dtype=float)

    result = sweep_crack_path_thresholds(
        surface_a,
        surface_b,
        thresholds=thresholds,
        dx=surface_a.dx,
        dy=surface_a.dy,
        propagation_axis="x",
        method="first_open_pixel",
    )

    assert np.array_equal(result["thresholds"], thresholds)
    assert result["effective_length"].shape == thresholds.shape
    assert result["projected_length"].shape == thresholds.shape
    assert result["tortuosity"].shape == thresholds.shape
    assert result["mean_abs_curvature"].shape == thresholds.shape
    assert np.all(np.isfinite(result["tortuosity"][:2]))
