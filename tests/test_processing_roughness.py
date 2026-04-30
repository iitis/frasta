"""Tests for minimal roughness parameter helpers."""

import numpy as np
import pytest

from frasta.core import Surface
from frasta.processing.roughness import (
    profile_roughness_parameters,
    surface_roughness_parameters,
)


def test_surface_roughness_symmetric_grid():
    """Surface parameters are computed from mean-centered valid heights."""
    surface = Surface(
        height=np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, np.nan]]),
        dx=1.0,
        dy=1.0,
    )

    result = surface_roughness_parameters(surface)

    assert result["Sa"] == pytest.approx(1.2)
    assert result["Sq"] == pytest.approx(np.sqrt(2.0))
    assert result["Sz"] == pytest.approx(4.0)
    assert set(result) == {"Sa", "Sq", "Sz"}


def test_surface_roughness_respects_mask():
    """An explicit mask restricts the evaluated surface region."""
    grid = np.array([[0.0, 10.0], [20.0, 30.0]])
    mask = np.array([[True, True], [False, False]])

    result = surface_roughness_parameters(grid, mask=mask)

    assert result["Sa"] == pytest.approx(5.0)
    assert result["Sq"] == pytest.approx(5.0)
    assert result["Sz"] == pytest.approx(10.0)


def test_surface_roughness_is_offset_invariant():
    """Mean centering makes surface summaries insensitive to constant offset."""
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])

    base = surface_roughness_parameters(grid)
    shifted = surface_roughness_parameters(grid + 100.0)

    assert shifted == pytest.approx(base)


def test_profile_roughness_symmetric_profile():
    """Profile parameters match simple analytical values."""
    profile = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = profile_roughness_parameters(profile)

    assert result["Ra"] == pytest.approx(1.2)
    assert result["Rq"] == pytest.approx(np.sqrt(2.0))
    assert result["Rz"] == pytest.approx(4.0)
    assert set(result) == {"Ra", "Rq", "Rz"}


def test_profile_roughness_is_offset_invariant():
    """Mean centering makes profile summaries insensitive to constant offset."""
    profile = np.array([1.0, 2.0, 3.0, 4.0])

    base = profile_roughness_parameters(profile)
    shifted = profile_roughness_parameters(profile - 50.0)

    assert shifted == pytest.approx(base)


def test_profile_roughness_rz_uses_five_highest_and_lowest():
    """Rz averages five highest and five lowest valid profile points."""
    profile = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)

    result = profile_roughness_parameters(profile)

    expected = np.mean([3, 4, 5, 6, 7]) - np.mean([-5, -4, -3, -2, -1])
    assert result["Rz"] == pytest.approx(expected)


def test_roughness_rejects_invalid_inputs():
    """Roughness helpers fail clearly for empty valid data."""
    with pytest.raises(ValueError):
        surface_roughness_parameters(np.full((2, 2), np.nan))

    with pytest.raises(ValueError):
        profile_roughness_parameters(np.array([np.nan, np.nan]))


def test_surface_roughness_rejects_mask_shape_mismatch():
    """Mask shape must match the evaluated grid shape."""
    with pytest.raises(ValueError):
        surface_roughness_parameters(np.ones((2, 2)), mask=np.ones((3, 3), dtype=bool))
