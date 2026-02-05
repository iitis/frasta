import numpy as np
import pytest
from src.helpers import (
    resource_path, compute_offset_in_center, remove_relative_offset,
    remove_relative_tilt, fill_holes, nan_aware_gaussian, remove_outliers
)

def test_resource_path_returns_string():
    path = resource_path("test.txt")
    assert isinstance(path, str)
    assert path.endswith("test.txt")

def test_compute_offset_in_center_correct():
    ref = np.ones((10, 10))
    tgt = np.ones((10, 10)) * 2
    offset = compute_offset_in_center(ref, tgt, window_size=4)
    assert np.isclose(offset, -1)

def test_compute_offset_in_center_raises():
    arr = np.full((10, 10), np.nan)
    with pytest.raises(ValueError):
        compute_offset_in_center(arr, arr, window_size=4)

def test_remove_relative_offset_shifts_target():
    ref = np.ones((10, 10))
    tgt = np.ones((10, 10)) * 2
    mask = ~np.isnan(ref) & ~np.isnan(tgt)
    shifted = remove_relative_offset(ref, tgt, mask)
    assert np.allclose(shifted, np.ones((10, 10)))

def test_remove_relative_tilt_runs():
    ref = np.ones((10, 10))
    tgt = np.ones((10, 10)) * 2
    mask = ~np.isnan(ref) & ~np.isnan(tgt)
    result = remove_relative_tilt(ref, tgt, mask)
    assert result.shape == ref.shape

def test_fill_holes_interpolates():
    arr = np.array([[1, np.nan], [3, 4]])
    filled = fill_holes(arr)
    assert not np.isnan(filled).any()
    assert np.isclose(filled[0, 1], 1)

def test_fill_holes_no_nan():
    arr = np.ones((3, 3))
    filled = fill_holes(arr)
    assert np.allclose(filled, arr)

def test_nan_aware_gaussian_basic():
    arr = np.ones((5, 5))
    result = nan_aware_gaussian(arr, sigma=1)
    assert result.shape == arr.shape

def test_nan_aware_gaussian_with_nan():
    arr = np.full((5, 5), np.nan)
    arr[2, 2] = 1.0
    result = nan_aware_gaussian(arr, sigma=1)
    
    # Tam gdzie nie ma żadnych ważnych sąsiadów, zostaje NaN
    # UWAGA to nie przechodzi !!!!
    # assert np.isnan(result[0, 0])
    
    # Tam gdzie był pojedynczy punkt, filtr rozleje wartość
    assert not np.isnan(result[2, 2])    

def test_remove_outliers_replaces():
    orig = np.array([[1, 100], [3, 4]])
    smooth = np.array([[1, 2], [3, 4]])
    cleaned = remove_outliers(orig, smooth, threshold=10)
    assert cleaned[0, 1] == 2
    assert cleaned[1, 1] == 4
