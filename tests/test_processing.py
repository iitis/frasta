"""Tests for frasta.processing module (alignment, filtering, interpolation)."""

import pytest
import numpy as np
from frasta.processing import (
    remove_relative_offset,
    remove_relative_tilt,
    fill_holes,
    nan_aware_gaussian,
    remove_outliers
)


class TestAlignment:
    """Test suite for alignment functions."""
    
    def test_remove_relative_offset_basic(self, sample_grid):
        """Test that offset removal centers data around zero."""
        # Create mask for valid data
        mask = ~np.isnan(sample_grid)
        adjusted = remove_relative_offset(sample_grid, sample_grid, mask)
        
        # After offset removal with same grid, should be identical
        valid_mask = ~np.isnan(adjusted) & ~np.isnan(sample_grid)
        if valid_mask.sum() > 0:
            diff = adjusted[valid_mask] - sample_grid[valid_mask]
            assert np.abs(np.mean(diff)) < 1e-10
    
    def test_remove_relative_offset_with_nan(self):
        """Test offset removal handles NaN values correctly."""
        ref = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=float)
        adj = np.array([[2, 3, 4], [5, np.nan, 7], [8, 9, 10]], dtype=float)
        mask = ~np.isnan(ref) & ~np.isnan(adj)
        
        result = remove_relative_offset(ref, adj, mask)
        
        # Result should have NaN in same locations
        assert np.isnan(result[1, 1])
        # Result should preserve shape
        assert result.shape == adj.shape
    
    def test_remove_relative_tilt_basic(self):
        """Test that tilt removal corrects difference between two tilted planes."""
        # Create two tilted planes with different tilts
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        reference = 2.0 * x + 3.0 * y + 10.0
        target = 2.0 * x + 3.0 * y + 15.0  # Same tilt, different offset
        mask = np.ones_like(reference, dtype=bool)
        
        # Remove relative tilt - should align them
        corrected = remove_relative_tilt(reference, target, mask)
        
        # After alignment, difference should be minimal
        difference = np.abs(corrected - reference)
        assert np.nanmean(difference) < 1.0  # Should be close


class TestFiltering:
    """Test suite for filtering functions."""
    
    def test_fill_holes_reduces_nans(self, sample_grid):
        """Test that fill_holes reduces number of NaN values."""
        nan_count_before = np.isnan(sample_grid).sum()
        filled = fill_holes(sample_grid.copy())
        nan_count_after = np.isnan(filled).sum()
        
        assert nan_count_after <= nan_count_before
    
    def test_fill_holes_preserves_shape(self, sample_grid):
        """Test that fill_holes preserves grid shape."""
        filled = fill_holes(sample_grid.copy())
        assert filled.shape == sample_grid.shape
    
    def test_fill_holes_preserves_valid_data(self):
        """Test that fill_holes doesn't change valid data significantly."""
        # Grid with a small hole
        grid = np.ones((5, 5), dtype=float)
        grid[2, 2] = np.nan
        
        filled = fill_holes(grid.copy())
        
        # Valid data should be roughly unchanged
        valid_mask = ~np.isnan(grid)
        assert np.allclose(filled[valid_mask], grid[valid_mask], rtol=0.1)
        
        # Hole should be filled
        assert not np.isnan(filled[2, 2])
    
    def test_nan_aware_gaussian_smooths_data(self):
        """Test that Gaussian filtering smooths data."""
        # Create noisy data
        np.random.seed(42)
        grid = np.random.randn(20, 20) + 10.0
        
        smoothed = nan_aware_gaussian(grid, sigma=2.0)
        
        # Smoothed data should have lower variance
        assert np.var(smoothed) < np.var(grid)
        assert smoothed.shape == grid.shape
    
    def test_nan_aware_gaussian_handles_nan(self):
        """Test that Gaussian filtering handles NaN values."""
        grid = np.ones((10, 10), dtype=float)
        grid[3:5, 3:5] = np.nan
        
        smoothed = nan_aware_gaussian(grid, sigma=1.0)
        
        # Should still have NaN in same general area
        assert smoothed.shape == grid.shape
        # Valid data should be smoothed
        assert np.all(smoothed[0:2, 0:2] > 0)
    
    def test_remove_outliers_clips_values(self):
        """Test that outlier removal replaces extreme values."""
        # Grid with outliers
        original = np.ones((10, 10), dtype=float) * 5.0
        original[2, 2] = 100.0  # Outlier
        original[7, 7] = -100.0  # Outlier
        
        # Smoothed version without outliers
        smoothed = np.ones((10, 10), dtype=float) * 5.0
        
        # Threshold: difference > 10 is outlier
        cleaned = remove_outliers(original, smoothed, threshold=10.0)
        
        # Outliers should be replaced with smoothed values
        assert cleaned[2, 2] == 5.0
        assert cleaned[7, 7] == 5.0
        # Normal values should be unchanged
        assert np.abs(cleaned[0, 0] - 5.0) < 0.1


class TestInterpolation:
    """Test suite for interpolation functions."""
    
    def test_fill_holes_is_interpolation(self):
        """Test that fill_holes performs reasonable interpolation."""
        # Create a simple grid with a hole
        grid = np.array([
            [10, 10, 10, 10, 10],
            [10, 20, 20, 20, 10],
            [10, 20, np.nan, 20, 10],  # Hole in middle
            [10, 20, 20, 20, 10],
            [10, 10, 10, 10, 10]
        ], dtype=float)
        
        filled = fill_holes(grid.copy())
        
        # Filled value should be close to neighbors (around 20)
        assert not np.isnan(filled[2, 2])
        assert 15.0 < filled[2, 2] < 25.0
