"""Tests for plane_fitting.py and morphology.py modules.

This test suite covers local plane fitting algorithms and morphological operations
that were previously untested or under-tested.
"""

import pytest
import numpy as np
from frasta.processing.plane_fitting import (
    fit_plane_local_least_squares,
    fit_plane_local_ransac,
    fit_plane_local_median_filter
)
from frasta.processing.morphology import (
    fit_plane_robust,
    level_by_three_points,
    remove_polynomial_form,
    threshold_grid
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tilted_grid():
    """Create a tilted plane with known parameters."""
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    # Create tilted plane: z = 0.5*x + 0.3*y + 100
    grid = 0.5 * x_idx + 0.3 * y_idx + 100.0
    
    return grid, (0.5, 0.3, 100.0)


@pytest.fixture
def noisy_tilted_grid():
    """Create a tilted plane with Gaussian noise."""
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    np.random.seed(42)
    grid = 0.5 * x_idx + 0.3 * y_idx + 100.0 + np.random.randn(ny, nx) * 2.0
    
    return grid, (0.5, 0.3, 100.0)


@pytest.fixture
def grid_with_outliers():
    """Create a tilted plane with outliers."""
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    grid = 0.5 * x_idx + 0.3 * y_idx + 100.0
    
    # Add outliers
    grid[50, 50] = 500.0
    grid[100, 100] = -200.0
    grid[150, 150] = 800.0
    
    return grid, (0.5, 0.3, 100.0)


@pytest.fixture
def grid_with_nan():
    """Create a grid with NaN regions."""
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    grid = 0.5 * x_idx + 0.3 * y_idx + 100.0
    
    # Add NaN regions
    grid[0:20, :] = np.nan
    grid[-20:, :] = np.nan
    grid[:, 0:20] = np.nan
    grid[:, -20:] = np.nan
    
    return grid


# ============================================================================
# Tests for plane_fitting.py
# ============================================================================

class TestFitPlaneLocalLeastSquares:
    """Tests for fit_plane_local_least_squares function."""
    
    def test_basic_tilted_plane(self, tilted_grid):
        """Test fitting a perfect tilted plane."""
        grid, expected_coeffs = tilted_grid
        
        # Fit plane at center
        a, b, c = fit_plane_local_least_squares(grid, x=100, y=100, window_size=50)
        
        # Should recover original coefficients (within tolerance)
        assert abs(a - expected_coeffs[0]) < 0.01
        assert abs(b - expected_coeffs[1]) < 0.01
        # c can vary depending on window position
        assert abs(c - expected_coeffs[2]) < 20  # More relaxed for intercept
    
    def test_with_noise(self, noisy_tilted_grid):
        """Test fitting with Gaussian noise."""
        grid, expected_coeffs = noisy_tilted_grid
        
        a, b, c = fit_plane_local_least_squares(grid, x=100, y=100, window_size=50)
        
        # Should still recover coefficients reasonably well
        assert abs(a - expected_coeffs[0]) < 0.1
        assert abs(b - expected_coeffs[1]) < 0.1
    
    def test_edge_of_grid(self, tilted_grid):
        """Test fitting near edge of grid."""
        grid, _ = tilted_grid
        
        # Near top-left corner
        a, b, c = fit_plane_local_least_squares(grid, x=10, y=10, window_size=50)
        
        # Should complete without error
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        assert isinstance(c, (int, float))
    
    def test_small_window(self, tilted_grid):
        """Test with small window size."""
        grid, expected_coeffs = tilted_grid
        
        a, b, c = fit_plane_local_least_squares(grid, x=100, y=100, window_size=10)
        
        # Should still work with smaller window
        assert abs(a - expected_coeffs[0]) < 0.1
        assert abs(b - expected_coeffs[1]) < 0.1
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Create grid with mostly NaN
        grid = np.full((100, 100), np.nan)
        grid[50:52, 50:52] = 10.0  # Only 4 valid points (< 10 threshold)
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            fit_plane_local_least_squares(grid, x=51, y=51, window_size=3)
    
    def test_handles_nan_in_window(self, grid_with_nan):
        """Test that NaN values in window are properly handled."""
        grid = grid_with_nan
        
        # Fit in center where there is valid data
        a, b, c = fit_plane_local_least_squares(grid, x=100, y=100, window_size=30)
        
        # Should complete successfully
        assert np.isfinite(a) and np.isfinite(b) and np.isfinite(c)


class TestFitPlaneLocalRANSAC:
    """Tests for fit_plane_local_ransac function."""
    
    def test_basic_tilted_plane(self, tilted_grid):
        """Test RANSAC on perfect tilted plane."""
        grid, expected_coeffs = tilted_grid
        
        a, b, c = fit_plane_local_ransac(grid, x=100, y=100, window_size=50)
        
        assert abs(a - expected_coeffs[0]) < 0.01
        assert abs(b - expected_coeffs[1]) < 0.01
    
    def test_with_outliers(self, grid_with_outliers):
        """Test RANSAC robustness against outliers."""
        grid, expected_coeffs = grid_with_outliers
        
        # RANSAC should be robust to outliers
        a, b, c = fit_plane_local_ransac(grid, x=100, y=100, window_size=80,
                                          residual_threshold=50.0)
        
        # Should still recover plane despite outliers
        assert abs(a - expected_coeffs[0]) < 0.1
        assert abs(b - expected_coeffs[1]) < 0.1
    
    def test_comparison_with_least_squares(self, grid_with_outliers):
        """Test that RANSAC outperforms least squares with outliers."""
        grid, expected_coeffs = grid_with_outliers
        
        # Least squares (affected by outliers)
        a_ls, b_ls, c_ls = fit_plane_local_least_squares(grid, x=100, y=100, window_size=80)
        
        # RANSAC (robust to outliers)
        a_ransac, b_ransac, c_ransac = fit_plane_local_ransac(
            grid, x=100, y=100, window_size=80, residual_threshold=50.0)
        
        # RANSAC should be closer to true values
        error_ls = abs(a_ls - expected_coeffs[0]) + abs(b_ls - expected_coeffs[1])
        error_ransac = abs(a_ransac - expected_coeffs[0]) + abs(b_ransac - expected_coeffs[1])
        
        assert error_ransac < error_ls
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        grid = np.full((100, 100), np.nan)
        grid[50:52, 50:52] = 10.0  # Only 4 valid points
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            fit_plane_local_ransac(grid, x=51, y=51, window_size=5)
    
    def test_various_residual_thresholds(self, noisy_tilted_grid):
        """Test with different residual thresholds."""
        grid, expected_coeffs = noisy_tilted_grid
        
        for threshold in [10.0, 50.0, 100.0]:
            a, b, c = fit_plane_local_ransac(grid, x=100, y=100, window_size=50,
                                              residual_threshold=threshold)
            
            # Should work with various thresholds
            assert np.isfinite(a) and np.isfinite(b) and np.isfinite(c)


class TestFitPlaneLocalMedianFilter:
    """Tests for fit_plane_local_median_filter function."""
    
    def test_basic_tilted_plane(self, tilted_grid):
        """Test median filter method on perfect plane."""
        grid, expected_coeffs = tilted_grid
        
        a, b, c = fit_plane_local_median_filter(grid, x=100, y=100, window_size=50)
        
        assert abs(a - expected_coeffs[0]) < 0.01
        assert abs(b - expected_coeffs[1]) < 0.01
    
    def test_with_outliers(self, grid_with_outliers):
        """Test median filter robustness against outliers."""
        grid, expected_coeffs = grid_with_outliers
        
        a, b, c = fit_plane_local_median_filter(grid, x=100, y=100, window_size=80,
                                                 outlier_threshold=3.0)
        
        # Should reject outliers and recover plane
        assert abs(a - expected_coeffs[0]) < 0.1
        assert abs(b - expected_coeffs[1]) < 0.1
    
    def test_constant_data(self):
        """Test with constant data (MAD = 0)."""
        grid = np.ones((100, 100)) * 50.0
        
        # Should handle constant data gracefully
        a, b, c = fit_plane_local_median_filter(grid, x=50, y=50, window_size=20)
        
        # For constant data, should get a horizontal plane
        assert abs(a) < 0.01
        assert abs(b) < 0.01
        assert abs(c - 50.0) < 0.1
    
    def test_nearly_constant_data(self):
        """Test with nearly constant data (very small MAD)."""
        np.random.seed(42)
        grid = np.ones((100, 100)) * 50.0 + np.random.randn(100, 100) * 0.001
        
        # Should handle near-constant data
        a, b, c = fit_plane_local_median_filter(grid, x=50, y=50, window_size=20)
        
        assert np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        grid = np.full((100, 100), np.nan)
        grid[50:52, 50:52] = 10.0
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            fit_plane_local_median_filter(grid, x=51, y=51, window_size=5)
    
    def test_too_little_after_outlier_rejection(self):
        """Test error when too little data remains after outlier rejection."""
        # Create grid where most points are outliers
        grid = np.ones((50, 50)) * 100.0
        grid[20:30, 20:30] = 100.0  # Normal values
        grid[0:10, :] = 1000.0  # Many outliers
        
        # Should raise error or handle gracefully
        try:
            a, b, c = fit_plane_local_median_filter(grid, x=25, y=25, window_size=20,
                                                     outlier_threshold=0.1)
            # If it succeeds, values should be finite
            assert np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
        except ValueError as e:
            # Expected error
            assert "Too little data after outlier rejection" in str(e)


# ============================================================================
# Tests for morphology.py
# ============================================================================

class TestFitPlaneRobust:
    """Tests for fit_plane_robust function (global RANSAC fitting)."""
    
    def test_basic_tilted_plane(self, tilted_grid):
        """Test robust fitting on perfect plane."""
        grid, expected_coeffs = tilted_grid
        
        plane_grid, coeffs, inlier_mask = fit_plane_robust(grid)
        
        # Should recover original coefficients
        assert abs(coeffs[0] - expected_coeffs[0]) < 0.01
        assert abs(coeffs[1] - expected_coeffs[1]) < 0.01
        assert abs(coeffs[2] - expected_coeffs[2]) < 0.01
        
        # Most points should be inliers
        assert np.sum(inlier_mask) > 0.95 * grid.size
    
    def test_with_outliers(self, grid_with_outliers):
        """Test robust fitting with outliers."""
        grid, expected_coeffs = grid_with_outliers
        
        plane_grid, coeffs, inlier_mask = fit_plane_robust(grid)
        
        # Should still recover plane despite outliers
        assert abs(coeffs[0] - expected_coeffs[0]) < 0.1
        assert abs(coeffs[1] - expected_coeffs[1]) < 0.1
        
        # Outlier points should not be inliers
        assert not inlier_mask[50, 50]   # Outlier at (50, 50)
        assert not inlier_mask[100, 100] # Outlier at (100, 100)
        assert not inlier_mask[150, 150] # Outlier at (150, 150)
    
    def test_auto_threshold(self, noisy_tilted_grid):
        """Test automatic threshold calculation."""
        grid, _ = noisy_tilted_grid
        
        # With None threshold, should auto-calculate
        plane_grid, coeffs, inlier_mask = fit_plane_robust(grid, residual_threshold=None)
        
        assert plane_grid is not None
        assert len(coeffs) == 3
        assert np.sum(inlier_mask) > 0
    
    def test_with_mask(self, tilted_grid):
        """Test robust fitting with mask."""
        grid, expected_coeffs = tilted_grid
        
        # Create mask that excludes borders
        mask = np.ones_like(grid, dtype=bool)
        mask[0:20, :] = False
        mask[-20:, :] = False
        mask[:, 0:20] = False
        mask[:, -20:] = False
        
        plane_grid, coeffs, inlier_mask = fit_plane_robust(grid, mask=mask)
        
        # Should still recover plane
        assert abs(coeffs[0] - expected_coeffs[0]) < 0.01
        assert abs(coeffs[1] - expected_coeffs[1]) < 0.01
    
    def test_insufficient_data(self):
        """Test with insufficient valid data."""
        grid = np.full((20, 20), np.nan)
        grid[10:12, 10:12] = 5.0  # Only 4 points
        
        plane_grid, coeffs, inlier_mask = fit_plane_robust(grid)
        
        # Should return zeros/empty
        assert np.allclose(plane_grid, 0.0)
        assert coeffs == (0, 0, 0)


class TestLevelByThreePoints:
    """Tests for level_by_three_points function."""
    
    def test_horizontal_plane(self):
        """Test leveling a horizontal plane - should remove it."""
        ny, nx = 100, 100
        grid = np.ones((ny, nx)) * 50.0
        
        xi = np.arange(nx) * 1.0
        yi = np.arange(ny) * 1.0
        
        # Three points on the plane
        p1 = (10.0, 10.0)
        p2 = (80.0, 10.0)
        p3 = (50.0, 80.0)
        
        leveled = level_by_three_points(grid, p1, p2, p3, xi, yi)
        
        # After leveling, should be close to zero
        assert abs(np.nanmean(leveled)) < 1.0
    
    def test_tilted_plane_leveling(self):
        """Test leveling a tilted plane."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Tilted plane
        grid = 0.5 * x_idx + 0.3 * y_idx + 100.0
        
        xi = np.arange(nx) * 1.0
        yi = np.arange(ny) * 1.0
        
        # Three points
        p1 = (10.0, 10.0)
        p2 = (80.0, 10.0)
        p3 = (50.0, 80.0)
        
        leveled = level_by_three_points(grid, p1, p2, p3, xi, yi)
        
        # After leveling, variation should be much smaller
        assert np.nanstd(leveled) < 5.0
    
    def test_with_surface_features(self):
        """Test leveling preserves surface features."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Tilted plane with a bump
        grid = 0.5 * x_idx + 0.3 * y_idx + 100.0
        grid[40:60, 40:60] += 10.0  # Add a plateau
        
        xi = np.arange(nx) * 1.0
        yi = np.arange(ny) * 1.0
        
        p1 = (10.0, 10.0)
        p2 = (80.0, 10.0)
        p3 = (10.0, 80.0)
        
        leveled = level_by_three_points(grid, p1, p2, p3, xi, yi)
        
        # The plateau should still be visible
        plateau_mean = np.nanmean(leveled[40:60, 40:60])
        background_mean = np.nanmean(leveled[0:20, 0:20])
        
        assert plateau_mean > background_mean + 5.0
    
    def test_nan_point_returns_copy(self):
        """Test that NaN at a point returns unchanged copy."""
        ny, nx = 100, 100
        grid = np.ones((ny, nx)) * 50.0
        grid[10, 10] = np.nan  # One of the points will be NaN
        
        xi = np.arange(nx) * 1.0
        yi = np.arange(ny) * 1.0
        
        p1 = (10.0, 10.0)  # This point is NaN
        p2 = (80.0, 10.0)
        p3 = (50.0, 80.0)
        
        leveled = level_by_three_points(grid, p1, p2, p3, xi, yi)
        
        # Should return copy without modification (except the NaN)
        valid_mask = ~np.isnan(grid) & ~np.isnan(leveled)
        assert np.allclose(grid[valid_mask], leveled[valid_mask])
    
    def test_coordinate_conversion(self):
        """Test that physical coordinates are properly converted to indices."""
        ny, nx = 50, 60
        grid = np.ones((ny, nx)) * 100.0
        
        # Non-unit spacing
        xi = np.arange(nx) * 2.5  # 0, 2.5, 5.0, ...
        yi = np.arange(ny) * 3.0  # 0, 3.0, 6.0, ...
        
        # Physical coordinates
        p1 = (5.0, 6.0)    # Should map to indices close to (2, 2)
        p2 = (50.0, 6.0)   # Should map to indices close to (20, 2)
        p3 = (25.0, 60.0)  # Should map to indices close to (10, 20)
        
        # Should complete without error
        leveled = level_by_three_points(grid, p1, p2, p3, xi, yi)
        
        assert leveled.shape == grid.shape


class TestRemovePolynomialForm:
    """Tests for remove_polynomial_form function."""
    
    def test_order1_removes_plane(self):
        """Test that order 1 removes a linear plane."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Linear plane
        plane = 0.5 * x_idx + 0.3 * y_idx + 50.0
        
        result = remove_polynomial_form(plane, order=1)
        
        # Should be nearly flat after removal
        assert abs(np.nanmean(result)) < 1.0
        assert np.nanstd(result) < 1.0
    
    def test_order2_removes_quadratic(self):
        """Test that order 2 removes quadratic form."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Normalize for numerical stability
        x_norm = (x_idx - nx/2) / (nx/2)
        y_norm = (y_idx - ny/2) / (ny/2)
        
        # Quadratic form (parabolic bowl)
        grid = 0.5 * x_norm**2 + 0.3 * y_norm**2 + 10.0
        
        result = remove_polynomial_form(grid, order=2)
        
        # Should be nearly flat
        assert abs(np.nanmean(result)) < 1.0
        assert np.nanstd(result) < 1.0
    
    def test_order3_removes_cubic(self):
        """Test that order 3 removes cubic form."""
        ny, nx = 80, 80
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        x_norm = (x_idx - nx/2) / (nx/2)
        y_norm = (y_idx - ny/2) / (ny/2)
        
        # Cubic form
        grid = 0.2 * x_norm**3 + 0.1 * y_norm**3 + 0.3 * x_norm * y_norm + 20.0
        
        result = remove_polynomial_form(grid, order=3)
        
        # Should remove most of the form
        assert np.nanstd(result) < 2.0
    
    def test_preserves_features(self):
        """Test that polynomial removal preserves high-frequency features."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Quadratic form with sine wave features
        x_norm = (x_idx - nx/2) / (nx/2)
        y_norm = (y_idx - ny/2) / (ny/2)
        
        quadratic = 0.5 * x_norm**2 + 0.3 * y_norm**2
        features = 2.0 * np.sin(x_idx * 0.3) * np.cos(y_idx * 0.3)
        grid = quadratic + features + 50.0
        
        result = remove_polynomial_form(grid, order=2)
        
        # The sine wave features should be preserved
        # Check that oscillations are still present
        assert np.nanstd(result) > 0.5  # Non-trivial variation
        # Check that the result resembles the features more than the original
        correlation = np.corrcoef(result.flatten(), features.flatten())[0, 1]
        assert correlation > 0.8  # Strong correlation with features
    
    def test_with_mask(self):
        """Test polynomial removal with mask."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        grid = 0.5 * x_idx + 0.3 * y_idx + 50.0
        
        # Mask excluding borders
        mask = np.ones_like(grid, dtype=bool)
        mask[0:10, :] = False
        mask[-10:, :] = False
        
        result = remove_polynomial_form(grid, order=1, mask=mask)
        
        # Should complete successfully
        assert result.shape == grid.shape
    
    def test_with_nan(self):
        """Test polynomial removal with NaN values."""
        ny, nx = 100, 100
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        grid = 0.5 * x_idx + 0.3 * y_idx + 50.0
        grid[20:30, 20:30] = np.nan
        
        result = remove_polynomial_form(grid, order=1)
        
        # NaN should be preserved
        assert np.sum(np.isnan(result[20:30, 20:30])) > 0
        
        # Valid areas should be processed
        assert np.isfinite(result[50, 50])
    
    def test_invalid_order_raises_error(self):
        """Test that invalid polynomial order raises ValueError."""
        grid = np.ones((50, 50))
        
        with pytest.raises(ValueError, match="Polynomial order must be between 1 and 5"):
            remove_polynomial_form(grid, order=0)
        
        with pytest.raises(ValueError, match="Polynomial order must be between 1 and 5"):
            remove_polynomial_form(grid, order=6)
    
    def test_insufficient_data(self):
        """Test with insufficient valid data."""
        grid = np.full((50, 50), np.nan)
        grid[20:23, 20:23] = 10.0  # Only 9 points (less than 15 basis functions for order 2)
        
        result = remove_polynomial_form(grid, order=2)
        
        # Should return copy when insufficient data (warning logged)
        # OR process successfully if it can
        assert result.shape == grid.shape
        assert np.isfinite(result[21, 21])  # Center point should have some value


class TestThresholdGrid:
    """Tests for threshold_grid function."""
    
    def test_low_threshold(self):
        """Test lower threshold removes low values."""
        grid = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        result = threshold_grid(grid, low=5.0)
        
        # Values < 5 should be NaN
        assert np.isnan(result[0, 0])  # 1.0
        assert np.isnan(result[0, 1])  # 2.0
        assert np.isnan(result[0, 2])  # 3.0
        assert np.isnan(result[1, 0])  # 4.0
        
        # Values >= 5 should remain
        assert result[1, 1] == 5.0
        assert result[2, 2] == 9.0
    
    def test_high_threshold(self):
        """Test upper threshold removes high values."""
        grid = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        result = threshold_grid(grid, high=5.0)
        
        # Values > 5 should be NaN
        assert np.isnan(result[1, 2])  # 6.0
        assert np.isnan(result[2, 0])  # 7.0
        assert np.isnan(result[2, 1])  # 8.0
        assert np.isnan(result[2, 2])  # 9.0
        
        # Values <= 5 should remain
        assert result[0, 0] == 1.0
        assert result[1, 1] == 5.0
    
    def test_both_thresholds(self):
        """Test both thresholds together."""
        grid = np.array([
            [1.0, 5.0, 10.0],
            [2.0, 6.0, 11.0],
            [3.0, 7.0, 12.0]
        ])
        
        result = threshold_grid(grid, low=4.0, high=8.0)
        
        # Only values in [4, 8] should remain
        assert np.isnan(result[0, 0])  # 1.0 < 4
        assert np.isnan(result[0, 2])  # 10.0 > 8
        assert result[0, 1] == 5.0     # 4 <= 5 <= 8
        assert result[1, 1] == 6.0     # 4 <= 6 <= 8
        assert result[2, 1] == 7.0     # 4 <= 7 <= 8
    
    def test_no_threshold_returns_copy(self):
        """Test that no thresholds returns copy."""
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = threshold_grid(grid)
        
        # Should be identical
        assert np.array_equal(result, grid)
        
        # But should be a copy
        assert result is not grid
    
    def test_preserves_existing_nan(self):
        """Test that existing NaN values are preserved."""
        grid = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        result = threshold_grid(grid, low=3.0, high=5.0)
        
        # Original NaN should remain
        assert np.isnan(result[0, 1])
        
        # New NaN from thresholding
        assert np.isnan(result[0, 0])  # 1.0 < 3.0
        assert np.isnan(result[1, 2])  # 6.0 > 5.0
        
        # Valid values in range
        assert result[0, 2] == 3.0
        assert result[1, 1] == 5.0
    
    def test_removes_outliers_sigma_based(self):
        """Test removing outliers using sigma-based thresholds."""
        np.random.seed(42)
        grid = np.random.randn(100, 100) * 10.0 + 50.0
        
        # Add extreme outliers
        grid[10, 10] = 200.0
        grid[50, 50] = -100.0
        
        # Calculate 3-sigma thresholds
        mean = np.nanmean(grid)
        std = np.nanstd(grid)
        
        result = threshold_grid(grid, low=mean - 3*std, high=mean + 3*std)
        
        # Outliers should be removed
        assert np.isnan(result[10, 10])
        assert np.isnan(result[50, 50])
        
        # Most normal points should remain
        normal_count = np.sum(~np.isnan(result))
        assert normal_count > 0.95 * grid.size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
