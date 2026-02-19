"""Tests for advanced filtering, morphology, and transforms modules."""

import numpy as np
import pytest
from frasta.processing.advanced_filtering import (
    bilateral_filter,
    median_filter_nan_aware,
    morphological_opening,
    morphological_closing,
    robust_gaussian_filter
)
from frasta.processing.morphology import (
    fit_plane_least_squares,
    level_by_plane,
    remove_polynomial_form,
    threshold_grid
)
from frasta.processing.transforms import (
    rotate_grid,
    rescale_grid,
    crop_to_valid_region,
    auto_register_surfaces
)


@pytest.fixture
def sample_grid():
    """Create a simple test grid."""
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a surface with some features
    Z = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(50, 50)
    
    # Add some NaN values
    Z[0:5, 0:5] = np.nan
    
    xi = x
    yi = y
    px_x = x[1] - x[0]
    px_y = y[1] - y[0]
    
    return Z, xi, yi, px_x, px_y


def test_bilateral_filter(sample_grid):
    """Test bilateral filtering."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    filtered = bilateral_filter(grid, sigma_spatial=1.0, sigma_range=0.5, 
                                dx=px_x, dy=px_y)
    
    assert filtered is not None
    assert filtered.shape == grid.shape
    # Should smooth but preserve some structure
    assert not np.allclose(filtered, grid, equal_nan=True)


def test_median_filter(sample_grid):
    """Test median filtering."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add a spike
    grid_with_spike = grid.copy()
    grid_with_spike[25, 25] = 100
    
    filtered = median_filter_nan_aware(grid_with_spike, size=1.0, 
                                      dx=px_x, dy=px_y)
    
    assert filtered is not None
    assert filtered.shape == grid.shape
    # Spike should be reduced
    assert filtered[25, 25] < grid_with_spike[25, 25]


def test_morphological_opening(sample_grid):
    """Test morphological opening."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    filtered = morphological_opening(grid, size=0.5, dx=px_x, dy=px_y)
    
    assert filtered is not None
    assert filtered.shape == grid.shape


def test_morphological_closing(sample_grid):
    """Test morphological closing."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    filtered = morphological_closing(grid, size=0.5, dx=px_x, dy=px_y)
    
    assert filtered is not None
    assert filtered.shape == grid.shape


def test_robust_gaussian_filter(sample_grid):
    """Test robust Gaussian filtering."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add some outliers
    grid_with_outliers = grid.copy()
    grid_with_outliers[10, 10] = 50
    grid_with_outliers[20, 20] = -50
    
    filtered = robust_gaussian_filter(grid_with_outliers, sigma=1.0, 
                                     dx=px_x, dy=px_y, iterations=2)
    
    assert filtered is not None
    assert filtered.shape == grid.shape


def test_fit_plane_least_squares(sample_grid):
    """Test plane fitting."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Create a tilted plane
    ny, nx = grid.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    plane = 0.1 * x_idx + 0.2 * y_idx + 5.0
    
    fitted_plane, coeffs = fit_plane_least_squares(plane)
    
    assert fitted_plane is not None
    assert len(coeffs) == 3
    # Should recover the plane parameters
    assert abs(coeffs[0] - 0.1) < 0.01
    assert abs(coeffs[1] - 0.2) < 0.01


def test_level_by_plane(sample_grid):
    """Test plane leveling."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add a tilt
    ny, nx = grid.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    tilted = grid + 0.1 * x_idx + 0.2 * y_idx
    
    leveled = level_by_plane(tilted)
    
    assert leveled is not None
    assert leveled.shape == grid.shape
    # Mean should be close to 0 after leveling
    assert abs(np.nanmean(leveled)) < 0.1


def test_remove_polynomial_form(sample_grid):
    """Test polynomial form removal."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add quadratic form
    ny, nx = grid.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_norm = (x_idx - nx/2) / (nx/2)
    y_norm = (y_idx - ny/2) / (ny/2)
    quadratic = grid + 0.5 * x_norm**2 + 0.3 * y_norm**2
    
    corrected = remove_polynomial_form(quadratic, order=2)
    
    assert corrected is not None
    assert corrected.shape == grid.shape
    # Should be closer to original
    assert np.nanstd(corrected - grid) < np.nanstd(quadratic - grid)


def test_threshold_grid(sample_grid):
    """Test grid thresholding."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add extreme values
    grid_with_outliers = grid.copy()
    grid_with_outliers[10, 10] = 100
    grid_with_outliers[20, 20] = -100
    
    thresholded = threshold_grid(grid_with_outliers, low=-10, high=10)
    
    assert thresholded is not None
    assert np.isnan(thresholded[10, 10])
    assert np.isnan(thresholded[20, 20])


def test_rotate_grid(sample_grid):
    """Test grid rotation."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    rotated, xi_new, yi_new, px_x_new, px_y_new = rotate_grid(
        grid, 45, xi, yi, px_x, px_y
    )
    
    assert rotated is not None
    assert rotated.shape == grid.shape
    # Pixel sizes should remain the same
    assert px_x_new == px_x
    assert px_y_new == px_y


def test_rescale_grid(sample_grid):
    """Test grid rescaling."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Upscale
    upscaled, xi_new, yi_new, px_x_new, px_y_new = rescale_grid(
        grid, 2.0, xi, yi, px_x, px_y
    )
    
    assert upscaled.shape[0] == grid.shape[0] * 2
    assert upscaled.shape[1] == grid.shape[1] * 2
    # Pixel size should be halved
    assert abs(px_x_new - px_x / 2) < 0.01


def test_crop_to_valid_region(sample_grid):
    """Test cropping to valid region."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Add NaN borders
    grid_with_borders = grid.copy()
    grid_with_borders[0:5, :] = np.nan
    grid_with_borders[-5:, :] = np.nan
    grid_with_borders[:, 0:5] = np.nan
    grid_with_borders[:, -5:] = np.nan
    
    cropped, xi_new, yi_new, px_x_new, px_y_new = crop_to_valid_region(
        grid_with_borders, xi, yi, px_x, px_y, margin=0
    )
    
    assert cropped.shape[0] < grid.shape[0]
    assert cropped.shape[1] < grid.shape[1]


def test_auto_register_surfaces(sample_grid):
    """Test automatic surface registration."""
    grid, xi, yi, px_x, px_y = sample_grid
    
    # Create shifted version
    shifted = np.roll(np.roll(grid, 5, axis=0), 3, axis=1)
    
    params = auto_register_surfaces(grid, shifted, method='correlation')
    
    assert params is not None
    assert 'translation' in params
    assert 'rmse' in params
    # Should detect the shift
    assert abs(params['translation'][0] - 5) <= 2
    assert abs(params['translation'][1] - 3) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
