"""Tests for transforms.py module - geometric transformations and registration.

This test suite covers rotation, scaling, cropping, and surface registration
algorithms including ICP and correlation-based methods.
"""

import pytest
import numpy as np
from frasta.processing.transforms import (
    rotate_grid,
    rescale_grid,
    crop_to_valid_region,
    auto_register_surfaces,
    apply_registration,
    _register_correlation,
    _register_icp
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_grid():
    """Create a simple grid with coordinate arrays."""
    ny, nx = 50, 60
    grid = np.random.randn(ny, nx) * 10.0 + 100.0
    
    xi = np.arange(nx) * 1.0
    yi = np.arange(ny) * 1.0
    dx = 1.0
    dy = 1.0
    
    return grid, xi, yi, dx, dy


@pytest.fixture
def grid_with_pattern():
    """Create a grid with a distinctive pattern for testing registration."""
    ny, nx = 100, 100
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    # Create a pattern with some features
    grid = 50.0 + 10.0 * np.sin(x_idx * 0.2) + 5.0 * np.cos(y_idx * 0.15)
    
    xi = np.arange(nx) * 2.0
    yi = np.arange(ny) * 2.0
    dx = 2.0
    dy = 2.0
    
    return grid, xi, yi, dx, dy


@pytest.fixture
def grid_with_border_nans():
    """Create a grid with NaN borders."""
    ny, nx = 80, 90
    grid = np.ones((ny, nx)) * 50.0
    
    # Add NaN borders
    grid[0:10, :] = np.nan
    grid[-10:, :] = np.nan
    grid[:, 0:15] = np.nan
    grid[:, -15:] = np.nan
    
    xi = np.arange(nx) * 1.0
    yi = np.arange(ny) * 1.0
    dx = 1.0
    dy = 1.0
    
    return grid, xi, yi, dx, dy


# ============================================================================
# Tests for rotate_grid
# ============================================================================

class TestRotateGrid:
    """Tests for rotate_grid function."""
    
    def test_rotate_0_degrees_unchanged(self, simple_grid):
        """Test that 0 degree rotation leaves grid unchanged."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, xi_new, yi_new, dx_new, dy_new = rotate_grid(
            grid, 0.0, xi, yi, dx, dy
        )
        
        # Should be essentially unchanged
        mask = ~np.isnan(grid) & ~np.isnan(rotated)
        assert np.allclose(rotated[mask], grid[mask], rtol=1e-5)
        assert dx_new == dx
        assert dy_new == dy
    
    def test_rotate_90_degrees(self, simple_grid):
        """Test 90 degree rotation."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, xi_new, yi_new, dx_new, dy_new = rotate_grid(
            grid, 90.0, xi, yi, dx, dy
        )
        
        # Should have same shape
        assert rotated.shape == grid.shape
        # Pixel sizes unchanged
        assert dx_new == dx
        assert dy_new == dy
    
    def test_rotate_180_degrees(self, simple_grid):
        """Test 180 degree rotation."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, xi_new, yi_new, dx_new, dy_new = rotate_grid(
            grid, 180.0, xi, yi, dx, dy
        )
        
        # Corner should be approximately flipped
        # (with interpolation, won't be exact)
        assert rotated.shape == grid.shape
    
    def test_rotate_360_degrees(self, simple_grid):
        """Test 360 degree rotation returns to original."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, xi_new, yi_new, dx_new, dy_new = rotate_grid(
            grid, 360.0, xi, yi, dx, dy
        )
        
        # Should be close to original (with some interpolation error)
        mask = ~np.isnan(grid) & ~np.isnan(rotated)
        assert np.allclose(rotated[mask], grid[mask], rtol=0.01)
    
    def test_rotate_negative_angle(self, simple_grid):
        """Test rotation with negative angle."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, xi_new, yi_new, dx_new, dy_new = rotate_grid(
            grid, -45.0, xi, yi, dx, dy
        )
        
        # Should work without error
        assert rotated.shape == grid.shape
        assert np.isfinite(rotated).any()
    
    def test_rotate_with_interpolation_orders(self, simple_grid):
        """Test different interpolation orders."""
        grid, xi, yi, dx, dy = simple_grid
        
        for order in [0, 1, 3]:
            rotated, _, _, _, _ = rotate_grid(
                grid, 45.0, xi, yi, dx, dy, order=order
            )
            assert rotated.shape == grid.shape
    
    def test_rotate_creates_edge_nans(self, simple_grid):
        """Test that rotation creates NaN at edges (corners)."""
        grid, xi, yi, dx, dy = simple_grid
        
        rotated, _, _, _, _ = rotate_grid(grid, 45.0, xi, yi, dx, dy)
        
        # Corners should have more NaN after rotation
        nan_before = np.isnan(grid).sum()
        nan_after = np.isnan(rotated).sum()
        
        assert nan_after >= nan_before


# ============================================================================
# Tests for rescale_grid
# ============================================================================

class TestRescaleGrid:
    """Tests for rescale_grid function."""
    
    def test_upscale_doubles_resolution(self, simple_grid):
        """Test upscaling by factor 2."""
        grid, xi, yi, dx, dy = simple_grid
        
        rescaled, xi_new, yi_new, dx_new, dy_new = rescale_grid(
            grid, 2.0, xi, yi, dx, dy
        )
        
        # Should double resolution
        assert rescaled.shape[0] == grid.shape[0] * 2
        assert rescaled.shape[1] == grid.shape[1] * 2
        
        # Pixel size should halve
        assert abs(dx_new - dx / 2) < 1e-10
        assert abs(dy_new - dy / 2) < 1e-10
    
    def test_downscale_halves_resolution(self, simple_grid):
        """Test downscaling by factor 0.5."""
        grid, xi, yi, dx, dy = simple_grid
        
        rescaled, xi_new, yi_new, dx_new, dy_new = rescale_grid(
            grid, 0.5, xi, yi, dx, dy
        )
        
        # Should halve resolution
        assert rescaled.shape[0] == grid.shape[0] // 2
        assert rescaled.shape[1] == grid.shape[1] // 2
        
        # Pixel size should double
        assert abs(dx_new - dx * 2) < 1e-10
        assert abs(dy_new - dy * 2) < 1e-10
    
    def test_scale_factor_1_preserves_size(self, simple_grid):
        """Test that scale factor 1.0 preserves size."""
        grid, xi, yi, dx, dy = simple_grid
        
        rescaled, xi_new, yi_new, dx_new, dy_new = rescale_grid(
            grid, 1.0, xi, yi, dx, dy
        )
        
        assert rescaled.shape == grid.shape
        assert dx_new == dx
        assert dy_new == dy
    
    def test_rescale_preserves_coordinate_range(self, simple_grid):
        """Test that coordinate range is preserved after rescaling."""
        grid, xi, yi, dx, dy = simple_grid
        
        rescaled, xi_new, yi_new, dx_new, dy_new = rescale_grid(
            grid, 2.0, xi, yi, dx, dy
        )
        
        # Coordinate range should be same
        assert abs(xi_new[0] - xi[0]) < 1e-10
        assert abs(xi_new[-1] - xi[-1]) < 1e-10
        assert abs(yi_new[0] - yi[0]) < 1e-10
        assert abs(yi_new[-1] - yi[-1]) < 1e-10
    
    def test_rescale_with_different_orders(self, simple_grid):
        """Test different interpolation orders."""
        grid, xi, yi, dx, dy = simple_grid
        
        for order in [0, 1, 3]:
            rescaled, _, _, _, _ = rescale_grid(
                grid, 1.5, xi, yi, dx, dy, order=order
            )
            assert rescaled.shape[0] == int(grid.shape[0] * 1.5)
            assert rescaled.shape[1] == int(grid.shape[1] * 1.5)
    
    def test_rescale_small_factor(self, simple_grid):
        """Test with very small scale factor."""
        grid, xi, yi, dx, dy = simple_grid
        
        rescaled, xi_new, yi_new, dx_new, dy_new = rescale_grid(
            grid, 0.2, xi, yi, dx, dy
        )
        
        # Should be much smaller
        assert rescaled.shape[0] < grid.shape[0] / 4
        assert rescaled.shape[1] < grid.shape[1] / 4


# ============================================================================
# Tests for crop_to_valid_region
# ============================================================================

class TestCropToValidRegion:
    """Tests for crop_to_valid_region function."""
    
    def test_crops_nan_borders(self, grid_with_border_nans):
        """Test that NaN borders are cropped."""
        grid, xi, yi, dx, dy = grid_with_border_nans
        
        cropped, xi_new, yi_new, dx_new, dy_new = crop_to_valid_region(
            grid, xi, yi, dx, dy, margin=0
        )
        
        # Should be smaller
        assert cropped.shape[0] < grid.shape[0]
        assert cropped.shape[1] < grid.shape[1]
        
        # Pixel sizes unchanged
        assert dx_new == dx
        assert dy_new == dy
    
    def test_crop_with_margin(self, grid_with_border_nans):
        """Test cropping with margin."""
        grid, xi, yi, dx, dy = grid_with_border_nans
        
        cropped_no_margin, _, _, _, _ = crop_to_valid_region(
            grid, xi, yi, dx, dy, margin=0
        )
        
        cropped_with_margin, _, _, _, _ = crop_to_valid_region(
            grid, xi, yi, dx, dy, margin=5
        )
        
        # With margin should be larger
        assert cropped_with_margin.shape[0] >= cropped_no_margin.shape[0]
        assert cropped_with_margin.shape[1] >= cropped_no_margin.shape[1]
    
    def test_all_valid_no_crop(self, simple_grid):
        """Test that grid with no NaN is not cropped."""
        grid, xi, yi, dx, dy = simple_grid
        
        cropped, xi_new, yi_new, dx_new, dy_new = crop_to_valid_region(
            grid, xi, yi, dx, dy
        )
        
        # Should be same size
        assert cropped.shape == grid.shape
    
    def test_all_nan_returns_unchanged(self):
        """Test that all-NaN grid is returned unchanged."""
        grid = np.full((50, 60), np.nan)
        xi = np.arange(60) * 1.0
        yi = np.arange(50) * 1.0
        dx = dy = 1.0
        
        cropped, xi_new, yi_new, dx_new, dy_new = crop_to_valid_region(
            grid, xi, yi, dx, dy
        )
        
        # Should be unchanged
        assert cropped.shape == grid.shape
    
    def test_single_valid_pixel(self):
        """Test with only one valid pixel."""
        grid = np.full((50, 60), np.nan)
        grid[25, 30] = 100.0
        
        xi = np.arange(60) * 1.0
        yi = np.arange(50) * 1.0
        dx = dy = 1.0
        
        cropped, xi_new, yi_new, dx_new, dy_new = crop_to_valid_region(
            grid, xi, yi, dx, dy, margin=0
        )
        
        # Should be cropped to 1x1
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == 100.0
    
    def test_coordinates_updated_correctly(self, grid_with_border_nans):
        """Test that coordinate arrays are updated correctly."""
        grid, xi, yi, dx, dy = grid_with_border_nans
        
        cropped, xi_new, yi_new, dx_new, dy_new = crop_to_valid_region(
            grid, xi, yi, dx, dy
        )
        
        # New coordinates should match new shape
        assert len(xi_new) == cropped.shape[1]
        assert len(yi_new) == cropped.shape[0]


# ============================================================================
# Tests for auto_register_surfaces
# ============================================================================

class TestAutoRegisterSurfaces:
    """Tests for auto_register_surfaces function."""
    
    def test_correlation_method(self, grid_with_pattern):
        """Test registration using correlation method."""
        grid, xi, yi, dx, dy = grid_with_pattern
        
        # Create shifted version
        shifted = np.roll(np.roll(grid, 5, axis=0), 3, axis=1)
        
        params = auto_register_surfaces(grid, shifted, method='correlation')
        
        # Should be close to the corrective shift needed for alignment
        assert abs(params['translation'][0] + 5) < 2
        assert abs(params['translation'][1] + 3) < 2
        assert params['rotation'] == 0.0  # Correlation doesn't estimate rotation
    
    def test_icp_method(self, grid_with_pattern):
        """Test registration using ICP method."""
        grid, xi, yi, dx, dy = grid_with_pattern
        
        # Create shifted version
        shifted = np.roll(np.roll(grid, 5, axis=0), -3, axis=1)
        
        params = auto_register_surfaces(grid, shifted, method='icp')
        
        # Should detect translation (approximately)
        assert 'translation' in params
        assert 'rotation' in params
        assert 'rmse' in params
        assert params['rmse'] < np.inf
    
    def test_invalid_method_raises_error(self, simple_grid):
        """Test that invalid method raises ValueError."""
        grid, xi, yi, dx, dy = simple_grid
        
        with pytest.raises(ValueError, match="Unknown registration method"):
            auto_register_surfaces(grid, grid, method='invalid')
    
    def test_registration_same_surface(self, grid_with_pattern):
        """Test registration of surface with itself."""
        grid, xi, yi, dx, dy = grid_with_pattern
        
        params = auto_register_surfaces(grid, grid, method='correlation')
        
        # Translation should be close to zero
        assert abs(params['translation'][0]) < 1.0
        assert abs(params['translation'][1]) < 1.0


class TestRegisterCorrelation:
    """Tests for _register_correlation function."""
    
    def test_detects_horizontal_shift(self):
        """Test detection of horizontal shift."""
        ny, nx = 100, 100
        grid = np.random.randn(ny, nx) * 5.0 + 50.0
        
        # Shift by 10 pixels in x
        shifted = np.roll(grid, 10, axis=1)
        # Zero out wrapped region
        shifted[:, :10] = np.nan
        
        params = _register_correlation(grid, shifted)
        
        # Should detect the corrective shift (approximately)
        assert abs(params['translation'][1] + 10) < 2
    
    def test_detects_vertical_shift(self):
        """Test detection of vertical shift."""
        ny, nx = 100, 100
        grid = np.random.randn(ny, nx) * 5.0 + 50.0
        
        # Shift by 8 pixels in y
        shifted = np.roll(grid, 8, axis=0)
        shifted[:8, :] = np.nan
        
        params = _register_correlation(grid, shifted)
        
        # Should detect the corrective shift
        assert abs(params['translation'][0] + 8) < 2
    
    def test_different_shapes_raises_error(self):
        """Test that different shapes raise ValueError."""
        grid1 = np.ones((50, 60))
        grid2 = np.ones((40, 60))
        
        with pytest.raises(ValueError, match="same-sized arrays"):
            _register_correlation(grid1, grid2)
    
    def test_handles_nan_regions(self):
        """Test handling of NaN regions."""
        grid = np.ones((80, 80)) * 100.0
        grid[10:20, 10:20] = np.nan
        
        shifted = np.roll(grid, 5, axis=0)
        shifted[:5, :] = np.nan
        
        # Should complete without error
        params = _register_correlation(grid, shifted)
        
        assert 'translation' in params
        assert np.isfinite(params['rmse'])

    def test_subpixel_negative_shift_keeps_finite_rmse(self):
        """Small negative subpixel shifts should not erase the overlap mask."""
        from scipy.ndimage import shift as ndimage_shift

        grid = np.random.randn(64, 64) * 3.0 + 25.0
        shifted = ndimage_shift(grid, (-0.4, -0.6), order=3, mode='constant', cval=np.nan)

        params = _register_correlation(grid, shifted)

        assert np.isfinite(params['rmse'])


class TestRegisterICP:
    """Tests for _register_icp function."""
    
    def test_basic_icp(self):
        """Test basic ICP registration."""
        ny, nx = 80, 80
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        
        # Create a surface with features
        grid = 50.0 + 5.0 * np.sin(x_idx * 0.1) + 3.0 * np.cos(y_idx * 0.15)
        
        # Shift it
        shifted = np.roll(np.roll(grid, 8, axis=0), -6, axis=1)
        
        params = _register_icp(grid, shifted)
        
        # Should estimate some translation
        assert 'translation' in params
        assert 'rotation' in params
        assert np.isfinite(params['rmse'])
    
    def test_insufficient_points_returns_default(self):
        """Test that insufficient points returns default params."""
        grid1 = np.full((20, 20), np.nan)
        grid1[10:12, 10:12] = 100.0  # Only 4 points
        
        grid2 = grid1.copy()
        
        params = _register_icp(grid1, grid2)
        
        # Should return default values
        assert params['translation'] == (0, 0)
        assert params['rotation'] == 0.0
        assert params['rmse'] == np.inf
    
    def test_subsampling_large_arrays(self):
        """Test that large arrays are subsampled."""
        # Create large grid
        grid = np.random.randn(200, 200) * 10.0 + 100.0
        shifted = np.roll(grid, 5, axis=0)
        
        # Should complete in reasonable time even with many points
        params = _register_icp(grid, shifted)
        
        assert params is not None
        assert 'rmse' in params


# ============================================================================
# Tests for apply_registration
# ============================================================================

class TestApplyRegistration:
    """Tests for apply_registration function."""
    
    def test_apply_no_transformation(self, simple_grid):
        """Test with zero translation and rotation."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(0.0, 0.0), rotation=0.0
        )
        
        # Should be essentially unchanged
        mask = ~np.isnan(grid) & ~np.isnan(transformed)
        assert np.allclose(transformed[mask], grid[mask], rtol=1e-5)
    
    def test_apply_translation_only(self, simple_grid):
        """Test applying translation only."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(5.0, 3.0), rotation=0.0
        )
        
        # Should have shifted data
        assert transformed.shape == grid.shape
        # Some edge regions should be NaN
        assert np.isnan(transformed[0:5, :]).all()  # Top shifted out
        assert np.isnan(transformed[:, 0:3]).all()  # Left shifted out
    
    def test_apply_negative_translation(self, simple_grid):
        """Test applying negative translation."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(-4.0, -6.0), rotation=0.0
        )
        
        # Edge regions should be NaN (opposite edges)
        assert np.isnan(transformed[-4:, :]).any()
        assert np.isnan(transformed[:, -6:]).any()
    
    def test_apply_rotation_only(self, simple_grid):
        """Test applying rotation only."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(0.0, 0.0), rotation=30.0
        )
        
        # Should have rotated (with edge NaN)
        assert transformed.shape == grid.shape
    
    def test_apply_rotation_and_translation(self, simple_grid):
        """Test applying both rotation and translation."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(3.0, 2.0), rotation=15.0
        )
        
        # Should complete successfully
        assert transformed.shape == grid.shape
        assert np.isfinite(transformed).any()
    
    def test_apply_preserves_nan_structure(self):
        """Test that NaN regions are preserved and extended properly."""
        grid = np.ones((50, 50)) * 100.0
        grid[10:15, 10:15] = np.nan  # Internal NaN region
        
        xi = np.arange(50) * 1.0
        yi = np.arange(50) * 1.0
        dx = dy = 1.0
        
        transformed, _, _, _, _ = apply_registration(
            grid, xi, yi, dx, dy, translation=(5.0, 5.0), rotation=0.0
        )
        
        # Original NaN should be shifted
        assert np.isnan(transformed[15:20, 15:20]).any()
        # New edge NaN from shift
        assert np.isnan(transformed[0:5, :]).all()
    
    def test_apply_with_small_translation(self, simple_grid):
        """Test with very small translation (should be ignored)."""
        grid, xi, yi, dx, dy = simple_grid
        
        transformed, xi_new, yi_new, dx_new, dy_new = apply_registration(
            grid, xi, yi, dx, dy, translation=(0.1, 0.2), rotation=0.0
        )
        
        # Should be nearly unchanged (below threshold)
        mask = ~np.isnan(grid) & ~np.isnan(transformed)
        assert np.allclose(transformed[mask], grid[mask], rtol=1e-3)
    
    def test_apply_handles_all_nan_grid(self):
        """Test with all-NaN grid."""
        grid = np.full((40, 40), np.nan)
        xi = np.arange(40) * 1.0
        yi = np.arange(40) * 1.0
        dx = dy = 1.0
        
        transformed, _, _, _, _ = apply_registration(
            grid, xi, yi, dx, dy, translation=(5.0, 5.0), rotation=0.0
        )
        
        # Should still be all NaN
        assert np.isnan(transformed).all()


# ============================================================================
# Integration tests
# ============================================================================

class TestTransformsIntegration:
    """Integration tests combining multiple transformations."""
    
    def test_rotate_then_rescale(self, simple_grid):
        """Test combining rotation and rescaling."""
        grid, xi, yi, dx, dy = simple_grid
        
        # Rotate first
        rotated, xi, yi, dx, dy = rotate_grid(grid, 30, xi, yi, dx, dy)
        
        # Then rescale
        rescaled, xi, yi, dx, dy = rescale_grid(rotated, 1.5, xi, yi, dx, dy)
        
        # Should complete successfully
        assert rescaled.shape[0] > grid.shape[0]
        assert rescaled.shape[1] > grid.shape[1]
    
    def test_crop_then_register(self, grid_with_pattern):
        """Test cropping then registering."""
        grid, xi, yi, dx, dy = grid_with_pattern
        
        # Add borders and crop
        grid_with_borders = np.copy(grid)
        grid_with_borders[0:10, :] = np.nan
        grid_with_borders[-10:, :] = np.nan
        grid_with_borders[:, 0:10] = np.nan
        grid_with_borders[:, -10:] = np.nan
        
        # Crop first
        cropped, xi_crop, yi_crop, dx, dy = crop_to_valid_region(
            grid_with_borders, xi, yi, dx, dy
        )
        
        # Create shifted version using scipy shift (more reliable than roll)
        from scipy.ndimage import shift as ndimage_shift
        shifted = ndimage_shift(cropped, (0, 3), order=3, mode='constant', cval=np.nan)
        shifted[:, :3] = np.nan
        
        # Register (should work on cropped data)
        params = auto_register_surfaces(cropped, shifted, method='correlation')
        
        # Should detect the corrective shift (approximately)
        assert abs(params['translation'][1] + 3) < 3  # More relaxed tolerance
    
    def test_register_then_apply(self, grid_with_pattern):
        """Test finding registration then applying it."""
        grid, xi, yi, dx, dy = grid_with_pattern
        
        # Create shifted version
        target = np.roll(np.roll(grid, 6, axis=0), -4, axis=1)
        target[:6, :] = np.nan
        target[:, -4:] = np.nan
        
        # Find registration
        params = auto_register_surfaces(grid, target, method='correlation')
        
        # Apply registration to target
        aligned, _, _, _, _ = apply_registration(
            target, xi, yi, dx, dy, 
            translation=params['translation'],
            rotation=params.get('rotation', 0.0)
        )
        
        # Should be more similar to reference after alignment.
        mask_before = np.isfinite(grid) & np.isfinite(target)
        mask_after = np.isfinite(grid) & np.isfinite(aligned)
        rmse_before = np.sqrt(np.mean((grid[mask_before] - target[mask_before]) ** 2))
        rmse_after = np.sqrt(np.mean((grid[mask_after] - aligned[mask_after]) ** 2))
        assert aligned.shape == grid.shape
        assert rmse_after < rmse_before * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
