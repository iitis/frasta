"""Tests for frasta.core.grid_data module."""

import pytest
import numpy as np
from frasta.core import Surface


class TestGridData:
    """Test suite for Surface class (formerly GridData)."""
    
    def test_init_creates_grid_data(self, sample_grid):
        """Test Surface initialization with basic parameters."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        assert gd.height.shape == sample_grid.shape
        assert gd.dx == 1.0
        assert gd.dy == 1.0
    
    def test_init_with_coordinates(self, sample_grid):
        """Test Surface initialization with coordinate arrays (generated as properties)."""
        gd = Surface(height=sample_grid, dx=0.5, dy=0.5)
        assert gd.xi is not None
        assert gd.yi is not None
        assert len(gd.xi) == 10
        assert len(gd.yi) == 10
    
    def test_crop_returns_cropped_grid(self, sample_grid):
        """Test that crop method returns a new Surface with cropped dimensions."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        cropped = gd.crop(6, 4)  # Height=6, Width=4
        
        assert cropped.height.shape == (6, 4)
        assert cropped is not gd  # New object
        assert cropped.dx == gd.dx
        assert cropped.dy == gd.dy
    
    def test_crop_with_coordinates(self, sample_grid):
        """Test that crop properly handles coordinate arrays."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        cropped = gd.crop(3, 3)  # 3x3 crop
        
        assert len(cropped.xi) == 3
        assert len(cropped.yi) == 3
        assert np.allclose(cropped.xi, np.arange(3) * 1.0)
        assert np.allclose(cropped.yi, np.arange(3) * 1.0)
    
    def test_copy_creates_independent_copy(self, sample_grid):
        """Test that copy method creates an independent copy."""
        gd = Surface(height=sample_grid.copy(), dx=1.0, dy=1.0)
        copied = gd.copy()
        
        # Modify original
        gd.height[0, 0] = 999
        
        # Copy should be unchanged
        assert copied.height[0, 0] != 999
        assert copied is not gd
        assert copied.dx == gd.dx
    
    def test_crop_boundary_conditions(self, sample_grid):
        """Test crop with boundary values."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        
        # Crop to full grid
        full = gd.crop(10, 10)
        assert full.height.shape == sample_grid.shape
        
        # Crop to single row
        single_row = gd.crop(1, 10)
        assert single_row.height.shape == (1, 10)
    
    def test_grid_preserves_nan_values(self, sample_grid):
        """Test that NaN values are preserved in Surface."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        
        # Count NaN values
        original_nans = np.isnan(sample_grid).sum()
        grid_nans = np.isnan(gd.height).sum()
        
        assert original_nans == grid_nans
        assert grid_nans > 0  # Fixture has NaN values
    
    def test_spatial_origin_default(self, sample_grid):
        """Test that default x0, y0 are zero."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0)
        assert gd.x0 == 0.0
        assert gd.y0 == 0.0
        assert gd.xi[0] == 0.0
        assert gd.yi[0] == 0.0
    
    def test_spatial_origin_custom(self, sample_grid):
        """Test Surface with custom spatial origin."""
        gd = Surface(height=sample_grid, dx=0.5, dy=0.5, x0=10.5, y0=20.0)
        
        assert gd.x0 == 10.5
        assert gd.y0 == 20.0
        assert gd.xi[0] == 10.5
        assert gd.yi[0] == 20.0
        assert np.allclose(gd.xi, 10.5 + np.arange(10) * 0.5)
        assert np.allclose(gd.yi, 20.0 + np.arange(10) * 0.5)
    
    def test_copy_preserves_origin(self, sample_grid):
        """Test that copy preserves x0 and y0."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0, x0=15.0, y0=25.0)
        copied = gd.copy()
        
        assert copied.x0 == 15.0
        assert copied.y0 == 25.0
        assert np.allclose(copied.xi, gd.xi)
        assert np.allclose(copied.yi, gd.yi)
    
    def test_crop_preserves_origin(self, sample_grid):
        """Test that crop preserves spatial origin."""
        gd = Surface(height=sample_grid, dx=1.0, dy=1.0, x0=10.0, y0=20.0)
        cropped = gd.crop(5, 5)
        
        # Origin should be preserved (crop doesn't shift coordinates)
        assert cropped.x0 == 10.0
        assert cropped.y0 == 20.0
        assert cropped.xi[0] == 10.0
        assert cropped.yi[0] == 20.0
