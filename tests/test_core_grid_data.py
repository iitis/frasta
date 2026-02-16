"""Tests for frasta.core.grid_data module."""

import pytest
import numpy as np
from frasta.core import GridData


class TestGridData:
    """Test suite for GridData class."""
    
    def test_init_creates_grid_data(self, sample_grid):
        """Test GridData initialization with basic parameters."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        assert gd.grid.shape == sample_grid.shape
        assert gd.px_x == 1.0
        assert gd.px_y == 1.0
    
    def test_init_with_coordinates(self, sample_grid):
        """Test GridData initialization with coordinate arrays."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=0.5, px_y=0.5)
        assert gd.xi is not None
        assert gd.yi is not None
        assert len(gd.xi) == 10
        assert len(gd.yi) == 10
    
    def test_crop_returns_cropped_grid(self, sample_grid):
        """Test that crop method returns a new GridData with cropped dimensions."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        cropped = gd.crop(6, 4)  # Height=6, Width=4
        
        assert cropped.grid.shape == (6, 4)
        assert cropped is not gd  # New object
        assert cropped.px_x ==gd.px_x
        assert cropped.px_y == gd.px_y
    
    def test_crop_with_coordinates(self, sample_grid):
        """Test that crop properly handles coordinate arrays."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        cropped = gd.crop(3, 3)  # 3x3 crop
        
        assert len(cropped.xi) == 3
        assert len(cropped.yi) == 3
        assert np.allclose(cropped.xi, xi[:3])
        assert np.allclose(cropped.yi, yi[:3])
    
    def test_copy_creates_independent_copy(self, sample_grid):
        """Test that copy method creates an independent copy."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid.copy(), xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        copied = gd.copy()
        
        # Modify original
        gd.grid[0, 0] = 999
        
        # Copy should be unchanged
        assert copied.grid[0, 0] != 999
        assert copied is not gd
        assert copied.px_x == gd.px_x
    
    def test_crop_boundary_conditions(self, sample_grid):
        """Test crop with boundary values."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        
        # Crop to full grid
        full = gd.crop(10, 10)
        assert full.grid.shape == sample_grid.shape
        
        # Crop to single row
        single_row = gd.crop(1, 10)
        assert single_row.grid.shape == (1, 10)
    
    def test_grid_preserves_nan_values(self, sample_grid):
        """Test that NaN values are preserved in GridData."""
        xi = np.linspace(0, 9, 10)
        yi = np.linspace(0, 9, 10)
        gd = GridData(sample_grid, xi=xi, yi=yi, px_x=1.0, px_y=1.0)
        
        # Count NaN values
        original_nans = np.isnan(sample_grid).sum()
        grid_nans = np.isnan(gd.grid).sum()
        
        assert original_nans == grid_nans
        assert grid_nans > 0  # Fixture has NaN values
