"""Interpolation algorithms for filling missing data.

This module provides functions for interpolating NaN holes in scan data using
various methods.
"""

import numpy as np
from scipy.interpolate import griddata

import logging
logger = logging.getLogger(__name__)


def fill_holes(grid, mask=None):
    """Fills NaN holes in a grid using nearest-neighbor interpolation.
    
    Interpolates missing values from surrounding valid data points.
    
    Args:
        grid (np.ndarray or None): 2D array with NaN holes to fill.
        mask (np.ndarray, optional): Boolean mask indicating where to fill holes.
            If None, fills all NaN values. Defaults to None.
            
    Returns:
        np.ndarray or None: Grid with holes filled, or None if input is None.
    """
    if grid is None:
        return None

    grid = grid.copy()
    tst = np.isnan(grid)

    if mask is not None:
        # Interpolate only where mask is True
        tst = tst & mask

    if not np.any(tst):
        return grid  # Nothing to fill

    grid_x, grid_y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))

    interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

    valid = ~np.isnan(grid)
    interp_values = griddata(
        (grid_x[valid], grid_y[valid]),
        grid[valid],
        interp_points,
        method='nearest'
    )

    grid[tst] = interp_values
    return grid
