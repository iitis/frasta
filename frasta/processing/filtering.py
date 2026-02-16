"""Filtering algorithms for scan data processing.

This module provides functions for smoothing and outlier removal that properly
handle NaN values in scan data.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)


def nan_aware_gaussian(grid, sigma, mask=None):
    """Applies Gaussian smoothing while ignoring NaN values.
    
    Performs weighted Gaussian filtering that properly handles missing data.
    Optionally restricts filtering to a masked region.
    
    Args:
        grid (np.ndarray or None): 2D array to smooth.
        sigma (float): Standard deviation for Gaussian kernel.
        mask (np.ndarray, optional): Boolean mask indicating region to smooth.
            If None, smooths entire grid. Defaults to None.
            
    Returns:
        np.ndarray or None: Smoothed grid, or None if input is None.
    """
    if grid is None:
        return None

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, 0, grid)
    weights = (~nan_mask).astype(float)

    # jeśli maska istnieje – przytnij do maski
    if mask is not None:
        filled = np.where(mask, filled, 0.0)
        weights = np.where(mask, weights, 0.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    weight_sum = gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed / weight_sum
        result[weight_sum == 0] = np.nan

    return result


def remove_outliers(original_grid, smoothed_grid, threshold, mask=None):
    """Replaces outliers with smoothed values if difference exceeds threshold.
    
    Identifies outlier points where the difference between original and smoothed
    grids exceeds the threshold, and replaces them with smoothed values.
    
    Args:
        original_grid (np.ndarray): Original grid with potential outliers.
        smoothed_grid (np.ndarray): Smoothed reference grid.
        threshold (float): Difference threshold for outlier detection.
        mask (np.ndarray, optional): Boolean mask restricting outlier removal.
            If None, processes entire grid. Defaults to None.
            
    Returns:
        np.ndarray: Grid with outliers replaced by smoothed values.
    """
    diff = np.abs(original_grid - smoothed_grid)
    mask_outlier = diff > threshold

    if mask is not None:
        mask_outlier = mask_outlier & mask

    cleaned = original_grid.copy()
    cleaned[mask_outlier] = smoothed_grid[mask_outlier]
    return cleaned
