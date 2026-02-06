"""Utility functions for scan data processing.

This module provides various helper functions for:
- Resource path resolution
- Performance measurement
- Scan alignment (offset and tilt correction)
- Data interpolation and hole filling
- Gaussian filtering with NaN handling
- Outlier detection and removal
"""

import sys
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from functools import wraps
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """Returns absolute path to a resource file.
    
    Works both in development (.py) and PyInstaller executable (.exe) environments.
    
    Args:
        relative_path (str): Relative path to the resource.
        
    Returns:
        str: Absolute path to the resource.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def measure_time(func):
    """Decorator for measuring and logging function execution time.
    
    Args:
        func (callable): Function to measure.
        
    Returns:
        callable: Wrapped function that logs execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f">>> {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper


def compute_offset_global(reference, target):
    """Computes global mean offset between two grids.
    
    Calculates the average difference where both grids have valid data (ignores NaN).
    
    Args:
        reference (np.ndarray): Reference grid.
        target (np.ndarray): Target grid to compare.
        
    Returns:
        float: Mean offset value.
        
    Raises:
        ValueError: If no common valid data exists across the grids.
    """
    mask = ~np.isnan(reference) & ~np.isnan(target)
    diff = reference - target
    masked_diff = diff[mask]

    if masked_diff.size == 0:
        raise ValueError("Lack of common valid data across the grid")

    offset = np.mean(masked_diff)
    return offset


def compute_offset_in_center(reference, target, window_size=100):
    """Computes mean offset in a central window region.
    
    Extracts a central window from both grids and calculates the mean difference.
    
    Args:
        reference (np.ndarray): Reference grid.
        target (np.ndarray): Target grid to compare.
        window_size (int, optional): Size of the central window in pixels. Defaults to 100.
        
    Returns:
        float: Mean offset in the central window.
        
    Raises:
        ValueError: If no valid data exists in the central window.
    """
    # Grid dimensions
    rows, cols = reference.shape
    # Center position
    center_row = rows // 2
    center_col = cols // 2
    half = window_size // 2
    # Extract central window
    ref_central = reference[center_row-half:center_row+half, center_col-half:center_col+half]
    target_central = target[center_row-half:center_row+half, center_col-half:center_col+half]
    # Mask: only where both are not NaN
    mask = ~np.isnan(ref_central) & ~np.isnan(target_central)
    diff = ref_central - target_central
    masked_diff = diff[mask]
    if masked_diff.size == 0:
        raise ValueError("No valid data in the central window")
    offset = np.mean(masked_diff)
    return offset

def remove_relative_offset(reference, target, mask):
    """Removes global offset between reference and target grids.
    
    Computes and removes the mean difference to align the target grid with the reference.
    
    Args:
        reference (np.ndarray): Reference grid.
        target (np.ndarray): Target grid to adjust.
        mask (np.ndarray): Boolean mask indicating valid regions.
        
    Returns:
        np.ndarray: Target grid with offset removed.
    """
    offset = compute_offset_global(reference, target)
    return target + offset

# def remove_relative_offset(reference, target, mask):
#     #mask = ~np.isnan(reference) & ~np.isnan(target)
#     difference = reference - target
#     masked_diff = difference[mask]
#     if masked_diff.size == 0:
#         raise ValueError("Brak ważnych danych do obliczenia offsetu")
#     offset = np.mean(masked_diff)
#     print('Wyznaczony offset:', offset)
#     return target + offset

def remove_relative_tilt(reference, target, mask):
    """Removes relative tilt between reference and target grids using linear regression.
    
    Fits a plane to the difference between grids and adds it to the target to correct tilt.
    
    Args:
        reference (np.ndarray): Reference grid.
        target (np.ndarray): Target grid to adjust.
        mask (np.ndarray): Boolean mask indicating valid regions for regression.
        
    Returns:
        np.ndarray: Target grid with tilt correction applied.
        
    Raises:
        ValueError: If no valid data is available for regression.
    """
    difference = reference - target
    rows, cols = difference.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    XX = X[mask].flatten()
    YY = Y[mask].flatten()
    ZZ = difference[mask].flatten()
    valid_mask = ~np.isnan(ZZ)
    XX, YY, ZZ = XX[valid_mask], YY[valid_mask], ZZ[valid_mask]
    if len(ZZ) == 0:
        raise ValueError("No valid data for regression - all points contained NaN")
    features = np.vstack((XX, YY)).T
    model = LinearRegression().fit(features, ZZ)
    tilt_plane = model.predict(np.vstack((X.flatten(), Y.flatten())).T).reshape(difference.shape)
    return target + tilt_plane


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
    cleaned[mask_outlier] = smoothed_grid[mask_outlier]  # lub +200 jeśli to był tylko test
    return cleaned

