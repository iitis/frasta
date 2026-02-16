"""Scan alignment and offset correction algorithms.

This module provides functions for aligning two scan datasets by removing
relative offset and tilt differences.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

import logging
logger = logging.getLogger(__name__)


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
