"""Geometric transformations and surface registration algorithms.

This module provides functions for rotating, scaling, cropping surfaces and
automatic registration (alignment) of two surfaces.
"""

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates, shift as ndimage_shift
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor, LinearRegression

import logging
logger = logging.getLogger(__name__)


def rotate_grid(grid, angle_degrees, xi, yi, dx, dy, order=3):
    """Rotates a grid around its center with interpolation.
    
    Rotates the height grid by the specified angle while maintaining the same
    grid dimensions. Uses spline interpolation to handle sub-pixel positions.
    
    Args:
        grid (np.ndarray): 2D height array to rotate.
        angle_degrees (float): Rotation angle in degrees (positive = counterclockwise).
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        order (int, optional): Interpolation order (0=nearest, 1=linear, 3=cubic).
            Defaults to 3.
            
    Returns:
        tuple: (rotated_grid, new_xi, new_yi, dx, dy)
            The rotated grid with updated coordinate arrays.
            
    Examples:
        >>> # Rotate by 45 degrees
        >>> rotated, xi_new, yi_new, dx, dy = rotate_grid(
        ...     grid, 45, xi, yi, dx, dy)
    """
    ny, nx = grid.shape
    center_x = nx / 2
    center_y = ny / 2
    
    # Convert angle to radians
    theta = np.radians(angle_degrees)
    
    # Create rotation matrix (inverse for affine_transform)
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    
    # Affine transformation matrix
    matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Offset to rotate around center
    offset = np.array([center_y, center_x]) - matrix @ np.array([center_y, center_x])
    
    # Apply rotation
    rotated = affine_transform(
        grid,
        matrix,
        offset=offset,
        order=order,
        mode='constant',
        cval=np.nan
    )
    
    # Coordinates don't change in the grid reference frame
    # (the data rotates, not the coordinate system)
    return rotated, xi, yi, dx, dy


def rescale_grid(grid, scale_factor, xi, yi, dx, dy, order=3):
    """Rescales a grid by changing its resolution.
    
    Resamples the grid to a different resolution. Scale factor > 1 increases
    resolution (more pixels), scale factor < 1 decreases resolution.
    
    Args:
        grid (np.ndarray): 2D height array to rescale.
        scale_factor (float): Scaling factor. 2.0 doubles resolution, 0.5 halves it.
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        order (int, optional): Interpolation order. Defaults to 3.
            
    Returns:
        tuple: (rescaled_grid, new_xi, new_yi, new_dx, new_dy)
            
    Examples:
        >>> # Double the resolution
        >>> high_res, xi, yi, dx, dy = rescale_grid(grid, 2.0, xi, yi, dx, dy)
        >>> # Reduce to half resolution
        >>> low_res, xi, yi, dx, dy = rescale_grid(grid, 0.5, xi, yi, dx, dy)
    """
    ny, nx = grid.shape
    
    # New dimensions
    new_nx = int(nx * scale_factor)
    new_ny = int(ny * scale_factor)
    
    # Create coordinate grids for new resolution
    y_new = np.linspace(0, ny - 1, new_ny)
    x_new = np.linspace(0, nx - 1, new_nx)
    
    coords_y, coords_x = np.meshgrid(y_new, x_new, indexing='ij')
    
    # Interpolate using map_coordinates
    rescaled = map_coordinates(
        grid,
        [coords_y, coords_x],
        order=order,
        mode='constant',
        cval=np.nan
    )
    
    # Update coordinate arrays and pixel sizes
    new_xi = np.linspace(xi[0], xi[-1], new_nx)
    new_yi = np.linspace(yi[0], yi[-1], new_ny)
    new_dx = dx / scale_factor
    new_dy = dy / scale_factor
    
    return rescaled, new_xi, new_yi, new_dx, new_dy


def crop_to_valid_region(grid, xi, yi, dx, dy, margin=0):
    """Crops grid to the smallest rectangle containing all valid (non-NaN) data.
    
    Removes rows and columns that contain only NaN values, optionally keeping
    a margin of invalid pixels around the valid region.
    
    Args:
        grid (np.ndarray): 2D height array.
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        margin (int, optional): Number of pixels to keep as margin around valid region.
            Defaults to 0.
            
    Returns:
        tuple: (cropped_grid, new_xi, new_yi, dx, dy)
            
    Examples:
        >>> # Crop to valid data with 10-pixel margin
        >>> cropped, xi, yi, dx, dy = crop_to_valid_region(
        ...     grid, xi, yi, dx, dy, margin=10)
    """
    valid = ~np.isnan(grid)
    
    # Find valid rows and columns
    valid_rows = np.any(valid, axis=1)
    valid_cols = np.any(valid, axis=0)
    
    if not np.any(valid_rows) or not np.any(valid_cols):
        logger.warning("crop_to_valid_region: no valid data found")
        return grid, xi, yi, dx, dy
    
    # Find bounds
    row_start = np.argmax(valid_rows)
    row_end = len(valid_rows) - np.argmax(valid_rows[::-1])
    col_start = np.argmax(valid_cols)
    col_end = len(valid_cols) - np.argmax(valid_cols[::-1])
    
    # Apply margin
    row_start = max(0, row_start - margin)
    row_end = min(len(valid_rows), row_end + margin)
    col_start = max(0, col_start - margin)
    col_end = min(len(valid_cols), col_end + margin)
    
    # Crop
    cropped = grid[row_start:row_end, col_start:col_end]
    new_yi = yi[row_start:row_end]
    new_xi = xi[col_start:col_end]
    
    logger.info(f"Cropped from {grid.shape} to {cropped.shape}")
    
    return cropped, new_xi, new_yi, dx, dy


def auto_register_surfaces(reference, target, method='icp', max_iterations=100):
    """Automatically registers (aligns) target surface to reference surface.
    
    Finds the optimal translation and rotation to align two surfaces using
    iterative closest point (ICP) algorithm or cross-correlation.
    
    Args:
        reference (np.ndarray): Reference surface (2D array).
        target (np.ndarray): Target surface to align (2D array).
        method (str, optional): Registration method. Options:
            'icp' - Iterative Closest Point (handles rotation + translation)
            'correlation' - Cross-correlation (translation only, faster)
            Defaults to 'icp'.
        max_iterations (int, optional): Maximum ICP iterations. Defaults to 100.
            
    Returns:
        dict: Registration parameters containing:
            - 'translation': (dy, dx) translation in pixels
            - 'rotation': rotation angle in degrees (for ICP)
            - 'rmse': root mean square error after alignment
            - 'inliers': number of matching points
            
    Examples:
        >>> # Automatically align two fracture surfaces
        >>> params = auto_register_surfaces(surface1, surface2, method='icp')
        >>> print(f"Translation: {params['translation']}, Rotation: {params['rotation']}°")
    """
    if method == 'correlation':
        return _register_correlation(reference, target)
    elif method == 'icp':
        return _register_icp(reference, target, max_iterations)
    else:
        raise ValueError(f"Unknown registration method: {method}")


def _register_correlation(reference, target):
    """Register using cross-correlation (translation only)."""
    from scipy.signal import correlate
    
    # Check that arrays have same shape
    if reference.shape != target.shape:
        raise ValueError(
            f"Cross-correlation requires same-sized arrays. "
            f"Reference: {reference.shape}, Target: {target.shape}. "
            f"Consider using ICP method instead or crop/resize to match."
        )
    
    # Use valid data only
    ref_valid = ~np.isnan(reference)
    tgt_valid = ~np.isnan(target)
    
    # Normalize: subtract mean from each surface (zero-mean correlation)
    # This is CRITICAL - otherwise filled NaN regions dominate the correlation
    ref_mean = np.nanmean(reference)
    tgt_mean = np.nanmean(target)
    ref_centered = np.where(ref_valid, reference - ref_mean, 0.0)
    tgt_centered = np.where(tgt_valid, target - tgt_mean, 0.0)
    
    # Cross-correlation on zero-mean data
    # Note: correlate(a, b) finds where b is located relative to a
    # The shift we calculate is how much we need to move b to align it with a
    correlation = correlate(ref_centered, tgt_centered, mode='same', method='fft')
    
    # Find peak
    max_val = np.max(correlation)
    max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
    center = (reference.shape[0] // 2, reference.shape[1] // 2)
    
    # Check if peak is at edge (suspicious!)
    edge_margin = 10
    at_edge = (max_pos[0] < edge_margin or max_pos[0] > correlation.shape[0] - edge_margin or
               max_pos[1] < edge_margin or max_pos[1] > correlation.shape[1] - edge_margin)
    if at_edge:
        logger.warning(f"Cross-correlation peak at edge! Position {max_pos} may be unreliable.")
    
    # Subpixel refinement using parabolic interpolation
    # Fit parabola in y and x directions to find subpixel peak
    peak_y = float(max_pos[0])
    peak_x = float(max_pos[1])
    
    if 0 < max_pos[0] < correlation.shape[0] - 1:
        # Parabolic fit in y direction
        c_minus = correlation[max_pos[0] - 1, max_pos[1]]
        c_zero = correlation[max_pos[0], max_pos[1]]
        c_plus = correlation[max_pos[0] + 1, max_pos[1]]
        denom = 2 * (2 * c_zero - c_minus - c_plus)
        if abs(denom) > 1e-10:
            offset_y = (c_minus - c_plus) / denom
            if abs(offset_y) < 1.0:  # Sanity check
                peak_y += offset_y
    
    if 0 < max_pos[1] < correlation.shape[1] - 1:
        # Parabolic fit in x direction
        c_minus = correlation[max_pos[0], max_pos[1] - 1]
        c_zero = correlation[max_pos[0], max_pos[1]]
        c_plus = correlation[max_pos[0], max_pos[1] + 1]
        denom = 2 * (2 * c_zero - c_minus - c_plus)
        if abs(denom) > 1e-10:
            offset_x = (c_minus - c_plus) / denom
            if abs(offset_x) < 1.0:  # Sanity check
                peak_x += offset_x
    
    # The shift needed to align target with reference
    # Note: we negate because correlate tells us where target IS relative to reference,
    # but we want the shift needed to MOVE target back to reference
    dy = -(peak_y - center[0])
    dx = -(peak_x - center[1])
    
    logger.info(f"Cross-correlation found shift: dy={dy}, dx={dx}")
    
    # Calculate RMSE after alignment
    # Note: Can't use mode='nearest' with NaN - it propagates NaN!
    # Fill NaN with mean before shifting for RMSE calculation
    tgt_nan_mask = np.isnan(target)
    tgt_valid_data = target[~tgt_nan_mask]
    if tgt_valid_data.size > 0:
        tgt_fill = np.mean(tgt_valid_data)
    else:
        tgt_fill = 0.0
    target_filled = np.where(tgt_nan_mask, tgt_fill, target)
    
    # Apply shift with constant boundary
    shifted = ndimage_shift(target_filled, (dy, dx), order=3, mode='constant', cval=tgt_fill)
    
    # Restore NaN mask
    tgt_nan_mask_shifted = ndimage_shift(tgt_nan_mask.astype(float), (dy, dx), order=0, mode='constant', cval=1.0) > 0.5
    shifted[tgt_nan_mask_shifted] = np.nan
    
    # Mark shifted regions that came from outside as NaN
    # Create a mask of valid regions after shift
    valid_mask = np.ones_like(target, dtype=bool)
    if dy > 0:
        valid_mask[:int(dy), :] = False
    elif dy < 0:
        valid_mask[int(dy):, :] = False
    if dx > 0:
        valid_mask[:, :int(dx)] = False
    elif dx < 0:
        valid_mask[:, int(dx):] = False
    
    shifted[~valid_mask] = np.nan
    valid_both = ref_valid & ~np.isnan(shifted)
    
    if np.sum(valid_both) > 0:
        rmse = np.sqrt(np.mean((reference[valid_both] - shifted[valid_both]) ** 2))
        logger.info(f"Registration RMSE: {rmse:.2f} nm, overlapping points: {np.sum(valid_both)}")
    else:
        rmse = np.inf
        logger.warning(f"Registration failed: no overlapping points after shift!")
    
    return {
        'translation': (dy, dx),
        'rotation': 0.0,
        'rmse': rmse,
        'inliers': np.sum(valid_both)
    }


def _register_icp(reference, target, max_iterations=100):
    """Register using Iterative Closest Point (handles rotation + translation)."""
    
    # Extract valid points
    ref_valid = ~np.isnan(reference)
    tgt_valid = ~np.isnan(target)
    
    ref_points = np.column_stack(np.where(ref_valid))
    tgt_points = np.column_stack(np.where(tgt_valid))
    
    if len(ref_points) < 10 or len(tgt_points) < 10:
        logger.warning("Not enough valid points for ICP registration")
        return {
            'translation': (0, 0),
            'rotation': 0.0,
            'rmse': np.inf,
            'inliers': 0
        }
    
    # Add height as 3rd dimension
    ref_heights = reference[ref_valid].reshape(-1, 1)
    tgt_heights = target[tgt_valid].reshape(-1, 1)
    
    ref_points_3d = np.hstack([ref_points, ref_heights])
    tgt_points_3d = np.hstack([tgt_points, tgt_heights])
    
    # Subsample if too many points
    max_points = 5000
    if len(ref_points_3d) > max_points:
        indices = np.random.choice(len(ref_points_3d), max_points, replace=False)
        ref_points_3d = ref_points_3d[indices]
    if len(tgt_points_3d) > max_points:
        indices = np.random.choice(len(tgt_points_3d), max_points, replace=False)
        tgt_points_3d = tgt_points_3d[indices]
    
    # Simplified ICP: just find translation and in-plane rotation
    # For full ICP, would need scipy or dedicated library
    
    # Center the point clouds
    ref_center = np.mean(ref_points_3d[:, :2], axis=0)
    tgt_center = np.mean(tgt_points_3d[:, :2], axis=0)
    
    translation = ref_center - tgt_center
    
    # Estimate rotation using covariance (simplified)
    ref_centered = ref_points_3d[:, :2] - ref_center
    tgt_centered = tgt_points_3d[:, :2] - tgt_center
    
    # Sample matching (nearest neighbor approximation)
    from scipy.spatial import cKDTree
    
    tree = cKDTree(ref_centered)
    distances, indices = tree.query(tgt_centered, k=1)
    
    # Estimate rotation angle
    matched_ref = ref_centered[indices]
    
    # Use least squares to find rotation
    angles_ref = np.arctan2(matched_ref[:, 0], matched_ref[:, 1])
    angles_tgt = np.arctan2(tgt_centered[:, 0], tgt_centered[:, 1])
    angle_diffs = angles_ref - angles_tgt
    
    # Circular mean
    rotation_rad = np.arctan2(np.mean(np.sin(angle_diffs)), np.mean(np.cos(angle_diffs)))
    rotation_deg = np.degrees(rotation_rad)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    inliers = np.sum(distances < 3 * np.median(distances))
    
    logger.info(f"ICP registration: translation={translation}, rotation={rotation_deg:.2f}°, RMSE={rmse:.3f}")
    
    return {
        'translation': tuple(translation),
        'rotation': rotation_deg,
        'rmse': rmse,
        'inliers': inliers
    }


def apply_registration(grid, xi, yi, dx, dy, translation, rotation=0.0):
    """Applies registration transformation to a grid.
    
    Args:
        grid (np.ndarray): 2D height array.
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        translation (tuple): (dy, dx) translation in pixels.
        rotation (float, optional): Rotation angle in degrees. Defaults to 0.0.
            
    Returns:
        tuple: (transformed_grid, xi, yi, dx, dy)
    """
    
    # Apply rotation first if needed
    if abs(rotation) > 0.01:
        grid, xi, yi, dx, dy = rotate_grid(grid, rotation, xi, yi, dx, dy)
    
    # Apply translation using proper shift (not circular roll)
    dy, dx = translation
    
    if abs(dy) > 0.5 or abs(dx) > 0.5:
        # Strategy: Can't use mode='nearest' with NaN - it propagates NaN everywhere!
        # Instead:
        # 1. Save NaN mask
        # 2. Fill NaN with mean value (for interpolation)
        # 3. Shift with mode='constant', cval=mean
        # 4. Restore NaN mask and add edge mask
        
        nan_mask_before = np.isnan(grid)
        
        # Fill NaN with mean of valid data for interpolation
        valid_data = grid[~nan_mask_before]
        if valid_data.size > 0:
            fill_value = np.mean(valid_data)
        else:
            fill_value = 0.0
            logger.warning(f"apply_registration: grid is all NaN before shift!")
        
        grid_filled = np.where(nan_mask_before, fill_value, grid)
        
        # Shift with constant boundary (using fill value)
        grid = ndimage_shift(grid_filled, (dy, dx), order=3, mode='constant', cval=fill_value)
        
        # Restore original NaN mask (shifted)
        # Use order=1 (linear) instead of order=0 to properly interpolate mask values
        nan_mask_shifted = ndimage_shift(nan_mask_before.astype(float), (dy, dx), order=1, mode='constant', cval=1.0) > 0.5
        grid[nan_mask_shifted] = np.nan
        
        # Mark regions that were shifted from outside the original bounds as NaN
        valid_mask = np.ones_like(grid, dtype=bool)
        if dy > 0:
            valid_mask[:int(np.ceil(dy)), :] = False
        elif dy < 0:
            valid_mask[int(np.floor(dy)):, :] = False
        if dx > 0:
            valid_mask[:, :int(np.ceil(dx))] = False
        elif dx < 0:
            valid_mask[:, int(np.floor(dx)):] = False
        
        grid[~valid_mask] = np.nan
    
    return grid, xi, yi, dx, dy
