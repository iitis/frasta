"""Morphological operations and geometric corrections for surface preprocessing.

This module provides functions for removing systematic geometric forms including
polynomial surfaces, plane leveling, and form removal operations.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

import logging
logger = logging.getLogger(__name__)


def fit_plane_least_squares(grid, mask=None):
    """Fits a plane to the surface using least-squares regression.
    
    Fits a plane of the form: z = ax + by + c
    
    Args:
        grid (np.ndarray): 2D height array.
        mask (np.ndarray, optional): Boolean mask indicating valid region.
            If None, uses all non-NaN points. Defaults to None.
            
    Returns:
        tuple: (plane_grid, coefficients) where:
            - plane_grid: 2D array with fitted plane values
            - coefficients: (a, b, c) plane parameters
            
    Examples:
        >>> plane, (a, b, c) = fit_plane_least_squares(grid)
        >>> leveled = grid - plane
    """
    ny, nx = grid.shape
    
    # Create coordinate grids
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    # Apply mask
    if mask is not None:
        valid = mask & ~np.isnan(grid)
    else:
        valid = ~np.isnan(grid)
    
    if np.sum(valid) < 3:
        logger.warning("fit_plane_least_squares: not enough valid points")
        return np.zeros_like(grid), (0, 0, 0)
    
    # Extract valid points
    x = x_idx[valid].flatten()
    y = y_idx[valid].flatten()
    z = grid[valid].flatten()
    
    # Build design matrix
    A = np.column_stack([x, y, np.ones_like(x)])
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    
    # Reconstruct plane for entire grid
    plane_grid = a * x_idx + b * y_idx + c
    
    logger.debug(f"Plane fit: z = {a:.6f}*x + {b:.6f}*y + {c:.6f}")
    
    return plane_grid, (a, b, c)


def fit_plane_robust(grid, mask=None, residual_threshold=None):
    """Fits a plane using RANSAC for robust outlier rejection.
    
    Uses RANSAC (Random Sample Consensus) to fit a plane while automatically
    excluding outliers. More robust than least-squares for noisy data.
    
    Args:
        grid (np.ndarray): 2D height array.
        mask (np.ndarray, optional): Boolean mask indicating valid region.
        residual_threshold (float, optional): RANSAC residual threshold.
            If None, uses 3 * median absolute deviation. Defaults to None.
            
    Returns:
        tuple: (plane_grid, coefficients, inlier_mask) where:
            - plane_grid: 2D array with fitted plane values
            - coefficients: (a, b, c) plane parameters
            - inlier_mask: Boolean mask of inlier points
    """
    ny, nx = grid.shape
    
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    if mask is not None:
        valid = mask & ~np.isnan(grid)
    else:
        valid = ~np.isnan(grid)
    
    if np.sum(valid) < 10:
        logger.warning("fit_plane_robust: not enough valid points")
        return np.zeros_like(grid), (0, 0, 0), valid
    
    x = x_idx[valid].flatten().reshape(-1, 1)
    y = y_idx[valid].flatten().reshape(-1, 1)
    z = grid[valid].flatten()
    
    # Combine x, y as features
    X = np.hstack([x, y])
    
    # Auto-threshold if not provided
    if residual_threshold is None:
        mad = np.median(np.abs(z - np.median(z)))
        residual_threshold = 3.0 * mad
        logger.debug(f"RANSAC threshold: {residual_threshold:.6f}")
    
    # Fit with RANSAC
    ransac = RANSACRegressor(residual_threshold=residual_threshold, random_state=42)
    ransac.fit(X, z)
    
    # Get coefficients
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    # Reconstruct plane
    plane_grid = a * x_idx + b * y_idx + c
    
    # Create inlier mask
    inlier_mask = np.zeros_like(grid, dtype=bool)
    valid_indices = np.where(valid)
    inlier_indices_flat = ransac.inlier_mask_
    inlier_mask[valid_indices[0][inlier_indices_flat], valid_indices[1][inlier_indices_flat]] = True
    
    n_inliers = np.sum(inlier_mask)
    n_total = np.sum(valid)
    logger.info(f"RANSAC plane fit: {n_inliers}/{n_total} inliers ({100*n_inliers/n_total:.1f}%)")
    
    return plane_grid, (a, b, c), inlier_mask


def level_by_plane(grid, mask=None, method='least_squares'):
    """Removes plane from grid (leveling operation).
    
    Args:
        grid (np.ndarray): 2D height array.
        mask (np.ndarray, optional): Boolean mask indicating valid region.
        method (str, optional): Fitting method. Options:
            'least_squares' - standard least squares (fast)
            'robust' - RANSAC-based robust fitting (slower, outlier-resistant)
            Defaults to 'least_squares'.
            
    Returns:
        np.ndarray: Leveled grid with plane removed.
        
    Examples:
        >>> leveled = level_by_plane(grid, method='robust')
    """
    if method == 'least_squares':
        plane, coeffs = fit_plane_least_squares(grid, mask)
    elif method == 'robust':
        plane, coeffs, inliers = fit_plane_robust(grid, mask)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return grid - plane


def level_by_three_points(grid, p1, p2, p3, xi, yi):
    """Levels grid by fitting plane through three specified points.
    
    Args:
        grid (np.ndarray): 2D height array.
        p1, p2, p3 (tuple): (x, y) coordinates of three points in physical units.
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
            
    Returns:
        np.ndarray: Leveled grid.
        
    Examples:
        >>> # Level using corners and center
        >>> leveled = level_by_three_points(
        ...     grid, (0, 0), (100, 0), (50, 50), xi, yi)
    """
    # Convert physical coordinates to indices
    def coord_to_index(x, y):
        ix = np.argmin(np.abs(xi - x))
        iy = np.argmin(np.abs(yi - y))
        return iy, ix
    
    i1, j1 = coord_to_index(*p1)
    i2, j2 = coord_to_index(*p2)
    i3, j3 = coord_to_index(*p3)
    
    # Get heights at these points
    z1 = grid[i1, j1]
    z2 = grid[i2, j2]
    z3 = grid[i3, j3]
    
    if np.isnan(z1) or np.isnan(z2) or np.isnan(z3):
        logger.warning("level_by_three_points: one or more points are NaN")
        return grid.copy()
    
    # Solve for plane coefficients: z = ax + by + c
    # Using three points
    A = np.array([
        [j1, i1, 1],
        [j2, i2, 1],
        [j3, i3, 1]
    ])
    b = np.array([z1, z2, z3])
    
    coeffs = np.linalg.solve(A, b)
    a, b, c = coeffs
    
    # Create plane
    ny, nx = grid.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    plane = a * x_idx + b * y_idx + c
    
    return grid - plane


def remove_polynomial_form(grid, order=2, mask=None):
    """Removes polynomial surface form from grid.
    
    Fits and removes a polynomial surface of specified order:
    - Order 1: plane (ax + by + c)
    - Order 2: quadratic (ax² + bxy + cy² + dx + ey + f)
    - Order 3: cubic (includes x³, x²y, xy², y³, etc.)
    
    Args:
        grid (np.ndarray): 2D height array.
        order (int, optional): Polynomial order (1-5). Defaults to 2.
        mask (np.ndarray, optional): Boolean mask indicating valid region.
            
    Returns:
        np.ndarray: Grid with polynomial form removed.
        
    Examples:
        >>> # Remove quadratic form (bending, warping)
        >>> flattened = remove_polynomial_form(grid, order=2)
        >>> # Remove cubic form
        >>> flattened = remove_polynomial_form(grid, order=3)
    """
    if order < 1 or order > 5:
        raise ValueError("Polynomial order must be between 1 and 5")
    
    ny, nx = grid.shape
    
    # Create coordinate grids
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    
    # Normalize coordinates to [-1, 1] for numerical stability
    x_norm = 2 * (x_idx - np.min(x_idx)) / (np.max(x_idx) - np.min(x_idx)) - 1
    y_norm = 2 * (y_idx - np.min(y_idx)) / (np.max(y_idx) - np.min(y_idx)) - 1
    
    # Build polynomial basis functions
    basis_functions = []
    
    for total_degree in range(order + 1):
        for x_degree in range(total_degree + 1):
            y_degree = total_degree - x_degree
            term = (x_norm ** x_degree) * (y_norm ** y_degree)
            basis_functions.append(term.ravel())
    
    # Stack into design matrix
    A_full = np.column_stack(basis_functions)
    
    # Apply mask
    if mask is not None:
        valid = mask & ~np.isnan(grid)
    else:
        valid = ~np.isnan(grid)
    
    if np.sum(valid) < len(basis_functions):
        logger.warning("remove_polynomial_form: not enough valid points")
        return grid.copy()
    
    valid_flat = valid.ravel()
    A_masked = A_full[valid_flat]
    z_masked = grid.ravel()[valid_flat]
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A_masked, z_masked, rcond=None)
    
    # Reconstruct polynomial surface
    polynomial_surface = (A_full @ coeffs).reshape(grid.shape)
    
    logger.info(f"Removed polynomial form of order {order} ({len(coeffs)} coefficients)")
    
    return grid - polynomial_surface


def threshold_grid(grid, low=None, high=None):
    """Applies threshold to grid, setting values outside range to NaN.
    
    Args:
        grid (np.ndarray): 2D height array.
        low (float, optional): Lower threshold. Values below this become NaN.
        high (float, optional): Upper threshold. Values above this become NaN.
            
    Returns:
        np.ndarray: Thresholded grid.
        
    Examples:
        >>> # Remove outliers beyond ±3 sigma
        >>> mean, std = np.nanmean(grid), np.nanstd(grid)
        >>> filtered = threshold_grid(grid, mean - 3*std, mean + 3*std)
    """
    result = grid.copy()
    
    if low is not None:
        result[result < low] = np.nan
        logger.debug(f"Thresholded {np.sum(grid < low)} pixels below {low}")
    
    if high is not None:
        result[result > high] = np.nan
        logger.debug(f"Thresholded {np.sum(grid > high)} pixels above {high}")
    
    return result
