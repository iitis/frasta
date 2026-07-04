"""Geometric transformations and surface registration algorithms.

This module provides functions for rotating, scaling, cropping surfaces and
automatic registration (alignment) of two surfaces.
"""

import numpy as np
from scipy.ndimage import (
    affine_transform,
    binary_closing,
    binary_opening,
    map_coordinates,
    shift as ndimage_shift,
)
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
    center_x = nx / 2.0
    center_y = ny / 2.0
    
    # Convert angle to radians
    theta = np.radians(angle_degrees)
    
    # Build the inverse rotation directly in index space while respecting
    # anisotropic sample spacing. ``affine_transform`` maps output indices back
    # to input indices, so we use the inverse rigid transform here.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    scale_col_to_row = float(dx) / float(dy)
    scale_row_to_col = float(dy) / float(dx)
    matrix = np.array(
        [
            [cos_theta, sin_theta * scale_col_to_row],
            [-sin_theta * scale_row_to_col, cos_theta],
        ],
        dtype=float,
    )
    
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


def auto_register_surfaces(
    reference,
    target,
    method='icp',
    max_iterations=100,
    refine=True,
    stable_region=False,
    reference_dx=1.0,
    reference_dy=1.0,
    target_dx=1.0,
    target_dy=1.0,
):
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
        refine (bool, optional): When using ICP, run final height-RMSE refinement.
            Defaults to True.
        stable_region (bool, optional): When using ICP, run a second pass on an
            automatically selected low-mismatch overlap region. Defaults to False.
            
    Returns:
        dict: Registration parameters containing:
            - 'translation': (dy, dx) translation in physical units
            - 'rotation': rotation angle in degrees (for ICP)
            - 'rmse': root mean square error after alignment
            - 'inliers': number of matching points
            
    Examples:
        >>> # Automatically align two fracture surfaces
        >>> params = auto_register_surfaces(surface1, surface2, method='icp')
        >>> print(f"Translation: {params['translation']}, Rotation: {params['rotation']}°")
    """
    same_spacing = (
        np.isclose(float(reference_dx), float(target_dx))
        and np.isclose(float(reference_dy), float(target_dy))
    )
    if method == 'correlation':
        if same_spacing:
            params = _register_correlation(
                reference,
                target,
                reference_dx=reference_dx,
                reference_dy=reference_dy,
                target_dx=target_dx,
                target_dy=target_dy,
            )
        else:
            logger.warning(
                "Cross-correlation requested for anisotropic or mismatched spacing; "
                "falling back to ICP in physical space."
            )
            params = _register_icp(
                reference,
                target,
                max_iterations,
                refine=refine,
                stable_region=stable_region,
                reference_dx=reference_dx,
                reference_dy=reference_dy,
                target_dx=target_dx,
                target_dy=target_dy,
            )
    elif method == 'icp':
        return _register_icp(
            reference,
            target,
            max_iterations,
            refine=refine,
            stable_region=stable_region,
            reference_dx=reference_dx,
            reference_dy=reference_dy,
            target_dx=target_dx,
            target_dy=target_dy,
        )
    else:
        raise ValueError(f"Unknown registration method: {method}")
    return params


def _detrend_surface_plane(grid: np.ndarray) -> np.ndarray:
    """Fit and subtract the best-fit plane from a 2D height grid.

    Removes the global tilt (linear trend) from the surface so that the
    cross-correlation is driven by topographic features rather than slope.
    NaN pixels are set to 0.0 in the output (neutral for FFT correlation).

    Args:
        grid (np.ndarray): 2D height array, may contain NaN.

    Returns:
        np.ndarray: Detrended grid with NaN regions zeroed out.
    """
    valid = ~np.isnan(grid)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        # Not enough points – fall back to mean centering
        mean_val = np.nanmean(grid)
        mean_val = mean_val if np.isfinite(mean_val) else 0.0
        return np.where(valid, grid - mean_val, 0.0)

    rows, cols = np.where(valid)
    z = grid[valid]
    A = np.column_stack([rows, cols, np.ones(n_valid)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        mean_val = float(np.mean(z))
        return np.where(valid, grid - mean_val, 0.0)

    r_all, c_all = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    plane = coeffs[0] * r_all + coeffs[1] * c_all + coeffs[2]
    return np.where(valid, grid - plane, 0.0)


def _register_correlation(
    reference,
    target,
    *,
    reference_dx=1.0,
    reference_dy=1.0,
    target_dx=1.0,
    target_dy=1.0,
):
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

    if not np.any(ref_valid) or not np.any(tgt_valid):
        logger.warning("Cross-correlation failed: one of the inputs has no valid data.")
        return {
            'translation': (0.0, 0.0),
            'rotation': 0.0,
            'rmse': np.inf,
            'inliers': 0,
        }

    # Remove the global plane (tilt + offset) from each surface before
    # correlating.  Simple mean subtraction fails when the surface is
    # dominated by a linear slope: after mean removal the dominant signal
    # is still the monotonic gradient, whose autocorrelation is nearly flat
    # and argmax can land anywhere.  Plane-detrending leaves only the
    # topographic residuals which are the true registration signal.
    ref_centered = _detrend_surface_plane(reference)
    tgt_centered = _detrend_surface_plane(target)
    correlation = correlate(ref_centered, tgt_centered, mode='same', method='fft')
    overlap = correlate(ref_valid.astype(float), tgt_valid.astype(float), mode='same', method='fft')

    # Ignore edge peaks with very small overlap and unrealistic large shifts.
    min_overlap = max(25.0, 0.15 * min(np.sum(ref_valid), np.sum(tgt_valid)))
    center = (reference.shape[0] // 2, reference.shape[1] // 2)
    max_shift_y = max(3, int(reference.shape[0] * 0.35))
    max_shift_x = max(3, int(reference.shape[1] * 0.35))
    row_indices, col_indices = np.indices(reference.shape)
    search_mask = (
        (overlap >= min_overlap)
        & (np.abs(row_indices - center[0]) <= max_shift_y)
        & (np.abs(col_indices - center[1]) <= max_shift_x)
    )
    if np.any(search_mask):
        safe_score = np.where(search_mask, correlation, -np.inf)
        max_pos = np.unravel_index(np.argmax(safe_score), safe_score.shape)
    else:
        max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)

    peak_y = float(max_pos[0])
    peak_x = float(max_pos[1])

    if 0 < max_pos[0] < correlation.shape[0] - 1:
        c_minus = correlation[max_pos[0] - 1, max_pos[1]]
        c_zero = correlation[max_pos[0], max_pos[1]]
        c_plus = correlation[max_pos[0] + 1, max_pos[1]]
        denom = 2 * (2 * c_zero - c_minus - c_plus)
        if abs(denom) > 1e-10:
            offset_y = (c_minus - c_plus) / denom
            if abs(offset_y) < 1.0:
                peak_y += offset_y

    if 0 < max_pos[1] < correlation.shape[1] - 1:
        c_minus = correlation[max_pos[0], max_pos[1] - 1]
        c_zero = correlation[max_pos[0], max_pos[1]]
        c_plus = correlation[max_pos[0], max_pos[1] + 1]
        denom = 2 * (2 * c_zero - c_minus - c_plus)
        if abs(denom) > 1e-10:
            offset_x = (c_minus - c_plus) / denom
            if abs(offset_x) < 1.0:
                peak_x += offset_x

    dy = (peak_y - center[0]) * float(reference_dy)
    dx = (peak_x - center[1]) * float(reference_dx)
    
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
    shift_rows = dy / float(target_dy)
    shift_cols = dx / float(target_dx)
    shifted = ndimage_shift(target_filled, (shift_rows, shift_cols), order=3, mode='constant', cval=tgt_fill)
    
    # Restore NaN mask
    tgt_nan_mask_shifted = ndimage_shift(
        tgt_nan_mask.astype(float),
        (shift_rows, shift_cols),
        order=0,
        mode='constant',
        cval=1.0,
    ) > 0.5
    shifted[tgt_nan_mask_shifted] = np.nan
    
    # Mark shifted regions that came from outside as NaN
    # Create a mask of valid regions after shift
    valid_mask = np.ones_like(target, dtype=bool)
    edge_rows = int(np.ceil(abs(shift_rows)))
    edge_cols = int(np.ceil(abs(shift_cols)))
    if shift_rows > 0 and edge_rows > 0:
        valid_mask[:edge_rows, :] = False
    elif shift_rows < 0 and edge_rows > 0:
        valid_mask[-edge_rows:, :] = False
    if shift_cols > 0 and edge_cols > 0:
        valid_mask[:, :edge_cols] = False
    elif shift_cols < 0 and edge_cols > 0:
        valid_mask[:, -edge_cols:] = False
    
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


def _subsample_registration_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Return a deterministic subsample of registration points."""
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def _estimate_rigid_transform_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the rigid 2D transform that maps source points onto target points."""
    source_mean = np.mean(source_xy, axis=0)
    target_mean = np.mean(target_xy, axis=0)
    source_centered = source_xy - source_mean
    target_centered = target_xy - target_mean

    covariance = source_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    translation = target_mean - source_mean @ rotation.T
    return rotation, translation


def _compose_row_vector_transform(
    rotation_total: np.ndarray,
    translation_total: np.ndarray,
    rotation_delta: np.ndarray,
    translation_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose rigid transforms that operate on row-vector coordinates."""
    rotation_composed = rotation_delta @ rotation_total
    translation_composed = translation_total @ rotation_delta.T + translation_delta
    return rotation_composed, translation_composed


def _downsample_grid_for_registration(grid: np.ndarray, max_dim: int = 160) -> np.ndarray:
    """Downsample a grid by integer strides for fast registration refinement."""
    if grid.size == 0:
        return grid
    stride_y = max(1, int(np.ceil(grid.shape[0] / max_dim)))
    stride_x = max(1, int(np.ceil(grid.shape[1] / max_dim)))
    return grid[::stride_y, ::stride_x]


def _registration_rmse_for_grid(
    reference: np.ndarray,
    target: np.ndarray,
    translation: tuple[float, float],
    rotation: float,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[float, int]:
    """Calculate overlap RMSE after applying a candidate registration."""
    if int(np.sum(np.isfinite(reference))) == 0 or int(np.sum(np.isfinite(target))) == 0:
        return np.inf, 0
    height, width = target.shape
    xi = np.arange(width, dtype=float)
    yi = np.arange(height, dtype=float)
    transformed, _, _, _, _ = apply_registration(
        target,
        xi,
        yi,
        dx,
        dy,
        translation,
        rotation=rotation,
    )
    common_height = min(reference.shape[0], transformed.shape[0])
    common_width = min(reference.shape[1], transformed.shape[1])
    if common_height == 0 or common_width == 0:
        return np.inf, 0

    reference_common = reference[:common_height, :common_width]
    transformed_common = transformed[:common_height, :common_width]
    overlap_mask = np.isfinite(reference_common) & np.isfinite(transformed_common)
    if not np.any(overlap_mask):
        return np.inf, 0
    rmse = float(
        np.sqrt(
            np.mean(
                (reference_common[overlap_mask] - transformed_common[overlap_mask]) ** 2
            )
        )
    )
    return rmse, int(np.sum(overlap_mask))


def _optimize_registration_on_grid(
    reference: np.ndarray,
    target: np.ndarray,
    translation: tuple[float, float],
    rotation: float,
    max_iterations: int,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[tuple[float, float], float]:
    """Refine rigid registration by minimizing height RMSE on the provided grids."""
    if int(np.sum(np.isfinite(reference))) < 10 or int(np.sum(np.isfinite(target))) < 10:
        return translation, rotation
    ref_valid = np.isfinite(reference)
    initial_guess = np.array([translation[0], translation[1], rotation], dtype=float)

    def objective(params: np.ndarray) -> float:
        """Evaluate height RMSE for a candidate rigid transform."""
        candidate_translation = (float(params[0]), float(params[1]))
        candidate_rotation = float(params[2])
        candidate_rmse, candidate_overlap = _registration_rmse_for_grid(
            reference,
            target,
            candidate_translation,
            candidate_rotation,
            dx=dx,
            dy=dy,
        )
        if candidate_overlap == 0 or not np.isfinite(candidate_rmse):
            return 1e12
        overlap_fraction = candidate_overlap / max(1, np.sum(ref_valid))
        return candidate_rmse - 0.05 * overlap_fraction

    refinement = minimize(
        objective,
        initial_guess,
        method="Powell",
        options={"maxiter": max_iterations, "xtol": 0.05, "ftol": 1e-3},
    )
    if refinement.success:
        return (float(refinement.x[0]), float(refinement.x[1])), float(refinement.x[2])
    return translation, rotation


def _compose_registration_parameters(
    base_translation: tuple[float, float],
    base_rotation: float,
    delta_translation: tuple[float, float],
    delta_rotation: float,
    grid_shape: tuple[int, int],
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[tuple[float, float], float]:
    """Compose two apply_registration-style rigid transforms for one grid shape."""
    center = np.array(
        [
            (grid_shape[0] / 2.0) * float(dy),
            (grid_shape[1] / 2.0) * float(dx),
        ],
        dtype=float,
    )

    def to_origin_transform(
        translation: tuple[float, float],
        rotation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation_rad = np.radians(rotation)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_rad), -np.sin(rotation_rad)],
                [np.sin(rotation_rad), np.cos(rotation_rad)],
            ],
            dtype=float,
        )
        translation_origin = np.array(translation, dtype=float) + center - center @ rotation_matrix.T
        return rotation_matrix, translation_origin

    base_rotation_matrix, base_translation_origin = to_origin_transform(base_translation, base_rotation)
    delta_rotation_matrix, delta_translation_origin = to_origin_transform(delta_translation, delta_rotation)
    combined_rotation, combined_translation_origin = _compose_row_vector_transform(
        base_rotation_matrix,
        base_translation_origin,
        delta_rotation_matrix,
        delta_translation_origin,
    )
    combined_rotation_deg = float(
        np.degrees(np.arctan2(combined_rotation[1, 0], combined_rotation[0, 0]))
    )
    combined_translation = tuple(
        (combined_translation_origin - center + center @ combined_rotation.T).tolist()
    )
    return combined_translation, combined_rotation_deg


def _build_stable_overlap_masks(
    reference: np.ndarray,
    aligned_target: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build masks for the low-mismatch overlap region in one coordinate frame."""
    common_height = min(reference.shape[0], aligned_target.shape[0])
    common_width = min(reference.shape[1], aligned_target.shape[1])
    if common_height == 0 or common_width == 0:
        return None, None

    reference_common = reference[:common_height, :common_width]
    transformed_common = aligned_target[:common_height, :common_width]
    overlap_mask = np.isfinite(reference_common) & np.isfinite(transformed_common)
    if int(np.sum(overlap_mask)) < 50:
        return None, None

    residual = reference_common - transformed_common
    residual_values = residual[overlap_mask]
    residual_median = float(np.median(residual_values))
    residual_mad = float(np.median(np.abs(residual_values - residual_median)))
    residual_sigma = 1.4826 * residual_mad
    if not np.isfinite(residual_sigma) or residual_sigma < 1e-8:
        residual_sigma = float(np.std(residual_values))
    if not np.isfinite(residual_sigma) or residual_sigma < 1e-8:
        return None, None

    stable_common = overlap_mask & (np.abs(residual - residual_median) <= 2.5 * residual_sigma)
    if int(np.sum(stable_common)) < 50:
        return None, None

    stable_common = binary_opening(stable_common, structure=np.ones((3, 3), dtype=bool))
    stable_common = binary_closing(stable_common, structure=np.ones((3, 3), dtype=bool))
    if int(np.sum(stable_common)) < 50:
        return None, None

    reference_mask = np.zeros_like(reference, dtype=bool)
    target_mask_transformed = np.zeros_like(aligned_target, dtype=bool)
    reference_mask[:common_height, :common_width] = stable_common
    target_mask_transformed[:common_height, :common_width] = stable_common

    if int(np.sum(reference_mask)) < 50 or int(np.sum(target_mask_transformed)) < 50:
        return None, None
    return reference_mask, target_mask_transformed


def _register_icp(
    reference,
    target,
    max_iterations=100,
    refine=True,
    stable_region=False,
    *,
    reference_dx=1.0,
    reference_dy=1.0,
    target_dx=1.0,
    target_dy=1.0,
):
    """Register using ICP with iterative 2D rigid fitting."""
    ref_valid = ~np.isnan(reference)
    tgt_valid = ~np.isnan(target)

    ref_rows, ref_cols = np.where(ref_valid)
    tgt_rows, tgt_cols = np.where(tgt_valid)
    ref_points = np.column_stack(
        (
            ref_rows.astype(float) * float(reference_dy),
            ref_cols.astype(float) * float(reference_dx),
        )
    )
    tgt_points = np.column_stack(
        (
            tgt_rows.astype(float) * float(target_dy),
            tgt_cols.astype(float) * float(target_dx),
        )
    )
    if len(ref_points) < 10 or len(tgt_points) < 10:
        logger.warning("Not enough valid points for ICP registration")
        return {
            'translation': (0, 0),
            'rotation': 0.0,
            'rmse': np.inf,
            'inliers': 0
        }

    ref_heights = reference[ref_valid].astype(float)
    tgt_heights = target[tgt_valid].astype(float)

    max_points = 5000
    ref_indices = _subsample_registration_points(np.arange(len(ref_points)), max_points)
    tgt_indices = _subsample_registration_points(np.arange(len(tgt_points)), max_points)
    ref_points = ref_points[ref_indices]
    tgt_points = tgt_points[tgt_indices]
    ref_heights = ref_heights[ref_indices]
    tgt_heights = tgt_heights[tgt_indices]

    combined_heights = np.concatenate([ref_heights, tgt_heights])
    height_scale = float(np.nanstd(combined_heights))
    if not np.isfinite(height_scale) or height_scale < 1e-8:
        height_scale = 1.0
    feature_height_weight = 3.0

    # Detrend heights by removing the best-fit plane before using them as
    # ICP features.  Without this, a tilted surface has a monotone height
    # gradient that is nearly redundant with the (row, col) coordinates and
    # causes ICP to converge to wrong positions on nearly-planar surfaces.
    def _detrend_heights(points_rc: np.ndarray, heights: np.ndarray) -> np.ndarray:
        if len(heights) < 3:
            return heights - np.median(heights)
        A = np.column_stack([points_rc, np.ones(len(points_rc))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
            return heights - (A @ coeffs)
        except np.linalg.LinAlgError:
            return heights - np.median(heights)

    ref_heights_dt = _detrend_heights(ref_points, ref_heights)
    tgt_heights_dt = _detrend_heights(tgt_points, tgt_heights)

    height_scale_dt = float(np.std(np.concatenate([ref_heights_dt, tgt_heights_dt])))
    if not np.isfinite(height_scale_dt) or height_scale_dt < 1e-8:
        height_scale_dt = 1.0

    ref_height_feature = feature_height_weight * (ref_heights_dt / height_scale_dt)
    tgt_height_feature = feature_height_weight * (tgt_heights_dt / height_scale_dt)
    ref_features = np.column_stack([ref_points, ref_height_feature])

    from scipy.spatial import cKDTree

    tree = cKDTree(ref_features)
    rotation_total = np.eye(2, dtype=float)
    translation_total = np.mean(ref_points, axis=0) - np.mean(tgt_points, axis=0)
    previous_rmse = np.inf
    correspondence_rmse = np.inf
    inliers = 0

    for _ in range(max_iterations):
        transformed_points = tgt_points @ rotation_total.T + translation_total
        transformed_features = np.column_stack([transformed_points, tgt_height_feature])
        distances, indices = tree.query(transformed_features, k=1)

        median_distance = float(np.median(distances))
        if not np.isfinite(median_distance):
            break
        if median_distance <= 0.0:
            inlier_mask = np.ones_like(distances, dtype=bool)
        else:
            inlier_mask = distances <= max(2.0, 2.5 * median_distance)
        if np.sum(inlier_mask) < 10:
            inlier_mask = np.ones_like(distances, dtype=bool)
        if np.sum(inlier_mask) < 10:
            break

        matched_reference = ref_points[indices[inlier_mask]]
        matched_source = transformed_points[inlier_mask]
        rotation_delta, translation_delta = _estimate_rigid_transform_2d(matched_source, matched_reference)
        rotation_total, translation_total = _compose_row_vector_transform(
            rotation_total,
            translation_total,
            rotation_delta,
            translation_delta,
        )

        transformed_points = tgt_points @ rotation_total.T + translation_total
        transformed_features = np.column_stack([transformed_points, tgt_height_feature])
        distances, _ = tree.query(transformed_features, k=1)
        median_distance = float(np.median(distances))
        if median_distance <= 0.0:
            inlier_mask = np.ones_like(distances, dtype=bool)
        else:
            inlier_mask = distances <= max(2.0, 2.5 * median_distance)
        inliers = int(np.sum(inlier_mask))
        if inliers == 0:
            break

        correspondence_rmse = float(np.sqrt(np.mean(distances[inlier_mask] ** 2)))
        delta_rotation = float(np.degrees(np.arctan2(rotation_delta[1, 0], rotation_delta[0, 0])))
        delta_translation = float(np.linalg.norm(translation_delta))
        if abs(previous_rmse - correspondence_rmse) < 1e-4 and abs(delta_rotation) < 0.01 and delta_translation < 0.01:
            break
        previous_rmse = correspondence_rmse

    rotation_deg = float(np.degrees(np.arctan2(rotation_total[1, 0], rotation_total[0, 0])))
    center = np.array(
        [
            (target.shape[0] / 2.0) * float(target_dy),
            (target.shape[1] / 2.0) * float(target_dx),
        ],
        dtype=float,
    )
    translation_apply = translation_total - center + center @ rotation_total.T
    translation = (float(translation_apply[0]), float(translation_apply[1]))

    same_spacing = (
        np.isclose(float(reference_dx), float(target_dx))
        and np.isclose(float(reference_dy), float(target_dy))
    )

    # Always do a small, cheap refinement on downsampled grids. This keeps the
    # fast ICP variant useful instead of stopping at the raw point-cloud fit.
    rmse = correspondence_rmse
    overlap_points = inliers
    if same_spacing:
        coarse_reference = _downsample_grid_for_registration(reference, max_dim=140)
        coarse_target = _downsample_grid_for_registration(target, max_dim=140)
        coarse_stride_y = max(1.0, reference.shape[0] / max(1, coarse_reference.shape[0]))
        coarse_stride_x = max(1.0, reference.shape[1] / max(1, coarse_reference.shape[1]))
        coarse_translation = translation
        coarse_translation, coarse_rotation = _optimize_registration_on_grid(
            coarse_reference,
            coarse_target,
            coarse_translation,
            rotation_deg,
            max_iterations=max(8, min(18, max_iterations // 2)),
            dx=float(reference_dx) * coarse_stride_x,
            dy=float(reference_dy) * coarse_stride_y,
        )
        translation = (float(coarse_translation[0]), float(coarse_translation[1]))
        rotation_deg = float(coarse_rotation)

        if refine:
            translation, rotation_deg = _optimize_registration_on_grid(
                reference,
                target,
                translation,
                rotation_deg,
                max_iterations=max(20, max_iterations),
                dx=float(reference_dx),
                dy=float(reference_dy),
            )

        grid_rmse, overlap_points = _registration_rmse_for_grid(
            reference,
            target,
            translation,
            rotation_deg,
            dx=float(reference_dx),
            dy=float(reference_dy),
        )
        rmse = grid_rmse if overlap_points > 0 else correspondence_rmse
        if overlap_points > 0:
            inliers = overlap_points

    if stable_region and np.isfinite(rmse):
        height, width = target.shape
        xi = np.arange(width, dtype=float)
        yi = np.arange(height, dtype=float)
        transformed_target, _, _, _, _ = apply_registration(
            target,
            xi,
            yi,
            float(target_dx),
            float(target_dy),
            translation,
            rotation=rotation_deg,
        )
        reference_mask, target_mask = _build_stable_overlap_masks(reference, transformed_target)
        if reference_mask is not None and target_mask is not None:
            reference_stable = np.where(reference_mask, reference, np.nan)
            target_stable = np.where(target_mask, transformed_target, np.nan)
            stable_params = _register_icp(
                reference_stable,
                target_stable,
                max_iterations=max(20, max_iterations // 2),
                refine=refine,
                stable_region=False,
                reference_dx=reference_dx,
                reference_dy=reference_dy,
                target_dx=target_dx,
                target_dy=target_dy,
            )
            translation, rotation_deg = _compose_registration_parameters(
                translation,
                rotation_deg,
                stable_params['translation'],
                stable_params['rotation'],
                target.shape,
                dx=float(target_dx),
                dy=float(target_dy),
            )
            grid_rmse, overlap_points = _registration_rmse_for_grid(
                reference,
                target,
                translation,
                rotation_deg,
                dx=float(reference_dx),
                dy=float(reference_dy),
            )
            rmse = grid_rmse if overlap_points > 0 else stable_params['rmse']
            inliers = overlap_points if overlap_points > 0 else stable_params['inliers']

    logger.info(
        "ICP registration: translation=%s, rotation=%.2f deg, RMSE=%.3f",
        translation,
        rotation_deg,
        rmse,
    )

    return {
        'translation': translation,
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
        translation (tuple): (dy, dx) translation in physical units.
        rotation (float, optional): Rotation angle in degrees. Defaults to 0.0.
            
    Returns:
        tuple: (transformed_grid, xi, yi, dx, dy)
    """
    
    shift_dy, shift_dx = translation
    if abs(rotation) <= 0.01 and abs(shift_dy) < (0.5 * dy) and abs(shift_dx) < (0.5 * dx):
        return grid, xi, yi, dx, dy

    theta = np.radians(rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    yi0 = float(yi[0]) if len(yi) else 0.0
    xi0 = float(xi[0]) if len(xi) else 0.0
    center_y = 0.5 * (float(yi[0]) + float(yi[-1])) if len(yi) else 0.0
    center_x = 0.5 * (float(xi[0]) + float(xi[-1])) if len(xi) else 0.0

    row_indices, col_indices = np.indices(grid.shape, dtype=np.float64)
    y_out = yi0 + row_indices * float(dy)
    x_out = xi0 + col_indices * float(dx)
    rel_y = y_out - center_y - float(shift_dy)
    rel_x = x_out - center_x - float(shift_dx)
    y_in = center_y + (cos_theta * rel_y + sin_theta * rel_x)
    x_in = center_x + (-sin_theta * rel_y + cos_theta * rel_x)
    row_in = (y_in - yi0) / float(dy)
    col_in = (x_in - xi0) / float(dx)

    valid_data = grid[np.isfinite(grid)]
    fill_value = float(np.mean(valid_data)) if valid_data.size > 0 else 0.0
    if valid_data.size == 0:
        logger.warning("apply_registration: grid is all NaN before transform!")
    grid_filled = np.where(np.isfinite(grid), grid, fill_value)
    transformed = map_coordinates(
        grid_filled,
        [row_in, col_in],
        order=3,
        mode='constant',
        cval=fill_value,
    )
    valid_weights = map_coordinates(
        np.isfinite(grid).astype(np.float32),
        [row_in, col_in],
        order=1,
        mode='constant',
        cval=0.0,
    )
    transformed[valid_weights < 0.999] = np.nan
    return transformed, xi, yi, dx, dy
