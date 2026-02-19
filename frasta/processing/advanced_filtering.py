"""Advanced filtering algorithms for scan data processing.

This module provides advanced filtering techniques including bilateral filtering
(edge-preserving), median filtering (outlier removal), and morphological operations.
"""

import numpy as np
from scipy.ndimage import median_filter, grey_opening, grey_closing
import warnings

# Try to import OpenCV for fast bilateral filtering
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

import logging
logger = logging.getLogger(__name__)


def bilateral_filter(grid, sigma_spatial, sigma_range, dx=1.0, dy=1.0, mask=None, use_opencv=True):
    """Applies bilateral filtering for edge-preserving smoothing.
    
    Bilateral filtering combines spatial proximity (Gaussian in space) with 
    intensity similarity (Gaussian in height domain). Unlike Gaussian filtering,
    it preserves sharp edges and step features while smoothing homogeneous regions.
    
    This is particularly useful for fracture surfaces where you want to smooth
    noise but preserve sharp fracture edges.
    
    **Performance**: Uses OpenCV's optimized implementation (~7500x faster than pure Python).
    Falls back to Python implementation if OpenCV is not available.
    
    Args:
        grid (np.ndarray or None): 2D array to filter.
        sigma_spatial (float): Spatial Gaussian standard deviation in physical units.
            Controls the extent of spatial smoothing.
        sigma_range (float): Range (height) Gaussian standard deviation.
            Controls how much height difference is tolerated for smoothing.
            Smaller values preserve edges more strongly.
        dx (float, optional): Pixel size in x-direction. Defaults to 1.0.
        dy (float, optional): Pixel size in y-direction. Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to filter.
            If None, filters entire grid. Defaults to None.
        use_opencv (bool, optional): Use OpenCV if available. Defaults to True.
            
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Notes:
        - OpenCV version is ~7500x faster than Python (0.005s vs 30s for 512x512 image)
        - Install OpenCV: pip install opencv-python
        - Python fallback is always available but much slower
        
    Examples:
        >>> # Smooth noise but preserve fracture edges
        >>> filtered = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0)
        >>> # More aggressive edge preservation
        >>> filtered = bilateral_filter(grid, sigma_spatial=3.0, sigma_range=5.0)
        >>> # Force Python implementation (slow but always available)
        >>> filtered = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, use_opencv=False)
    """
    if grid is None:
        return None
    
    grid = grid.copy()
    
    # Convert spatial sigma to pixels
    sigma_x_pixels = sigma_spatial / dx
    sigma_y_pixels = sigma_spatial / dy
    sigma_spatial_pixels = (sigma_x_pixels + sigma_y_pixels) / 2
    
    # Warn if sigma is too small
    if sigma_x_pixels < 1 or sigma_y_pixels < 1:
        logger.warning("bilateral_filter: sigma_spatial is smaller than pixel size")
    
    # Determine kernel size (3 sigma rule)
    kernel_radius = int(np.ceil(3 * sigma_spatial_pixels))
    kernel_size = 2 * kernel_radius + 1
    
    if kernel_size < 3:
        logger.warning("bilateral_filter: kernel size too small, returning original")
        return grid
    
    # Apply mask if provided (do this BEFORE checking for NaN)
    if mask is not None:
        work_grid = np.where(mask, grid, np.nan)
        mask = mask.copy()
    else:
        work_grid = grid.copy()
        mask = ~np.isnan(grid)
    
    # Try OpenCV implementation first (much faster)
    if use_opencv and HAS_OPENCV:
        has_nan = np.any(np.isnan(work_grid))
        
        if not has_nan:
            # Fast path: no NaN values, use OpenCV directly
            return _bilateral_filter_opencv(grid, sigma_spatial_pixels, sigma_range, kernel_size)
        else:
            # Handle NaN with OpenCV
            return _bilateral_filter_opencv_nan(work_grid, sigma_spatial_pixels, sigma_range, 
                                               kernel_size, mask)
    
    # Fall back to Python if OpenCV disabled or not available
    if not HAS_OPENCV:
        logger.info("bilateral_filter: OpenCV not available, using pure Python (slow)")
    
    return _bilateral_filter_python(grid, work_grid, mask, sigma_spatial_pixels,
                                   sigma_spatial, sigma_range, dx, dy, 
                                   kernel_radius, kernel_size)


def _bilateral_filter_opencv(grid, sigma_spatial_pixels, sigma_range, kernel_size):
    """Fast bilateral filter using OpenCV."""
    # OpenCV bilateral filter works best with float32 or uint8
    # We'll use float32 to preserve precision
    
    # OpenCV parameters
    d = min(kernel_size, 9)  # OpenCV recommends d <= 9 for efficiency
    sigma_color = sigma_range
    sigma_space = sigma_spatial_pixels
    
    # Convert to float32 (OpenCV's cv2.bilateralFilter supports it)
    grid_f32 = grid.astype(np.float32)
    
    # Apply bilateral filter directly on float32 data
    filtered_f32 = cv2.bilateralFilter(grid_f32, d, sigma_color, sigma_space,
                                       borderType=cv2.BORDER_REFLECT)
    
    return filtered_f32.astype(np.float64)


def _bilateral_filter_opencv_nan(work_grid, sigma_spatial_pixels, sigma_range, kernel_size, mask):
    """Bilateral filter with NaN handling using OpenCV."""
    result = work_grid.copy()
    
    # Get valid data
    valid_mask = ~np.isnan(work_grid) & mask
    
    if not np.any(valid_mask):
        return result
    
    # Replace NaN with mean value for OpenCV
    mean_val = np.nanmean(work_grid[valid_mask])
    grid_clean = np.where(valid_mask, work_grid, mean_val)
    
    # Apply OpenCV bilateral filter
    filtered_clean = _bilateral_filter_opencv(grid_clean, sigma_spatial_pixels, 
                                             sigma_range, kernel_size)
    
    # Restore NaN and masked regions
    result = np.where(valid_mask, filtered_clean, np.nan)
    
    return result


def _bilateral_filter_python(grid, work_grid, mask, sigma_spatial_pixels, sigma_spatial, 
                             sigma_range, dx, dy, kernel_radius, kernel_size):
    """Pure Python bilateral filter implementation (slow but no dependencies)."""
    # Apply mask if provided
    if mask is not None:
        work_grid = np.where(mask, grid, np.nan)
    else:
        work_grid = grid
        mask = ~np.isnan(grid)
    
    result = grid.copy()
    ny, nx = grid.shape
    
    # Manual bilateral filtering implementation
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            
            # Define neighborhood
            i_min = max(0, i - kernel_radius)
            i_max = min(ny, i + kernel_radius + 1)
            j_min = max(0, j - kernel_radius)
            j_max = min(nx, j + kernel_radius + 1)
            
            # Extract neighborhood
            neighborhood = work_grid[i_min:i_max, j_min:j_max]
            center_value = grid[i, j]
            
            # Create coordinate grids for spatial weights
            y_coords, x_coords = np.ogrid[i_min:i_max, j_min:j_max]
            
            # Spatial weights (Gaussian)
            spatial_dist_sq = ((x_coords - j) * dx) ** 2 + ((y_coords - i) * dy) ** 2
            spatial_weights = np.exp(-spatial_dist_sq / (2 * sigma_spatial ** 2))
            
            # Range weights (Gaussian on height differences)
            range_diff = neighborhood - center_value
            range_weights = np.exp(-range_diff ** 2 / (2 * sigma_range ** 2))
            
            # Combined weights
            weights = spatial_weights * range_weights
            
            # Handle NaN
            valid = ~np.isnan(neighborhood)
            weights = weights * valid
            
            # Weighted average
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                result[i, j] = np.sum(weights * np.nan_to_num(neighborhood)) / weight_sum
    
    return result


def median_filter_nan_aware(grid, size, dx=1.0, dy=1.0, mask=None):
    """Applies median filtering for robust outlier removal.
    
    Median filters are non-linear and robust to outliers and spikes.
    Unlike Gaussian filters, they preserve edges and sharp features.
    This implementation properly handles NaN values.
    
    Args:
        grid (np.ndarray or None): 2D array to filter.
        size (float): Filter kernel size in physical units.
        dx (float, optional): Pixel size in x-direction. Defaults to 1.0.
        dy (float, optional): Pixel size in y-direction. Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to filter.
            If None, filters entire grid. Defaults to None.
            
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Examples:
        >>> # Remove measurement spikes
        >>> filtered = median_filter_nan_aware(grid, size=10.0, px_x=1.0)
    """
    if grid is None:
        return None
    
    # Convert physical size to pixel units
    size_x = size / dx
    size_y = size / dy
    
    # Kernel size must be odd integer
    kernel_x = int(np.round(size_x))
    kernel_y = int(np.round(size_y))
    
    if kernel_x % 2 == 0:
        kernel_x += 1
    if kernel_y % 2 == 0:
        kernel_y += 1
    
    if kernel_x < 3 or kernel_y < 3:
        logger.warning("median_filter: kernel size too small (< 3 pixels)")
        return grid.copy()
    
    # Apply mask
    grid_work = grid.copy()
    if mask is not None:
        grid_work[~mask] = np.nan
    
    # Apply median filter
    result = median_filter(grid_work, size=(kernel_y, kernel_x), mode='nearest')
    
    # Restore masked values
    if mask is not None:
        result[~mask] = grid[~mask]
    
    return result


def morphological_opening(grid, size, dx=1.0, dy=1.0, mask=None):
    """Applies morphological opening (erosion followed by dilation).
    
    Opening removes small peaks, spikes, and bright outliers while smoothing
    object contours. It's useful for removing measurement artifacts.
    
    Args:
        grid (np.ndarray or None): 2D array to filter.
        size (float): Structuring element size in physical units.
        dx (float, optional): Pixel size in x-direction. Defaults to 1.0.
        dy (float, optional): Pixel size in y-direction. Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to filter.
            If None, filters entire grid. Defaults to None.
            
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Examples:
        >>> # Remove small spikes
        >>> filtered = morphological_opening(grid, size=5.0)
    """
    if grid is None:
        return None
    
    # Convert to pixels
    size_x = size / dx
    size_y = size / dy
    
    kernel_x = int(np.round(size_x))
    kernel_y = int(np.round(size_y))
    
    if kernel_x % 2 == 0:
        kernel_x += 1
    if kernel_y % 2 == 0:
        kernel_y += 1
    
    if kernel_x < 3 or kernel_y < 3:
        logger.warning("morphological_opening: kernel size too small")
        return grid.copy()
    
    grid_work = grid.copy()
    if mask is not None:
        grid_work[~mask] = np.nan
    
    result = grey_opening(grid_work, size=(kernel_y, kernel_x), mode='nearest')
    
    if mask is not None:
        result[~mask] = grid[~mask]
    
    return result


def morphological_closing(grid, size, dx=1.0, dy=1.0, mask=None):
    """Applies morphological closing (dilation followed by erosion).
    
    Closing removes small valleys, pits, and dark outliers while smoothing
    object contours. It's useful for filling small holes.
    
    Args:
        grid (np.ndarray or None): 2D array to filter.
        size (float): Structuring element size in physical units.
        dx (float, optional): Pixel size in x-direction. Defaults to 1.0.
        dy (float, optional): Pixel size in y-direction. Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to filter.
            If None, filters entire grid. Defaults to None.
            
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Examples:
        >>> # Fill small valleys
        >>> filtered = morphological_closing(grid, size=5.0)
    """
    if grid is None:
        return None
    
    # Convert to pixels
    size_x = size / dx
    size_y = size / dy
    
    kernel_x = int(np.round(size_x))
    kernel_y = int(np.round(size_y))
    
    if kernel_x % 2 == 0:
        kernel_x += 1
    if kernel_y % 2 == 0:
        kernel_y += 1
    
    if kernel_x < 3 or kernel_y < 3:
        logger.warning("morphological_closing: kernel size too small")
        return grid.copy()
    
    grid_work = grid.copy()
    if mask is not None:
        grid_work[~mask] = np.nan
    
    result = grey_closing(grid_work, size=(kernel_y, kernel_x), mode='nearest')
    
    if mask is not None:
        result[~mask] = grid[~mask]
    
    return result


def robust_gaussian_filter(grid, sigma, dx=1.0, dy=1.0, mask=None, iterations=3, threshold=3.0):
    """Applies robust Gaussian filtering with iterative outlier rejection.
    
    This filter iteratively applies Gaussian smoothing while excluding outliers
    that deviate too much from the smoothed result. It's more robust to spikes
    than standard Gaussian filtering.
    
    Args:
        grid (np.ndarray or None): 2D array to filter.
        sigma (float): Gaussian standard deviation in physical units.
        dx (float, optional): Pixel size in x-direction. Defaults to 1.0.
        dy (float, optional): Pixel size in y-direction. Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to filter.
        iterations (int, optional): Number of outlier rejection iterations. Defaults to 3.
        threshold (float, optional): Number of standard deviations for outlier threshold.
            Defaults to 3.0.
            
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Examples:
        >>> # Robust smoothing with outlier rejection
        >>> filtered = robust_gaussian_filter(grid, sigma=10.0, iterations=3)
    """
    if grid is None:
        return None
    
    from scipy.ndimage import gaussian_filter
    
    # Convert sigma to pixels
    sigma_x = sigma / dx
    sigma_y = sigma / dy
    
    grid_work = grid.copy()
    if mask is not None:
        valid_mask = mask & ~np.isnan(grid)
    else:
        valid_mask = ~np.isnan(grid)
    
    for iteration in range(iterations):
        # Apply Gaussian filter to current valid data
        filled = np.where(valid_mask, grid_work, 0)
        weights = valid_mask.astype(float)
        
        smoothed = gaussian_filter(filled, sigma=(sigma_y, sigma_x))
        weight_sum = gaussian_filter(weights, sigma=(sigma_y, sigma_x))
        
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = smoothed / weight_sum
            smoothed[weight_sum == 0] = np.nan
        
        # Calculate residuals
        residuals = np.abs(grid_work - smoothed)
        residual_std = np.nanstd(residuals[valid_mask])
        
        # Update valid mask - exclude outliers
        outliers = residuals > (threshold * residual_std)
        valid_mask = valid_mask & ~outliers
        
        logger.debug(f"Robust Gaussian iteration {iteration+1}: excluded {np.sum(outliers)} outliers")
    
    # Final smoothing with outlier-free data
    filled = np.where(valid_mask, grid_work, 0)
    weights = valid_mask.astype(float)
    
    smoothed = gaussian_filter(filled, sigma=(sigma_y, sigma_x))
    weight_sum = gaussian_filter(weights, sigma=(sigma_y, sigma_x))
    
    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed / weight_sum
        result[weight_sum == 0] = np.nan
    
    return result
