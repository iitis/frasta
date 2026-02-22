"""Local plane fitting algorithms for tilt correction.

Provides methods for fitting a plane to a LOCAL WINDOW around a clicked point.
These are used for interactive tilt correction in the GUI.

For GLOBAL plane fitting to entire surfaces, see morphology.py module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

import logging
logger = logging.getLogger(__name__)


def fit_plane_local_least_squares(grid: np.ndarray, x: int, y: int, window_size: int = 100) -> tuple[float, float, float]:
    """Fits a plane to a local window of the grid using least squares regression.
    
    This function uses linear regression to fit a plane to the non-NaN values 
    in a square window of size (2*window_size+1) around the specified point.

    Args:
        grid (np.ndarray): The 2D array representing the grid data.
        x (int): The x-coordinate of the window center.
        y (int): The y-coordinate of the window center.
        window_size (int, optional): The half-size of the window. Defaults to 100.

    Returns:
        tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

    Raises:
        ValueError: If there are not enough valid data points to fit a plane.
    """
    h, w = grid.shape
    s = window_size
    xmin = max(0, x - s)
    xmax = min(w, x + s + 1)
    ymin = max(0, y - s)
    ymax = min(h, y + s + 1)

    window = grid[ymin:ymax, xmin:xmax]

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    zz = window

    # Convert to 1D and reject NaN
    X = xx.flatten()
    Y = yy.flatten()
    Z = zz.flatten()
    mask = ~np.isnan(Z)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    if len(Z) < 10:
        raise ValueError("Not enough valid data to fit the plane")

    A = np.vstack((X, Y)).T
    model = LinearRegression().fit(A, Z)
    a, b = model.coef_
    c = model.intercept_

    return a, b, c


def fit_plane_local_ransac(grid: np.ndarray, x: int, y: int, window_size: int = 100, 
                     residual_threshold: float = 200.0) -> tuple[float, float, float]:
    """Fits a plane to a local window of the grid using RANSAC robust regression.

    This function applies RANSAC regression to fit a plane to the non-NaN values 
    in a square window of size (2*window_size+1) around the specified point, 
    making it resistant to outliers.

    Args:
        grid (np.ndarray): The 2D array representing the grid data.
        x (int): The x-coordinate of the window center.
        y (int): The y-coordinate of the window center.
        window_size (int, optional): The half-size of the window. Defaults to 100.
        residual_threshold (float, optional): RANSAC residual threshold. Defaults to 200.0.

    Returns:
        tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

    Raises:
        ValueError: If there are not enough valid data points to fit a plane.
    """
    h, w = grid.shape
    s = window_size
    xmin = max(0, x - s)
    xmax = min(w, x + s + 1)
    ymin = max(0, y - s)
    ymax = min(h, y + s + 1)

    window = grid[ymin:ymax, xmin:xmax]
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    zz = window

    X = xx.flatten()
    Y = yy.flatten()
    Z = zz.flatten()
    mask = ~np.isnan(Z)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    if len(Z) < 10:
        raise ValueError("Not enough valid data to fit the plane")

    A = np.vstack((X, Y)).T

    # Use RANSAC to be robust against outliers
    base_model = LinearRegression()
    model = RANSACRegressor(
        base_model, 
        min_samples=min(10, len(Z)), 
        residual_threshold=residual_threshold, 
        random_state=42
    )
    model.fit(A, Z)
    a, b = model.estimator_.coef_
    c = model.estimator_.intercept_

    return a, b, c


def fit_plane_local_median_filter(grid: np.ndarray, x: int, y: int, window_size: int = 100, 
                            outlier_threshold: float = 300.0) -> tuple[float, float, float]:
    """Fits a plane to a local window of the grid using median filter for outlier removal.

    This function fits a plane to the non-NaN values in a square window of size 
    (2*window_size+1) around the specified point, excluding outliers based on 
    the median absolute deviation.

    Args:
        grid (np.ndarray): The 2D array representing the grid data.
        x (int): The x-coordinate of the window center.
        y (int): The y-coordinate of the window center.
        window_size (int, optional): The half-size of the window. Defaults to 100.
        outlier_threshold (float, optional): The threshold multiplier for outlier removal. Defaults to 300.0.

    Returns:
        tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

    Raises:
        ValueError: If there are not enough valid data points to fit a plane.
    """
    h, w = grid.shape
    s = window_size
    xmin = max(0, x - s)
    xmax = min(w, x + s + 1)
    ymin = max(0, y - s)
    ymax = min(h, y + s + 1)

    window = grid[ymin:ymax, xmin:xmax]
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    zz = window

    X = xx.flatten()
    Y = yy.flatten()
    Z = zz.flatten()
    mask = ~np.isnan(Z)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    if len(Z) < 10:
        raise ValueError("Not enough valid data to fit the plane")

    # Remove outliers based on median
    z_median = np.median(Z)
    mad = np.median(np.abs(Z - z_median))  # Median Absolute Deviation

    epsilon = 1e-8
    if mad < epsilon:
        # MAD is too small, fallback to std-based outlier detection or treat all as inliers
        std = np.std(Z)
        if std < epsilon:
            # Data is nearly constant, treat all as inliers
            robust_mask = np.ones_like(Z, dtype=bool)
        else:
            # Use standard deviation for outlier detection
            robust_mask = np.abs(Z - z_median) < 3 * std
    else:
        robust_mask = np.abs(Z - z_median) < outlier_threshold * mad
    
    X = X[robust_mask]
    Y = Y[robust_mask]
    Z = Z[robust_mask]

    if len(Z) < 10:
        raise ValueError("Too little data after outlier rejection")

    A = np.vstack((X, Y)).T
    model = LinearRegression().fit(A, Z)
    a, b = model.coef_
    c = model.intercept_

    return a, b, c
