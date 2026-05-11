"""
Examples of using advanced processing functions in FRASTA.

This file demonstrates the new filtering, morphology, and transformation
functions added from the EFS-toolbox project.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frasta.core import Surface
from frasta.processing import (
    # Advanced filtering
    bilateral_filter,
    median_filter_nan_aware,
    morphological_opening,
    morphological_closing,
    robust_gaussian_filter,
    # Morphology and leveling
    fit_plane_least_squares,
    fit_plane_robust,
    level_by_plane,
    remove_polynomial_form,
    threshold_grid,
    # Transforms
    rotate_grid,
    rescale_grid,
    crop_to_valid_region,
    auto_register_surfaces,
    apply_registration,
)


def example_bilateral_filtering():
    """Example: Edge-preserving smoothing with bilateral filter."""
    print("=== Bilateral Filter Example ===")
    
    # For demo, create synthetic data
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/10) * np.cos(Y/10) + 0.5 * np.random.randn(200, 200)
    
    grid_data = Surface(height=Z, dx=0.5, dy=0.5)
    
    # Apply bilateral filter - smooths noise but preserves edges
    filtered = bilateral_filter(
        grid_data.height,
        sigma_spatial=5.0,  # spatial smoothing scale (in physical units)
        sigma_range=10.0,   # height difference tolerance
        dx=grid_data.dx,
        dy=grid_data.dy
    )
    
    # Create new Surface with filtered result
    filtered_data = Surface(height=filtered, dx=grid_data.dx, dy=grid_data.dy)
    
    print(f"Original noise: {np.nanstd(grid_data.height):.3f}")
    print(f"Filtered noise: {np.nanstd(filtered):.3f}")
    print("Bilateral filter preserves edges while smoothing!")


def example_median_filter():
    """Example: Removing measurement spikes with median filter."""
    print("\n=== Median Filter Example ===")
    
    # Create data with spikes
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    Z = 5 * np.random.randn(200, 200)
    
    # Add measurement spikes
    Z[50, 50] = 100
    Z[100, 100] = -100
    Z[150, 150] = 80
    
    grid_data = Surface(height=Z, dx=0.5, dy=0.5)
    
    # Apply median filter - robust to outliers
    filtered = median_filter_nan_aware(
        grid_data.height,
        size=2.0,  # kernel size in physical units
        dx=grid_data.dx,
        dy=grid_data.dy
    )
    
    print(f"Spike at (50,50): {grid_data.height[50,50]:.1f} -> {filtered[50,50]:.1f}")
    print(f"Spike at (100,100): {grid_data.height[100,100]:.1f} -> {filtered[100,100]:.1f}")
    print("Spikes removed!")


def example_plane_leveling():
    """Example: Remove tilt/plane from surface."""
    print("\n=== Plane Leveling Example ===")
    
    # Create tilted surface
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    Z = 0.1 * x_idx + 0.05 * y_idx + 2 * np.random.randn(ny, nx)
    
    x = np.arange(nx) * 0.5
    y = np.arange(ny) * 0.5
    grid_data = Surface(height=Z, dx=0.5, dy=0.5)
    
    print(f"Before leveling - mean: {np.nanmean(grid_data.height):.3f}")
    
    # Remove plane (tilt)
    leveled = level_by_plane(grid_data.height, method='least_squares')
    
    print(f"After leveling - mean: {np.nanmean(leveled):.3f}")
    print("Tilt removed!")


def example_polynomial_correction():
    """Example: Remove curved form (bending, warping)."""
    print("\n=== Polynomial Form Removal Example ===")
    
    # Create surface with quadratic bending
    ny, nx = 200, 200
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_norm = (x_idx - nx/2) / (nx/2)
    y_norm = (y_idx - ny/2) / (ny/2)
    
    # Parabolic form + noise
    Z = 5 * (x_norm**2 + y_norm**2) + 0.5 * np.random.randn(ny, nx)
    
    x = np.arange(nx) * 0.5
    y = np.arange(ny) * 0.5
    grid_data = Surface(height=Z, dx=0.5, dy=0.5)
    
    print(f"Before correction - range: {np.nanmax(grid_data.height) - np.nanmin(grid_data.height):.3f}")
    
    # Remove quadratic form
    corrected = remove_polynomial_form(grid_data.height, order=2)
    
    print(f"After correction - range: {np.nanmax(corrected) - np.nanmin(corrected):.3f}")
    print("Curved form removed!")


def example_surface_rotation():
    """Example: Rotate surface."""
    print("\n=== Surface Rotation Example ===")
    
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 0.5 * Y  # Linear trend
    
    grid_data = Surface(height=Z, dx=1.0, dy=1.0)
    
    # Rotate by 45 degrees
    rotated, xi, yi, px_x, px_y = rotate_grid(
        grid_data.height, 45, grid_data.xi, grid_data.yi,
        grid_data.dx, grid_data.dy
    )
    
    print(f"Original shape: {grid_data.height.shape}")
    print(f"Rotated shape: {rotated.shape}")
    print("Surface rotated by 45°!")


def example_auto_registration():
    """Example: Automatically align two fracture surfaces."""
    print("\n=== Automatic Surface Registration Example ===")
    
    # Create reference surface
    x = np.linspace(0, 100, 150)
    y = np.linspace(0, 100, 150)
    X, Y = np.meshgrid(x, y)
    reference = np.sin(X/15) * np.cos(Y/15)
    
    # Create shifted + rotated target
    target = np.roll(np.roll(reference, 10, axis=0), 5, axis=1)
    target = target + 0.2 * np.random.randn(*target.shape)
    
    # Automatically find alignment
    params = auto_register_surfaces(reference, target, method='correlation')
    
    print(f"Detected translation: {params['translation']}")
    print(f"Registration RMSE: {params['rmse']:.4f}")
    print(f"Inlier points: {params['inliers']}")
    
    # Apply registration
    grid_data = Surface(height=target, dx=x[1]-x[0], dy=y[1]-y[0])
    aligned, xi, yi, px_x, px_y = apply_registration(
        grid_data.height, grid_data.xi, grid_data.yi,
        grid_data.dx, grid_data.dy,
        translation=params['translation'],
        rotation=params['rotation']
    )
    
    print("Surfaces automatically aligned!")


def example_rescaling():
    """Example: Change surface resolution."""
    print("\n=== Grid Rescaling Example ===")
    
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    Z = np.random.randn(100, 100)
    
    grid_data = Surface(height=Z, dx=1.0, dy=1.0)
    
    print(f"Original: {grid_data.height.shape}, pixel size: {grid_data.dx:.2f}")
    
    # Double resolution
    high_res, xi, yi, px_x, px_y = rescale_grid(
        grid_data.height, 2.0, grid_data.xi, grid_data.yi,
        grid_data.dx, grid_data.dy
    )
    
    print(f"High-res: {high_res.shape}, pixel size: {px_x:.2f}")
    
    # Half resolution
    low_res, xi, yi, px_x, px_y = rescale_grid(
        grid_data.height, 0.5, grid_data.xi, grid_data.yi,
        grid_data.dx, grid_data.dy
    )
    
    print(f"Low-res: {low_res.shape}, pixel size: {px_x:.2f}")


def example_robust_filtering():
    """Example: Robust Gaussian with outlier rejection."""
    print("\n=== Robust Gaussian Filter Example ===")
    
    # Create data with outliers
    x = np.linspace(0, 100, 150)
    y = np.linspace(0, 100, 150)
    Z = 2 * np.random.randn(150, 150)
    
    # Add strong outliers
    for _ in range(20):
        i, j = np.random.randint(0, 150, 2)
        Z[i, j] = np.random.choice([-50, 50])
    
    grid_data = Surface(height=Z, dx=x[1]-x[0], dy=y[1]-y[0])
    
    # Regular Gaussian would be affected by outliers
    # Robust Gaussian iteratively excludes them
    filtered = robust_gaussian_filter(
        grid_data.height,
        sigma=5.0,
        dx=grid_data.dx,
        dy=grid_data.dy,
        iterations=3,
        threshold=3.0
    )
    
    print(f"Original std: {np.nanstd(grid_data.height):.3f}")
    print(f"Filtered std: {np.nanstd(filtered):.3f}")
    print("Outliers rejected during smoothing!")


if __name__ == '__main__':
    print("FRASTA Advanced Processing Examples")
    print("=" * 50)
    
    example_bilateral_filtering()
    example_median_filter()
    example_plane_leveling()
    example_polynomial_correction()
    example_surface_rotation()
    example_auto_registration()
    example_rescaling()
    example_robust_filtering()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
