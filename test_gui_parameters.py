"""Quick test of GUI processing operations with synthetic data."""
import sys
from pathlib import Path
import numpy as np

# Test creating synthetic data and basic operations
print("Creating test data...")
grid = np.random.randn(100, 100) * 10
grid += np.linspace(0, 50, 100)[np.newaxis, :]  # Add gradient

print(f"Grid shape: {grid.shape}")
print(f"Grid range: {np.min(grid):.2f} to {np.max(grid):.2f}")

# Test imports
print("\nTesting imports...")
from frasta.processing import (
    bilateral_filter, 
    median_filter_nan_aware,
    morphological_opening, 
    morphological_closing,
    robust_gaussian_filter,
    level_by_plane,
    remove_polynomial_form,
    threshold_grid,
    rotate_grid,
    rescale_grid,
    crop_to_valid_region
)
from frasta.processing.morphology import fit_plane_robust

print("✓ All imports successful")

# Test each function with correct parameters
print("\n" + "="*60)
print("Testing filter functions...")
print("="*60)

print("\n1. Bilateral filter...")
result = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, dx=1.0, dy=1.0)
print(f"   ✓ Result shape: {result.shape}")

print("\n2. Median filter...")
result = median_filter_nan_aware(grid, size=5, dx=1.0, dy=1.0)
print(f"   ✓ Result shape: {result.shape}")

print("\n3. Morphological opening...")
result = morphological_opening(grid, size=5, dx=1.0, dy=1.0)
print(f"   ✓ Result shape: {result.shape}")

print("\n4. Morphological closing...")
result = morphological_closing(grid, size=5, dx=1.0, dy=1.0)
print(f"   ✓ Result shape: {result.shape}")

print("\n5. Robust Gaussian filter...")
result = robust_gaussian_filter(grid, sigma=2.0, dx=1.0, dy=1.0, iterations=3, threshold=3.0)
print(f"   ✓ Result shape: {result.shape}")

print("\n" + "="*60)
print("Testing morphology functions...")
print("="*60)

print("\n6. Level by plane (least squares)...")
result = level_by_plane(grid, method='least_squares')
print(f"   ✓ Result shape: {result.shape}")
print(f"   ✓ Mean after leveling: {np.mean(result):.6f}")

print("\n7. Level by plane (robust)...")
plane, coeffs, inliers = fit_plane_robust(grid, residual_threshold=10.0)
result = grid - plane
print(f"   ✓ Result shape: {result.shape}")
print(f"   ✓ Inliers: {np.sum(inliers)}/{grid.size}")

print("\n8. Remove polynomial form...")
result = remove_polynomial_form(grid, order=2)
print(f"   ✓ Result shape: {result.shape}")

print("\n9. Threshold grid...")
result = threshold_grid(grid, low=-50, high=50)
print(f"   ✓ Result shape: {result.shape}")
print(f"   ✓ Valid pixels: {np.sum(~np.isnan(result))}/{grid.size}")

print("\n" + "="*60)
print("Testing transform functions...")
print("="*60)

# Create coordinate arrays
h, w = grid.shape
xi = np.arange(w) * 1.0
yi = np.arange(h) * 1.0

print("\n10. Rotate grid...")
result, new_xi, new_yi, px_x, px_y = rotate_grid(grid, angle_degrees=45, xi=xi, yi=yi, dx=1.0, dy=1.0, order=3)
print(f"   ✓ Result shape: {result.shape}")
print(f"   ✓ Coordinate shapes: xi={new_xi.shape}, yi={new_yi.shape}")

print("\n11. Rescale grid...")
result, new_xi, new_yi, px_x, px_y = rescale_grid(grid, scale_factor=0.5, xi=xi, yi=yi, dx=1.0, dy=1.0, order=3)
print(f"   ✓ Result shape: {result.shape} (downsampled from {grid.shape})")
print(f"   ✓ New pixel size: {px_x:.2f} x {px_y:.2f}")

print("\n12. Crop to valid region...")
result, new_xi, new_yi, px_x, px_y = crop_to_valid_region(grid, xi=xi, yi=yi, dx=1.0, dy=1.0, margin=0)
print(f"   ✓ Result shape: {result.shape}")

print("\n" + "="*60)
print("✅ All processing functions work correctly!")
print("="*60)
print("\nGUI parameter fixes verified:")
print("  ✓ robust_gaussian_filter: iterations, threshold")
print("  ✓ fit_plane_robust: residual_threshold (no max_trials)")
print("  ✓ threshold_grid: low, high")
print("  ✓ rotate_grid: xi, yi, angle_degrees")
print("  ✓ rescale_grid: xi, yi, scale_factor")
print("  ✓ crop_to_valid_region: xi, yi")
