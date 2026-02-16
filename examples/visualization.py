"""
Visualization examples for advanced processing functions.

This script creates comparison plots showing the effects of different filters
and processing techniques. Results are saved in the output/ subdirectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frasta.processing import (
    bilateral_filter,
    median_filter_nan_aware,
    level_by_plane,
    remove_polynomial_form,
    rotate_grid,
    auto_register_surfaces,
)

# Create output directory for results
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_surface(size=200):
    """Create a test surface with features and noise."""
    x = np.linspace(0, 100, size)
    y = np.linspace(0, 100, size)
    X, Y = np.meshgrid(x, y)
    
    # Base topography
    Z = 10 * np.sin(X/20) * np.cos(Y/20) + 5 * np.sin(X/10)
    
    # Add noise
    Z += 2 * np.random.randn(size, size)
    
    # Add some spikes (measurement artifacts)
    for _ in range(10):
        i, j = np.random.randint(20, size-20, 2)
        Z[i, j] += np.random.choice([-30, 30])
    
    return Z, x, y


def demo_filters():
    """Compare different filtering techniques."""
    print("Creating filter comparison plots...")
    
    Z, x, y = create_test_surface()
    px = x[1] - x[0]
    
    # Apply different filters
    bilateral = bilateral_filter(Z, sigma_spatial=2.0, sigma_range=5.0, px_x=px, px_y=px)
    median = median_filter_nan_aware(Z, size=2.0, px_x=px, px_y=px)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = np.nanpercentile(Z, [5, 95])
    
    im1 = axes[0].imshow(Z, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original (with noise & spikes)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(bilateral, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Bilateral Filter\n(edges preserved)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    im3 = axes[2].imshow(median, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('Median Filter\n(spikes removed)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'filter_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def demo_leveling():
    """Demonstrate plane and polynomial leveling."""
    print("Creating leveling comparison plots...")
    
    # Create tilted and curved surface
    size = 200
    ny, nx = size, size
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_norm = (x_idx - nx/2) / (nx/2)
    y_norm = (y_idx - ny/2) / (ny/2)
    
    # Add tilt + curvature + roughness
    Z = (0.3 * x_idx + 0.2 * y_idx +          # tilt
         20 * (x_norm**2 + y_norm**2) +       # parabolic form
         3 * np.random.randn(ny, nx))         # roughness
    
    # Apply corrections
    leveled = level_by_plane(Z, method='least_squares')
    flattened = remove_polynomial_form(Z, order=2)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(Z, cmap='terrain')
    axes[0].set_title('Original\n(tilted + curved)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(leveled, cmap='terrain')
    axes[1].set_title(f'Plane Leveled\n(mean={np.nanmean(leveled):.3f})', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    im3 = axes[2].imshow(flattened, cmap='terrain')
    axes[2].set_title(f'Polynomial Corrected (order=2)\n(range reduced)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'leveling_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def demo_registration():
    """Demonstrate automatic surface registration."""
    print("Creating registration demonstration...")
    
    # Create reference surface
    x = np.linspace(0, 100, 150)
    y = np.linspace(0, 100, 150)
    X, Y = np.meshgrid(x, y)
    
    reference = 10 * np.sin(X/15) * np.cos(Y/15) + np.sin(X/8)
    
    # Create shifted version
    shift_y, shift_x = 10, 15
    target = np.roll(np.roll(reference, shift_y, axis=0), shift_x, axis=1)
    target += 1.0 * np.random.randn(*target.shape)  # add noise
    
    # Auto-register
    params = auto_register_surfaces(reference, target, method='correlation')
    detected_shift = params['translation']
    
    # Apply registration
    from frasta.processing import apply_registration
    px = x[1] - x[0]
    aligned, _, _, _, _ = apply_registration(
        target, x, y, px, px,
        translation=detected_shift
    )
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    im1 = axes[0, 0].imshow(reference, cmap='viridis')
    axes[0, 0].set_title('Reference Surface', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(target, cmap='viridis')
    axes[0, 1].set_title(f'Target (shifted by {shift_y}, {shift_x})', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[1, 0].imshow(aligned, cmap='viridis')
    axes[1, 0].set_title(f'After Auto-Registration\nDetected: {detected_shift}', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Difference map
    difference = np.abs(reference - aligned)
    im4 = axes[1, 1].imshow(difference, cmap='hot')
    axes[1, 1].set_title(f'Difference Map\nRMSE={params["rmse"]:.3f}', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'registration_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def demo_rotation():
    """Demonstrate grid rotation."""
    print("Creating rotation demonstration...")
    
    # Create anisotropic surface
    x = np.linspace(0, 100, 150)
    y = np.linspace(0, 100, 150)
    X, Y = np.meshgrid(x, y)
    
    # Linear gradient + noise (directional)
    Z = 0.5 * X + 0.1 * Y + 2 * np.random.randn(150, 150)
    
    px = x[1] - x[0]
    
    # Rotate
    rotated_45, _, _, _, _ = rotate_grid(Z, 45, x, y, px, px)
    rotated_90, _, _, _, _ = rotate_grid(Z, 90, x, y, px, px)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(Z, cmap='viridis')
    axes[0].set_title('Original (0°)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(rotated_45, cmap='viridis')
    axes[1].set_title('Rotated 45°', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    im3 = axes[2].imshow(rotated_90, cmap='viridis')
    axes[2].set_title('Rotated 90°', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'rotation_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def demo_edge_preservation():
    """Demonstrate edge preservation with bilateral filter."""
    print("Creating edge preservation comparison...")
    
    # Create surface with sharp edge
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    X, Y = np.meshgrid(x, y)
    
    # Create step function with noise
    Z = np.zeros((200, 200))
    Z[:, 100:] = 20  # step
    Z += 3 * np.random.randn(200, 200)  # noise
    
    px = x[1] - x[0]
    
    # Compare filters
    from frasta.processing import nan_aware_gaussian
    gaussian = nan_aware_gaussian(Z, sigma=2.0, mask=None)
    bilateral = bilateral_filter(Z, sigma_spatial=2.0, sigma_range=5.0, px_x=px, px_y=px)
    
    # Plot cross-sections
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2D views
    axes[0, 0].imshow(gaussian, cmap='viridis')
    axes[0, 0].set_title('Gaussian Filter\n(edge blurred)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bilateral, cmap='viridis')
    axes[0, 1].set_title('Bilateral Filter\n(edge preserved)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Cross-sections at row 100
    row = 100
    axes[1, 0].plot(Z[row, :], 'k-', alpha=0.3, label='Original (noisy)')
    axes[1, 0].plot(gaussian[row, :], 'b-', linewidth=2, label='Gaussian')
    axes[1, 0].set_title('Gaussian Filter - Cross Section', fontsize=12)
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Height')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(Z[row, :], 'k-', alpha=0.3, label='Original (noisy)')
    axes[1, 1].plot(bilateral[row, :], 'r-', linewidth=2, label='Bilateral')
    axes[1, 1].set_title('Bilateral Filter - Cross Section', fontsize=12)
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Height')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'edge_preservation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("FRASTA Advanced Processing - Visual Demonstrations")
    print("=" * 60)
    print()
    
    demo_filters()
    demo_leveling()
    demo_registration()
    demo_rotation()
    demo_edge_preservation()
    
    print()
    print("=" * 60)
    print("All visualizations completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - filter_comparison.png")
    print("  - leveling_comparison.png")
    print("  - registration_demo.png")
    print("  - rotation_demo.png")
    print("  - edge_preservation.png")
    print("=" * 60)
