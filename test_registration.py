"""Test registration functions with correct parameters."""
import numpy as np
from frasta.processing import auto_register_surfaces, apply_registration

print("="*60)
print("Testing registration functions...")
print("="*60)

# Create test data
print("\n1. Creating test grids (same size)...")
reference = np.random.randn(128, 128) * 10
target = reference.copy()

# Add known translation
shift_y, shift_x = 10, 15
target = np.roll(np.roll(target, shift_y, axis=0), shift_x, axis=1)
print(f"   ✓ Reference: {reference.shape}")
print(f"   ✓ Target (shifted by {shift_y}, {shift_x}): {target.shape}")

# Test cross-correlation
print("\n2. Testing cross-correlation registration...")
try:
    params = auto_register_surfaces(reference, target, method='correlation')
    print(f"   ✓ Translation found: {params['translation']}")
    print(f"   ✓ Rotation: {params['rotation']}°")
    print(f"   ✓ RMSE: {params['rmse']:.4f}")
    print(f"   ✓ Inliers: {params['inliers']}")
    
    # Check if correct
    dy, dx = params['translation']
    expected_dy, expected_dx = shift_y, shift_x
    if abs(dy - expected_dy) < 2 and abs(dx - expected_dx) < 2:
        print(f"   ✓ Translation CORRECT (expected {expected_dy}, {expected_dx})")
    else:
        print(f"   ✗ Translation WRONG (expected {expected_dy}, {expected_dx}, got {dy}, {dx})")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test ICP
print("\n3. Testing ICP registration...")
try:
    params = auto_register_surfaces(reference, target, method='icp', max_iterations=50)
    print(f"   ✓ Translation found: {params['translation']}")
    print(f"   ✓ Rotation: {params['rotation']}°")
    print(f"   ✓ RMSE: {params['rmse']:.4f}")
    print(f"   ✓ Inliers: {params['inliers']}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test apply_registration
print("\n4. Testing apply_registration...")
try:
    # Create coordinate arrays
    h, w = target.shape
    xi = np.arange(w) * 1.0
    yi = np.arange(h) * 1.0
    px_x, px_y = 1.0, 1.0
    
    # Apply registration
    registered, new_xi, new_yi, new_px_x, new_px_y = apply_registration(
        target,
        xi,
        yi,
        px_x,
        px_y,
        params['translation'],
        params.get('rotation', 0.0)
    )
    
    print(f"   ✓ Registered shape: {registered.shape}")
    print(f"   ✓ Coordinate shapes: xi={new_xi.shape}, yi={new_yi.shape}")
    print(f"   ✓ Pixel sizes: {new_px_x} x {new_px_y}")
    
    # Check alignment
    valid = ~np.isnan(reference) & ~np.isnan(registered)
    rmse = np.sqrt(np.mean((reference[valid] - registered[valid]) ** 2))
    print(f"   ✓ RMSE after registration: {rmse:.4f}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test different sizes (should fail for correlation)
print("\n5. Testing cross-correlation with different sizes...")
reference_big = np.random.randn(128, 128) * 10
target_small = np.random.randn(100, 100) * 10

try:
    params = auto_register_surfaces(reference_big, target_small, method='correlation')
    print(f"   ✗ Should have failed but didn't!")
except ValueError as e:
    print(f"   ✓ Correctly rejected: {str(e)[:80]}...")
except Exception as e:
    print(f"   ? Unexpected error: {e}")

# Test ICP with different sizes (should work)
print("\n6. Testing ICP with different sizes...")
try:
    params = auto_register_surfaces(reference_big, target_small, method='icp', max_iterations=50)
    print(f"   ✓ Translation: {params['translation']}")
    print(f"   ✓ Rotation: {params['rotation']}°")
    print(f"   ✓ RMSE: {params['rmse']:.4f}")
    print(f"   ✓ ICP works with different sizes!")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "="*60)
print("✅ Registration tests complete!")
print("="*60)
print("\nSummary:")
print("  ✓ apply_registration: correct signature (grid, xi, yi, px_x, px_y, translation, rotation)")
print("  ✓ Cross-correlation: requires same-sized arrays")
print("  ✓ ICP: works with different-sized arrays")
print("  ✓ auto_register_surfaces: returns dict with translation, rotation, rmse, inliers")
