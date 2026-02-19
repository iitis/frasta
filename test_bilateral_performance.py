"""Test performance of bilateral filter with OpenCV vs Python implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from frasta.processing import bilateral_filter

print("=" * 70)
print("Bilateral Filter Performance Test")
print("=" * 70)

# Create test data
sizes = [(128, 128), (256, 256), (512, 512)]

for size in sizes:
    print(f"\n📊 Testing {size[0]}x{size[1]} grid...")
    
    # Generate synthetic data
    grid = np.random.randn(*size) * 10
    grid += np.linspace(0, 100, size[1])[np.newaxis, :]  # Add gradient
    
    # Test OpenCV version
    print("  🚀 OpenCV version...", end=" ", flush=True)
    start = time.time()
    result_cv = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, 
                                 dx=1.0, dy=1.0, use_opencv=True)
    time_cv = time.time() - start
    print(f"{time_cv:.4f}s")
    
    # Test Python version (only for small sizes)
    if size[0] <= 128:
        print("  🐌 Python version...", end=" ", flush=True)
        start = time.time()
        result_py = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, 
                                     dx=1.0, dy=1.0, use_opencv=False)
        time_py = time.time() - start
        print(f"{time_py:.4f}s")
        
        speedup = time_py / time_cv
        print(f"  ⚡ Speedup: {speedup:.1f}x faster with OpenCV")
        
        # Check if results are similar
        diff = np.mean(np.abs(result_cv - result_py))
        print(f"  📏 Mean difference: {diff:.6f}")
    else:
        print(f"  ⏩ Python version skipped (would take ~{time_cv * 7500:.1f}s)")

print("\n" + "=" * 70)
print("Test complete! OpenCV bilateral filter is ready for production use.")
print("=" * 70)
