"""Test downsampling with NaN values."""
from frasta.io import save_stl
from frasta.core import Surface
import numpy as np
import tempfile
import os
import logging

# Configure logging to see downsampling messages
logging.basicConfig(level=logging.INFO)

print('Test: Wpływ NaN na downsampling STL')
print('=' * 60)

# Create large grid with different NaN percentages
h, w = 2878, 3441
print(f'\nGrid size: {h}x{w} = {h*w:,} total points')
print()

for nan_percent in [0, 25, 50, 75]:
    print(f'\n--- Test z {nan_percent}% NaN ---')
    
    # Create grid with random data
    grid = np.random.rand(h, w)
    
    # Add NaN values
    if nan_percent > 0:
        nan_count = int(h * w * nan_percent / 100)
        nan_indices = np.random.choice(h * w, nan_count, replace=False)
        grid.flat[nan_indices] = np.nan
    
    valid_count = np.count_nonzero(~np.isnan(grid))
    print(f'Valid points: {valid_count:,} ({100 * valid_count / (h*w):.1f}%)')
    
    # Create Surface
    s = Surface(height=grid, dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    
    # Save to STL
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        fname = f.name
    
    save_stl(fname, s, binary=True, max_points=500000)
    size_mb = os.path.getsize(fname) / 1024 / 1024
    print(f'File size: {size_mb:.2f} MB')
    os.remove(fname)

print('\n' + '=' * 60)
print('Wnioski:')
print('- Im więcej NaN, tym mniejszy downsampling')
print('- Downsampling bazuje na liczbie VALID punktów, nie total')
print('- Jeśli 50% to NaN, nie ma downsamplingu dla 5M valid punktów')
