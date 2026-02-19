"""Quick test of STL downsampling."""
from frasta.io import save_stl
from frasta.core import Surface
import numpy as np
import tempfile
import os
import logging

# Configure logging to see downsampling messages
logging.basicConfig(level=logging.INFO)

print('Creating large grid (2878x3441)...')
s = Surface(height=np.random.rand(2878, 3441), dx=1.0, dy=1.0, x0=0.0, y0=0.0)
print(f'Grid shape: {s.height.shape}')
print(f'Total points: {s.height.shape[0] * s.height.shape[1]:,}')

with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
    fname = f.name

print(f'\nSaving to {fname}...')
print('With max_points=500000 (default)')
save_stl(fname, s, binary=True, max_points=500000)
size_mb = os.path.getsize(fname) / 1024 / 1024
print(f'✓ File saved: {size_mb:.2f} MB')

# Try with smaller max_points
print(f'\nTrying with max_points=100000...')
fname2 = fname.replace('.stl', '_small.stl')
save_stl(fname2, s, binary=True, max_points=100000)
size_mb2 = os.path.getsize(fname2) / 1024 / 1024
print(f'✓ File saved: {size_mb2:.2f} MB')

os.remove(fname)
os.remove(fname2)
print('\n✓ Test complete')
