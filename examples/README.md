# FRASTA Examples

This directory contains example scripts demonstrating the advanced processing capabilities of FRASTA-toolbox.

## Contents

### [`advanced_processing.py`](advanced_processing.py)
Interactive examples showing usage of advanced filtering, morphology, and transformation functions:
- Bilateral filtering (edge-preserving smoothing)
- Median filtering (spike removal)
- Plane leveling (tilt removal)
- Polynomial form removal (curvature correction)
- Surface rotation
- Automatic surface registration
- Grid rescaling
- Robust Gaussian filtering

**Run:**
```bash
python examples/advanced_processing.py
```

---

### [`visualization.py`](visualization.py)
Creates visual comparisons demonstrating the effects of different processing techniques. Generates publication-quality plots showing:
- Filter comparison (original vs bilateral vs median)
- Leveling and polynomial correction effects
- Automatic registration demonstration
- Grid rotation examples
- Edge preservation comparison (Gaussian vs Bilateral)

**Run:**
```bash
python examples/visualization.py
```

**Output:** Results are saved in [`output/`](output/) subdirectory as PNG files.

---

### [`data/synthetic_alicona_demo.al3d`](data/synthetic_alicona_demo.al3d)
Small synthetic Alicona AL3D file for parser and GUI smoke tests.

You can regenerate it with:
```bash
python examples/data/generate_synthetic_al3d.py
```

---

### Crack-path demo datasets
Deterministic NPZ datasets for the contact-map and crack-path workflow:
- `data/crack_path_straight_demo.npz`
- `data/crack_path_wavy_demo.npz`
- `data/crack_path_y_axis_demo.npz`
- `data/crack_path_realistic_demo.npz`

Each file has a companion JSON manifest with the recommended threshold,
propagation direction, front side, and expected tortuosity metrics. The
direction may be axis-aligned or described by an explicit in-plane angle.

Generate them with:
```bash
python examples/data/generate_crack_path_demo_data.py
```

---

## Quick Start

```bash
# From project root directory
cd examples

# Run interactive examples
python advanced_processing.py

# Generate visualizations
python visualization.py

# Check output
ls output/
```

---

## Generated Visualizations

After running `visualization.py`, you'll find in `output/`:

1. **`filter_comparison.png`** - Comparison of noise reduction techniques
2. **`leveling_comparison.png`** - Tilt and curvature removal
3. **`registration_demo.png`** - Automatic surface alignment (4-panel)
4. **`rotation_demo.png`** - Grid rotation at different angles
5. **`edge_preservation.png`** - Edge-preserving filtering demonstration

---

## Requirements

All examples require the standard FRASTA dependencies:
- numpy
- scipy
- scikit-learn
- matplotlib (for visualizations)

These are installed automatically with `pip install -r requirements.txt` from the project root.

---

## Documentation

For detailed documentation on the processing functions used in these examples, see:
- [Advanced Processing Guide](../docs/ADVANCED_PROCESSING.md)
- [Quick Reference](../docs/QUICK_REFERENCE.md)

---

## Extending Examples

Feel free to modify these scripts to test with your own data:

```python
# Load your data
from frasta.io.loaders import load_csv_data
grid, xi, yi, px_x, px_y = load_csv_data('your_scan.csv')

# Apply processing
from frasta.processing import bilateral_filter
filtered = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, 
                           px_x=px_x, px_y=px_y)
```

---

## Troubleshooting

**Import errors:** Make sure to run from examples directory or add parent to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Matplotlib backend issues:** If plots don't display, try:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

## License

Same as FRASTA-toolbox main project (see [LICENSE](../LICENSE)).
