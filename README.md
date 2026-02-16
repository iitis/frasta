# FRASTA-toolbox

FRASTA-toolbox is an open-source desktop application for fracture-surface topography analysis based on the FRASTA (Fracture Surface Topography Analysis) methodology. The software supports interactive import, preprocessing, alignment, and comparative analysis of opposing fracture surfaces represented as structured 3D grids.

The toolbox provides tools for masking, interpolation-based hole filling, manual surface alignment with live difference maps, and cross-sectional profile analysis with synchronized 2D and 3D visualization. It is designed to support reproducible fracture-surface analysis workflows and to translate established FRASTA concepts into a practical, accessible research tool.

FRASTA-toolbox is implemented in Python using PyQt5 and pyqtgraph, and is intended for use in materials science, fracture mechanics, tribology, biomedical engineering, and related research domains.

## Project Structure (NEW!)

- **Data structures**: `frasta/core/`
- **Analysis algorithms**: `frasta/processing/`
- **File I/O**: `frasta/io/`
- **GUI components**: `frasta/gui/`
- **Utilities**: `frasta/utils/`

## Input data format

FRASTA-toolbox currently supports structured grid data exported as text-based XYZ files, where each row corresponds to a single grid point (X, Y, Z). Additional internal formats (NPZ, HDF5) are supported for faster reload and reproducible workflows.

## Typical workflow (GUI)

1. Import one or more fracture-surface scans in CSV (XYZ grid) format.
2. Apply basic preprocessing:
   - define region of interest (ROI),
   - adjust value range using histogram thresholding,
   - fill missing data if necessary.
3. **Apply advanced processing** (NEW!):
   - Use **Processing** menu for filtering, leveling, transforms
   - Interactive dialogs guide parameter selection
   - See [GUI Integration Guide](docs/GUI_INTEGRATION.md) for details
4. Align two opposing fracture surfaces using interactive translation and rotation.
5. Place cross-sectional profiles to inspect local deviations and contacts.
6. Export aligned data, profiles, and measurements for further analysis or documentation.

## Advanced Processing (NEW!)

FRASTA-toolbox now includes advanced processing algorithms adapted from the EFS-toolbox project:

### 🔹 Advanced Filtering
- **Bilateral filter** - edge-preserving smoothing (preserves fracture edges)
- **Median filter** - robust outlier removal (removes measurement spikes)
- **Morphological operations** - opening/closing for structural processing
- **Robust Gaussian** - smoothing with iterative outlier rejection

### 🔹 Morphology & Leveling
- **Plane leveling** - remove tilt (least-squares or RANSAC-robust)
- **Polynomial form removal** - remove bending, warping, curvature (order 1-5)
- **Three-point leveling** - level by reference points
- **Thresholding** - value-based masking

### 🔹 Geometric Transformations
- **Rotation** - rotate surfaces with interpolation
- **Rescaling** - change resolution (upsampling/downsampling)
- **Cropping** - automatic crop to valid regions
- **Auto-registration** - automatic surface alignment (ICP, cross-correlation)

**Documentation:** 
- 📖 [Advanced Processing Guide](docs/ADVANCED_PROCESSING.md) - detailed API documentation
- �️ [GUI Integration Guide](docs/GUI_INTEGRATION.md) - using advanced processing in the GUI
- �📋 [Quick Reference](docs/QUICK_REFERENCE.md) - cheat sheet for all functions
- 💡 [Examples](examples/) - interactive demos and visualizations

**Quick Example:**
```python
from frasta.processing import bilateral_filter, level_by_plane, auto_register_surfaces

# Edge-preserving smoothing
smoothed = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, px_x=1.0, px_y=1.0)

# Remove tilt
leveled = level_by_plane(grid, method='robust')

# Automatic surface alignment
params = auto_register_surfaces(surface1, surface2, method='icp')
```

## Examples

The [`examples/`](examples/) directory contains ready-to-run demonstration scripts:

- **`advanced_processing.py`** - Interactive examples for all 16 processing functions
- **`visualization.py`** - Generate publication-quality comparison plots

**Run examples:**
```bash
python examples/advanced_processing.py    # Interactive demos
python examples/visualization.py          # Generate visualizations (saved to examples/output/)
```

See [examples/README.md](examples/README.md) for details.

## Configuration

### Windows

* create virtual environment:
`python -m venv .venv`

* activate:
`.venv\Scripts\activate.bat`

* instal packages:
`.venv\Scripts\pip.exe install -r requirements.txt`

* generating of requirements.txt:
`.venv\Scripts\pip.exe freeze > requirements.txt`

### Linux

* create virtual environment:
`python -m venv .venv`

* activate:
`sh .venv/bin/activate`

* instal packages:
`./.venv/bin/pip install -r requirements.txt`

* generating of requirements.txt:
`./.venv/bin/pip freeze > requirements.txt`

## Other useful commands:

* creating distribution package:
`./.venv/bin/python -m PyInstaller --add-data "icons;icons" main.py`

* running tests:
`./.venv/bin/python -m pytest -v -s`


