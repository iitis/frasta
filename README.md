# FRASTA-toolbox

FRASTA-toolbox is an open-source desktop application for fracture-surface topography analysis based on the FRASTA (Fracture Surface Topography Analysis) methodology. The software supports interactive import, preprocessing, alignment, and comparative analysis of opposing fracture surfaces represented as structured 3D grids.

The toolbox provides tools for masking, interpolation-based hole filling, manual surface alignment with live difference maps, and cross-sectional profile analysis with synchronized 2D and 3D visualization. It is designed to support reproducible fracture-surface analysis workflows and to translate established FRASTA concepts into a practical, accessible research tool.

FRASTA-toolbox is implemented in Python using PyQt5 and pyqtgraph, and is intended for use in materials science, fracture mechanics, tribology, biomedical engineering, and related research domains.

## Project structure

- **Data structures**: `frasta/core/`
- **Analysis algorithms**: `frasta/processing/`
- **File I/O**: `frasta/io/`
- **GUI components**: `frasta/gui/`
- **Utilities**: `frasta/utils/`

## Input data format

FRASTA-toolbox currently supports structured grid data exported as text-based XYZ files, where each row corresponds to a single grid point (X, Y, Z). Additional internal formats (NPZ, HDF5) are supported for faster reload and reproducible workflows.

### Supported formats

- **CSV, TXT, DAT**: Text-based XYZ data. Each row stores one point as `X Y Z`, `X,Y,Z`, `X;Y;Z`, or tab-separated values. Coordinates are converted to micrometers at import according to the units selected by the user.
- **NPZ**: Compressed NumPy archive used for saving and reloading one or more gridded scans. Each scan stores `height`, `dx`, `dy`, `x0`, `y0`, and a scan name.
- **HDF5**: Hierarchical storage for one or more gridded scans. Each scan is stored in a `tab_XX` group with datasets for `name`, `height`, `dx`, `dy`, `x0`, and `y0`.
- **STL**: Mesh import/export support. On import, STL meshes are sampled into a regular height map; on export, valid grid cells are converted to a triangular mesh.

The main FRASTA workflow assumes regular height-map data. Unstructured point clouds or volumetric scans should first be converted to a structured grid before analysis. This conversion may introduce interpolation or smoothing effects that should be considered when interpreting results.

## Typical workflow (GUI)

1. Import one or more fracture-surface scans in CSV (XYZ grid) format.
2. Apply basic preprocessing:
   - define region of interest (ROI),
   - adjust value range using histogram thresholding,
   - fill missing data if necessary.
3. Apply advanced processing:
   - Use **Processing** menu for filtering, leveling, transforms
   - Interactive dialogs guide parameter selection
   - See [GUI Integration Guide](docs/GUI_INTEGRATION.md) for details
4. Align two opposing fracture surfaces using interactive translation and rotation.
5. Place cross-sectional profiles to inspect local deviations and contacts.
6. Export aligned data, profiles, and measurements for further analysis or documentation.

## Advanced processing

FRASTA-toolbox now includes advanced processing algorithms adapted from the EFS-toolbox project:

### Advanced filtering
- **Bilateral filter** - edge-preserving smoothing (preserves fracture edges)
- **Median filter** - robust outlier removal (removes measurement spikes)
- **Morphological operations** - opening/closing for structural processing
- **Robust Gaussian** - smoothing with iterative outlier rejection

### Morphology and leveling
- **Plane leveling** - remove tilt (least-squares or RANSAC-robust)
- **Polynomial form removal** - remove bending, warping, curvature (order 1-5)
- **Three-point leveling** - level by reference points
- **Thresholding** - value-based masking

### Geometric transformations
- **Rotation** - rotate surfaces with interpolation
- **Rescaling** - change resolution (upsampling/downsampling)
- **Cropping** - automatic crop to valid regions
- **Auto-registration** - automatic surface alignment (ICP, cross-correlation)

**Documentation:**
- [Advanced Processing Guide](docs/ADVANCED_PROCESSING.md) - detailed API documentation
- [GUI Integration Guide](docs/GUI_INTEGRATION.md) - using advanced processing in the GUI
- [Methods Overview](docs/METHODS.md) - computational workflow and assumptions
- [Quick Reference](docs/QUICK_REFERENCE.md) - cheat sheet for all functions
- [Examples](examples/) - interactive demos and visualizations

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

## Requirements

FRASTA-toolbox is developed and tested primarily with Python 3.10 or newer. A standard desktop Python installation is sufficient for numerical processing and 2D views. The 3D views require an active desktop session with working OpenGL support.

Core dependencies are listed in `requirements.txt` and include PyQt5, pyqtgraph, NumPy/SciPy-related packages, h5py, scikit-image, scikit-learn, trimesh, PyOpenGL, and OpenCV.

### Operating systems

- **Windows**: Primary development environment. Use the Windows installation commands below.
- **Linux**: Supported when Qt and OpenGL desktop dependencies are available.
- **macOS**: Supported in principle with a local Python/Qt installation. Run the application from the repository root so that icons and other resources are resolved correctly.

### Hardware

- CPU: standard desktop or laptop CPU.
- RAM: depends on grid size; large scans require proportionally more memory.
- GPU/OpenGL: required for interactive 3D visualization. The core numerical processing does not require a dedicated GPU.

## Installation

### Windows

* create virtual environment:
`python -m venv .venv`

* activate:
`.venv\Scripts\activate.bat`

* install packages:
`.venv\Scripts\pip.exe install -r requirements.txt`

* generate `requirements.txt`:
`.venv\Scripts\pip.exe freeze > requirements.txt`

### Linux and macOS

* create virtual environment:
`python -m venv .venv`

* activate:
`sh .venv/bin/activate`

* install packages:
`./.venv/bin/pip install -r requirements.txt`

* generate `requirements.txt`:
`./.venv/bin/pip freeze > requirements.txt`

## Other useful commands:

* create distribution package on Windows:
`.venv\Scripts\python.exe -m PyInstaller --add-data "icons;icons" main.py`

* create distribution package on Linux or macOS:
`./.venv/bin/python -m PyInstaller --add-data "icons:icons" main.py`

* run tests:
`./.venv/bin/python -m pytest -v -s`

## Troubleshooting

### Icons or resources are missing

Run the application from the repository root:

```bash
python main.py
```

When packaging with PyInstaller, include the `icons` directory using the platform-specific `--add-data` syntax shown above.

### 3D views fail to open

Check that the system has a working OpenGL-capable desktop session. Remote, headless, or software-rendered sessions may not provide the OpenGL features required by `pyqtgraph.opengl`.

### Qt platform plugin errors

Recreate the virtual environment and reinstall dependencies from `requirements.txt`. On Linux, also check that the system Qt/X11 or Wayland libraries required by PyQt5 are installed.

### Large scans are slow

Large regular grids increase both memory use and processing time. Crop invalid borders, downsample where appropriate, and use NPZ or HDF5 for repeated loading instead of re-importing text XYZ files.

## Developer documentation

For contributors and developers:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design principles
- **[Coding conventions](agent-docs/conventions/README.md)** - Detailed coding standards by module
  - [Processing algorithms](agent-docs/conventions/processing/algorithms.md) - Required reading for algorithm development
  - [File I/O patterns](agent-docs/conventions/io/) - Loader/exporter conventions
  - [GUI development](agent-docs/conventions/gui/) - Dialog and widget patterns
  - [General standards](agent-docs/conventions/general/) - Naming, imports, logging
