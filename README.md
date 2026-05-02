# FRASTA-toolbox

FRASTA-toolbox is an open-source desktop application for fracture-surface topography analysis based on the FRASTA (Fracture Surface Topography Analysis) approach. The software supports interactive data import, preprocessing, alignment, and comparative analysis of opposing fracture surfaces represented as structured height maps (regular 3D grids).

The toolbox provides tools for masking, interpolation-based hole filling, manual surface alignment with live difference maps, and cross-sectional profile analysis with synchronized 2D and 3D visualization. It is designed to support reproducible fracture-surface analysis workflows and to make established FRASTA concepts accessible as a practical research tool.

FRASTA-toolbox is implemented in Python using PyQt5 and pyqtgraph, and is intended for applications in materials science, fracture mechanics, tribology, biomedical engineering, and related research domains.

## Start here

If you are new to the project, use the documentation in this order:

1. Read the [Quick Start Guide](docs/QUICK_START_GUI.md) to launch the GUI and walk through the first demo workflows.
2. Return to this README for installation details, supported formats, and troubleshooting.
3. Use the [documentation index](docs/README.md) to choose more detailed user or developer guides.

## What the software does

FRASTA-toolbox helps you:

- import structured fracture-surface scans,
- clean and level the data,
- align opposing surfaces,
- inspect difference and contact maps,
- extract cross-sectional profiles,
- export results for later analysis or reporting.

## Who this README is for

This README is the main entry point for:

- first-time users who need installation and a quick orientation,
- returning users who need a reminder about supported data formats,
- contributors who need links to the detailed user and developer guides.

## Project structure

- **Data structures**: `frasta/core/`
- **Analysis algorithms**: `frasta/processing/`
- **File I/O**: `frasta/io/`
- **GUI components**: `frasta/gui/`
- **Utilities**: `frasta/utils/`

## Input data format

FRASTA-toolbox currently operates on structured grid data exported as text-based XYZ files, where each row corresponds to a single grid point (X, Y, Z). Additional internal formats (NPZ, HDF5) are supported for faster reload and reproducible workflows.

### Supported formats

- **CSV, TXT, DAT**: Text-based XYZ data. Each row stores one point as `X Y Z`, `X,Y,Z`, `X;Y;Z`, or tab-separated values. Coordinates are converted to micrometers at import according to the units selected by the user.
- **NPZ**: Compressed NumPy archive used for saving and reloading one or more gridded scans. Each scan stores `height`, `dx`, `dy`, `x0`, `y0`, and a scan name.
- **HDF5**: Hierarchical storage for one or more gridded scans. Each scan is stored in a `tab_XX` group with datasets for `name`, `height`, `dx`, `dy`, `x0`, and `y0`.
- **STL**: Mesh import/export support. On import, STL meshes are sampled into a regular height map; on export, valid grid cells are converted to a triangular mesh. STL stores triangles rather than the original measurement grid, so it should be treated as an exchange or visualization format. It may not preserve all information about missing measurement points or local sampling neighborhoods.

The main FRASTA workflow assumes regular height-map data. This matches the typical output of optical surface measurement systems used for fracture topography, which commonly export regularly sampled height maps. Unstructured point clouds or volumetric scans should first be converted to a structured grid before analysis. This conversion may introduce interpolation or smoothing effects that should be considered when interpreting results.

## Typical workflow (GUI)

The usual GUI workflow is:

1. Import one or more fracture-surface scans in CSV (XYZ grid) format.
2. Apply basic preprocessing:
   - Define a region of interest (ROI) from `Edit -> ROI settings...`.
   - Choose whether the ROI should stay shared across scans or become independent per scan.
   - Adjust ROI type, position, and size numerically in physical units, with the dialog offering the native unit and nearby smaller or larger units when they are meaningful.
   - Use `Tools -> Scan info...` whenever you need a quick summary of spacing, extents, valid-data coverage, and stored metadata for the active scan.
   - Adjust the value range using histogram thresholding.
   - Use the mouse wheel over the histogram to zoom its visible value range.
   - Drag the histogram background horizontally to pan the zoomed range.
   - Use the manual `Min` / `Max` fields beside the histogram for precise threshold values.
   - Use the `Hide below Min` and `Hide above Max` toggles to choose independently whether low and high out-of-range values should be masked or rendered with boundary colors.
   - Choose grayscale or a false-color palette directly from the main toolbar.
   - Fill missing data if necessary.
3. Apply advanced processing steps:
   - Use the **Processing** menu for filtering, leveling, and transforms.
   - The active ROI is used automatically by local processing and registration operations; geometric transforms still apply to the full grid.
   - In shared mode, all scans use the same ROI geometry. In per-scan mode, each tab keeps its own ROI position and size.
   - Interactive dialogs guide parameter selection.
   - Processed and registered results consistently offer a choice between replacing the current scan and creating a derived tab.
   - See the [GUI Integration Guide](docs/GUI_INTEGRATION.md) for details.
4. Align two opposing fracture surfaces using interactive translation and rotation.
   - The comparison window can prefill manual alignment with a fast ICP-based translation+rotation estimate.
5. Place cross-sectional profiles to inspect local deviations and contacts.
6. Export aligned data, profiles, and measurements for further analysis or documentation.

If you want a guided walkthrough with demo datasets, start with the
[Quick Start Guide](docs/QUICK_START_GUI.md).

## Advanced processing

FRASTA-toolbox includes advanced processing algorithms for surface preparation and alignment:

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
- **Auto-registration** - automatic surface alignment for height-map data (cross-correlation and ICP-based translation/rotation estimation)
  with optional slower ICP refinement for small, distinctive ROIs and optional automatic rejection of mismatched areas
- **Minimal roughness summaries** - basic amplitude parameters for grids
  (`Sa`, `Sq`, `Sz`) and extracted profiles (`Ra`, `Rq`, `Rz`) available from
  the Python processing API and from lightweight GUI summary dialogs

## Documentation map

Use these guides depending on what you need:

- [Quick Start Guide](docs/QUICK_START_GUI.md) - best starting point for first-time GUI users
- [Documentation Index](docs/README.md) - overview of the user and developer documentation
- [Advanced Processing Guide](docs/ADVANCED_PROCESSING.md) - detailed API documentation
- [GUI Integration Guide](docs/GUI_INTEGRATION.md) - using advanced processing in the GUI
- [Methods Overview](docs/METHODS.md) - computational workflow and assumptions
- [Quick Reference](docs/QUICK_REFERENCE.md) - cheat sheet for all functions
- [Examples](examples/) - interactive demos and visualizations

**Quick Example:**
```python
from frasta.processing import bilateral_filter, level_by_plane, auto_register_surfaces
from frasta.processing import surface_roughness_parameters, profile_roughness_parameters

# Edge-preserving smoothing
smoothed = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, px_x=1.0, px_y=1.0)

# Remove tilt
leveled = level_by_plane(grid, method='robust')

# Automatic surface alignment
params = auto_register_surfaces(surface1, surface2, method='icp')

# Basic roughness summaries
surface_metrics = surface_roughness_parameters(surface1)
profile_metrics = profile_roughness_parameters(profile_heights)
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

If you only want to get the GUI running, the fastest path is:

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run `python main.py` from the repository root.

The platform-specific commands are listed below.

### Windows

- Create a virtual environment:
  `python -m venv .venv`
- Activate it:
  `.venv\Scripts\activate.bat`
- Install packages:
  `.venv\Scripts\pip.exe install -r requirements.txt`
- Regenerate `requirements.txt` when needed:
  `.venv\Scripts\pip.exe freeze > requirements.txt`

### Linux and macOS

- Create a virtual environment:
  `python -m venv .venv`
- Activate it:
  `source .venv/bin/activate`
- Install packages:
  `./.venv/bin/pip install -r requirements.txt`
- Regenerate `requirements.txt` when needed:
  `./.venv/bin/pip freeze > requirements.txt`

## Other useful commands

- Create a distribution package on Windows:
  `.venv\Scripts\python.exe -m PyInstaller --add-data "icons;icons" main.py`
- Create a distribution package on Linux or macOS:
  `./.venv/bin/python -m PyInstaller --add-data "icons:icons" main.py`
- Run tests:
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
- Review the module structure in `frasta/core`, `frasta/io`, `frasta/processing`, and `frasta/gui` before adding new features.
- Keep processing logic in `frasta/processing` and keep GUI code focused on orchestration and presentation.
- Follow the patterns already used in nearby modules when extending loaders, processing functions, dialogs, or viewers.
