# Computational Methods

This document summarizes the computational assumptions used by FRASTA-toolbox.
It is intended to make the implemented workflow easier to inspect and reproduce.

## Data Model

FRASTA-toolbox represents a measured fracture surface as a regular two-dimensional
height map. Internally, each scan is stored as a `Surface` object containing:

- `height`: a 2D NumPy array of height values.
- `dx`, `dy`: pixel spacing in the X and Y directions.
- `x0`, `y0`: physical coordinates of the first grid point.
- `unit`: internal unit, currently micrometers.
- optional metadata and display limits.

Missing or invalid data are represented by `NaN`. Processing functions are
expected to preserve or explicitly handle these values.

## Input Gridding

Text-based XYZ files are imported by reading points with Cartesian coordinates
`X`, `Y`, and `Z`. The loader estimates the regular grid spacing from unique
coordinate values and maps each point to the nearest grid index. If multiple
points map to the same cell, their height values are averaged.

The import dialog lets the user specify whether XY and Z values are in
millimeters or micrometers. Values stored internally are converted to
micrometers. STL files are handled differently: the mesh is sampled from above
onto a regular XY grid and converted into a height map.

## Missing Data Interpolation

Missing values can be filled with nearest-neighbor interpolation. The current
hole-filling routine identifies `NaN` cells, optionally restricts the operation
to a user-specified mask, and interpolates from surrounding valid grid points.

This step is intended for small holes and measurement gaps. Interpolating large
regions or converting unstructured point clouds to a grid may smooth local
features and should be considered during interpretation.

## Preprocessing

Preprocessing tools include masking, histogram-based value selection,
thresholding, filtering, leveling, and geometric transformations.

Value thresholding sets values below or above user-defined limits to `NaN`.
Plane leveling subtracts a fitted plane from the height map. Robust leveling
uses RANSAC to reduce the influence of outliers. Polynomial form removal fits
and subtracts a polynomial background of selected order.

Filtering operations are implemented as pure processing functions over NumPy
arrays. They do not modify input arrays in place.

## Surface Alignment

FRASTA-toolbox supports interactive alignment and automatic registration.

Interactive alignment applies user-controlled translation and in-plane rotation
to one surface while showing a live difference map. This makes the manual
choice of alignment parameters visible and repeatable: once the same parameters
are used on the same input grids, the transformation is deterministic.

Automatic registration is available in two modes:

- **Cross-correlation** estimates translation between same-sized grids. It
  subtracts mean height values, fills invalid regions for the correlation step,
  finds the correlation peak, and applies a subpixel peak refinement.
- **ICP mode** estimates translation and in-plane rotation from valid points.
  The implementation is a simplified registration routine for height-map data,
  not a general-purpose full 3D point-cloud ICP solver.

The resulting parameters include translation, rotation, RMSE, and the number of
overlapping or matching points used in the estimate.

## Difference Map

After two surfaces are brought into a common grid frame, the difference map is
computed as the point-wise height difference:

```text
D(x, y) = H_ref(x, y) - H_adj(x, y)
```

where `H_ref` is the reference surface and `H_adj` is the adjusted surface after
alignment. Cells where either surface is invalid are excluded or displayed as
invalid values depending on the view.

The difference map is used as visual feedback during alignment and as a basis
for later profile and contact analysis.

## Contact Map

The profile-analysis tool computes a binary contact map from the difference
between the reference surface and the adjusted surface after applying a vertical
separation parameter:

```text
D_sep(x, y) = H_ref(x, y) - (H_adj(x, y) + separation)
```

Cells are marked as contact where `D_sep > 0` and both input values are valid.
The separation parameter is controlled by the user and should be interpreted as
a modeling or inspection threshold rather than an automatically inferred
material property.

For the current view, the software can report contact area and an approximate
volume measure based on the selected binary region and the pixel area.

## Cross-Sectional Profiles

Users can place section lines across aligned surfaces. The software samples
both surfaces along the selected line, plots the corresponding profiles, and can
apply optional tilt and offset correction. Local line fits are used to report
profile angles and angular differences between the surfaces.

Profile data, selected points, contact maps, and related analysis metadata can
be exported for later inspection.

## Reproducibility

For a fixed input dataset and fixed user-defined parameters, the implemented
processing operations are deterministic. This includes thresholding, grid
transformations, difference-map computation, and contact-map generation.

Manual operations affect reproducibility through the selected parameters. The
recommended practice is to save processed datasets and exported profile
analysis metadata after each completed workflow.

## Limitations

- The primary data model is a regular height map. Native unstructured
  point-cloud and volumetric workflows are not implemented.
- Interpolation can smooth local features when missing regions are large or
  point-cloud data are converted to a regular grid.
- Contact detection is threshold-based and depends on user-selected separation
  and alignment parameters.
- Automatic registration complements, but does not fully replace, expert visual
  inspection of fracture-surface alignment.
