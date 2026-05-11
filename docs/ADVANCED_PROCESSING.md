# Advanced Processing in FRASTA-toolbox

This guide summarizes the advanced-processing functions available in
`advanced_filtering`, `morphology`, and `transforms`. These routines extend the
core FRASTA workflow with additional filtering, leveling, geometric correction,
and automatic registration helpers.

## Modules

### 1. Advanced filtering (`advanced_filtering.py`)

#### Bilateral filter - `bilateral_filter()`

Preserves edges while smoothing local noise.

```python
from frasta.processing import bilateral_filter

filtered = bilateral_filter(
    grid,
    sigma_spatial=5.0,  # spatial smoothing scale in physical units
    sigma_range=10.0,   # tolerated height difference
    px_x=grid_data.px_x,
    px_y=grid_data.px_y
)
```

**Use cases:**
- Noise reduction while preserving sharp fracture edges
- Preprocessing before feature extraction
- Situations where standard Gaussian smoothing is too aggressive

#### Median filter - `median_filter_nan_aware()`

Robust removal of isolated outliers and spikes.

```python
from frasta.processing import median_filter_nan_aware

filtered = median_filter_nan_aware(
    grid,
    size=10.0,  # kernel size in physical units
    px_x=1.0,
    px_y=1.0
)
```

**Use cases:**
- Removing measurement spikes
- Reducing noise while preserving edges reasonably well
- Preprocessing before roughness or fractal analysis

#### Morphological opening and closing

```python
from frasta.processing import morphological_opening, morphological_closing

opened = morphological_opening(grid, size=5.0, px_x=1.0)
closed = morphological_closing(grid, size=5.0, px_x=1.0)
```

**Use cases:**
- Removing small positive artifacts
- Filling small valleys or gaps
- Structural cleanup before later processing

#### Robust Gaussian filter - `robust_gaussian_filter()`

Gaussian-like smoothing with iterative outlier rejection.

```python
from frasta.processing import robust_gaussian_filter

filtered = robust_gaussian_filter(
    grid,
    sigma=10.0,
    px_x=1.0,
    iterations=3,
    threshold=3.0
)
```

**Use cases:**
- Smoothing data that contains outliers
- More robust alternative to standard Gaussian smoothing
- Cases where median filtering changes the structure too much

---

### 2. Morphology and geometric correction (`morphology.py`)

#### Plane leveling - `level_by_plane()`

Removes overall sample tilt.

```python
from frasta.processing import level_by_plane

leveled = level_by_plane(grid, method='least_squares')
leveled_robust = level_by_plane(grid, method='robust')
```

**Use cases:**
- Correcting specimen tilt
- Removing global linear slope
- Preparing data for roughness or profile analysis

#### Polynomial form removal - `remove_polynomial_form()`

Removes curvature, warping, or other smooth background form.

```python
from frasta.processing import remove_polynomial_form

corrected = remove_polynomial_form(grid, order=2)
corrected_cubic = remove_polynomial_form(grid, order=3)
```

**Parameter `order`:**
- `order=1`: plane; similar to tilt removal
- `order=2`: quadratic form; typical for bending or warping
- `order=3`: cubic form; use for more complex trends
- `order=4-5`: use with care; may remove meaningful structure

**Note:** These operations are useful as compact form-removal tools, but they
should not be interpreted as a complete ISO-compliant surface-metrology
workflow on their own.

#### Three-point leveling - `level_by_three_points()`

```python
from frasta.processing import level_by_three_points

leveled = level_by_three_points(
    grid,
    p1=(0, 0),
    p2=(100, 0),
    p3=(50, 50),
    xi=xi,
    yi=yi
)
```

**Use case:** Useful when specific reference points are known in advance.

#### Thresholding - `threshold_grid()`

Masks values outside user-selected bounds.

```python
from frasta.processing import threshold_grid

mean = np.nanmean(grid)
std = np.nanstd(grid)
filtered = threshold_grid(grid, low=mean - 3 * std, high=mean + 3 * std)
```

---

### 3. Geometric transforms (`transforms.py`)

#### Rotation - `rotate_grid()`

```python
from frasta.processing import rotate_grid

rotated, xi, yi, px_x, px_y = rotate_grid(
    grid,
    angle_degrees=45,
    xi=xi,
    yi=yi,
    px_x=px_x,
    px_y=px_y,
    order=3  # 0=nearest, 1=linear, 3=cubic
)
```

**Use cases:**
- Reorienting surfaces before comparison
- Matching scan orientation between datasets

#### Rescaling - `rescale_grid()`

```python
from frasta.processing import rescale_grid

high_res, xi, yi, px_x, px_y = rescale_grid(grid, 2.0, xi, yi, px_x, px_y)
low_res, xi, yi, px_x, px_y = rescale_grid(grid, 0.5, xi, yi, px_x, px_y)
```

**Use cases:**
- Matching resolution before comparison
- Upsampling for visualization
- Downsampling for faster computation

#### Cropping - `crop_to_valid_region()`

```python
from frasta.processing import crop_to_valid_region

cropped, xi, yi, px_x, px_y = crop_to_valid_region(
    grid, xi, yi, px_x, px_y, margin=10
)
```

**Use cases:**
- Removing empty borders
- Reducing data size before later processing

#### Automatic registration - `auto_register_surfaces()`

Automatically estimates relative translation and, in ICP mode, in-plane
rotation between two surfaces.

```python
from frasta.processing import auto_register_surfaces, apply_registration

params = auto_register_surfaces(
    reference_grid,
    target_grid,
    method='correlation'  # or 'icp'
)

print(f"Translation: {params['translation']}")
print(f"Rotation: {params['rotation']} deg")
print(f"RMSE: {params['rmse']}")

aligned, xi, yi, px_x, px_y = apply_registration(
    target_grid, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)
```

**Methods:**
- `'correlation'`: cross-correlation; translation only; fast
- `'icp'`: simplified ICP-style routine; translation plus rotation; slower

**ICP options:**
- `refine=True`: slower final refinement using height RMSE
- `stable_region=True`: second ICP pass on an automatically selected
  low-mismatch overlap region

**Use cases:**
- Aligning opposing fracture surfaces
- Estimating scan-to-scan offsets before manual inspection

---

## Typical Workflows

### Workflow 1: Cleaning raw data

```python
from frasta.processing import (
    median_filter_nan_aware,
    level_by_plane,
    threshold_grid
)

cleaned = median_filter_nan_aware(raw_grid, size=5.0, px_x=px_x)
leveled = level_by_plane(cleaned, method='robust')
mean, std = np.nanmean(leveled), np.nanstd(leveled)
final = threshold_grid(leveled, low=mean - 3 * std, high=mean + 3 * std)
```

### Workflow 2: Removing geometric form

```python
from frasta.processing import level_by_plane, remove_polynomial_form

leveled = level_by_plane(grid)
flattened = remove_polynomial_form(leveled, order=2)
```

### Workflow 3: Edge-preserving denoising

```python
from frasta.processing import bilateral_filter

smoothed = bilateral_filter(
    grid,
    sigma_spatial=5.0,
    sigma_range=10.0,
    px_x=px_x,
    px_y=px_y
)
```

### Workflow 4: Automatic surface alignment

```python
from frasta.processing import (
    auto_register_surfaces,
    apply_registration,
    remove_relative_offset
)

params = auto_register_surfaces(surf1, surf2, method='icp')

aligned = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)

aligned_corrected = remove_relative_offset(surf1, aligned[0], mask)
```

---

## Comparison with Basic Routines

| Task | Basic routine | Advanced routine | Main advantage |
|------|---------------|------------------|----------------|
| Smoothing | `nan_aware_gaussian` | `bilateral_filter` | Better edge preservation |
| Outlier removal | `remove_outliers` | `median_filter_nan_aware` | More robust to spikes |
| Leveling | `remove_relative_tilt` | `level_by_plane` + `remove_polynomial_form` | More correction options |
| Alignment | Manual translation/rotation | `auto_register_surfaces` | Automated parameter proposal |

---

## Cautions

1. **Bilateral filter performance**: the Python implementation is slower than
   the other routines. For large datasets, consider downsampling first.
2. **Polynomial order selection**:
   - Too high an order may remove real surface features.
   - `order=2` is a good default in many cases.
   - `order=3` should be used only when the surface clearly warrants it.
3. **ICP registration**:
   - Requires reasonably overlapping regions.
   - For large initial mismatch, start with `'correlation'`.
   - Runtime increases on larger surfaces even with internal subsampling.
4. **Interpolation order**:
   - `order=0` (nearest): fastest, least smooth
   - `order=1` (linear): good compromise
   - `order=3` (cubic): smoothest, but may introduce artifacts around `NaN`

---

## Source Context

Some of these routines were adapted from work previously developed in the
EFS-toolbox project and then integrated into FRASTA-toolbox in a form
appropriate for the current grid-based workflow.

---

## Example Script

See the complete working example:

[`examples/advanced_processing.py`](../examples/advanced_processing.py)

```bash
python examples/advanced_processing.py
```
