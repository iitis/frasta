# FRASTA Advanced Processing - Quick Reference

This cheat sheet summarizes the most commonly used advanced-processing
functions in FRASTA-toolbox.

Use it as a quick reminder after you already know the workflow. If you are
launching the software for the first time, start with
[`QUICK_START_GUI.md`](QUICK_START_GUI.md) instead.

## Filtering

```python
from frasta.processing import (
    bilateral_filter,           # edge-preserving smoothing
    median_filter_nan_aware,    # outlier removal
    robust_gaussian_filter,     # robust smoothing
    morphological_opening,      # remove peaks
    morphological_closing       # fill valleys
)

# Edge-preserving smoothing; useful for fracture surfaces
smoothed = bilateral_filter(
    grid, sigma_spatial=5.0, sigma_range=10.0, px_x=px_x, px_y=px_y
)

# Remove isolated spikes
cleaned = median_filter_nan_aware(grid, size=5.0, px_x=px_x, px_y=px_y)

# Robust Gaussian smoothing with iterative outlier rejection
filtered = robust_gaussian_filter(
    grid, sigma=10.0, px_x=px_x, iterations=3, threshold=3.0
)
```

---

## Leveling and Corrections

```python
from frasta.processing import (
    level_by_plane,          # remove tilt
    remove_polynomial_form,  # remove curvature
    threshold_grid           # value masking
)

# Remove tilt
leveled = level_by_plane(grid, method='least_squares')  # fast
leveled = level_by_plane(grid, method='robust')         # more outlier-resistant

# Remove curvature or warping
flattened = remove_polynomial_form(grid, order=2)  # quadratic
flattened = remove_polynomial_form(grid, order=3)  # cubic

# Threshold outliers
mean, std = np.nanmean(grid), np.nanstd(grid)
filtered = threshold_grid(grid, low=mean - 3 * std, high=mean + 3 * std)
```

---

## Geometric Transforms

```python
from frasta.processing import (
    rotate_grid,             # rotate
    rescale_grid,            # change resolution
    crop_to_valid_region,    # crop
    auto_register_surfaces,  # auto-align
    apply_registration       # apply transform
)

# Rotate
rotated, xi, yi, px_x, px_y = rotate_grid(grid, 45, xi, yi, px_x, px_y)

# Change resolution
high_res, xi, yi, px_x, px_y = rescale_grid(grid, 2.0, xi, yi, px_x, px_y)
low_res, xi, yi, px_x, px_y = rescale_grid(grid, 0.5, xi, yi, px_x, px_y)

# Crop to valid data
cropped, xi, yi, px_x, px_y = crop_to_valid_region(grid, xi, yi, px_x, px_y)

# Automatic alignment
params = auto_register_surfaces(surf1, surf2, method='correlation')
aligned, xi, yi, px_x, px_y = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)
```

---

## Typical Scenarios

### Scenario 1: Cleaning raw data

```python
# Pipeline: median -> level -> threshold
cleaned = median_filter_nan_aware(raw, size=5.0, px_x=px_x)
leveled = level_by_plane(cleaned, method='robust')
mean, std = np.nanmean(leveled), np.nanstd(leveled)
final = threshold_grid(leveled, low=mean - 3 * std, high=mean + 3 * std)
```

### Scenario 2: Preprocessing for roughness analysis

```python
# Remove geometric form while preserving roughness-scale variations
leveled = level_by_plane(grid)
flattened = remove_polynomial_form(leveled, order=2)
# Roughness descriptors such as Sa and Sq can be computed afterwards
```

### Scenario 3: Edge-preserving smoothing

```python
# Prefer bilateral filtering to standard Gaussian smoothing on fracture surfaces
smoothed = bilateral_filter(
    grid, sigma_spatial=5.0, sigma_range=10.0, px_x=px_x, px_y=px_y
)
```

### Scenario 4: Automatic alignment of two surfaces

```python
# 1. Estimate alignment parameters
params = auto_register_surfaces(surf1, surf2, method='correlation')

# 2. Apply the transform
aligned, xi, yi, px_x, px_y = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation']
)

# 3. Add ICP refinement if needed
params_fine = auto_register_surfaces(surf1, aligned, method='icp')
```

---

## Parameter Hints

### Bilateral filter
- `sigma_spatial`: approximately 5-10x the pixel size
- `sigma_range`: approximately 1-2x the noise level
- Smaller `sigma_range` values preserve sharper edges

### Median filter
- `size`: approximately 3-5x the pixel size for spike removal
- Increase `size` when noise spikes are larger

### Polynomial removal
- `order=1`: tilt only; roughly equivalent to plane leveling
- `order=2`: standard correction for bending or warping
- `order=3`: use only when the surface clearly shows more complex curvature
- `order>3`: rarely needed; may remove real surface features

### Auto-registration methods
- `'correlation'`: translation only; fast; good for initial alignment
- `'icp'`: translation plus rotation; slower; usually more precise

---

## Practical Tips

1. Always inspect the result visually; automated routines are helpful but should
   be treated as aids rather than unquestioned ground truth.
2. Save parameters used in each workflow for reproducibility.
3. Bilateral filtering is slow; consider downsampling very large grids first.
4. Robust methods such as RANSAC and robust Gaussian filtering are slower but
   usually better when outliers are present.
5. Start polynomial correction with `order=2` and increase the order only when
   the surface clearly requires it.
6. ICP requires reasonable overlap; use correlation first for rough alignment.

---

## Approximate Performance on a 500x500 Grid

| Function | Time | Notes |
|---------|------|-------|
| `median_filter` | ~0.1 s | Fast |
| `bilateral_filter` | ~30 s | Slow; Python implementation |
| `level_by_plane` | <0.01 s | Very fast |
| `remove_polynomial_form` (order=2) | ~0.05 s | Fast |
| `auto_register` (correlation) | ~0.5 s | Moderate |
| `auto_register` (ICP) | ~2-5 s | Slower |

---

## See Also

Run the example script:

```bash
python examples/advanced_processing.py
```

For fuller descriptions, see [Advanced Processing Guide](ADVANCED_PROCESSING.md).
