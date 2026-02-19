# Processing Module Conventions

**Purpose:** Detailed coding standards, patterns, and best practices for implementing algorithms in `frasta/processing/`. This is your go-to reference when adding new processing functions.

**Audience:** Developers (human and AI) adding or modifying processing algorithms.

---

## 📋 Table of Contents

1. [Function Signature Pattern](#function-signature-pattern)
2. [Handling NaN Values](#handling-nan-values)
3. [Mask Parameter Implementation](#mask-parameter-implementation)
4. [Physical Units vs Pixels](#physical-units-vs-pixels)
5. [Input Validation](#input-validation)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Documentation Standards](#documentation-standards)
9. [Testing Requirements](#testing-requirements)
10. [Complete Examples](#complete-examples)

---

## Function Signature Pattern

### Standard Template

```python
def my_processing_function(grid, param1, param2, px_x=1.0, px_y=1.0, mask=None):
    """[One-line summary of what the function does.]
    
    [2-3 paragraph detailed explanation including:
    - Algorithm description
    - When to use this function
    - Key characteristics/benefits
    - Any mathematical background if relevant]
    
    Args:
        grid (np.ndarray or None): 2D input array. May contain NaN values.
        param1 (float): [Description with physical units, e.g., "Spatial radius in micrometers"]
        param2 (int): [Description]
        px_x (float, optional): Pixel size in x-direction (micrometers). Defaults to 1.0.
        px_y (float, optional): Pixel size in y-direction (micrometers). Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to process.
            If None, processes entire grid. Defaults to None.
            
    Returns:
        np.ndarray or None: [Description of output], or None if input is None.
        
    Examples:
        >>> # Basic usage
        >>> filtered = my_processing_function(grid, param1=5.0, param2=3)
        >>> 
        >>> # With physical dimensions
        >>> filtered = my_processing_function(grid, param1=10.0, param2=5,
        ...                                   px_x=0.5, px_y=0.5)
        >>> 
        >>> # Process only masked region
        >>> mask = create_circular_mask(...)
        >>> filtered = my_processing_function(grid, param1=5.0, param2=3, mask=mask)
        
    Notes:
        - [Any important implementation details]
        - [Performance characteristics]
        - [Known limitations]
    """
    # Implementation follows patterns below
    pass
```

### Key Rules

1. **First parameter is always `grid`** (np.ndarray, not Surface)
2. **Algorithm parameters come next** (required, no defaults)
3. **Physical parameters** (`px_x`, `px_y`) always have default=1.0
4. **`mask` parameter always last**, default=None
5. **Return type matches input type** (ndarray → ndarray, None → None)

### Parameter Ordering Priority

```python
# CORRECT ✅ - Follows standard order
def bilateral_filter(grid, sigma_spatial, sigma_range, px_x=1.0, px_y=1.0, mask=None):
    pass

# WRONG ❌ - Mask not last
def bilateral_filter(grid, sigma_spatial, sigma_range, mask=None, px_x=1.0, px_y=1.0):
    pass

# WRONG ❌ - Physical params are required instead of optional
def bilateral_filter(grid, sigma_spatial, sigma_range, px_x, px_y, mask=None):
    pass
```

---

## Handling NaN Values

### Rule 1: Always Accept NaN Input

Processing functions **must gracefully handle NaN values** in the input grid. NaN represents missing/invalid data and is fundamental to the data model.

### Rule 2: Preserve NaN Where Appropriate

```python
# PATTERN 1: Preserve all NaN values (filtering operations)
def my_filter(grid, sigma, px_x=1.0, px_y=1.0, mask=None):
    """Filters should preserve NaN locations."""
    if grid is None:
        return None
    
    # Identify NaN locations BEFORE processing
    nan_mask = np.isnan(grid)
    
    # Option A: Work on filled version
    filled = np.where(nan_mask, 0, grid)
    weights = (~nan_mask).astype(float)
    # ... apply weighted filtering ...
    result = smoothed / weight_sum
    result[weight_sum == 0] = np.nan  # Restore NaN
    
    # Option B: Process only valid region
    result = grid.copy()
    valid = ~nan_mask
    result[valid] = process(grid[valid])
    # NaN remains NaN automatically
    
    return result

# PATTERN 2: Fill NaN values (interpolation operations)
def fill_holes(grid, mask=None):
    """Interpolation explicitly fills NaN."""
    if grid is None:
        return None
    
    grid = grid.copy()
    nan_locs = np.isnan(grid)
    
    if mask is not None:
        nan_locs = nan_locs & mask  # Only fill masked NaN
    
    if not np.any(nan_locs):
        return grid  # Nothing to fill
    
    # Interpolate from valid neighbors
    grid[nan_locs] = interpolate_from_neighbors(grid, nan_locs)
    return result
```

### Rule 3: NaN-Aware Operations

```python
# CORRECT ✅ - Use np.nanmean, np.nanstd, etc.
mean_val = np.nanmean(grid[mask])
std_val = np.nanstd(grid[mask])

# WRONG ❌ - Regular mean propagates NaN
mean_val = np.mean(grid[mask])  # Returns NaN if any element is NaN!

# CORRECT ✅ - Check for all-NaN before computing
if np.all(np.isnan(grid[mask])):
    logger.warning("No valid data in masked region")
    return grid  # or handle appropriately

mean_val = np.nanmean(grid[mask])
```

### Rule 4: Weighted Filtering Pattern (NaN-Safe)

This is the **standard pattern** for implementing filters that respect NaN:

```python
def nan_aware_filter_example(grid, sigma, px_x=1.0, px_y=1.0, mask=None):
    """Standard pattern for NaN-aware filtering."""
    if grid is None:
        return None
    
    # Step 1: Identify valid data
    nan_mask = np.isnan(grid)
    
    # Step 2: Create filled version and weights
    filled = np.where(nan_mask, 0, grid)
    weights = (~nan_mask).astype(float)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        filled = np.where(mask, filled, 0.0)
        weights = np.where(mask, weights, 0.0)
    
    # Step 4: Filter both values and weights
    from scipy.ndimage import gaussian_filter
    smoothed_values = gaussian_filter(filled, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)
    
    # Step 5: Normalize (handle division by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed_values / smoothed_weights
        result[smoothed_weights == 0] = np.nan
    
    return result
```

**Real example from codebase:** See `filtering.py::nan_aware_gaussian()`

---

## Mask Parameter Implementation

The `mask` parameter allows users to restrict processing to a region of interest (ROI). **Every processing function should support masking.**

### Pattern 1: Apply Before Processing

```python
def filter_with_early_mask(grid, sigma, px_x=1.0, px_y=1.0, mask=None):
    """Mask applied BEFORE main processing."""
    if grid is None:
        return None
    
    # Create working copy
    work_grid = grid.copy()
    
    # Apply mask: set non-masked regions to NaN
    if mask is not None:
        work_grid = np.where(mask, grid, np.nan)
    
    # Now process work_grid (NaN-aware algorithm)
    result = apply_filter(work_grid, sigma)
    
    return result
```

### Pattern 2: Apply After Processing

```python
def transform_with_late_mask(grid, angle, px_x=1.0, px_y=1.0, mask=None):
    """Mask applied AFTER main processing (for transforms)."""
    if grid is None:
        return None
    
    # Apply transformation to entire grid
    result = rotate_array(grid, angle)
    
    # Apply mask: restore original values outside mask
    if mask is not None:
        result = np.where(mask, result, grid)
        # Or set to NaN outside mask:
        # result = np.where(mask, result, np.nan)
    
    return result
```

### Pattern 3: Mask-Conditional Processing

```python
def selective_processing(grid, threshold, px_x=1.0, px_y=1.0, mask=None):
    """Only process pixels that meet condition AND are in mask."""
    if grid is None:
        return None
    
    result = grid.copy()
    
    # Build effective mask
    valid = ~np.isnan(grid)
    if mask is not None:
        valid = valid & mask
    
    # Apply operation only to valid pixels
    result[valid] = process(grid[valid], threshold)
    
    return result
```

### Mask Testing

```python
# Test that mask restricts processing
def test_function_respects_mask():
    grid = np.ones((100, 100)) * 10.0
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True  # Only center region
    
    result = my_filter(grid, sigma=5.0, mask=mask)
    
    # Center should be modified
    assert not np.allclose(result[40:60, 40:60], 10.0)
    
    # Edges should be unchanged (or NaN, depending on function)
    assert np.allclose(result[0:10, 0:10], 10.0) or np.all(np.isnan(result[0:10, 0:10]))
```

**Real example:** See `advanced_filtering.py::bilateral_filter()` lines 88-93

---

## Physical Units vs Pixels

### Rule: Always Accept Physical Units as Input

Users think in **physical dimensions (micrometers)**, not pixels. Functions should accept physical parameters and convert internally.

### Conversion Pattern

```python
def spatial_filter(grid, radius_um, px_x=1.0, px_y=1.0, mask=None):
    """
    Args:
        radius_um (float): Filter radius in micrometers (physical units).
        px_x (float): Pixel size in x-direction (micrometers).
        px_y (float): Pixel size in y-direction (micrometers).
    """
    # Step 1: Convert physical units to pixels
    radius_x_pixels = radius_um / px_x
    radius_y_pixels = radius_um / px_y
    
    # Step 2: Use pixel values in algorithm
    kernel_size_x = int(np.ceil(2 * radius_x_pixels + 1))
    kernel_size_y = int(np.ceil(2 * radius_y_pixels + 1))
    
    # Step 3: Apply filter with pixel-based kernel
    result = apply_kernel(grid, kernel_size_x, kernel_size_y)
    
    return result
```

### Handling Anisotropic Pixels

If `px_x != px_y`, you may need anisotropic kernels:

```python
def anisotropic_aware_filter(grid, sigma_um, px_x=1.0, px_y=1.0, mask=None):
    """Handles different pixel sizes in x and y."""
    # Convert sigma to pixels
    sigma_x = sigma_um / px_x
    sigma_y = sigma_um / px_y
    
    if np.abs(sigma_x - sigma_y) / max(sigma_x, sigma_y) > 0.1:
        # Significant anisotropy - need anisotropic kernel
        logger.info(f"Using anisotropic filter: sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f}")
        result = scipy.ndimage.gaussian_filter(grid, sigma=(sigma_y, sigma_x))
    else:
        # Nearly isotropic - use average
        sigma_avg = (sigma_x + sigma_y) / 2
        result = scipy.ndimage.gaussian_filter(grid, sigma=sigma_avg)
    
    return result
```

### Warnings for Invalid Conversions

```python
def my_filter(grid, sigma_spatial, px_x=1.0, px_y=1.0, mask=None):
    """Validate that physical parameters make sense."""
    # Convert to pixels
    sigma_x_pixels = sigma_spatial / px_x
    sigma_y_pixels = sigma_spatial / px_y
    
    # Warn if sigma is too small
    if sigma_x_pixels < 1.0 or sigma_y_pixels < 1.0:
        logger.warning(
            f"sigma_spatial ({sigma_spatial} μm) is smaller than pixel size "
            f"(px_x={px_x}, px_y={px_y}). Results may be inaccurate."
        )
    
    # Warn if kernel would be too large
    kernel_radius = int(np.ceil(3 * max(sigma_x_pixels, sigma_y_pixels)))
    if kernel_radius > min(grid.shape) / 4:
        logger.warning(
            f"Filter kernel ({kernel_radius} pixels) is large relative to grid size "
            f"{grid.shape}. Consider using smaller sigma_spatial."
        )
    
    # Continue with processing...
```

**Real example:** See `advanced_filtering.py::bilateral_filter()` lines 63-68

---

## Input Validation

### Rule 1: Handle None Input Gracefully

```python
def my_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    """Always check for None first."""
    if grid is None:
        return None  # Propagate None (don't raise error)
    
    # Rest of function...
```

**Rationale:** GUI may pass None for empty tabs. Returning None is cleaner than raising exceptions.

### Rule 2: Validate Array Dimensions

```python
def my_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    if grid is None:
        return None
    
    # Check dimensionality
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {grid.shape}")
    
    # Check minimum size
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        logger.warning(f"Grid very small: {grid.shape}. Results may be poor.")
        return grid  # Or raise error if algorithm requires minimum size
    
    # Check mask shape
    if mask is not None:
        if mask.shape != grid.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match grid shape {grid.shape}"
            )
```

### Rule 3: Validate Parameters

```python
def my_function(grid, sigma, iterations, px_x=1.0, px_y=1.0, mask=None):
    if grid is None:
        return None
    
    # Range checks
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    
    if px_x <= 0 or px_y <= 0:
        raise ValueError(f"Pixel sizes must be positive, got px_x={px_x}, px_y={px_y}")
    
    # Warn for unusual values (don't raise)
    if sigma > 100:
        logger.warning(f"Large sigma value: {sigma}. Are you sure?")
```

### Rule 4: Check for Sufficient Valid Data

```python
def my_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    if grid is None:
        return None
    
    # Determine effective region
    valid = ~np.isnan(grid)
    if mask is not None:
        valid = valid & mask
    
    n_valid = np.sum(valid)
    
    # Check minimum data requirement
    if n_valid < 10:  # Or algorithm-specific threshold
        logger.warning(
            f"Insufficient valid data: {n_valid} points. "
            f"Returning original grid."
        )
        return grid  # Can't process reliably
    
    if n_valid < 0.01 * grid.size:  # Less than 1% valid
        logger.warning(
            f"Very sparse data: {n_valid}/{grid.size} valid points "
            f"({100*n_valid/grid.size:.2f}%)"
        )
    
    # Continue processing...
```

**Real example:** See `morphology.py::fit_plane_least_squares()` lines 32-35

---

## Error Handling

### Rule 1: Prefer Graceful Degradation

```python
def my_robust_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    """Try multiple strategies, fall back gracefully."""
    if grid is None:
        return None
    
    try:
        # Try optimal algorithm
        result = fast_algorithm(grid, param)
        return result
    except Exception as e:
        logger.warning(f"Fast algorithm failed: {e}. Falling back to slow method.")
        
        try:
            # Fall back to robust alternative
            result = slow_robust_algorithm(grid, param)
            return result
        except Exception as e2:
            logger.error(f"All algorithms failed: {e2}. Returning original grid.")
            return grid  # Last resort
```

**Real example:** See `advanced_filtering.py::bilateral_filter()` - OpenCV vs Python fallback

### Rule 2: Use Appropriate Exception Types

```python
# Input validation errors (user's fault) → ValueError
if sigma <= 0:
    raise ValueError(f"sigma must be positive, got {sigma}")

# Shape mismatches → ValueError
if mask.shape != grid.shape:
    raise ValueError(f"Shape mismatch: mask {mask.shape} vs grid {grid.shape}")

# Missing dependencies (installation issue) → ImportError or skip gracefully
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available, using slower Python implementation")
    # Continue with fallback

# Numerical issues → RuntimeError or RuntimeWarning
if np.all(np.isnan(result)):
    raise RuntimeError("Algorithm produced all-NaN result")
```

### Rule 3: Informative Error Messages

```python
# WRONG ❌
raise ValueError("Invalid input")

# CORRECT ✅
raise ValueError(
    f"sigma_spatial ({sigma_spatial}) must be positive and less than "
    f"grid dimensions ({grid.shape[0]} x {grid.shape[1]} pixels)"
)
```

---

## Performance Optimization

### Rule 1: Prefer Vectorized NumPy Operations

```python
# SLOW ❌ - Explicit Python loops
for i in range(rows):
    for j in range(cols):
        result[i, j] = grid[i, j] ** 2 + 2 * grid[i, j]

# FAST ✅ - Vectorized NumPy
result = grid ** 2 + 2 * grid
```

### Rule 2: Use Optimized Libraries When Available

**Order of preference:**
1. **OpenCV** (if applicable) - fastest
2. **scipy.ndimage** - fast, battle-tested
3. **NumPy vectorized** - fast
4. **numba @jit** - compile Python to native code
5. **Pure Python loops** - only as last resort

```python
# Example: Bilateral filter implementation hierarchy
def bilateral_filter(grid, sigma_spatial, sigma_range, px_x=1.0, px_y=1.0, 
                    mask=None, use_opencv=True):
    # Try OpenCV first (~7500x faster)
    if use_opencv and HAS_OPENCV:
        return _bilateral_filter_opencv(grid, ...)
    
    # Fall back to Python (always available, but slow)
    logger.info("OpenCV not available, using pure Python (slow)")
    return _bilateral_filter_python(grid, ...)
```

### Rule 3: Profile Before Optimizing

```python
import time

def my_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    start = time.time()
    
    # ... processing ...
    
    elapsed = time.time() - start
    logger.debug(f"my_function took {elapsed:.3f}s for {grid.shape} grid")
    
    return result
```

Or use the decorator pattern:

```python
from ..utils.decorators import timeit

@timeit
def my_expensive_function(grid, param, px_x=1.0, px_y=1.0, mask=None):
    # ... processing ...
    return result
```

### Rule 4: Document Performance Characteristics

```python
def bilateral_filter(grid, sigma_spatial, sigma_range, px_x=1.0, px_y=1.0, 
                    mask=None, use_opencv=True):
    """...
    
    Notes:
        - OpenCV version: ~0.005s for 512x512 image (use_opencv=True)
        - Python version: ~30s for 512x512 image (use_opencv=False)
        - **Always install OpenCV for production use**: pip install opencv-python
    """
```

### Rule 5: Avoid Unnecessary Copies

```python
# WASTEFUL ❌
def bad_function(grid):
    grid1 = grid.copy()
    grid2 = grid1.copy()
    grid3 = grid2.copy()  # Why 3 copies?
    return process(grid3)

# EFFICIENT ✅
def good_function(grid):
    result = grid.copy()  # One copy to preserve input
    process_inplace(result)  # Modify the copy
    return result

# EVEN BETTER ✅ (if algorithm allows)
def best_function(grid):
    # Create new array only for result
    result = np.zeros_like(grid)
    compute_into(grid, out=result)  # Many NumPy funcs support 'out' parameter
    return result
```

---

## Documentation Standards

### Docstring Template

Every function must have a complete docstring following this structure:

```python
def function_name(grid, param1, param2, px_x=1.0, px_y=1.0, mask=None):
    """[One-line summary (< 80 chars).]
    
    [Detailed explanation - 2-3 paragraphs:
    - What does this function do?
    - When should you use it?
    - How does it work (high-level algorithm)?
    - What makes it different from similar functions?]
    
    Args:
        grid (np.ndarray or None): [Description]. [Special notes about NaN handling].
        param1 (float): [Description with units]. [Valid range if applicable].
        param2 (str): [Description]. [Options if enum-like].
        px_x (float, optional): Pixel size in x-direction (micrometers). Defaults to 1.0.
        px_y (float, optional): Pixel size in y-direction (micrometers). Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask indicating region to process.
            If None, processes entire grid. Defaults to None.
            
    Returns:
        [type]: [Description of output]. [Relationship to input].
        
    Raises:
        ValueError: [When and why].
        RuntimeError: [When and why].
        
    Examples:
        >>> # Example 1: Basic usage
        >>> result = function_name(grid, param1=5.0, param2="option1")
        >>> 
        >>> # Example 2: With mask
        >>> mask = create_mask(...)
        >>> result = function_name(grid, param1=5.0, param2="option2", mask=mask)
        >>> 
        >>> # Example 3: Edge case
        >>> result = function_name(small_grid, param1=1.0, param2="option3")
        
    Notes:
        - [Important implementation detail]
        - [Performance characteristic]
        - [Known limitation]
        - [Reference to paper/algorithm if applicable]
        
    See Also:
        - :func:`related_function1` - [How it's related]
        - :func:`related_function2` - [How it's related]
    """
    pass
```

### Documentation Best Practices

1. **One-line summary** must fit in 80 chars and end with period
2. **Detailed explanation** should answer: What? When? How? Why?
3. **Args** must specify type and include units for physical quantities
4. **Examples** should be runnable (or clearly marked as pseudocode)
5. **Notes** for performance, limitations, or algorithm references
6. **See Also** to connect related functions

**Real example:** See `advanced_filtering.py::bilateral_filter()` lines 20-67

---

## Testing Requirements

### Test Categories

Every processing function should have tests for:

1. **Basic functionality** - Does it work at all?
2. **Shape preservation** - Output shape matches input?
3. **NaN handling** - NaN values handled correctly?
4. **Mask support** - Mask restricts processing?
5. **Edge cases** - Empty arrays, all-NaN, single pixel, etc.
6. **Parameter validation** - Invalid params raise errors?

### Test Template

```python
# In tests/test_processing.py

import numpy as np
import pytest
from frasta.processing import my_function


class TestMyFunction:
    """Test suite for my_function."""
    
    def test_basic_functionality(self):
        """Test that function works on simple input."""
        grid = np.random.randn(100, 100)
        result = my_function(grid, param=5.0)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        shapes = [(10, 10), (50, 100), (200, 150)]
        
        for shape in shapes:
            grid = np.random.randn(*shape)
            result = my_function(grid, param=5.0)
            assert result.shape == grid.shape, f"Shape mismatch for {shape}"
    
    def test_handles_none_input(self):
        """Test that None input returns None."""
        result = my_function(None, param=5.0)
        assert result is None
    
    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        grid = np.random.randn(100, 100)
        grid[20:30, 20:30] = np.nan
        
        result = my_function(grid, param=5.0)
        
        # Check that NaN region is still NaN (or filled, depending on function)
        # For filters: should preserve NaN
        assert np.all(np.isnan(result[20:30, 20:30]))
        
        # Valid region should be processed
        assert not np.all(np.isnan(result[0:10, 0:10]))
    
    def test_handles_all_nan(self):
        """Test behavior when input is all NaN."""
        grid = np.full((50, 50), np.nan)
        result = my_function(grid, param=5.0)
        
        # Should return all NaN or original grid
        assert np.all(np.isnan(result)) or np.array_equal(result, grid)
    
    def test_respects_mask(self):
        """Test that mask parameter restricts processing."""
        grid = np.ones((100, 100)) * 10.0
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Only center
        
        result = my_function(grid, param=5.0, mask=mask)
        
        # Center should be modified
        center_modified = not np.allclose(result[45:55, 45:55], 10.0)
        
        # Edges should be unchanged or NaN
        edge_unchanged = (np.allclose(result[0:10, 0:10], 10.0) or 
                          np.all(np.isnan(result[0:10, 0:10])))
        
        assert center_modified, "Masked region was not processed"
        assert edge_unchanged, "Unmasked region was unexpectedly modified"
    
    def test_pixel_size_conversion(self):
        """Test that physical units are converted correctly."""
        grid = np.random.randn(100, 100)
        
        # Same physical size, different pixel sizes
        result1 = my_function(grid, param=10.0, px_x=1.0, px_y=1.0)
        result2 = my_function(grid, param=10.0, px_x=2.0, px_y=2.0)
        
        # Results should differ (10μm is 10px vs 5px)
        assert not np.allclose(result1, result2, equal_nan=True)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        grid = np.random.randn(50, 50)
        
        # Negative parameter
        with pytest.raises(ValueError):
            my_function(grid, param=-1.0)
        
        # Zero pixel size
        with pytest.raises(ValueError):
            my_function(grid, param=5.0, px_x=0.0)
    
    def test_mask_shape_mismatch(self):
        """Test that mismatched mask shape raises error."""
        grid = np.random.randn(100, 100)
        wrong_mask = np.ones((50, 50), dtype=bool)
        
        with pytest.raises(ValueError, match="shape.*match"):
            my_function(grid, param=5.0, mask=wrong_mask)
    
    def test_does_not_modify_input(self):
        """Test that input array is not modified in-place."""
        grid = np.random.randn(50, 50)
        grid_original = grid.copy()
        
        result = my_function(grid, param=5.0)
        
        # Input should be unchanged
        np.testing.assert_array_equal(grid, grid_original)
        
        # Result should be different (unless trivial operation)
        assert not np.array_equal(result, grid)  # Adjust for your function
```

### Fixtures for Common Test Data

```python
# In tests/conftest.py

import pytest
import numpy as np

@pytest.fixture
def simple_grid():
    """Simple 100x100 grid with random data."""
    return np.random.randn(100, 100)

@pytest.fixture
def grid_with_nans():
    """Grid with NaN holes."""
    grid = np.random.randn(100, 100)
    grid[20:30, 20:30] = np.nan
    grid[70:75, 80:85] = np.nan
    return grid

@pytest.fixture
def circular_mask():
    """Circular mask (radius 25, center at 50,50)."""
    y, x = np.ogrid[-50:50, -50:50]
    mask = x**2 + y**2 <= 25**2
    return mask

@pytest.fixture
def ramp_grid():
    """Linear ramp for testing leveling functions."""
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    xx, yy = np.meshgrid(x, y)
    return 0.5 * xx + 0.3 * yy + 10.0
```

---

## Complete Examples

### Example 1: Simple Filter

```python
def median_filter(grid, size, px_x=1.0, px_y=1.0, mask=None):
    """Applies median filter for robust outlier removal.
    
    Median filtering replaces each pixel with the median of its neighborhood.
    More robust than mean filtering for removing spike noise and outliers
    while preserving edges better than Gaussian smoothing.
    
    Args:
        grid (np.ndarray or None): 2D input array.
        size (int): Window size in pixels (must be odd). Larger values = more smoothing.
        px_x (float, optional): Pixel size in x-direction (micrometers). Defaults to 1.0.
        px_y (float, optional): Pixel size in y-direction (micrometers). Defaults to 1.0.
        mask (np.ndarray, optional): Boolean mask for ROI. Defaults to None.
        
    Returns:
        np.ndarray or None: Filtered grid, or None if input is None.
        
    Examples:
        >>> # Remove spikes with 5x5 window
        >>> cleaned = median_filter(grid, size=5)
    """
    from scipy.ndimage import median_filter as scipy_median
    
    # Handle None
    if grid is None:
        return None
    
    # Validate
    if size < 3 or size % 2 == 0:
        raise ValueError(f"size must be odd and >= 3, got {size}")
    
    # Apply mask
    if mask is not None:
        if mask.shape != grid.shape:
            raise ValueError(f"Mask shape {mask.shape} != grid shape {grid.shape}")
        work_grid = np.where(mask, grid, np.nan)
    else:
        work_grid = grid.copy()
    
    # Median filter (NaN-aware via footprint)
    result = scipy_median(work_grid, size=size)
    
    return result
```

### Example 2: Grid Transformation

```python
def rotate_grid(grid, angle_degrees, xi, yi, px_x, px_y, order=3):
    """Rotates a grid around its center with interpolation.
    
    Rotates the height grid by the specified angle while maintaining the same
    grid dimensions. Uses spline interpolation to handle sub-pixel positions.
    NaN values are preserved in the output.
    
    Args:
        grid (np.ndarray): 2D height array to rotate.
        angle_degrees (float): Rotation angle in degrees (positive = counterclockwise).
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        px_x (float): Pixel size in x-direction (micrometers).
        px_y (float): Pixel size in y-direction (micrometers).
        order (int, optional): Interpolation order (0=nearest, 1=linear, 3=cubic).
            Defaults to 3.
            
    Returns:
        tuple: (rotated_grid, new_xi, new_yi, px_x, px_y)
            The rotated grid with updated coordinate arrays.
            
    Examples:
        >>> # Rotate by 45 degrees
        >>> rotated, xi_new, yi_new, px_x, px_y = rotate_grid(
        ...     grid, 45, xi, yi, px_x, px_y)
        >>> 
        >>> # Rotate with linear interpolation (faster)
        >>> rotated, *coords = rotate_grid(grid, 30, xi, yi, px_x, px_y, order=1)
    """
    from scipy.ndimage import affine_transform
    
    ny, nx = grid.shape
    center_x = nx / 2
    center_y = ny / 2
    
    # Convert angle to radians
    theta = np.radians(angle_degrees)
    
    # Create rotation matrix (inverse for affine_transform)
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    
    # Rotation matrix about center
    matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Offset to rotate around center
    offset = np.array([center_y, center_x]) - matrix @ np.array([center_y, center_x])
    
    # Apply transformation
    rotated = affine_transform(
        grid,
        matrix,
        offset=offset,
        order=order,
        cval=np.nan,  # Fill with NaN outside bounds
        prefilter=True
    )
    
    # Coordinate arrays remain the same (rotation is in-plane)
    return rotated, xi, yi, px_x, px_y
```

### Example 3: Leveling Operation

```python
def level_by_plane(grid, mask=None, method='least_squares'):
    """Removes plane from grid (leveling operation).
    
    Fits a plane to the surface and subtracts it, removing overall tilt.
    Useful for removing systematic sample tilt or scanner tilt artifacts.
    
    Args:
        grid (np.ndarray): 2D height array.
        mask (np.ndarray, optional): Boolean mask for fitting region.
            If None, uses all non-NaN points. Defaults to None.
        method (str, optional): Fitting method:
            - 'least_squares': Standard least-squares (fast, sensitive to outliers)
            - 'robust': RANSAC-based robust fitting (slower, outlier-resistant)
            Defaults to 'least_squares'.
            
    Returns:
        np.ndarray: Leveled grid with plane removed.
        
    Examples:
        >>> # Remove tilt with least squares
        >>> leveled = level_by_plane(grid)
        >>> 
        >>> # Robust fitting (ignores outliers)
        >>> leveled = level_by_plane(grid, method='robust')
        >>> 
        >>> # Fit only to masked region
        >>> mask = create_flat_region_mask(...)
        >>> leveled = level_by_plane(grid, mask=mask, method='least_squares')
        
    Notes:
        - Least squares: O(n) time, assumes no outliers
        - Robust RANSAC: O(n log n) time, handles up to ~30% outliers
        - NaN values are automatically excluded from fitting
    """
    # Implementation using existing fit_plane functions
    if method == 'least_squares':
        plane, coeffs = fit_plane_least_squares(grid, mask=mask)
    elif method == 'robust':
        plane, coeffs, inliers = fit_plane_robust(grid, mask=mask)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'least_squares' or 'robust'.")
    
    # Subtract plane
    leveled = grid - plane
    
    logger.info(f"Plane removed: z = {coeffs[0]:.6f}*x + {coeffs[1]:.6f}*y + {coeffs[2]:.6f}")
    
    return leveled
```

---

## Quick Checklist for New Functions

Before submitting a new processing function, verify:

- [ ] Function signature follows standard pattern
- [ ] Accepts `np.ndarray`, not `Surface`
- [ ] Returns same type as input (None → None, array → array)
- [ ] Handles NaN values gracefully (preserves or fills appropriately)
- [ ] Implements `mask` parameter correctly
- [ ] Uses physical units (micrometers) with `px_x`, `px_y` conversion
- [ ] Does NOT modify input arrays in-place
- [ ] Validates input parameters (shape, range, types)
- [ ] Has complete docstring with Args/Returns/Examples
- [ ] Has unit tests (basic, shape, NaN, mask, parameters)
- [ ] Uses vectorized NumPy operations (no unnecessary loops)
- [ ] Logs important events (warnings, timing, algorithm selection)
- [ ] Added to module's `__init__.py` exports
- [ ] Example added to `examples/advanced_processing.py` (if major feature)

---

## Reference: Standard Imports for Processing Modules

```python
"""Module docstring."""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter  # or other scipy
from scipy.interpolate import griddata  # if interpolation
from sklearn.linear_model import LinearRegression, RANSACRegressor  # if regression

# Optional optimized libraries
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

import logging
logger = logging.getLogger(__name__)
```

---

## Questions or Edge Cases?

**Q: Should I accept Surface objects directly?**  
**A:** No. Processing functions work with `np.ndarray`. GUI converts `Surface.height` to array before calling.

**Q: What if my function needs to return multiple outputs?**  
**A:** Return tuple:
```python
def my_analysis(grid, px_x=1.0, px_y=1.0, mask=None):
    result_grid = ...
    statistics = {...}
    return result_grid, statistics
```

**Q: Can I use global state or cache?**  
**A:** Avoid it. Processing functions should be **stateless**. If caching is essential for performance, document it clearly and use function-level private cache dict.

**Q: How do I handle optional dependencies (OpenCV, etc.)?**  
**A:** Try-import at module level, set HAS_XXX flag, provide fallback, document in function docstring.

**Q: Should I log at INFO or DEBUG level?**  
**A:** 
- **DEBUG**: Timing, intermediate values, iteration counts
- **INFO**: Algorithm selection, major steps, completion status
- **WARNING**: Unusual inputs, degraded quality, fallback to slower method
- **ERROR**: Failure that prevents completion

---

This document is a **living reference**. Update it when you discover new patterns or better practices!

**Last updated:** 2026-02-18
