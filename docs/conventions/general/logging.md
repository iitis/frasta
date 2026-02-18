# Logging Standards

**Purpose:** Guidelines for using Python's logging module consistently.

**Audience:** All developers.

---

## Logger Setup

```python
import logging
logger = logging.getLogger(__name__)
```

**Always use `__name__`** - this creates hierarchical loggers like `frasta.processing.filtering`.

---

## Log Levels

### DEBUG - Detailed Diagnostic

```python
logger.debug(f"Processing grid of shape {grid.shape}")
logger.debug(f"Converted sigma: {sigma_um}μm = {sigma_pixels}px")
logger.debug(f"Iteration {i}/{max_iter}: residual = {residual:.6f}")
```

**When:** Detailed information for debugging, intermediate values, loop iterations.

### INFO - Confirmations

```python
logger.info(f"Loaded {name}: {grid.shape} grid, px={px_x:.2f}μm")
logger.info(f"Applied bilateral filter: sigma_spatial={sigma_spatial}, sigma_range={sigma_range}")
logger.info(f"RANSAC plane fit: {n_inliers}/{n_total} inliers ({100*n_inliers/n_total:.1f}%)")
```

**When:** Normal operations completed successfully, algorithm choices, major milestones.

### WARNING - Something Unexpected

```python
logger.warning(f"sigma_spatial ({sigma_spatial}) is smaller than pixel size ({px_x})")
logger.warning(f"Insufficient valid data: {n_valid} points. Returning original grid.")
logger.warning("OpenCV not available, using slower Python implementation")
```

**When:** Unusual inputs, degraded performance, fallback methods, potential data quality issues.

### ERROR - Operation Failed

```python
logger.error(f"Failed to load file {fname}: {e}")
logger.error(f"All algorithms failed for grid {grid.shape}: {e}")
```

**When:** Recoverable errors, failed operations that can be caught.

---

## Formatting

### Include Context

```python
# GOOD ✅
logger.info(f"Loaded {name}: {grid.shape} grid, px={px_x:.2f}μm")

# BAD ❌
logger.info("Loaded file")
```

### Use f-strings

```python
# MODERN ✅
logger.debug(f"Processing {n_points} points")

# OLD ❌
logger.debug("Processing %d points" % n_points)
```

### Show Units

```python
# GOOD ✅
logger.info(f"Filter radius: {radius_um:.2f}μm ({radius_px:.1f}px)")

# UNCLEAR ❌
logger.info(f"Filter radius: {radius}")
```

---

## Avoid Logging in Tight Loops

```python
# BAD ❌ - Logs millions of times
for i in range(1000000):
    logger.debug(f"Processing {i}")

# GOOD ✅ - Log at milestones
for i in range(1000000):
    if i % 100000 == 0:
        logger.debug(f"Processed {i}/1000000")
```

---

## Exception Logging

```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)  # Include traceback
    raise
```

---

**Last Updated:** 2026-02-18
