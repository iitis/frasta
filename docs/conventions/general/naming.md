# Naming Conventions

**Purpose:** Standard naming patterns across FRASTA-toolbox.

**Audience:** All developers.

---

## Python Naming Standards

### Modules and Packages

```python
# Lowercase, underscores for readability
frasta/processing/advanced_filtering.py  ✅
frasta/processing/AdvancedFiltering.py   ❌
```

### Classes

```python
# PascalCase
class Surface:           ✅
class ScanTab:            ✅
class grid_data:          ❌
```

### Functions and Methods

```python
# snake_case
def bilateral_filter():   ✅
def load_csv_data():      ✅
def bilateralFilter():    ❌
```

### Variables

```python
# snake_case
grid_size = 100           ✅
px_x = 0.5               ✅
gridSize = 100           ❌
```

### Constants

```python
# UPPER_CASE with underscores
MAX_ITERATIONS = 100      ✅
DEFAULT_SIGMA = 5.0       ✅
max_iterations = 100      ❌
```

### Private Members

```python
class MyClass:
    def __init__(self):
        self._internal_state = 0    # Leading underscore
    
    def _helper_method(self):       # Leading underscore
        pass
```

---

## Domain-Specific Conventions

### Physical Quantities

Always include units in variable names when ambiguous:

```python
sigma_um = 5.0           # Micrometers ✅
sigma_pixels = 10        # Pixels ✅
sigma = 5.0              # Ambiguous ❌

radius_mm = 2.0          # Millimeters ✅
radius = 2.0             # Ambiguous ❌
```

### Coordinate Arrays

```python
xi = np.array([...])     # X-coordinates (1D) ✅
yi = np.array([...])     # Y-coordinates (1D) ✅
x_coords = ...           # Also acceptable ✅
x = ...                  # Too generic ❌
```

### Pixel Sizes

```python
px_x = 0.5               # Standard ✅
px_y = 0.5               # Standard ✅
pixel_size_x = 0.5       # Verbose but OK ✅
pxx = 0.5                # Unclear ❌
```

### Grids and Arrays

```python
grid = np.array([[...]])      # 2D height data ✅
mask = np.array([[...]])      # Boolean mask ✅
weights = np.array([[...]])   # Weight array ✅
arr = np.array([[...]])       # Generic ❌
```

---

## File Naming

### Python Modules

```python
# Descriptive, snake_case
advanced_filtering.py    ✅
bilateral_filter.py      ✅ (if single-purpose)
filter.py                ❌ (too generic)
advFiltering.py          ❌ (wrong case)
```

### Test Files

```python
# Prefix with 'test_'
test_advanced_filtering.py   ✅
advanced_filtering_test.py   ❌
```

### Documentation

```python
# PascalCase or UPPERCASE for markdown
ARCHITECTURE.md          ✅
README.md                ✅
algorithms.md            ✅
Architecture.MD          ❌ (inconsistent extension case)
```

---

## Abbreviations

### Standard Abbreviations (OK to Use)

```python
px = pixel size          ✅
um = micrometers         ✅
idx = index              ✅
coord = coordinate       ✅
param = parameter        ✅
config = configuration   ✅
src = source             ✅
dst = destination        ✅
tmp = temporary          ✅
```

### Avoid Unless Standard

```python
# Spell out for clarity
grid_size = ...          ✅
g_sz = ...               ❌

iterations = ...         ✅
iters = ...              ❌ (unless very local scope)

threshold = ...          ✅
thresh = ...             ⚠️ (OK in local scope)
```

---

**Last Updated:** 2026-02-18
