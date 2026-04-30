# Naming Conventions

**Purpose:** Standard naming patterns across FRASTA-toolbox.

**Audience:** All developers.

---

## Python Naming Standards

### Modules and Packages

```python
# Lowercase, underscores for readability
frasta/processing/advanced_filtering.py  OK
frasta/processing/AdvancedFiltering.py   BAD
```

### Classes

```python
# PascalCase
class Surface:           OK
class ScanTab:            OK
class grid_data:          BAD
```

### Functions and Methods

```python
# snake_case
def bilateral_filter():   OK
def load_csv_data():      OK
def bilateralFilter():    BAD
```

### Variables

```python
# snake_case
grid_size = 100           OK
px_x = 0.5               OK
gridSize = 100           BAD
```

### Constants

```python
# UPPER_CASE with underscores
MAX_ITERATIONS = 100      OK
DEFAULT_SIGMA = 5.0       OK
max_iterations = 100      BAD
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
sigma_um = 5.0           # Micrometers OK
sigma_pixels = 10        # Pixels OK
sigma = 5.0              # Ambiguous BAD

radius_mm = 2.0          # Millimeters OK
radius = 2.0             # Ambiguous BAD
```

### Coordinate Arrays

```python
xi = np.array([...])     # X-coordinates (1D) OK
yi = np.array([...])     # Y-coordinates (1D) OK
x_coords = ...           # Also acceptable OK
x = ...                  # Too generic BAD
```

### Pixel Sizes

```python
px_x = 0.5               # Standard OK
px_y = 0.5               # Standard OK
pixel_size_x = 0.5       # Verbose but OK OK
pxx = 0.5                # Unclear BAD
```

### Grids and Arrays

```python
grid = np.array([[...]])      # 2D height data OK
mask = np.array([[...]])      # Boolean mask OK
weights = np.array([[...]])   # Weight array OK
arr = np.array([[...]])       # Generic BAD
```

---

## File Naming

### Python Modules

```python
# Descriptive, snake_case
advanced_filtering.py    OK
bilateral_filter.py      OK (if single-purpose)
filter.py                BAD (too generic)
advFiltering.py          BAD (wrong case)
```

### Test Files

```python
# Prefix with 'test_'
test_advanced_filtering.py   OK
advanced_filtering_test.py   BAD
```

### Documentation

```python
# PascalCase or UPPERCASE for markdown
ARCHITECTURE.md          OK
README.md                OK
algorithms.md            OK
Architecture.MD          BAD (inconsistent extension case)
```

---

## Abbreviations

### Standard Abbreviations (OK to Use)

```python
px = pixel size          OK
um = micrometers         OK
idx = index              OK
coord = coordinate       OK
param = parameter        OK
config = configuration   OK
src = source             OK
dst = destination        OK
tmp = temporary          OK
```

### Avoid Unless Standard

```python
# Spell out for clarity
grid_size = ...          OK
g_sz = ...               BAD

iterations = ...         OK
iters = ...              BAD (unless very local scope)

threshold = ...          OK
thresh = ...             WARNING (OK in local scope)
```

---

**Last Updated:** 2026-02-18