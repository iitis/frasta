# Data Structure Conventions

**Purpose:** Guidelines for data structures in `frasta/core/`.

**Audience:** Developers extending or modifying core data types.

---

## Surface Usage (Replaces GridData)

### What Surface Is

`Surface` is a **unified container** for 2D surface data:

```python
class Surface:
    def __init__(self, height, dx, dy, x0=0.0, y0=0.0, mask=None, unit="um", 
                 metadata=None, vmin=None, vmax=None):
        self.height = height  # 2D numpy array
        self.dx = dx          # Pixel size in x (um)
        self.dy = dy          # Pixel size in y (um)
        self.x0 = x0          # Origin/offset for X coordinates
        self.y0 = y0          # Origin/offset for Y coordinates
        self.mask = mask      # Boolean mask (optional)
        self.unit = unit      # Physical unit
        self.metadata = {}    # Additional metadata
        self.vmin = vmin      # Display range min (optional)
        self.vmax = vmax      # Display range max (optional)
    
    @property
    def xi(self):  # Generated from x0, dx and shape
        return self.x0 + np.arange(self.nx) * self.dx
    
    @property
    def yi(self):  # Generated from y0, dy and shape
        return self.y0 + np.arange(self.ny) * self.dy
```

### When to Use

- OK **Loading/Saving:** Loaders return Surface, exporters accept (name, Surface) tuples
- OK **GUI components:** Pass Surface between tabs and dialogs
- OK **Storing visualization parameters:** vmin/vmax for display ranges
- OK **Simple utility methods:** crop, copy
- OK **Preserving spatial positioning:** x0, y0 for alignment and comparison

### When NOT to Use

- BAD Processing functions (use `np.ndarray` instead)
- BAD Business logic / algorithms
- BAD Complex state management

### Spatial Coordinates

`Surface` preserves spatial positioning through `x0` and `y0`:

```python
# Example: Data starting at X=10.5, Y=20.0
surf = Surface(
    height=data,
    dx=0.5, dy=0.5,
    x0=10.5, y0=20.0
)

# Coordinate arrays respect the origin
print(surf.xi)  # [10.5, 11.0, 11.5, 12.0, ...]
print(surf.yi)  # [20.0, 20.5, 21.0, 21.5, ...]
```

**Important:** Always preserve `x0` and `y0` when:
- Loading data from files (CSV, NPZ, H5)
- Creating Surface from real-world measurements
- Aligning/comparing multiple scans

### Usage in GUI

```python
# Loading data
surface = load_csv_data(fname, units_xy='um', units_z='um')

# Setting data in tab
tab = ScanTab()
tab.setSurface(surface)

# Getting data from tab
surface = tab.getSurface()

# Processing (extract arrays)
from ..processing import bilateral_filter
result = bilateral_filter(
    surface.height,
    sigma_spatial=5.0,
    sigma_range=10.0,
    px_x=surface.dx,
    px_y=surface.dy
)

# Update surface
surface.height = result
tab.setSurface(surface)
```

### Usage in Dialogs

```python
# ProfileViewer - accepts two Surface objects
viewer = ProfileViewer()
viewer.set_surfaces(surface1, surface2)

# OverlayViewer - accepts two Surface objects
overlay = OverlayViewer(surface1, surface2)
```

### Extending Surface

If you need additional fields:

```python
class ExtendedSurface(Surface):
    """Extended version with custom metadata."""
    
    def __init__(self, height, dx, dy, custom_field=None, **kwargs):
        super().__init__(height, dx, dy, **kwargs)
        self.custom_field =9custom_field
```

---

## Creating New Data Structures

### When to Create

Only create new core data structures when:
1. Multiple modules need the same structure
2. Structure has clear, single responsibility
3. Simple containers (no business logic)

### Pattern

```python
class MyDataStructure:
    """Brief description of purpose.
    
    Attributes:
        field1 (type): Description.
        field2 (type): Description.
    """
    
    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2
    
    def copy(self):
        """Return deep copy."""
        return MyDataStructure(
            field1=self.field1.copy(),
            field2=self.field2.copy()
        )
```

---

**Last Updated:** 2026-02-18