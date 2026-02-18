# Data Structure Conventions

**Purpose:** Guidelines for data structures in `frasta/core/`.

**Audience:** Developers extending or modifying core data types.

---

## GridData Usage

### What GridData Is

`GridData` is a **simple container** for scan data:

```python
class GridData:
    def __init__(self, grid, xi, yi, px_x, px_y, vmin=None, vmax=None):
        self.grid = grid      # 2D numpy array
        self.xi = xi          # 1D x-coordinates
        self.yi = yi          # 1D y-coordinates
        self.px_x = px_x      # Pixel size in x (μm)
        self.px_y = px_y      # Pixel size in y (μm)
        self.vmin = vmin      # Display range min (optional)
        self.vmax = vmax      # Display range max (optional)
```

### When to Use

- ✅ GUI components passing data between tabs
- ✅ Storing visualization parameters (vmin/vmax)
- ✅ Simple utility methods (crop, copy)

### When NOT to Use

- ❌ Processing functions (use `np.ndarray` instead)
- ❌ Business logic / algorithms
- ❌ Complex state management

### Extending GridData

If you need additional fields:

```python
class ExtendedGridData(GridData):
    """Extended version with metadata."""
    
    def __init__(self, grid, xi, yi, px_x, px_y, vmin=None, vmax=None, metadata=None):
        super().__init__(grid, xi, yi, px_x, px_y, vmin, vmax)
        self.metadata = metadata or {}
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
