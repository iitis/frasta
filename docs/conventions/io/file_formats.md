# File Format Specifications

**Purpose:** Detailed specifications for all supported file formats in FRASTA-toolbox.

**Audience:** Developers implementing loaders/exporters or debugging file I/O issues.

---

## 📋 Table of Contents

1. [NPZ Format](#npz-format)
2. [HDF5 Format](#hdf5-format)
3. [CSV Format](#csv-format)
4. [STL Format](#stl-format)
5. [Unit Conventions](#unit-conventions)

---

## NPZ Format

**Extension:** `.npz`  
**Type:** Compressed NumPy arrays  
**Use:** Fast save/load, internal format, batch processing

### Structure

```python
{
    'frasta_info': 1,              # Version marker
    'frasta_cnt': N,               # Number of scans
    'name_00': str,                # Scan 0 name
    'grid_00': ndarray (2D),       # Scan 0 height data
    'xi_00': ndarray (1D),         # Scan 0 x-coordinates
    'yi_00': ndarray (1D),         # Scan 0 y-coordinates
    'px_00': float,                # Scan 0 x pixel size
    'py_00': float,                # Scan 0 y pixel size
    # ... repeat for scans 01, 02, etc.
}
```

### Conventions

- **Key naming:** `{field}_{index:02d}` (e.g., `grid_03`)
- **Coordinates in micrometers (μm)**
- **NaN values preserved** in grid arrays
- **Compression:** Always use `np.savez_compressed()`

### Example Implementation

```python
def save_npz(fname, scans):
    """Save list of (name, Surface) tuples."""
    save_dict = {'frasta_info': 1, 'frasta_cnt': len(scans)}
    
    for i, (name, surface) in enumerate(scans):
        save_dict[f"name_{i:02}"] = name
        save_dict[f"grid_{i:02}"] = surface.height
        save_dict[f"xi_{i:02}"] = surface.xi
        save_dict[f"yi_{i:02}"] = surface.yi
        save_dict[f"px_{i:02}"] = surface.dx
        save_dict[f"py_{i:02}"] = surface.dy
    
    np.savez_compressed(fname, **save_dict)
```

---

## HDF5 Format

**Extension:** `.h5`, `.hdf5`  
**Type:** Hierarchical Data Format  
**Use:** Large datasets, metadata-rich exports, archival

### Structure

```
/
├── @frasta_info = 1           # Root attribute
├── @frasta_cnt = N            # Number of scans
├── tab_00/                    # Group for scan 0
│   ├── name (dataset)         # UTF-8 encoded string
│   ├── grid (dataset)         # 2D float array, gzip compressed
│   ├── xi (dataset)           # 1D float array
│   ├── yi (dataset)           # 1D float array
│   ├── px_x (scalar)          # Float
│   └── px_y (scalar)          # Float
├── tab_01/
│   └── ...
└── ...
```

### Conventions

- **Group naming:** `tab_{index:02d}`
- **Compression:** Use gzip for all datasets
- **String encoding:** UTF-8 for names
- **Coordinates in micrometers (μm)**
- **Optional metadata:** Can add custom attributes to groups

### Example Implementation

```python
def save_h5(fname, scans):
    """Save with HDF5 structure."""
    with h5py.File(fname, 'w') as f:
        f.attrs['frasta_info'] = 1
        f.attrs['frasta_cnt'] = len(scans)
        
        for i, (name, surface) in enumerate(scans):
            group = f.create_group(f"tab_{i:02}")
            group.create_dataset("name", data=name.encode("utf-8"))
            group.create_dataset("grid", data=surface.height, compression="gzip")
            group.create_dataset("xi", data=surface.xi, compression="gzip")
            group.create_dataset("yi", data=surface.yi, compression="gzip")
            group.create_dataset("px_x", data=surface.dx)
            group.create_dataset("px_y", data=surface.dy)
```

---

## CSV Format

**Extension:** `.csv`, `.txt`, `.dat`  
**Type:** Plain text XYZ coordinates  
**Use:** Import from measurement devices, interoperability

### Structure

```
# Comment lines starting with # are ignored
X,Y,Z
0.0,0.0,10.5
0.5,0.0,10.3
1.0,0.0,10.7
0.0,0.5,10.2
...
```

### Conventions

- **Delimiter:** Auto-detect (`,`, `;`, tab, space)
- **Header:** Optional, if present should be in first non-comment line
- **Column order:** X, Y, Z (height/depth)
- **Input units:** User-specified (mm or μm), converted to μm internally
- **Grid structure:** Assumed regular grid, coordinates sorted
- **Missing values:** Empty cells or non-numeric → NaN

### Parsing Algorithm

1. Read file, skip comment lines
2. Detect delimiter (most frequent: `,`, `;`, `\t`, space)
3. Parse all numeric columns as X, Y, Z
4. Determine grid structure from unique X, Y values
5. Fill grid with Z values at corresponding (X, Y)
6. Missing grid cells → NaN

### Example

```python
def load_csv_data(fname, xy_units='um', z_units='um', progress_callback=None):
    """Load CSV with auto-delimiter detection."""
    df = pd.read_csv(fname, sep=r'[;,\t ]+', engine='python', 
                     header=None, names=['x', 'y', 'z'])
    
    # Convert units to micrometers
    if xy_units == 'mm':
        df['x'] *= 1000
        df['y'] *= 1000
    if z_units == 'mm':
        df['z'] *= 1000
    
    # Build grid from coordinates
    # ... (grid construction logic)
    
    # Return Surface object
    return Surface(
        height=grid,
        dx=px_x,
        dy=px_y,
        x0=xi[0],
        y0=yi[0],
        unit="µm"
    )
```

---

## STL Format

**Extension:** `.stl`  
**Type:** Triangular mesh (ASCII or binary)  
**Use:** 3D printing, CAD export, surface visualization

### Conventions for Import

- **Units in STL:** Assumed millimeters
- **Conversion to internal:** Multiply by 1000 → micrometers
- **Grid creation:** Ray-casting from above, sample Z at regular XY grid
- **Resolution:** Auto-determined or user-specified

### Conventions for Export

- **Units in STL:** Millimeters (divide internal μm by 1000)
- **Mesh generation:** Create triangular mesh from 2D height map
- **NaN handling:** Exclude NaN regions from mesh
- **Format:** Binary STL preferred (smaller files)
- **Downsampling:** Large grids are automatically downsampled
  - Based on **valid (non-NaN) points count**, not total grid size
  - Default: max 500k valid points
  - Example: 10M grid with 50% NaN → 5M valid → downsampled only if >500k valid

### Example Export

```python
def save_stl(fname, surface, binary=True, max_points=500000):
    """Export height map as STL mesh."""
    # Extract coordinates from Surface (includes x0, y0)
    xi = surface.xi
    yi = surface.yi
    grid = surface.height
    
    # Downsample large grids automatically
    h, w = grid.shape
    if h * w > max_points:
        stride = int(np.ceil(np.sqrt(h * w / max_points)))
        grid = grid[::stride, ::stride]
        xi = xi[::stride]
        yi = yi[::stride]
    
    # Convert μm to mm
    xi_mm = xi / 1000.0
    yi_mm = yi / 1000.0
    grid_mm = grid / 1000.0
    
    # Create mesh (trimesh library)
    xx, yy = np.meshgrid(xi_mm, yi_mm)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), grid_mm.ravel()])
    
    # Remove NaN vertices
    valid = ~np.isnan(vertices[:, 2])
    vertices = vertices[valid]
    
    # Create faces (triangulation)
    # ... (mesh triangulation logic)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(fname, file_type='stl_ascii' if not binary else 'stl')
```

---

## Unit Conventions

### Universal Standard: Micrometers (μm)

**All internal processing uses micrometers.**

| Quantity | Internal Unit | Common Input Units |
|----------|---------------|-------------------|
| X/Y coordinates | μm | mm, μm |
| Z heights | μm | mm, μm |
| Pixel sizes (px_x, px_y) | μm | mm, μm |
| Filter sigmas | μm | μm |
| Physical dimensions | μm | μm |

### Conversion Rules

```python
# Input (user specifies unit)
if input_unit == 'mm':
    value_um = value_mm * 1000.0
elif input_unit == 'um':
    value_um = value_um
else:
    raise ValueError(f"Unknown unit: {input_unit}")

# Output (STL export)
value_mm = value_um / 1000.0
```

### Heuristic for Unit Detection

When units are ambiguous (no user input), use heuristics:

```python
def suggest_units(fname):
    """Suggest units based on typical values."""
    sample = load_sample(fname)
    
    # Check XY step size
    px_x_raw = median_step(sample['x'])
    px_y_raw = median_step(sample['y'])
    
    # If step < 0.1, likely mm; if step > 1, likely μm
    suggested_xy = 'mm' if px_x_raw < 0.1 else 'um'
    
    # Check Z range
    z_range = sample['z'].max() - sample['z'].min()
    suggested_z = 'mm' if z_range < 1 else 'um'
    
    return suggested_xy, suggested_z
```

**Real example:** See `io/loaders.py::suggest_units()`

---

## Format Selection Guidelines

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| Save work in progress | NPZ | Fast, preserves all data including NaN |
| Archive/share datasets | HDF5 | Compressed, metadata support, widespread |
| Import from device | CSV | Device output, human-readable |
| Export for 3D printing | STL | Standard 3D format |
| Export for visualization | STL | Import to Blender, MeshLab, etc. |
| Batch processing | NPZ | Fastest I/O |

---

## Validation Requirements

All loaders should validate:

1. **File exists and is readable**
2. **Format is correct** (magic numbers, headers)
3. **Required fields present** (grid, coordinates, pixel sizes)
4. **Data types correct** (arrays are numeric, shapes match)
5. **Units are reasonable** (pixel sizes > 0, grid not empty)
6. **Warn on suspicious values** (very large/small pixel sizes)

---

## Error Handling

```python
def load_format(fname):
    """Standard error handling pattern."""
    try:
        # Attempt load
        data = parse_file(fname)
        
        # Validate
        if not validate_data(data):
            raise ValueError("Invalid data structure")
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fname}")
    except ValueError as e:
        raise ValueError(f"Invalid {format} file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading {fname}: {e}")
        raise RuntimeError(f"Failed to load file: {e}")
```

---

## Testing Requirements

Every format should have tests for:

- **Round-trip:** Save then load, verify data matches
- **Unit conversion:** Verify mm ↔ μm conversion
- **NaN preservation:** NaN values survive save/load
- **Edge cases:** Empty arrays, single-point grids
- **Corrupted files:** Graceful error messages

---

**Last Updated:** 2026-02-18
