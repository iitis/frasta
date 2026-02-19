# Loader and Exporter Conventions

**Purpose:** Patterns for implementing file I/O functions in `frasta/io/`.

**Audience:** Developers adding support for new file formats.

---

## 📋 Table of Contents

1. [Loader Function Pattern](#loader-function-pattern)
2. [Exporter Function Pattern](#exporter-function-pattern)
3. [Error Handling](#error-handling)
4. [Progress Reporting](#progress-reporting)
5. [Unit Conversion](#unit-conversion)

---

## Loader Function Pattern

### Standard Signature

```python
def load_format_data(fname, progress_callback=None, **format_specific_params):
    """Loads scan data from FORMAT.
    
    Args:
        fname (str): Path to file.
        progress_callback (callable, optional): Function(progress: int) for 0-100 updates.
        **format_specific_params: Format-specific parameters (e.g., units, encoding).
        
    Returns:
        Surface or list[Surface]: 
            - Single scan: Surface object
            - Multiple scans: list of Surface objects
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid or data is malformed.
        RuntimeError: For unexpected errors.
    """
    pass
```

### Implementation Template

```python
def load_my_format(fname, progress_callback=None):
    """Load MY_FORMAT file."""
    import logging
    from ..core import Surface
    
    logger = logging.getLogger(__name__)
    
    # Step 1: Validate file exists
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")
    
    if progress_callback:
        progress_callback(10)
    
    try:
        # Step 2: Parse file
        with open(fname, 'r') as f:
            data = parse_my_format(f)
        
        if progress_callback:
            progress_callback(40)
        
        # Step 3: Validate structure
        validate_data(data)
        
        if progress_callback:
            progress_callback(60)
        
        # Step 4: Convert to standard format
        grid, xi, yi, px_x, px_y = convert_to_grid(data)
        
        if progress_callback:
            progress_callback(80)
        
        # Step 5: Apply unit conversions
        grid, xi, yi, px_x, px_y = convert_units(grid, xi, yi, px_x, px_y)
        
        # Step 6: Generate name from filename
        name = os.path.splitext(os.path.basename(fname))[0]
        
        # Step 7: Create Surface object
        surface = Surface(
            height=grid,
            dx=px_x,
            dy=px_y,
            x0=xi[0] if len(xi) > 0 else 0.0,
            y0=yi[0] if len(yi) > 0 else 0.0,
            unit="µm",
            metadata={"name": name}
        )
        
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Loaded {name}: {grid.shape} grid, px={px_x:.2f}μm")
        
        return surface
        
    except ValueError as e:
        raise ValueError(f"Invalid MY_FORMAT file: {e}")
    except Exception as e:
        logger.error(f"Error loading {fname}: {e}")
        raise RuntimeError(f"Failed to load: {e}")
```

### Key Rules

1. **Return Surface objects:** Single `Surface` or `list[Surface]` for multiple scans
2. **Preserve spatial positioning:** Set `x0` and `y0` from coordinate arrays
3. **Store metadata:** Use `Surface.metadata` for scan name and other info
4. **Coordinates in micrometers:** Convert at I/O boundary
5. **NaN for missing data:** Use `np.nan`, not 0 or sentinel values
6. **Log important info:** File loaded, dimensions, warnings
7. **Progress at key milestones:** 10%, 40%, 60%, 80%, 100%

---

## Exporter Function Pattern

### Standard Signature

```python
def save_format(fname, scans, **format_specific_params):
    """Saves scan data to FORMAT.
    
    Args:
        fname (str): Path to output file.
        scans (list): List of (name, Surface) tuples.
        **format_specific_params: Format-specific parameters.
        
    Raises:
        ValueError: If scans list is empty or malformed.
        RuntimeError: If write fails.
    """
    pass
```

### Implementation Template

```python
def save_my_format(fname, scans, compression=True):
    """Save to MY_FORMAT."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Step 1: Validate input
    if not scans:
        raise ValueError("scans list is empty")
    
    if not all(isinstance(s[1], Surface) for s in scans):
        raise ValueError("Each scan must be (name, Surface)")
    
    try:
        # Step 2: Write file
        with open(fname, 'wb') as f:
            write_header(f, len(scans))
            
            for name, surface in scans:
                # Convert units if necessary for format
                grid_mm = surface.height / 1000.0  # example: μm -> mm
                xi_mm = surface.xi / 1000.0
                yi_mm = surface.yi / 1000.0
                
                write_scan(f, name, grid_mm, xi_mm, yi_mm, 
                          surface.dx / 1000.0, surface.dy / 1000.0)
        
        logger.info(f"Saved {len(scans)} scan(s) to {fname}")
        
    except Exception as e:
        logger.error(f"Error saving to {fname}: {e}")
        raise RuntimeError(f"Failed to save: {e}")
```

### Key Rules

1. **Accept list of (name, Surface) tuples:** Even if format supports only one
2. **Extract data from Surface:** Use `surface.height`, `surface.xi`, `surface.yi`, etc.
3. **Convert units at boundary:** Internal μm → format's expected units
4. **Handle NaN appropriately:** Skip, fill, or format-specific encoding
5. **Atomic writes if possible:** Write to temp file, then rename
6. **Log on success:** Number of scans, file size

---

## Error Handling

### Exception Hierarchy

```python
# File not found (user error)
raise FileNotFoundError(f"File not found: {fname}")

# Format invalid (user provided wrong file)
raise ValueError(f"Not a valid MY_FORMAT file: missing header")

# Data malformed (file corrupted)
raise ValueError(f"Corrupted data: {reason}")

# Unexpected errors (bugs/system issues)
raise RuntimeError(f"Unexpected error: {e}")
```

### Informative Messages

```python
# BAD ❌
raise ValueError("Invalid file")

# GOOD ✅
raise ValueError(
    f"Invalid MY_FORMAT file '{fname}': "
    f"expected magic number 0x1234, got 0x5678"
)
```

---

## Progress Reporting

### Pattern

```python
def load_with_progress(fname, progress_callback=None):
    """Use progress_callback if provided."""
    
    # Check if callback provided
    if progress_callback:
        progress_callback(0)
    
    # Parse file
    data = parse(fname)
    
    if progress_callback:
        progress_callback(50)
    
    # Process data
    result = process(data)
    
    if progress_callback:
        progress_callback(100)
    
    return result
```

### Milestones

- **0-10%:** File opening
- **10-40%:** Parsing/reading
- **40-60%:** Validation
- **60-80%:** Conversion to internal format
- **80-100%:** Finalization

---

## Unit Conversion

### Import (File → Internal)

```python
def convert_to_internal_units(value, unit):
    """Convert input units to micrometers."""
    if unit == 'mm':
        return value * 1000.0
    elif unit == 'um' or unit == 'μm':
        return value
    elif unit == 'nm':
        return value / 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")
```

### Export (Internal → File)

```python
def convert_from_internal_units(value_um, target_unit):
    """Convert from micrometers to target unit."""
    if target_unit == 'mm':
        return value_um / 1000.0
    elif target_unit == 'um':
        return value_um
    else:
        raise ValueError(f"Unknown target unit: {target_unit}")
```

---

## Testing Requirements

```python
# In tests/test_io.py
from frasta.core import Surface

class TestMyFormatLoader:
    def test_loads_valid_file(self):
        """Test loading a known-good file."""
        surface = load_my_format('test_data/valid.myformat')
        
        assert isinstance(surface, Surface)
        assert surface.metadata.get('name') == "valid"
        assert surface.height.shape == (100, 100)
        assert surface.dx > 0
    
    def test_raises_on_missing_file(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_my_format('nonexistent.myformat')
    
    def test_unit_conversion(self):
        """Test that units are converted correctly."""
        # Load file known to be in mm
        surface = load_my_format('test_data/mm_units.myformat')
        
        # Should be converted to μm internally
        assert surface.dx > 100  # If original was ~0.1mm
    
    def test_preserves_nan(self):
        """Test that NaN values are preserved."""
        surface = load_my_format('test_data/with_nan.myformat')
        
        assert np.any(np.isnan(surface.height))
    
    def test_roundtrip(self):
        """Test save then load preserves data."""
        original_surface = Surface(
            height=np.random.rand(10, 10),
            dx=1.5, dy=1.5,
            x0=0.0, y0=0.0,
            metadata={"name": "test"}
        )
        original = [("test", original_surface)]
        
        save_my_format('temp.myformat', original)
        loaded = load_my_format('temp.myformat')
        
        assert loaded.metadata.get('name') == "test"
        np.testing.assert_array_almost_equal(loaded.height, original_surface.height)
```

---

## Checklist for New Format

- [ ] Loader follows standard signature
- [ ] Loader returns Surface object(s)
- [ ] Exporter follows standard signature
- [ ] Exporter accepts (name, Surface) tuples
- [ ] Preserves spatial positioning (x0, y0)
- [ ] Converts units to micrometers
- [ ] Handles NaN appropriately
- [ ] Progress reporting implemented
- [ ] Error messages are informative
- [ ] Validates file structure
- [ ] Tests: valid file, missing file, corrupted file, round-trip
- [ ] Added to `io/__init__.py` exports
- [ ] Documented in `file_formats.md`

---

**Last Updated:** 2026-02-19
