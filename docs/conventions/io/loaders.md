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
        tuple or list: 
            - Single scan: (name, grid, xi, yi, px_x, px_y)
            - Multiple scans: [(name1, grid1, ...), (name2, grid2, ...), ...]
        
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
        
        if progress_callback:
            progress_callback(100)
        
        # Step 6: Generate name from filename
        name = os.path.splitext(os.path.basename(fname))[0]
        
        logger.info(f"Loaded {name}: {grid.shape} grid, px={px_x:.2f}μm")
        
        return name, grid, xi, yi, px_x, px_y
        
    except ValueError as e:
        raise ValueError(f"Invalid MY_FORMAT file: {e}")
    except Exception as e:
        logger.error(f"Error loading {fname}: {e}")
        raise RuntimeError(f"Failed to load: {e}")
```

### Key Rules

1. **Always return same structure:** `(name, grid, xi, yi, px_x, px_y)` or list thereof
2. **Coordinates in micrometers:** Convert at I/O boundary
3. **NaN for missing data:** Use `np.nan`, not 0 or sentinel values
4. **Name from filename:** Extract base name, remove extension
5. **Log important info:** File loaded, dimensions, warnings
6. **Progress at key milestones:** 10%, 40%, 60%, 80%, 100%

---

## Exporter Function Pattern

### Standard Signature

```python
def save_format(fname, scans, **format_specific_params):
    """Saves scan data to FORMAT.
    
    Args:
        fname (str): Path to output file.
        scans (list): List of (name, grid, xi, yi, px_x, px_y) tuples.
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
    
    if not all(len(s) == 6 for s in scans):
        raise ValueError("Each scan must be (name, grid, xi, yi, px_x, px_y)")
    
    try:
        # Step 2: Convert units if necessary
        converted_scans = [convert_for_export(s) for s in scans]
        
        # Step 3: Write file
        with open(fname, 'wb') as f:
            write_header(f, len(converted_scans))
            
            for name, grid, xi, yi, px_x, px_y in converted_scans:
                write_scan(f, name, grid, xi, yi, px_x, px_y)
        
        logger.info(f"Saved {len(scans)} scan(s) to {fname}")
        
    except Exception as e:
        logger.error(f"Error saving to {fname}: {e}")
        raise RuntimeError(f"Failed to save: {e}")
```

### Key Rules

1. **Accept list of scans:** Even if format supports only one
2. **Convert units at boundary:** Internal μm → format's expected units
3. **Handle NaN appropriately:** Skip, fill, or format-specific encoding
4. **Atomic writes if possible:** Write to temp file, then rename
5. **Log on success:** Number of scans, file size

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

class TestMyFormatLoader:
    def test_loads_valid_file(self):
        """Test loading a known-good file."""
        name, grid, xi, yi, px_x, px_y = load_my_format('test_data/valid.myformat')
        
        assert name == "valid"
        assert grid.shape == (100, 100)
        assert len(xi) == 100
        assert px_x > 0
    
    def test_raises_on_missing_file(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_my_format('nonexistent.myformat')
    
    def test_unit_conversion(self):
        """Test that units are converted correctly."""
        # Load file known to be in mm
        _, grid, xi, yi, px_x, px_y = load_my_format('test_data/mm_units.myformat')
        
        # Should be converted to μm internally
        assert px_x > 100  # If original was ~0.1mm
    
    def test_preserves_nan(self):
        """Test that NaN values are preserved."""
        _, grid, *_ = load_my_format('test_data/with_nan.myformat')
        
        assert np.any(np.isnan(grid))
    
    def test_roundtrip(self):
        """Test save then load preserves data."""
        original = [("test", grid, xi, yi, px_x, px_y)]
        
        save_my_format('temp.myformat', original)
        loaded = load_my_format('temp.myformat')
        
        assert loaded[0] == "test"
        np.testing.assert_array_almost_equal(loaded[1], grid)
```

---

## Checklist for New Format

- [ ] Loader follows standard signature
- [ ] Exporter follows standard signature
- [ ] Returns correct tuple structure
- [ ] Converts units to micrometers
- [ ] Handles NaN appropriately
- [ ] Progress reporting implemented
- [ ] Error messages are informative
- [ ] Validates file structure
- [ ] Tests: valid file, missing file, corrupted file, round-trip
- [ ] Added to `io/__init__.py` exports
- [ ] Documented in `file_formats.md`

---

**Last Updated:** 2026-02-18
