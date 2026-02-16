# Migration Guide: src/ → frasta/

**Status: ✅ Migration Completed**

This guide documents the migration from the old `src/` structure to the new `frasta/` modular architecture. The old `src/` directory has been removed.

## Quick Reference

### Core Data Structures
```python
# OLD
from src.gridData import GridData

# NEW
from frasta.core import GridData
```

### Processing/Analysis
```python
# OLD
from src.helpers import (
    fill_holes,
    remove_relative_offset,
    remove_relative_tilt,
    nan_aware_gaussian,
    remove_outliers,
    compute_offset_global,
    compute_offset_in_center
)

# NEW
from frasta.processing import (
    fill_holes,
    remove_relative_offset,
    remove_relative_tilt,
    nan_aware_gaussian,
    remove_outliers,
    compute_offset_global,
    compute_offset_in_center
)
```

### Utilities
```python
# OLD
from src.helpers import resource_path, measure_time

# NEW
from frasta.utils import resource_path, measure_time
```

### File I/O
```python
# OLD
# GridWorker was in src.frasta_gui
# No dedicated loader/exporter functions

# NEW
from frasta.io import (
    load_csv_data,
    load_npz_data,
    load_h5_data,
    save_npz,
    save_h5,
    suggest_units
)
```

### GUI Components
```python
# OLD
from src.frasta_gui import MainWindow
from src.scanTab import ScanTab
from src.aboutDialog import AboutDialog
from src.overlayViewer import OverlayViewer
from src.profileViewer import ProfileViewer  
from src.grid3DViewer import Grid3DViewer, show_3d_viewer
from src.limitedGLView import LimitedGLView
from src.lodSurface import LODSurface
from src.responsiveInfiniteLine import ResponsiveInfiniteLine

# NEW
from frasta.gui import MainWindow, ScanTab
from frasta.gui.dialogs import AboutDialog, OverlayViewer, ProfileViewer
from frasta.gui.viewers import Grid3DViewer, show_3d_viewer, LimitedGLView, LODSurface
from frasta.gui.widgets import ResponsiveInfiniteLine
```

## Automated Migration

### Using Find & Replace (VS Code / PyCharm)

1. **Find:** `from src\.gridData import`  
   **Replace:** `from frasta.core import`

2. **Find:** `from src\.helpers import ([\s\S]*?fill_holes[\s\S]*?)`  
   **Replace:** `from frasta.processing import $1`

3. **Find:** `from src\.helpers import ([\s\S]*?resource_path[\s\S]*?)`  
   **Replace:** `from frasta.utils import $1`

4. **Find:** `from src\.frasta_gui import MainWindow`  
   **Replace:** `from frasta.gui import MainWindow`

5. **Find:** `from src\.scanTab import`  
   **Replace:** `from frasta.gui import`

6. **Find:** `from src\.(aboutDialog|overlayViewer|profileViewer) import`  
   **Replace:** `from frasta.gui.dialogs.$1 import`

7. **Find:** `from src\.(grid3DViewer|limitedGLView|lodSurface) import`  
   **Replace:** `from frasta.gui.viewers import`

8. **Find:** `from src\.responsiveInfiniteLine import`  
   **Replace:** `from frasta.gui.widgets import`

### Python Script for Batch Migration

```python
#!/usr/bin/env python3
"""Migrate imports from src/ to frasta/ structure."""
import re
from pathlib import Path

REPLACEMENTS = [
    (r'from src\.gridData import', 'from frasta.core import'),
    (r'from src\.helpers import', 'from frasta.processing import'),
    (r'from src\.frasta_gui import MainWindow', 'from frasta.gui import MainWindow'),
    (r'from src\.scanTab import', 'from frasta.gui import'),
    (r'from src\.aboutDialog import', 'from frasta.gui.dialogs import'),
    (r'from src\.overlayViewer import', 'from frasta.gui.dialogs import'),
    (r'from src\.profileViewer import', 'from frasta.gui.dialogs import'),
    (r'from src\.grid3DViewer import', 'from frasta.gui.viewers import'),
    (r'from src\.limitedGLView import', 'from frasta.gui.viewers import'),
    (r'from src\.lodSurface import', 'from frasta.gui.viewers import'),
    (r'from src\.responsiveInfiniteLine import', 'from frasta.gui.widgets import'),
]

def migrate_file(filepath: Path):
    """Migrate a single Python file."""
    content = filepath.read_text(encoding='utf-8')
    original = content
    
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        print(f"✓ Migrated: {filepath}")
        return True
    return False

def migrate_directory(root: Path):
    """Migrate all Python files in directory."""
    migrated = 0
    for pyfile in root.rglob("*.py"):
        if migrate_file(pyfile):
            migrated += 1
    print(f"\nTotal files migrated: {migrated}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python migrate_imports.py <directory>")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    migrate_directory(root_dir)
```

## Common Migration Issues

### Issue 1: helpers.py split into multiple modules

**Problem:**
```python
from src.helpers import fill_holes, resource_path
```

**Solution:**
```python
from frasta.processing import fill_holes
from frasta.utils import resource_path
```

### Issue 2: GridWorker moved to io

**Problem:**
```python
# Old GridWorker usage in tests
from src.frasta_gui import GridWorker
```

**Solution:**
GridWorker is now internal to `frasta.gui.main_window`. Use `frasta.io.load_csv_data` directly:

```python
from frasta.io import load_csv_data

# Direct usage
grid, xi, yi, px_x, px_y = load_csv_data(
    filename,
    units_xy='um',
    units_z='um',
    progress_callback=lambda p: print(f"Progress: {p}%")
)
```

### Issue 3: Circular imports

**Problem:**
You might encounter circular imports if old code had complex interdependencies.

**Solution:**
- Import only what you need
- Use `from frasta.core import GridData` instead of `import frasta.core`
- Consider restructuring code to remove circular dependencies

## Testing After Migration

1. **Run tests:**
   ```bash
   .venv/Scripts/python -m pytest tests/ -v
   ```

2. **Check imports:**
   ```python
   python -c "from frasta.core import GridData; from frasta.processing import fill_holes; print('OK')"
   ```

3. **Run application:**
   ```bash
   .venv/Scripts/python main.py
   ```

## Migration Complete ✅

The old `src/` directory has been removed. All code now uses the new `frasta/` modular structure.

For version control rollback, use git:
```bash
git revert <commit-hash>
```

## Questions?

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation of the new structure.
