# Import Organization

**Purpose:** Standards for organizing imports in Python modules.

**Audience:** All developers.

---

## Import Order

Follow PEP 8 with FRASTA-specific conventions:

```python
"""Module docstring."""

# 1. Standard library imports
import os
import sys
import logging

# 2. Third-party imports (alphabetically)
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression

# 3. Optional third-party (try/except)
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 4. Local application imports (relative)
from ..core import Surface
from ..processing import bilateral_filter, median_filter
from .dialogs import FilterDialog
from .workers import GridWorker

# 5. Logging setup (if needed)
logger = logging.getLogger(__name__)
```

---

## Lazy Imports in GUI

GUI modules should import processing functions **only when needed**:

```python
# In main_window.py

# DON'T do this at module level BAD
from ..processing import bilateral_filter, median_filter, ...  # 20+ imports

# DO this in methods OK
def apply_bilateral_filter(self):
    from ..processing import bilateral_filter  # Import on demand
    result = bilateral_filter(...)
```

**Why?** Reduces startup time and allows processing to be used independently.

---

## Import Aliases

### Standard Aliases (Use These)

```python
import numpy as np                    OK
import pandas as pd                   OK
import matplotlib.pyplot as plt       OK
from PyQt5 import QtWidgets, QtCore   OK
```

### Avoid Ambiguous Aliases

```python
import numpy as n              BAD
import pandas as p             BAD
from scipy import *            BAD (star imports)
```

---

## Relative vs Absolute Imports

### Within Package (Use Relative)

```python
# In frasta/processing/advanced_filtering.py
from ..core import Surface           OK (relative)
from frasta.core import Surface      BAD (absolute in package)
```

### From External Code (Use Absolute)

```python
# In examples/ or tests/
from frasta.processing import bilateral_filter   OK
```

---

## Optional Dependencies

```python
# Module level - check availability
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available, using fallback implementation")

# Usage
def my_function(..., use_opencv=True):
    if use_opencv and HAS_OPENCV:
        return opencv_implementation(...)
    else:
        return python_implementation(...)
```

---

**Last Updated:** 2026-02-18