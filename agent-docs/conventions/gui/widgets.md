# Widget Development Conventions

**Purpose:** Guidelines for creating custom PyQt5 widgets.

**Audience:** Developers adding reusable GUI components.

---

## Custom Widget Pattern

```python
from PyQt5 import QtWidgets, QtCore

class MyCustomWidget(QtWidgets.QWidget):
    """Custom widget for specific purpose."""
    
    # Define custom signals
    value_changed = QtCore.pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Build widget layout."""
        layout = QtWidgets.QVBoxLayout()
        
        # Add controls
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)
        
        self.setLayout(layout)
    
    def _on_value_changed(self, value):
        """Internal handler - emit custom signal."""
        self.value_changed.emit(float(value))
    
    def get_value(self):
        """Public API for getting value."""
        return self.slider.value()
    
    def set_value(self, value):
        """Public API for setting value."""
        self.slider.setValue(int(value))
```

## Key Principles

1. **Encapsulation:** Widget manages its own state
2. **Signals for communication:** Use signals instead of callbacks
3. **Public API:** Provide clear `get_*()` and `set_*()` methods
4. **Internal methods:** Prefix with `_` (e.g., `_on_value_changed`)
5. **Documentation:** Document signals and public methods

---

**Last Updated:** 2026-02-18