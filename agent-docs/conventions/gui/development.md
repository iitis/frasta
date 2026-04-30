# GUI Development Guide

**Purpose:** Patterns and best practices for developing GUI components in `frasta/gui/`.

**Audience:** Developers adding dialogs, views, or GUI features.

---

## Table of Contents

1. [Parameter Dialog Pattern](#parameter-dialog-pattern)
2. [Worker Thread Pattern](#worker-thread-pattern)
3. [Connecting Menu to Processing](#connecting-menu-to-processing)
4. [Signal/Slot Best Practices](#signalslot-best-practices)
5. [Tab Management](#tab-management)

---

## Parameter Dialog Pattern

### Standard Structure

```python
from PyQt5 import QtWidgets

class MyProcessingDialog(QtWidgets.QDialog):
    """Dialog for collecting processing parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("My Processing Operation")
        self.params = {}
        self.init_ui()
    
    def init_ui(self):
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout()
        
        # Parameter inputs
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QFormLayout()
        
        # Example: Double spin box
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 100.0)
        self.sigma_spin.setValue(5.0)
        self.sigma_spin.setSingleStep(0.5)
        param_layout.addRow("Sigma:", self.sigma_spin)
        
        # Example: Combo box
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Method A", "Method B"])
        param_layout.addRow("Method:", self.method_combo)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # OK/Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Return dict of user-selected parameters."""
        return {
            'sigma': self.sigma_spin.value(),
            'method': self.method_combo.currentText()
        }
```

### Usage in MainWindow

```python
def on_my_processing_action(self):
    """Handle menu action."""
    dialog = MyProcessingDialog(self)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        params = dialog.get_parameters()
        self.apply_processing(params)

def apply_processing(self, params):
    """Apply processing with parameters."""
    from ..processing import my_function
    
    current_tab = self.get_current_tab()
    if current_tab is None:
        return
    
    result = my_function(
        current_tab.grid,
        sigma=params['sigma'],
        px_x=current_tab.px_x,
        px_y=current_tab.px_y
    )
    
    current_tab.grid = result
    current_tab.update_display()
```

---

## Worker Thread Pattern

### For Long-Running Operations

```python
from PyQt5 import QtCore

class ProcessingWorker(QtCore.QThread):
    """Worker thread for heavy computation."""
    
    # Signals
    progress = QtCore.pyqtSignal(int)  # 0-100
    finished = QtCore.pyqtSignal(object)  # Result
    error = QtCore.pyqtSignal(str)  # Error message
    
    def __init__(self, grid, params):
        super().__init__()
        self.grid = grid
        self.params = params
    
    def run(self):
        """Execute in background thread."""
        try:
            from ..processing import expensive_operation
            
            # Perform computation
            result = expensive_operation(
                self.grid,
                **self.params,
                progress_callback=self.emit_progress
            )
            
            # Emit success
            self.finished.emit(result)
            
        except Exception as e:
            # Emit error
            self.error.emit(str(e))
    
    def emit_progress(self, value):
        """Callback for progress updates."""
        self.progress.emit(value)
```

### Usage

```python
def start_long_operation(self):
    """Start worker and show progress."""
    current_tab = self.get_current_tab()
    
    # Create progress dialog
    progress = QtWidgets.QProgressDialog(
        "Processing...", "Cancel", 0, 100, self
    )
    progress.setWindowModality(QtCore.Qt.WindowModal)
    
    # Create worker
    self.worker = ProcessingWorker(current_tab.grid, params)
    
    # Connect signals
    self.worker.progress.connect(progress.setValue)
    self.worker.finished.connect(self.on_processing_complete)
    self.worker.error.connect(self.on_processing_error)
    self.worker.finished.connect(progress.close)
    self.worker.error.connect(progress.close)
    
    # Start
    self.worker.start()

def on_processing_complete(self, result):
    """Handle successful completion."""
    current_tab = self.get_current_tab()
    current_tab.grid = result
    current_tab.update_display()

def on_processing_error(self, error_msg):
    """Handle error."""
    QtWidgets.QMessageBox.critical(
        self, "Processing Error", f"Failed: {error_msg}"
    )
```

---

## Connecting Menu to Processing

### Step 1: Add Menu Action

```python
# In MainWindow.__init__()
def create_menus(self):
    """Create menu bar."""
    menubar = self.menuBar()
    
    # Processing menu
    process_menu = menubar.addMenu("Processing")
    
    # Add action
    my_action = QtWidgets.QAction("My Filter", self)
    my_action.triggered.connect(self.on_my_filter)
    process_menu.addAction(my_action)
```

### Step 2: Implement Handler

```python
def on_my_filter(self):
    """Handle 'My Filter' menu action."""
    # Check if tab is active
    current_tab = self.get_current_tab()
    if current_tab is None:
        QtWidgets.QMessageBox.warning(
            self, "No Data", "Please load a scan first."
        )
        return
    
    # Show parameter dialog (optional)
    dialog = MyFilterDialog(self)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return  # User cancelled
    
    params = dialog.get_parameters()
    
    # Apply processing
    from ..processing import my_filter
    
    result = my_filter(
        current_tab.grid,
        sigma=params['sigma'],
        px_x=current_tab.px_x,
        px_y=current_tab.px_y,
        mask=current_tab.get_mask()  # If ROI is active
    )
    
    # Update display
    current_tab.grid = result
    current_tab.update_display()
```

---

## Signal/Slot Best Practices

### Rule 1: Use Typed Signals

```python
# GOOD OK
class MyWidget(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(float)  # Explicit type
    
# BAD BAD
class MyWidget(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal()  # No type info
```

### Rule 2: Disconnect Before Reconnect

```python
# Avoid duplicate connections
try:
    self.button.clicked.disconnect()
except TypeError:
    pass  # Not connected yet
self.button.clicked.connect(self.on_click)
```

### Rule 3: Use Lambda for Parameters

```python
# Pass parameters via lambda
for i, item in enumerate(items):
    button = QtWidgets.QPushButton(f"Item {i}")
    button.clicked.connect(lambda checked, idx=i: self.on_item_click(idx))
```

---

## Tab Management

### Get Current Tab

```python
def get_current_tab(self):
    """Get currently active ScanTab."""
    current_widget = self.tab_widget.currentWidget()
    if isinstance(current_widget, ScanTab):
        return current_widget
    return None
```

### Iterate All Tabs

```python
def process_all_tabs(self):
    """Apply operation to all tabs."""
    for i in range(self.tab_widget.count()):
        tab = self.tab_widget.widget(i)
        if isinstance(tab, ScanTab):
            # Process tab
            self.process_single_tab(tab)
```

### Add New Tab

```python
def add_scan_tab(self, name, surface):
    """Create and add new scan tab from Surface object."""
    tab = ScanTab(parent=self)
    tab.setSurface(surface)
    
    index = self.tab_widget.addTab(tab, name)
    self.tab_widget.setCurrentIndex(index)
```

---

## Checklist for New GUI Feature

- [ ] Dialog inherits from appropriate Qt class
- [ ] Parameters collected via `get_parameters()` method
- [ ] Long operations use QThread workers
- [ ] Progress reporting for operations > 1 second
- [ ] Error handling with user-friendly messages
- [ ] Menu action connected properly
- [ ] Works when no tab is active (graceful warning)
- [ ] Updates display after processing
- [ ] Respects ROI/mask if applicable

---

**Last Updated:** 2026-02-18