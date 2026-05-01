"""Advanced processing dialogs for FRASTA-toolbox.

Provides parameter dialogs for advanced filtering, morphology operations,
and geometric transformations.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui


class FilterDialog(QtWidgets.QDialog):
    """Dialog for advanced filtering operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Filtering")
        self.filter_type = None
        self.params = {}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Filter type selection
        type_group = QtWidgets.QGroupBox("Filter Type")
        type_layout = QtWidgets.QVBoxLayout()
        
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems([
            "Bilateral Filter",
            "Median Filter",
            "Morphological Opening",
            "Morphological Closing",
            "Robust Gaussian Filter"
        ])
        self.filter_combo.currentIndexChanged.connect(self.update_parameter_panel)
        type_layout.addWidget(self.filter_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Parameter panel (dynamic)
        self.param_group = QtWidgets.QGroupBox("Parameters")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.update_parameter_panel()
        
    def update_parameter_panel(self):
        """Update parameter inputs based on selected filter."""
        # Clear existing widgets
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        filter_name = self.filter_combo.currentText()
        
        if filter_name == "Bilateral Filter":
            self.add_param_spin("Spatial Sigma (px)", 1.0, 20.0, 5.0, 0.5, "sigma_spatial")
            self.add_param_spin("Range Sigma", 1.0, 50.0, 10.0, 1.0, "sigma_range")
            self.add_info("Edge-preserving smoothing. Preserves fracture features while reducing noise.")
            
        elif filter_name == "Median Filter":
            self.add_param_int("Window Size", 3, 15, 5, "size")
            self.add_info("Robust outlier removal. Removes measurement spikes without blurring.")
            
        elif filter_name == "Morphological Opening":
            self.add_param_int("Structure Size", 3, 15, 5, "size")
            self.add_info("Removes peaks smaller than structure size. Good for removing noise peaks.")
            
        elif filter_name == "Morphological Closing":
            self.add_param_int("Structure Size", 3, 15, 5, "size")
            self.add_info("Fills valleys smaller than structure size. Good for filling small holes.")
            
        elif filter_name == "Robust Gaussian Filter":
            self.add_param_spin("Sigma (px)", 0.5, 10.0, 2.0, 0.5, "sigma")
            self.add_param_int("Max Iterations", 1, 10, 3, "max_iterations")
            self.add_param_spin("Outlier Threshold (σ)", 1.0, 5.0, 3.0, 0.5, "outlier_threshold")
            self.add_info("Gaussian smoothing with iterative outlier rejection.")
    
    def add_param_spin(self, label, min_val, max_val, default, step, key):
        """Add a double spin box parameter."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", spin)
        
    def add_param_int(self, label, min_val, max_val, default, key):
        """Add an integer spin box parameter."""
        spin = QtWidgets.QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setProperty("param_key", key)
        # Ensure odd values only for window sizes
        if "size" in key.lower() or "window" in label.lower():
            spin.setSingleStep(2)  # Step by 2 to keep odd
            if default % 2 == 0:
                spin.setValue(default + 1)
        self.param_layout.addRow(label + ":", spin)
        
    def add_info(self, text):
        """Add informational text."""
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-style: italic;")
        self.param_layout.addRow(label)
    
    def get_filter_config(self):
        """Get selected filter type and parameters."""
        filter_map = {
            "Bilateral Filter": "bilateral",
            "Median Filter": "median",
            "Morphological Opening": "opening",
            "Morphological Closing": "closing",
            "Robust Gaussian Filter": "robust_gaussian"
        }
        
        filter_type = filter_map[self.filter_combo.currentText()]
        params = {}
        
        for i in range(self.param_layout.rowCount()):
            widget = self.param_layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
            if widget and widget.widget():
                w = widget.widget()
                if hasattr(w, 'property'):
                    key = w.property("param_key")
                    if key:
                        if isinstance(w, QtWidgets.QDoubleSpinBox):
                            params[key] = w.value()
                        elif isinstance(w, QtWidgets.QSpinBox):
                            params[key] = w.value()
        
        return filter_type, params


class MorphologyDialog(QtWidgets.QDialog):
    """Dialog for morphology and leveling operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Morphology & Leveling")
        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Operation type selection
        type_group = QtWidgets.QGroupBox("Operation")
        type_layout = QtWidgets.QVBoxLayout()
        
        self.op_combo = QtWidgets.QComboBox()
        self.op_combo.addItems([
            "Level by Plane (Least Squares)",
            "Level by Plane (Robust RANSAC)",
            "Remove Polynomial Form",
            "Threshold Grid"
        ])
        self.op_combo.currentIndexChanged.connect(self.update_parameter_panel)
        type_layout.addWidget(self.op_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Parameter panel
        self.param_group = QtWidgets.QGroupBox("Parameters")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.update_parameter_panel()
        
    def update_parameter_panel(self):
        """Update parameter inputs based on selected operation."""
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        op_name = self.op_combo.currentText()
        
        if "Level by Plane" in op_name:
            if "Robust" in op_name:
                self.add_param_spin("Inlier Threshold", 0.1, 100.0, 10.0, 1.0, "residual_threshold")
                self.add_info("RANSAC-based robust plane fitting. Threshold for inlier detection (lower = stricter).")
            else:
                self.add_info("Least-squares plane fitting. Fast but sensitive to outliers.")
                
        elif op_name == "Remove Polynomial Form":
            self.add_param_int("Polynomial Order", 1, 5, 2, "order")
            self.add_info("Remove polynomial surface. Order 1=plane, 2=parabolic, 3+=higher order.")
            
        elif op_name == "Threshold Grid":
            self.add_param_spin("Lower Bound", -1000.0, 1000.0, -100.0, 10.0, "lower")
            self.add_param_spin("Upper Bound", -1000.0, 1000.0, 100.0, 10.0, "upper")
            self.add_info("Mask values outside specified range. Values become NaN.")
    
    def add_param_spin(self, label, min_val, max_val, default, step, key):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", spin)
        
    def add_param_int(self, label, min_val, max_val, default, key):
        spin = QtWidgets.QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", spin)
        
    def add_info(self, text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-style: italic;")
        self.param_layout.addRow(label)
    
    def get_operation_config(self):
        """Get selected operation and parameters."""
        op_map = {
            "Level by Plane (Least Squares)": "level_ls",
            "Level by Plane (Robust RANSAC)": "level_robust",
            "Remove Polynomial Form": "polynomial",
            "Threshold Grid": "threshold"
        }
        
        op_type = op_map[self.op_combo.currentText()]
        params = {}
        
        for i in range(self.param_layout.rowCount()):
            widget = self.param_layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
            if widget and widget.widget():
                w = widget.widget()
                if hasattr(w, 'property'):
                    key = w.property("param_key")
                    if key:
                        if isinstance(w, QtWidgets.QDoubleSpinBox):
                            params[key] = w.value()
                        elif isinstance(w, QtWidgets.QSpinBox):
                            params[key] = w.value()
        
        return op_type, params


class TransformDialog(QtWidgets.QDialog):
    """Dialog for geometric transformations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Geometric Transforms")
        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Transform type selection
        type_group = QtWidgets.QGroupBox("Transform Type")
        type_layout = QtWidgets.QVBoxLayout()
        
        self.transform_combo = QtWidgets.QComboBox()
        self.transform_combo.addItems([
            "Rotate Grid",
            "Rescale Grid",
            "Crop to Valid Region"
        ])
        self.transform_combo.currentIndexChanged.connect(self.update_parameter_panel)
        type_layout.addWidget(self.transform_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Parameter panel
        self.param_group = QtWidgets.QGroupBox("Parameters")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.update_parameter_panel()
        
    def update_parameter_panel(self):
        """Update parameter inputs based on selected transform."""
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        transform_name = self.transform_combo.currentText()
        
        if transform_name == "Rotate Grid":
            self.add_param_spin("Angle (degrees)", -180.0, 180.0, 0.0, 1.0, "angle")
            self.add_param_combo("Interpolation", ["Nearest", "Linear", "Cubic"], "order")
            self.add_info("Rotate grid by specified angle. Cubic interpolation is recommended.")
            
        elif transform_name == "Rescale Grid":
            self.add_param_spin("Scale Factor", 0.1, 10.0, 1.0, 0.1, "scale")
            self.add_param_combo("Interpolation", ["Nearest", "Linear", "Cubic"], "order")
            self.add_info("Rescale grid resolution. Values >1 increase resolution, <1 decrease.")
            
        elif transform_name == "Crop to Valid Region":
            self.add_param_int("Margin (pixels)", 0, 50, 0, "margin")
            self.add_info("Automatically crop to non-NaN region with optional margin.")
    
    def add_param_spin(self, label, min_val, max_val, default, step, key):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", spin)
        
    def add_param_int(self, label, min_val, max_val, default, key):
        spin = QtWidgets.QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", spin)
        
    def add_param_combo(self, label, items, key):
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setProperty("param_key", key)
        self.param_layout.addRow(label + ":", combo)
        
    def add_info(self, text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-style: italic;")
        self.param_layout.addRow(label)
    
    def get_transform_config(self):
        """Get selected transform and parameters."""
        transform_map = {
            "Rotate Grid": "rotate",
            "Rescale Grid": "rescale",
            "Crop to Valid Region": "crop"
        }
        
        transform_type = transform_map[self.transform_combo.currentText()]
        params = {}
        
        for i in range(self.param_layout.rowCount()):
            widget = self.param_layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
            if widget and widget.widget():
                w = widget.widget()
                if hasattr(w, 'property'):
                    key = w.property("param_key")
                    if key:
                        if isinstance(w, QtWidgets.QDoubleSpinBox):
                            params[key] = w.value()
                        elif isinstance(w, QtWidgets.QSpinBox):
                            params[key] = w.value()
                        elif isinstance(w, QtWidgets.QComboBox):
                            order_map = {"Nearest": 0, "Linear": 1, "Cubic": 3}
                            params[key] = order_map.get(w.currentText(), 1)
        
        return transform_type, params


class RegistrationDialog(QtWidgets.QDialog):
    """Dialog for automatic surface registration."""
    
    def __init__(self, scan_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Surface Registration")
        self.scan_names = scan_names
        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Scan selection
        scan_group = QtWidgets.QGroupBox("Select Surfaces")
        scan_layout = QtWidgets.QFormLayout()
        
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems(self.scan_names)
        scan_layout.addRow("Reference Surface:", self.ref_combo)
        
        self.mov_combo = QtWidgets.QComboBox()
        self.mov_combo.addItems(self.scan_names)
        if len(self.scan_names) > 1:
            self.mov_combo.setCurrentIndex(1)
        scan_layout.addRow("Moving Surface:", self.mov_combo)
        
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)
        
        # Method selection
        method_group = QtWidgets.QGroupBox("Registration Method")
        method_layout = QtWidgets.QVBoxLayout()
        
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems([
            "Cross-Correlation (translation only)",
            "ICP (translation + rotation)"
        ])
        method_layout.addWidget(self.method_combo)

        self.refine_checkbox = QtWidgets.QCheckBox("Refine ICP alignment (slower)")
        self.refine_checkbox.setChecked(False)
        self.refine_checkbox.setEnabled(False)
        method_layout.addWidget(self.refine_checkbox)

        self.stable_region_checkbox = QtWidgets.QCheckBox("Auto reject mismatched areas (ICP)")
        self.stable_region_checkbox.setChecked(False)
        self.stable_region_checkbox.setEnabled(False)
        method_layout.addWidget(self.stable_region_checkbox)
        self.method_combo.currentIndexChanged.connect(self._update_refine_option_state)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Info
        info_label = QtWidgets.QLabel(
            "Automatic registration estimates alignment parameters for the moving surface. "
            "Cross-correlation updates translation only, while ICP estimates both translation and in-plane rotation. "
            "The stable-region option adds a second ICP pass on automatically selected low-mismatch overlap areas."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self._update_refine_option_state()

    def _update_refine_option_state(self):
        """Enable the ICP refinement option only for the ICP method."""
        is_icp = "ICP" in self.method_combo.currentText()
        self.refine_checkbox.setEnabled(is_icp)
        self.stable_region_checkbox.setEnabled(is_icp)
    
    def get_registration_config(self):
        """Get registration configuration."""
        ref_idx = self.ref_combo.currentIndex()
        mov_idx = self.mov_combo.currentIndex()
        method = "correlation" if "Cross-Correlation" in self.method_combo.currentText() else "icp"
        refine = bool(self.refine_checkbox.isChecked()) if method == "icp" else False
        stable_region = bool(self.stable_region_checkbox.isChecked()) if method == "icp" else False

        return ref_idx, mov_idx, method, refine, stable_region
