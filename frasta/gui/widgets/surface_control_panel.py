"""Surface control panel widgets for 3D viewer.

Provides reusable control panel widgets for managing surface visualization
settings including rendering mode, colormap, and value ranges.
"""

from pyqtgraph.Qt import QtWidgets, QtCore


class SurfaceControlWidget(QtWidgets.QWidget):
    """Control panel for configuring a single surface visualization.
    
    Provides controls for:
    - Surface visibility toggle
    - Rendering mode selection (surface/wireframe/mesh)
    - Colormap selection
    - Value range controls (auto/manual, lo/hi spinboxes)
    
    Signals:
        visibility_changed (bool): Emitted when visibility checkbox is toggled.
        mode_changed (int): Emitted when rendering mode is changed.
        colormap_changed (int): Emitted when colormap is changed.
        auto_range_toggled (bool): Emitted when auto-range checkbox is toggled.
        range_value_changed (): Emitted when manual range values change.
    """
    
    visibility_changed = QtCore.pyqtSignal(bool)
    mode_changed = QtCore.pyqtSignal(int)
    colormap_changed = QtCore.pyqtSignal(int)
    auto_range_toggled = QtCore.pyqtSignal(bool)
    range_lo_changed = QtCore.pyqtSignal(float)
    range_hi_changed = QtCore.pyqtSignal(float)
    
    def __init__(self, label_text="Surface", default_mode='surface', parent=None):
        """Initialize surface control widget.
        
        Args:
            label_text (str): Label for the surface (e.g., "Ref surface", "Adj surface").
            default_mode (str): Default rendering mode ('surface', 'wireframe', 'mesh').
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self._setup_ui(label_text, default_mode)
        self._connect_signals()
        
    def _setup_ui(self, label_text, default_mode):
        """Create and layout all control widgets."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Visibility checkbox
        self.checkbox = QtWidgets.QCheckBox(label_text + ":")
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)
        layout.addSpacing(12)
        
        # Rendering mode combo
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Surface (shaded)", userData='surface')
        self.combo_mode.addItem("Wireframe", userData='wireframe')
        self.combo_mode.addItem("Mesh", userData='mesh')
        idx = self.combo_mode.findData(default_mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)
            
        # Colormap combo
        self.combo_colormap = QtWidgets.QComboBox()
        self.combo_colormap.addItems(["None", "RG", "B&W", "viridis", "plasma", "magma"])
        self.combo_colormap.setCurrentText('RG')
        
        layout.addWidget(QtWidgets.QLabel("mode:"))
        layout.addWidget(self.combo_mode)
        layout.addSpacing(12)
        layout.addWidget(QtWidgets.QLabel("colormap:"))
        layout.addWidget(self.combo_colormap)
        
        # Range controls
        self.chk_auto = QtWidgets.QCheckBox("Auto")
        self.chk_auto.setChecked(True)
        
        self.spin_lo = QtWidgets.QDoubleSpinBox()
        self.spin_hi = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_lo, self.spin_hi):
            sp.setDecimals(6)
            sp.setRange(-1e12, 1e12)
            sp.setSingleStep(0.1)
            sp.setEnabled(False)
            
        layout.addWidget(QtWidgets.QLabel(label_text[:3] + " lo/hi:"))
        layout.addWidget(self.spin_lo)
        layout.addWidget(self.spin_hi)
        layout.addWidget(self.chk_auto)
        layout.addStretch(1)
        
    def _connect_signals(self):
        """Connect internal widget signals to public signals."""
        self.checkbox.stateChanged.connect(
            lambda state: self.visibility_changed.emit(bool(state)))
        self.combo_mode.currentIndexChanged.connect(self.mode_changed.emit)
        self.combo_colormap.currentIndexChanged.connect(self.colormap_changed.emit)
        self.chk_auto.toggled.connect(self.auto_range_toggled.emit)
        self.spin_lo.valueChanged.connect(self.range_lo_changed.emit)
        self.spin_hi.valueChanged.connect(self.range_hi_changed.emit)
        
    def get_mode(self):
        """Get current rendering mode ('surface', 'wireframe', or 'mesh')."""
        return self.combo_mode.currentData()
        
    def get_colormap(self):
        """Get current colormap name or None if "None" is selected."""
        txt = self.combo_colormap.currentText()
        return None if txt == "None" else txt
        
    def is_auto_range(self):
        """Check if auto-range is enabled."""
        return self.chk_auto.isChecked()
        
    def get_range(self):
        """Get current manual range values as (lo, hi) tuple."""
        return (self.spin_lo.value(), self.spin_hi.value())
        
    def set_range(self, lo, hi, block_signals=True):
        """Set range spinbox values.
        
        Args:
            lo (float): Low value.
            hi (float): High value.
            block_signals (bool): Whether to block signals during update.
        """
        if block_signals:
            self.spin_lo.blockSignals(True)
            self.spin_hi.blockSignals(True)
        self.spin_lo.setValue(lo)
        self.spin_hi.setValue(hi)
        if block_signals:
            self.spin_lo.blockSignals(False)
            self.spin_hi.blockSignals(False)
            
    def set_auto_range(self, auto):
        """Enable or disable auto-range mode.
        
        Args:
            auto (bool): True to enable auto-range, False for manual.
        """
        self.chk_auto.setChecked(auto)
        self.spin_lo.setEnabled(not auto)
        self.spin_hi.setEnabled(not auto)
        
    def set_range_enabled(self, enabled):
        """Enable or disable range spinboxes (useful for linked mode).
        
        Args:
            enabled (bool): Whether spinboxes should be enabled.
        """
        if not self.chk_auto.isChecked():
            self.spin_lo.setEnabled(enabled)
            self.spin_hi.setEnabled(enabled)


class ControlsPanel(QtWidgets.QWidget):
    """Complete control panel for 3D viewer with both surface controls.
    
    Contains:
    - Reference surface controls
    - Adjusted surface controls
    - Profile line and section plane visibility toggles
    - Range linking checkbox
    """
    
    def __init__(self, parent=None):
        """Initialize the complete controls panel.
        
        Args:
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                          QtWidgets.QSizePolicy.Fixed)
        self._setup_ui()
        
    def _setup_ui(self):
        """Create and layout all control widgets."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        
        # Reference surface controls
        self.ref_controls = SurfaceControlWidget("Ref surface", parent=self)
        self.ref_controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Fixed)
        self.ref_controls.setMaximumHeight(self.ref_controls.sizeHint().height())
        main_layout.addWidget(self.ref_controls)
        
        # Adjusted surface controls
        self.adj_controls = SurfaceControlWidget("Adj surface", parent=self)
        self.adj_controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Fixed)
        self.adj_controls.setMaximumHeight(self.adj_controls.sizeHint().height())
        
        # Add link checkbox to adjusted controls layout
        self.chk_link_ranges = QtWidgets.QCheckBox("Link ranges")
        self.chk_link_ranges.setChecked(False)
        self.adj_controls.layout().addWidget(self.chk_link_ranges)
        
        main_layout.addWidget(self.adj_controls)
        
        # Additional controls (profile line and section plane)
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)
        ctrl_layout.addWidget(self.checkbox_line)
        ctrl_layout.addWidget(self.checkbox_plane)
        
        main_layout.addLayout(ctrl_layout)
    
    def update_visibility(self, has_adjusted_surface=False, has_profile_line=False):
        """Update visibility of controls based on available data.
        
        Args:
            has_adjusted_surface (bool): Whether adjusted surface data is present.
            has_profile_line (bool): Whether profile line data is present.
        """
        # Adjusted surface controls and link checkbox
        self.adj_controls.setVisible(has_adjusted_surface)
        self.chk_link_ranges.setVisible(has_adjusted_surface)
        
        # Profile line and section plane controls
        self.checkbox_line.setVisible(has_profile_line)
        self.checkbox_plane.setVisible(has_profile_line)
