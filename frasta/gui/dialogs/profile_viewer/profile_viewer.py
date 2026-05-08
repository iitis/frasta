"""Profile viewer for cross-sectional analysis of aligned scans.

This module provides tools for interactive cross-sectional analysis of two
aligned scan datasets, including profile plotting, contact point detection,
and 3D visualization of profile locations.
"""

import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtCore import QPointF

from .data_manager import DataManager
from .profile_analyzer import ProfileAnalyzer
from .roi_handler import ROIHandler
from .plot_interactions import PlotInteractions
from .visualization_manager import VisualizationManager

import logging
logger = logging.getLogger(__name__)


def create_image_view():
    """Creates a simplified Image View without histogram and ROI controls.
    
    Returns:
        pg.ImageView: Configured image view widget.
    """
    view = pg.ImageView()
    view.ui.histogram.hide()
    view.ui.roiBtn.hide()
    view.ui.menuBtn.hide()
    return view


class ProfileViewer(QtWidgets.QMainWindow):
    """Main window for interactive cross-sectional profile analysis.
    
    Provides tools for:
    - Loading aligned scan pairs from HDF5
    - Interactive profile line placement and adjustment
    - Real-time profile plotting with offset correction
    - Contact point detection and separation analysis
    - 3D visualization of profile locations
    
    Attributes:
        ref_pixel_um (QPointF): Reference scan pixel size in micrometers.
        adj_pixel_um (QPointF): Adjusted scan pixel size in micrometers.
        sigma (float): Smoothing parameter.
        separation (int): Vertical separation between profiles.
        reference_grid (np.ndarray): Reference scan data.
        adjusted_grid (np.ndarray): Adjusted scan data.
    """
    
    def __init__(self, parent=None):
        """Initialize the profile viewer window.
        
        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Interactive cross-sectional analysis")
        self.setGeometry(100, 100, 1000, 600)
        
        # Parameters and metadata
        self.ref_pixel_um = QPointF(1.0, 1.0)
        self.adj_pixel_um = QPointF(1.0, 1.0)
        self.sigma = 5.0
        self.separation = 0
        self.binary_contact = None
        self._preview_win = None
        
        # State variables
        self.line_drag_active = False
        self.line_drag_which = None
        self.cursor_lines = []
        self.annotations = []
        self.mytest = []
        self.image_marker = None
        self.image_boundary_item = None
        self.saved_points = []
        self.saved_point_markers = []
        
        # Initialize managers
        self.data_manager = DataManager(self)
        self.profile_analyzer = ProfileAnalyzer(self)
        self.roi_handler = ROIHandler(self)
        self.plot_interactions = PlotInteractions(self)
        self.visualization_manager = VisualizationManager(self)
        
        # Setup UI
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()
        
        # Progress bar in status bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    # ==========================================================================
    # UI Setup
    # ==========================================================================
    
    def _setup_menu(self):
        """Setup menu bar with actions."""
        menubar = self.menuBar()
        
        view_menu = menubar.addMenu('View')
        
        self.open_3d_action = QtWidgets.QAction('Show 3D view', self)
        self.open_3d_action.triggered.connect(self.visualization_manager.show_3d_view)
        view_menu.addAction(self.open_3d_action)
        
        view_menu.addSeparator()
        
        self.load_profiles_action = QtWidgets.QAction('Load profiles...', self)
        self.load_profiles_action.triggered.connect(self.data_manager.load_profiles)
        view_menu.addAction(self.load_profiles_action)
        
        self.save_profiles_action = QtWidgets.QAction('Save profiles...', self)
        self.save_profiles_action.triggered.connect(self.data_manager.save_profiles)
        view_menu.addAction(self.save_profiles_action)

        self.roughness_action = QtWidgets.QAction('Profile roughness summary...', self)
        self.roughness_action.triggered.connect(self.show_profile_roughness_summary)
        view_menu.addAction(self.roughness_action)
        
        view_menu.addSeparator()
        
        self.exit_action = QtWidgets.QAction('Exit', self)
        self.exit_action.triggered.connect(self.close)
        view_menu.addAction(self.exit_action)
    
    def _setup_ui(self):
        """Setup user interface widgets and layouts."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        central_widget.setLayout(layout)

        controls_group = QtWidgets.QGroupBox("Profile Controls", central_widget)
        controls_layout = QtWidgets.QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        # Left column - profile plot
        plot_panel = QtWidgets.QWidget(splitter)
        center_layout = QtWidgets.QVBoxLayout(plot_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, 1))
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        center_layout.addWidget(self.plot_widget)
        splitter.addWidget(plot_panel)

        # Right column - binary map and compact controls
        right_panel = QtWidgets.QWidget(splitter)
        right_panel.setMinimumWidth(260)
        right_panel.setMaximumWidth(560)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.image_view = create_image_view()
        self.image_view.setMinimumWidth(240)
        self.image_view.setMinimumHeight(240)
        self.image_view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        right_layout.addWidget(self.image_view)

        vb = self.image_view.getView()
        vb.setRange(xRange=(0, 1000), padding=0)

        self.spinbox_separation = QtWidgets.QDoubleSpinBox()
        self.spinbox_separation.setRange(-10000.0, 10000.0)
        self.spinbox_separation.setDecimals(2)
        self.spinbox_separation.setSingleStep(0.1)
        self.spinbox_separation.setValue(self.separation)
        self.spinbox_separation.setMaximumWidth(110)

        self.spinbox_window_mm = QtWidgets.QDoubleSpinBox()
        self.spinbox_window_mm.setRange(0.001, 5.0)
        self.spinbox_window_mm.setValue(0.5)
        self.spinbox_window_mm.setSingleStep(0.001)
        self.spinbox_window_mm.setDecimals(3)
        self.spinbox_window_mm.setMaximumWidth(110)

        self.checkbox_snap = QtWidgets.QCheckBox("Snap to plot")
        self.checkbox_snap.setChecked(True)

        self.checkbox_tilt = QtWidgets.QCheckBox("Tilt correction")
        self.checkbox_tilt.setChecked(True)
        controls_layout.addWidget(QtWidgets.QLabel("Separation [μm]:"))
        controls_layout.addWidget(self.spinbox_separation)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QtWidgets.QLabel("Window size [mm]:"))
        controls_layout.addWidget(self.spinbox_window_mm)
        controls_layout.addWidget(self.checkbox_snap)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.checkbox_tilt)
        controls_layout.addStretch(1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([720, 420])
        layout.addWidget(controls_group, 0)
        layout.addWidget(splitter, 1)
    
    def _connect_signals(self):
        """Connect widget signals to handler methods."""
        # Image view interactions
        self.image_view.getView().mousePressEvent = self.roi_handler.on_image_click
        self.image_view.getView().mouseReleaseEvent = self.roi_handler.on_image_mouse_release
        self.image_view.getView().mouseMoveEvent = self.roi_handler.on_image_mouse_move
        self.image_view.getView().sigRangeChanged.connect(
            self.visualization_manager.on_range_changed
        )
        
        # Plot interactions
        self.plot_widget.scene().sigMouseMoved.connect(
            self.plot_interactions.on_mouse_move
        )
        self.plot_widget.scene().sigMouseClicked.connect(
            self.plot_interactions.on_plot_click
        )
        
        # Control widgets
        self.spinbox_separation.valueChanged.connect(self.update_plot)
        self.spinbox_window_mm.valueChanged.connect(self.update_plot)
        self.checkbox_tilt.stateChanged.connect(self.profile_analyzer.toggle_tilt)
    
    # ==========================================================================
    # Main Methods
    # ==========================================================================

    def show_profile_roughness_summary(self):
        """Show minimal roughness parameters for the current profiles."""
        if not hasattr(self, 'reference_profile') or not hasattr(self, 'adjusted_profile'):
            QtWidgets.QMessageBox.warning(self, "No data", "No profile data available.")
            return

        from ....processing import profile_roughness_parameters

        try:
            ref_metrics = profile_roughness_parameters(self.reference_profile)
            adj_metrics = profile_roughness_parameters(self.adjusted_profile)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Profile roughness summary", str(exc))
            return

        lines = [
            "Minimal profile roughness summary",
            "",
            "Values are in current height units.",
            "",
            "Reference profile:",
        ]
        for name in ("Ra", "Rq", "Rz"):
            lines.append(f"{name}: {ref_metrics[name]:.6g}")

        lines.extend(["", "Adjusted profile:"])
        for name in ("Ra", "Rq", "Rz"):
            lines.append(f"{name}: {adj_metrics[name]:.6g}")

        QtWidgets.QMessageBox.information(
            self,
            "Profile roughness summary",
            "\n".join(lines),
        )
    
    def closeEvent(self, event):
        """Handle window close event.
        
        Args:
            event: Close event.
        """
        if hasattr(self.parent(), '_profile_viewer'):
            self.parent()._profile_viewer = None
        event.accept()

    def resizeEvent(self, event):
        """Refit the FRASTA map after window resizes when data is present.

        The map fit depends on the current viewport aspect ratio, so the fit
        must be recomputed after splitter moves and window resizes.

        Args:
            event: Qt resize event.
        """
        super().resizeEvent(event)
        if self.binary_contact is None:
            return
        QtCore.QTimer.singleShot(0, self.visualization_manager.fit_contact_image_view_to_image)

    def update_separation_spinbox_range(self) -> None:
        """Adapt the separation-control range to the loaded height data.

        The separation value uses the same height unit as the current scan, so
        a fixed +/-1000 range may be too small for tall surfaces. The range is
        expanded from the actual surface amplitudes rather than the absolute
        height offset, so the control remains large enough without becoming
        effectively unbounded for scans stored far from zero.
        """
        spans = []
        for values in (
            getattr(self, "reference_grid_smooth", None),
            getattr(self, "adjusted_grid_corrected", None),
        ):
            if values is None:
                continue
            finite = np.asarray(values, dtype=float)[np.isfinite(values)]
            if finite.size:
                spans.append(float(np.max(finite) - np.min(finite)))

        if not spans:
            return

        data_span = max(max(spans), 1.0)
        separation_limit = max(5000.0, 2.0 * data_span)

        current_value = self.spinbox_separation.value()
        self.spinbox_separation.blockSignals(True)
        self.spinbox_separation.setRange(-separation_limit, separation_limit)
        self.spinbox_separation.setValue(
            min(max(current_value, -separation_limit), separation_limit)
        )
        self.spinbox_separation.blockSignals(False)
    
    def update_plot(self):
        """Update binary contact map and profile plot.
        
        Returns:
            tuple: Shape of binary contact map (height, width).
        """
        self.separation = self.spinbox_separation.value()
        valid_mask = ~np.isnan(self.reference_grid_smooth) & ~np.isnan(
            self.adjusted_grid_corrected
        )
        
        difference = self.reference_grid_smooth - (
            self.adjusted_grid_corrected + self.separation
        )
        binary_contact = (difference > 0) & valid_mask
        
        self.image_view.setImage(
            binary_contact.T.astype(np.uint8), 
            autoRange=False, 
            autoLevels=True
        )
        self.visualization_manager.fit_contact_image_view_to_image()
        
        self.roi_handler.update_profile_from_roi()
        self.binary_contact = binary_contact
        self.visualization_manager.update_volume_info()
        
        return binary_contact.shape
    
    # ==========================================================================
    # Public API Methods (for external access)
    # ==========================================================================
    
    def set_surfaces(self, surface1, surface2):
        """Set scan data from Surface objects.
        
        Args:
            surface1 (Surface): Reference surface object.
            surface2 (Surface): Adjusted surface object.
        """
        self.data_manager.set_surfaces(surface1, surface2)
    
    def set_data(self, grid1, grid2, px1_um, py1_um, px2_um, py2_um):
        """Set scan data from grid arrays and pixel sizes.
        
        Args:
            grid1 (np.ndarray): Reference grid data.
            grid2 (np.ndarray): Adjusted grid data.
            px1_um (float): Reference pixel size in X (micrometers).
            py1_um (float): Reference pixel size in Y (micrometers).
            px2_um (float): Adjusted pixel size in X (micrometers).
            py2_um (float): Adjusted pixel size in Y (micrometers).
        """
        self.data_manager.set_data(grid1, grid2, px1_um, py1_um, px2_um, py2_um)
