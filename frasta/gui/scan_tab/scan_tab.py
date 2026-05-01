"""Scan tab widget for displaying and editing 2D scan data.

This module provides the ScanTab widget which displays scan data with interactive
histogram controls, supports various editing operations (masking, hole filling,
rotation, flipping), and provides tools for setting zero points and removing tilt.

The widget delegates specific functionality to specialized components:
- HistogramManager: Histogram display and threshold controls
- InteractiveHandler: Mouse click handling and interactive modes
- TransformOperations: Geometric transformations
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh

from ...core import Surface
from ...utils import get_lookup_table
from ...processing import fill_holes, remove_outliers, nan_aware_gaussian
from ..widgets import HistogramViewBox

from .histogram_manager import HistogramManager
from .interactive_handler import InteractiveHandler
from .transform_operations import TransformOperations

import logging
logger = logging.getLogger(__name__)


class ScanTab(QtWidgets.QWidget):
    """Widget for displaying and interacting with a single scan dataset.
    
    Provides an image view with histogram-based contrast adjustment, interactive
    tools for zero point selection, tilt removal, hole filling, and various
    geometric transformations.
    
    Attributes:
        image_view (pg.ImageView): Main image display widget.
        hist_widget (pg.PlotWidget): Histogram display for contrast adjustment.
        grid (np.ndarray): Current 2D scan data.
        xi (np.ndarray): X-coordinate array.
        yi (np.ndarray): Y-coordinate array.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        histogram_manager (HistogramManager): Manages histogram display.
        interactive_handler (InteractiveHandler): Handles mouse interactions.
    """
    
    def __init__(self, parent=None):
        """Initialize the scan tab widget.
        
        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        
        # Setup UI components
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        self.image_view.getView().setMenuEnabled(False)
        self.image_view.ui.graphicsView.setBackground((34, 34, 34))

        self.hist_widget = pg.PlotWidget(viewBox=HistogramViewBox())
        self.hist_widget.setMaximumHeight(120)
        self.hist_widget.setMenuEnabled(False)
        self.hist_widget.setMouseEnabled(x=False, y=False)
        self._updating_threshold_controls = False

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        histogram_layout = QtWidgets.QHBoxLayout()
        histogram_layout.addWidget(self.hist_widget, stretch=1)
        histogram_layout.addWidget(self._create_histogram_controls())
        main_layout.addWidget(self.image_view, stretch=1)
        main_layout.addLayout(histogram_layout)
        self.setLayout(main_layout)

        # Data attributes
        self.grid = None
        self.masked = None
        self.xi = None
        self.yi = None
        self.dx = None
        self.dy = None

        # Display settings
        self.is_colormap = False
        self.current_colormap = None
        self.hide_below_range = True
        self.hide_above_range = True

        # Initialize managers and handlers
        self.histogram_manager = HistogramManager(self.hist_widget, self._on_threshold_range_changed)
        self.interactive_handler = InteractiveHandler(self)
        
        # Connect mouse events
        self.image_view.getView().scene().sigMouseClicked.connect(
            self.interactive_handler.handle_mouse_click
        )

    def _create_histogram_controls(self) -> QtWidgets.QWidget:
        """Create manual threshold controls displayed beside the histogram."""
        control_widget = QtWidgets.QWidget(self)
        control_widget.setMinimumWidth(180)
        control_widget.setMaximumWidth(200)
        control_layout = QtWidgets.QFormLayout(control_widget)
        control_layout.setContentsMargins(8, 0, 0, 0)
        control_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.range_min_spin = QtWidgets.QDoubleSpinBox(control_widget)
        self.range_min_spin.setDecimals(3)
        self.range_min_spin.setRange(-1e12, 1e12)
        self.range_min_spin.setKeyboardTracking(False)
        self.range_min_spin.valueChanged.connect(self._on_manual_threshold_changed)

        self.range_max_spin = QtWidgets.QDoubleSpinBox(control_widget)
        self.range_max_spin.setDecimals(3)
        self.range_max_spin.setRange(-1e12, 1e12)
        self.range_max_spin.setKeyboardTracking(False)
        self.range_max_spin.valueChanged.connect(self._on_manual_threshold_changed)

        self.hide_below_range_checkbox = QtWidgets.QCheckBox("Hide below Min", control_widget)
        self.hide_below_range_checkbox.setChecked(True)
        self.hide_below_range_checkbox.toggled.connect(self._on_out_of_range_visibility_toggled)

        self.hide_above_range_checkbox = QtWidgets.QCheckBox("Hide above Max", control_widget)
        self.hide_above_range_checkbox.setChecked(True)
        self.hide_above_range_checkbox.toggled.connect(self._on_out_of_range_visibility_toggled)

        control_layout.addRow("Min:", self.range_min_spin)
        control_layout.addRow("Max:", self.range_max_spin)
        control_layout.addRow(self.hide_below_range_checkbox)
        control_layout.addRow(self.hide_above_range_checkbox)
        return control_widget

    # ==========================================================================
    # Data Management
    # ==========================================================================
    
    def get_surface(self) -> Surface:
        """Get Surface object from current grid data.
        
        Returns:
            Surface: Surface object containing current scan data
        """
        # Handle grid that might contain NaN
        valid_data = self.grid[~np.isnan(self.grid)]
        if valid_data.size > 0:
            grid_min = float(np.min(valid_data))
            grid_max = float(np.max(valid_data))
        else:
            # All NaN, use dummy values
            grid_min = 0.0
            grid_max = 1.0
        
        # Extract origin from stored xi, yi arrays
        x0 = self.xi[0] if hasattr(self, 'xi') and len(self.xi) > 0 else 0.0
        y0 = self.yi[0] if hasattr(self, 'yi') and len(self.yi) > 0 else 0.0
        
        data = Surface(
            height=self.grid,
            dx=self.dx,
            dy=self.dy,
            x0=x0,
            y0=y0,
            vmin=grid_min,
            vmax=grid_max
        )
        
        # Get current threshold values if available
        vmin, vmax = self.histogram_manager.get_threshold_range()
        if vmin is not None and vmax is not None:
            data.vmin = vmin
            data.vmax = vmax
        
        return data
    
    def set_surface(self, data: Surface):
        """Set scan data from a Surface object.
        
        Args:
            data (Surface): Surface object containing scan data.
        """
        self.grid = data.height
        self.xi = data.xi
        self.yi = data.yi
        self.dx = data.dx
        self.dy = data.dy
        logger.debug(f"grid: {self.grid.shape}, xmin: {self.xi[0]}, ymin: {self.yi[0]}, px_x: {self.dx}, px_y: {self.dy}")
        
        # Update histogram first to set threshold lines
        self.histogram_manager.update_histogram(
            self.grid,
            colormap_name=self.get_colormap_name(),
        )
        self._sync_threshold_controls()
        self.update_image()
        
        # Then set the threshold line values if provided
        if data.vmin is not None and data.vmax is not None:
            self.histogram_manager.set_threshold_values(data.vmin, data.vmax)
            self._sync_threshold_controls()

    # ==========================================================================
    # Display Methods
    # ==========================================================================
    
    def update_histogram(self, was_data_negated: bool = False):
        """Update histogram display.
        
        Args:
            was_data_negated (bool): Whether data was recently inverted
        """
        self.histogram_manager.update_histogram(
            self.grid,
            was_data_negated,
            self.get_colormap_name(),
        )
        self._sync_threshold_controls()

    def _on_threshold_range_changed(self, vmin: float, vmax: float):
        """Handle threshold changes coming from histogram lines."""
        self._sync_threshold_controls(vmin, vmax)
        self.update_image(vmin, vmax)

    def _on_manual_threshold_changed(self, _value: float):
        """Handle manual threshold edits from spin boxes."""
        if self._updating_threshold_controls:
            return
        self.histogram_manager.set_threshold_values(
            self.range_min_spin.value(),
            self.range_max_spin.value(),
        )
        vmin, vmax = self.histogram_manager.get_threshold_range()
        self._on_threshold_range_changed(vmin, vmax)

    def _on_out_of_range_visibility_toggled(self, _checked: bool):
        """Switch masking independently below Min and above Max."""
        self.hide_below_range = self.hide_below_range_checkbox.isChecked()
        self.hide_above_range = self.hide_above_range_checkbox.isChecked()
        self.histogram_manager.set_out_of_range_visibility(
            self.hide_below_range,
            self.hide_above_range,
        )
        self.update_image()

    def _sync_threshold_controls(self, vmin: float = None, vmax: float = None):
        """Synchronize manual threshold controls with histogram state."""
        if vmin is None or vmax is None:
            vmin, vmax = self.histogram_manager.get_threshold_range()
        data_min, data_max = self.histogram_manager.get_data_range()
        if vmin is None or vmax is None or data_min is None or data_max is None:
            return

        self._updating_threshold_controls = True
        try:
            self.range_min_spin.setRange(data_min, data_max)
            self.range_max_spin.setRange(data_min, data_max)
            self.range_min_spin.setValue(vmin)
            self.range_max_spin.setValue(vmax)
        finally:
            self._updating_threshold_controls = False
    
    def update_image(self, vmin: float = None, vmax: float = None):
        """Update the displayed image based on current grid and value range.
        
        Args:
            vmin (float, optional): Minimum value for display range
            vmax (float, optional): Maximum value for display range
        """
        logger.debug(f"update_image called: grid is None? {self.grid is None}")
        if self.grid is None:
            logger.warning("update_image: self.grid is None!")
            return
        
        # Get threshold range
        if vmin is None or vmax is None:
            vmin, vmax = self.histogram_manager.get_threshold_range()
            if vmin is None or vmax is None:
                # Handle case where grid might be all NaN
                valid_data = self.grid[~np.isnan(self.grid)]
                if valid_data.size > 0:
                    vmin = float(np.min(valid_data))
                    vmax = float(np.max(valid_data))
                else:
                    # Grid is all NaN, use dummy range
                    logger.warning(f"update_image: grid is all NaN! shape={self.grid.shape}")
                    vmin = 0.0
                    vmax = 1.0

        logger.debug(f"update_image: grid.shape={self.grid.shape}, vmin={vmin}, vmax={vmax}")
        
        # Debug: check actual data range before masking
        valid_grid_data = self.grid[~np.isnan(self.grid)]
        if valid_grid_data.size > 0:
            actual_min = float(np.min(valid_grid_data))
            actual_max = float(np.max(valid_grid_data))
            logger.debug(f"update_image: grid actual range: [{actual_min:.2f}, {actual_max:.2f}]")
            logger.debug(f"update_image: threshold range: [{vmin:.2f}, {vmax:.2f}]")
            
            # Check if threshold range makes sense
            if vmin > actual_max or vmax < actual_min:
                logger.error(f"update_image: threshold range [{vmin:.2f}, {vmax:.2f}] is outside data range [{actual_min:.2f}, {actual_max:.2f}]!")
                # Use actual data range instead
                vmin = actual_min
                vmax = actual_max
                logger.warning(f"update_image: using actual data range instead")
        
        # IMPORTANT: grid.T creates a VIEW, not a copy!
        # Make a copy immediately to avoid accidentally modifying self.grid.
        image_data = self.grid.T.copy()
        invalid_mask = np.isnan(image_data)
        if self.hide_below_range:
            image_data[image_data < vmin] = np.nan
        else:
            image_data[image_data < vmin] = vmin
        if self.hide_above_range:
            image_data[image_data > vmax] = np.nan
        else:
            image_data[image_data > vmax] = vmax
        image_data[invalid_mask] = np.nan
        self.masked = image_data
        
        nan_count = np.isnan(self.masked).sum()
        total_count = self.masked.size
        logger.debug(f"update_image: masked has {nan_count}/{total_count} NaN values")
        
        if np.isnan(self.masked).all():
            logger.warning("update_image: masked is all NaN, using zeros")
            self.masked = np.zeros_like(self.masked)
        
        # Apply colormap
        image_item = self.image_view.getImageItem()
        if self.is_colormap:
            cmap_name = self.current_colormap or 'metrology'
            lut = get_lookup_table(cmap_name, 256)
            image_item.setLookupTable(lut)
        else:
            image_item.setLookupTable(None)
        
        if vmax <= vmin:
            vmax = vmin + 1e-9
        self.image_view.setImage(
            self.masked,
            autoLevels=False,
            autoRange=False,
            levels=(vmin, vmax),
        )
        self.interactive_handler.clear_seed_points()
    
    def toggle_colormap(self):
        """Toggle between grayscale and color display."""
        self.is_colormap = not self.is_colormap
        if self.is_colormap and self.current_colormap is None:
            self.current_colormap = 'metrology'
        self.update_histogram()
        self.update_image()

    def set_colormap(self, name: str):
        """Set grayscale or a named colormap for the 2D scan view.

        Args:
            name (str): Display mode name. ``Gray`` disables the lookup table;
                any other value is interpreted as a colormap name.
        """
        if name in ("Gray", "None", "", None):
            self.is_colormap = False
            self.current_colormap = None
        else:
            self.is_colormap = True
            self.current_colormap = str(name).lower()
        self.update_histogram()
        self.update_image()

    def get_colormap_name(self) -> str:
        """Return current 2D display colormap label."""
        if not self.is_colormap or self.current_colormap is None:
            return "Gray"
        if self.current_colormap == "metrology":
            return "Metrology"
        return self.current_colormap

    # ==========================================================================
    # Interactive Mode Methods
    # ==========================================================================
    
    def set_zero_point_mode(self):
        """Enable zero point selection mode."""
        self.interactive_handler.set_zero_point_mode()
    
    def set_tilt_mode(self):
        """Enable tilt correction mode."""
        self.interactive_handler.set_tilt_mode()

    # ==========================================================================
    # Transform Methods
    # ==========================================================================
    
    def flip_scan(self, direction: str = 'UD', parent=None):
        """Flip scan vertically or horizontally.
        
        Args:
            direction (str): 'UD' for up/down, 'LR' for left/right
            parent (QWidget): Parent widget for error messages
        """
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = TransformOperations.flip_scan(self.grid, direction, parent)
        self.update_image()
    
    def scan_rot90(self, parent=None):
        """Rotate scan 90 degrees counter-clockwise.
        
        Args:
            parent (QWidget): Parent widget for error messages
        """
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = TransformOperations.rotate_90(self.grid, parent)
        self.update_image()
    
    def invert_scan(self, parent=None):
        """Invert Z values (negate height).
        
        Args:
            parent (QWidget): Parent widget for error messages
        """
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = TransformOperations.invert_z(self.grid, parent)
        self.update_histogram(was_data_negated=True)
        self.update_image()
    
    def delete_unmasked(self, mask: np.ndarray):
        """Delete data outside mask (set to NaN).
        
        Args:
            mask (np.ndarray): Boolean mask (True = keep, False = delete)
        """
        if self.grid is not None:
            self.grid = TransformOperations.delete_unmasked(self.grid, mask)
            self.update_image()
            self.update_histogram()

    # ==========================================================================
    # Processing Methods
    # ==========================================================================
    
    def repair_grid(self, mask: np.ndarray = None):
        """Repair grid by removing holes and outliers.
        
        Args:
            mask (np.ndarray, optional): Mask indicating region to repair
        """
        dialog, ed_sigma, ed_thresh = self._create_repair_dialog()
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        sigma = ed_sigma.value()
        threshold = ed_thresh.value()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # Process
        grid_filled = fill_holes(self.grid, mask=mask)
        grid_smooth = nan_aware_gaussian(grid_filled, sigma, mask=mask)
        grid_cleaned = remove_outliers(grid_filled, grid_smooth, threshold, mask=mask)

        if mask is not None:
            self.grid[mask] = grid_cleaned[mask]
        else:
            self.grid = grid_cleaned

        self.update_image()
        QtWidgets.QApplication.restoreOverrideCursor()
    
    def fill_holes(self, parent=None):
        """Fill holes in scan data using interpolation.
        
        Args:
            parent (QWidget): Parent widget for error messages
        """
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return

        tst = np.isnan(self.grid)
        if not np.any(tst):
            return
        
        # Fill regions marked by seed points
        seed_points = self.interactive_handler.get_seed_points()
        for (iy, ix) in seed_points:
            if tst[iy, ix]:
                filled = flood(tst, seed_point=(iy, ix))
                tst[filled] = False

        if not np.any(tst):
            return

        logger.debug(f"grid.shape: {self.grid.shape}, xi len: {len(self.xi)}, yi len: {len(self.yi)}")

        grid_x, grid_y = np.meshgrid(self.xi, self.yi)

        logger.debug(f"grid_x.shape: {grid_x.shape}, grid_y.shape: {grid_y.shape}, tst.shape: {tst.shape}")

        interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

        valid = ~np.isnan(self.grid)
        interp_values = griddata(
            (grid_x[valid], grid_y[valid]),
            self.grid[valid],
            interp_points,
            method='nearest'
        )

        self.grid[tst] = interp_values
        self.update_image()
    
    def _create_repair_dialog(self, sigma: int = 25, threshold: int = 100):
        """Create dialog for repair grid parameters.
        
        Args:
            sigma (int): Default sigma value
            threshold (int): Default threshold value
            
        Returns:
            tuple: (dialog, sigma_spinbox, threshold_spinbox)
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select actions")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        ch_sigma = QtWidgets.QLabel("sigma:")
        ed_sigma = QtWidgets.QSpinBox()
        ed_sigma.setRange(0, 100)
        ed_sigma.setValue(sigma)
        
        ch_thresh = QtWidgets.QLabel("threshold:")
        ed_thresh = QtWidgets.QSpinBox()
        ed_thresh.setRange(0, 10000)
        ed_thresh.setValue(threshold)
        
        ch_newtab = QtWidgets.QCheckBox("create new tab:")
        lbl_newtab = QtWidgets.QLabel("tab label:")
        ed_label = QtWidgets.QLineEdit("name")
        ch_newtab.setDisabled(True)
        ed_label.setDisabled(True)
        
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(ok_btn)
        hl.addWidget(cancel_btn)
        
        fl = QtWidgets.QFormLayout()
        fl.addRow(ch_sigma, ed_sigma)
        fl.addRow(ch_thresh, ed_thresh)
        fl.addWidget(ch_newtab)
        fl.addRow(lbl_newtab, ed_label)
        layout.addLayout(fl)
        layout.addLayout(hl)
        
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        return dialog, ed_sigma, ed_thresh

    # ==========================================================================
    # Legacy/Utility Methods (kept for backward compatibility)
    # ==========================================================================
    
    def grid_to_mesh_vectorized(self, grid: np.ndarray, dx: float = 1.0, dy: float = 1.0):
        """Convert grid to mesh (vertices and faces).
        
        Args:
            grid (np.ndarray): Grid data
            dx (float): Pixel size in x
            dy (float): Pixel size in y
            
        Returns:
            tuple: (vertices, faces) arrays for mesh
        """
        h, w = grid.shape

        # XY grid
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_coords = x_indices * dx
        y_coords = y_indices * dy
        z_coords = grid

        # All vertices
        vertices = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)

        # Mask of valid points (not NaN)
        valid_mask = ~np.isnan(vertices[:, 2])
        index_map = -np.ones(h * w, dtype=int)
        index_map[valid_mask] = np.arange(np.count_nonzero(valid_mask))

        # Triangle indices
        idx_tl = np.ravel_multi_index((np.arange(h - 1)[:, None], np.arange(w - 1)[None, :]), dims=(h, w))
        idx_tr = idx_tl + 1
        idx_bl = idx_tl + w
        idx_br = idx_bl + 1

        # Flattened and combined
        idx_tl = idx_tl.ravel()
        idx_tr = idx_tr.ravel()
        idx_bl = idx_bl.ravel()
        idx_br = idx_br.ravel()

        # Only where all 4 are valid
        valid_quad = (index_map[idx_tl] >= 0) & (index_map[idx_tr] >= 0) & \
                    (index_map[idx_bl] >= 0) & (index_map[idx_br] >= 0)

        # Two triangles per square
        faces_a = np.stack([index_map[idx_tl], index_map[idx_tr], index_map[idx_br]], axis=1)[valid_quad]
        faces_b = np.stack([index_map[idx_tl], index_map[idx_br], index_map[idx_bl]], axis=1)[valid_quad]
        faces = np.vstack([faces_a, faces_b])

        # Filtered vertices
        vertices = vertices[valid_mask]

        return vertices.astype(np.float32), faces.astype(np.int32)
    
    def save_as_mesh(self, grid: np.ndarray, dx: float = 1.38, dy: float = 1.38):
        """Save grid as mesh file.
        
        Args:
            grid (np.ndarray): Grid data
            dx (float): Pixel size in x
            dy (float): Pixel size in y
        """
        v, f = self.grid_to_mesh_vectorized(grid, dx, dy)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.export("mesh_output.obj")
