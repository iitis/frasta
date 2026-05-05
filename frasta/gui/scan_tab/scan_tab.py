"""Scan tab widget for displaying and editing 2D scan data.

This module provides the ScanTab widget which displays scan data with interactive
histogram controls, supports various editing operations (masking, hole filling,
rotation, flipping), and provides tools for setting zero points and removing tilt.

The widget delegates specific functionality to specialized components:
- HistogramManager: Histogram display and threshold controls
- InteractiveHandler: Mouse click handling and interactive modes
- TransformOperations: Geometric transformations
"""

from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh

from ...core import Surface
from ...utils import get_colormap, get_lookup_table
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
        self.image_view.getView().setAspectLocked(True)
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
        self.unit = "µm"

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
            unit=self.unit,
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
        self.unit = getattr(data, "unit", "µm")
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
        self._apply_physical_image_rect()
        self.interactive_handler.clear_seed_points()

    def _apply_physical_image_rect(self):
        """Map the image to physical coordinates using scan spacing and origin."""
        if self.grid is None:
            return

        image_item = self.image_view.getImageItem()
        if image_item is None:
            return

        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        width = self.grid.shape[1] * dx
        height = self.grid.shape[0] * dy
        image_item.setRect(QtCore.QRectF(x0 - dx / 2.0, y0 - dy / 2.0, width, height))

    def physical_to_indices(self, x_phys: float, y_phys: float) -> tuple[int, int]:
        """Convert physical coordinates to nearest grid indices."""
        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        x_idx = int(round((x_phys - x0) / dx))
        y_idx = int(round((y_phys - y0) / dy))
        return x_idx, y_idx

    def indices_to_physical(self, x_idx: int, y_idx: int) -> tuple[float, float]:
        """Convert grid indices to physical coordinates at pixel centers."""
        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        return x0 + x_idx * dx, y0 + y_idx * dy
    
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

    def export_2d_image(self) -> None:
        """Export either the full grid or the current viewport as a PNG image."""
        if self.grid is None:
            QtWidgets.QMessageBox.information(self, "No data", "Load a scan first.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export 2D image")
        form = QtWidgets.QFormLayout(dialog)

        combo_source = QtWidgets.QComboBox(dialog)
        combo_source.addItem("Full grid", userData="full")
        combo_source.addItem("Current viewport", userData="viewport")

        chk_transparent = QtWidgets.QCheckBox("Transparent hidden pixels", dialog)
        chk_transparent.setChecked(True)

        form.addRow("Source:", combo_source)
        form.addRow("", chk_transparent)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 2D image",
            "frasta-2d-image.png",
            "PNG Images (*.png)",
        )
        if not output_path:
            return

        image = self.build_export_image(
            source=combo_source.currentData() or "full",
            transparent_background=chk_transparent.isChecked(),
        )

        # print(f"Exporting 2D image to {output_path} (source={combo_source.currentData()}, transparent={chk_transparent.isChecked()})")
        # print(f"Image size: {image.width()}x{image.height()}, format: {image.format()}")
        
        self._save_png_image(image, output_path, "2D image")

    def export_2d_colorbar(self) -> None:
        """Export a standalone colorbar for the current 2D display settings."""
        if self.grid is None:
            QtWidgets.QMessageBox.information(self, "No data", "Load a scan first.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export 2D colorbar")
        form = QtWidgets.QFormLayout(dialog)

        combo_source = QtWidgets.QComboBox(dialog)
        combo_source.addItem("Full grid", userData="full")
        combo_source.addItem("Current viewport", userData="viewport")

        spin_width = QtWidgets.QSpinBox(dialog)
        spin_width.setRange(32, 4096)
        spin_width.setValue(220)

        spin_height = QtWidgets.QSpinBox(dialog)
        spin_height.setRange(64, 4096)
        spin_height.setValue(900)

        chk_transparent = QtWidgets.QCheckBox("Transparent background", dialog)
        chk_transparent.setChecked(True)

        chk_histogram = QtWidgets.QCheckBox("Include histogram", dialog)
        chk_histogram.setChecked(True)

        form.addRow("Source:", combo_source)
        form.addRow("Width [px]:", spin_width)
        form.addRow("Height [px]:", spin_height)
        form.addRow("", chk_transparent)
        form.addRow("", chk_histogram)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 2D colorbar",
            "frasta-2d-colorbar.png",
            "PNG Images (*.png)",
        )
        if not output_path:
            return

        image = self.build_export_colorbar(
            source=combo_source.currentData() or "full",
            width=spin_width.value(),
            height=spin_height.value(),
            transparent_background=chk_transparent.isChecked(),
            include_histogram=chk_histogram.isChecked(),
        )
        self._save_png_image(image, output_path, "2D colorbar")

    def build_export_image(
        self,
        source: str = "full",
        transparent_background: bool = True,
    ) -> QtGui.QImage:
        """Build a PNG-ready image matching the current 2D display settings.

        Args:
            source: ``"full"`` for the whole grid or ``"viewport"`` for the
                current visible window.
            transparent_background: If True, hidden or invalid pixels use alpha 0.

        Returns:
            RGBA image representing the current 2D view configuration.
        """
        image_data, vmin, vmax = self._build_display_array(source)
        return self._display_array_to_qimage(image_data, vmin, vmax, transparent_background)

    def build_export_colorbar(
        self,
        source: str = "full",
        width: int = 220,
        height: int = 900,
        transparent_background: bool = True,
        include_histogram: bool = True,
    ) -> QtGui.QImage:
        """Build a standalone colorbar image for the current 2D display.

        Args:
            source: ``"full"`` or ``"viewport"`` to match the exported image scope.
            width: Output image width in pixels.
            height: Output image height in pixels.
            transparent_background: If True, use an alpha-0 canvas.
            include_histogram: If True, draw the data histogram beside the bar.

        Returns:
            QImage containing the colorbar, labels, and optional histogram.
        """
        width = max(32, int(width))
        height = max(64, int(height))
        image_data, vmin, vmax = self._build_display_array(source)
        values = self._get_histogram_values(image_data)

        image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32_Premultiplied)
        background = QtGui.QColor(0, 0, 0, 0) if transparent_background else QtGui.QColor(255, 255, 255, 255)
        image.fill(background)

        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

            text_color = QtGui.QColor(255, 255, 255) if transparent_background else QtGui.QColor(20, 20, 20)
            border_color = QtGui.QColor(255, 255, 255, 180) if transparent_background else QtGui.QColor(40, 40, 40)
            hist_color = QtGui.QColor(180, 180, 180, 170)

            title_font = QtGui.QFont()
            title_font.setBold(True)
            title_font.setPointSize(11)
            painter.setFont(title_font)
            painter.setPen(text_color)
            title_rect = QtCore.QRectF(12, 8, width - 24, 28)
            painter.drawText(title_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, self.unit)

            bar_top = 48
            bar_bottom = max(bar_top + 40, height - 24)
            bar_height = bar_bottom - bar_top
            bar_width = max(28, min(40, width // 5))
            bar_left = 12
            bar_rect = QtCore.QRect(bar_left, bar_top, bar_width, bar_height)

            gradient = self._build_colorbar_gradient(bar_rect)
            painter.fillRect(bar_rect, gradient)
            painter.setPen(QtGui.QPen(border_color, 1.0))
            painter.drawRect(bar_rect)

            if include_histogram and values.size > 0:
                hist_left = bar_rect.right() + 4
                hist_width = max(20, min(60, width // 4))
                hist_rect = QtCore.QRect(hist_left, bar_top, hist_width, bar_height)
                self._draw_colorbar_histogram(painter, hist_rect, values, vmin, vmax, hist_color)

            label_left = bar_rect.right() + (max(20, min(60, width // 4)) + 12 if include_histogram else 12)
            self._draw_colorbar_ticks(painter, bar_rect, label_left, width, vmin, vmax, text_color, border_color)
        finally:
            painter.end()

        return image

    def _build_display_array(self, source: str) -> tuple[np.ndarray, float, float]:
        """Return the display array used for image and colorbar export."""
        grid = self._extract_export_grid(source)
        vmin, vmax = self.histogram_manager.get_threshold_range()
        if vmin is None or vmax is None:
            data_min, data_max = self.histogram_manager.get_data_range()
            vmin = 0.0 if data_min is None else float(data_min)
            vmax = 1.0 if data_max is None else float(data_max)
        if vmax <= vmin:
            vmax = vmin + 1e-9

        image_data = grid.T.copy()
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
        return image_data, float(vmin), float(vmax)

    def _display_array_to_qimage(
        self,
        image_data: np.ndarray,
        vmin: float,
        vmax: float,
        transparent_background: bool,
    ) -> QtGui.QImage:
        """Convert a display-ready array into an ARGB image."""
        if image_data.size == 0:
            return QtGui.QImage(1, 1, QtGui.QImage.Format_ARGB32_Premultiplied)

        normalized = np.clip((image_data - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
        rgba = np.zeros(image_data.shape + (4,), dtype=np.uint8)
        valid_mask = np.isfinite(image_data)

        if self.is_colormap and self.current_colormap is not None:
            lut = get_lookup_table(self.get_colormap_name(), 256)
            lut_index = np.zeros(image_data.shape, dtype=np.int32)
            lut_index[valid_mask] = np.round(normalized[valid_mask] * 255.0).astype(np.int32, copy=False)
            sampled = lut[lut_index[valid_mask]]
            if sampled.shape[1] >= 4:
                rgba[valid_mask] = sampled[:, :4]
            else:
                rgba[valid_mask, :3] = sampled[:, :3]
                rgba[valid_mask, 3] = 255
        else:
            gray = np.round(normalized * 255.0).astype(np.uint8, copy=False)
            rgba[..., 0] = gray
            rgba[..., 1] = gray
            rgba[..., 2] = gray
            rgba[..., 3] = np.where(valid_mask, 255, 0 if transparent_background else 255).astype(np.uint8)

        if self.is_colormap and self.current_colormap is not None:
            rgba[..., 3] = np.where(valid_mask, 255, 0 if transparent_background else 255).astype(np.uint8)
        elif not transparent_background:
            rgba[~valid_mask, :3] = 255

        if transparent_background:
            rgba[~valid_mask, 3] = 0
        else:
            rgba[~valid_mask, :3] = 255
            rgba[~valid_mask, 3] = 255

        image = QtGui.QImage(rgba.shape[1], rgba.shape[0], QtGui.QImage.Format_ARGB32)
        for row_index in range(rgba.shape[0]):
            scanline = image.scanLine(row_index)
            scanline.setsize(image.bytesPerLine())
            row_buffer = np.frombuffer(scanline, dtype=np.uint8, count=image.bytesPerLine())
            row_buffer[: rgba.shape[1] * 4 : 4] = rgba[row_index, :, 2]
            row_buffer[1 : rgba.shape[1] * 4 : 4] = rgba[row_index, :, 1]
            row_buffer[2 : rgba.shape[1] * 4 : 4] = rgba[row_index, :, 0]
            row_buffer[3 : rgba.shape[1] * 4 : 4] = rgba[row_index, :, 3]
        return image

    def _extract_export_grid(self, source: str) -> np.ndarray:
        """Extract either the full grid or the current viewport fragment."""
        if self.grid is None:
            return np.empty((0, 0), dtype=np.float32)
        if source != "viewport":
            return np.asarray(self.grid, dtype=np.float32)

        x_min, x_max, y_min, y_max = self._current_viewport_indices()
        return np.asarray(self.grid[y_min:y_max + 1, x_min:x_max + 1], dtype=np.float32)

    def _current_viewport_indices(self) -> tuple[int, int, int, int]:
        """Return current visible index bounds in the 2D view."""
        view = self.image_view.getView()
        x_range, y_range = view.viewRange()
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0

        x_start = int(np.floor((min(x_range) - (x0 - dx / 2.0)) / dx))
        x_stop = int(np.ceil((max(x_range) - (x0 - dx / 2.0)) / dx)) - 1
        y_start = int(np.floor((min(y_range) - (y0 - dy / 2.0)) / dy))
        y_stop = int(np.ceil((max(y_range) - (y0 - dy / 2.0)) / dy)) - 1

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_stop = min(self.grid.shape[1] - 1, x_stop)
        y_stop = min(self.grid.shape[0] - 1, y_stop)
        return x_start, x_stop, y_start, y_stop

    def _get_histogram_values(self, image_data: np.ndarray) -> np.ndarray:
        """Return finite displayed values contributing to the colorbar histogram."""
        return np.asarray(image_data[np.isfinite(image_data)], dtype=np.float32)

    def _build_colorbar_gradient(self, bar_rect: QtCore.QRect) -> QtGui.QLinearGradient:
        """Return a vertical gradient matching the active 2D colormap."""
        gradient = QtGui.QLinearGradient(
            float(bar_rect.left()),
            float(bar_rect.bottom()),
            float(bar_rect.left()),
            float(bar_rect.top()),
        )
        if not self.is_colormap or self.current_colormap is None:
            gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0))
            gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255))
            return gradient

        cmap = get_colormap(self.get_colormap_name())
        if cmap is None:
            gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0))
            gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255))
            return gradient

        positions, colors = cmap.getStops(mode="byte")
        for position, rgba in zip(positions, colors):
            gradient.setColorAt(
                float(position),
                QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
            )
        return gradient

    def _draw_colorbar_histogram(
        self,
        painter: QtGui.QPainter,
        hist_rect: QtCore.QRect,
        values: np.ndarray,
        vmin: float,
        vmax: float,
        color: QtGui.QColor,
    ) -> None:
        """Draw a normalized histogram beside the colorbar."""
        if values.size < 1 or vmax <= vmin:
            return
        counts, edges = np.histogram(values, bins=min(128, max(16, hist_rect.height() // 4)), range=(vmin, vmax))
        if counts.size < 1 or np.max(counts) <= 0:
            return

        counts = counts.astype(np.float32)
        counts /= float(np.max(counts))
        if counts.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= np.sum(kernel)
            counts = np.convolve(counts, kernel, mode="same")
            counts /= max(float(np.max(counts)), 1e-9)
        path = QtGui.QPainterPath()
        path.moveTo(hist_rect.left(), hist_rect.bottom())
        for idx, count in enumerate(counts):
            y0 = hist_rect.bottom() - (edges[idx] - vmin) / (vmax - vmin) * hist_rect.height()
            y1 = hist_rect.bottom() - (edges[idx + 1] - vmin) / (vmax - vmin) * hist_rect.height()
            x = hist_rect.left() + count * hist_rect.width()
            if idx == 0:
                path.lineTo(x, y0)
            path.lineTo(x, y1)
        path.lineTo(hist_rect.left(), hist_rect.top())
        path.closeSubpath()
        painter.fillPath(path, QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(color.darker(130), 1.0))
        painter.drawPath(path)

    def _draw_colorbar_ticks(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        label_left: int,
        image_width: int,
        vmin: float,
        vmax: float,
        text_color: QtGui.QColor,
        tick_color: QtGui.QColor,
    ) -> None:
        """Draw major and minor tick marks with labels beside the colorbar."""
        painter.setPen(text_color)
        label_font = QtGui.QFont()
        label_font.setPointSize(10)
        painter.setFont(label_font)

        major_count = 6
        minor_per_interval = 1
        for major_index in range(major_count):
            fraction = major_index / float(max(1, major_count - 1))
            y = bar_rect.bottom() - fraction * bar_rect.height()
            value = vmin + fraction * (vmax - vmin)
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.right() + 1, int(y), bar_rect.right() + 10, int(y))
            painter.setPen(text_color)
            text_rect = QtCore.QRectF(label_left, y - 12, image_width - label_left - 8, 24)
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                self._format_export_value(value),
            )

            if major_index >= major_count - 1:
                continue
            next_fraction = (major_index + 1) / float(max(1, major_count - 1))
            for minor_index in range(1, minor_per_interval + 1):
                local_fraction = minor_index / float(minor_per_interval + 1)
                minor_fraction = fraction + (next_fraction - fraction) * local_fraction
                minor_y = bar_rect.bottom() - minor_fraction * bar_rect.height()
                painter.setPen(QtGui.QPen(tick_color, 1.0))
                painter.drawLine(bar_rect.right() + 1, int(minor_y), bar_rect.right() + 6, int(minor_y))

    @staticmethod
    def _format_export_value(value: float) -> str:
        """Format numeric labels compactly for export annotations."""
        if not np.isfinite(value):
            return "nan"
        magnitude = abs(float(value))
        if magnitude >= 1e4 or (0 < magnitude < 1e-3):
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _save_png_image(self, image: QtGui.QImage, output_path: str, label: str) -> None:
        """Save an image as PNG and show a detailed error on failure."""
        path = Path(output_path)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")

        writer = QtGui.QImageWriter(str(path), b"png")
        if not writer.write(image):
            QtWidgets.QMessageBox.warning(
                self,
                "Export failed",
                f"Could not save the {label}.\n{writer.errorString()}",
            )

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
