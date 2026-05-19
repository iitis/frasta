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
import trimesh

from ...core import Surface
from ...utils import get_colormap, get_gradient_stops, get_lookup_table
from ...processing import fill_holes, remove_outliers, nan_aware_gaussian
from ..orientation import (
    build_image_rect,
    grid_to_image_data,
    indices_to_physical as orientation_indices_to_physical,
    physical_to_indices as orientation_physical_to_indices,
)
from ..colorbar_common import (
    build_colorbar_tick_layout,
    build_reference_label_specs,
    build_tick_label_rect,
    format_colorbar_value,
    get_colorbar_font_sizes,
    layout_reference_label_positions,
)
from ..colorbar_renderer import ColorbarRenderConfig, ExportColorbarRenderer
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

    MAX_DISPLAY_PIXELS = 2_000_000

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
        image_item = self.image_view.getImageItem()
        if hasattr(image_item, "setAutoDownsample"):
            image_item.setAutoDownsample(True)

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
        self.orientation = "default"

        # Display settings
        self.is_colormap = False
        self.current_colormap = None
        self.colormap_curve_strength = 0.0
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

        self.colormap_curve_spin = QtWidgets.QDoubleSpinBox(control_widget)
        self.colormap_curve_spin.setDecimals(2)
        self.colormap_curve_spin.setRange(0.0, 8.0)
        self.colormap_curve_spin.setSingleStep(0.25)
        self.colormap_curve_spin.setKeyboardTracking(False)
        self.colormap_curve_spin.setToolTip(
            "Stretch low and high colormap regions without changing the numeric range."
        )
        self.colormap_curve_spin.valueChanged.connect(self._on_colormap_curve_changed)

        control_layout.addRow("Min:", self.range_min_spin)
        control_layout.addRow("Max:", self.range_max_spin)
        control_layout.addRow("Color curve:", self.colormap_curve_spin)
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
            vmax=grid_max,
            orientation=getattr(self, "orientation", "default"),
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
        self.orientation = getattr(data, "orientation", "default")
        self.unit = getattr(data, "unit", "µm")
        logger.debug(f"grid: {self.grid.shape}, xmin: {self.xi[0]}, ymin: {self.yi[0]}, px_x: {self.dx}, px_y: {self.dy}")
        
        # Update histogram first to set threshold lines
        self.histogram_manager.update_histogram(
            self.grid,
            colormap_name=self.get_colormap_name(),
            colormap_curve_strength=self.colormap_curve_strength,
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
            self.colormap_curve_strength,
        )
        self._sync_threshold_controls()

    def _on_threshold_range_changed(self, vmin: float, vmax: float):
        """Handle threshold changes coming from histogram lines."""
        self._sync_threshold_controls(vmin, vmax)
        self.update_image(vmin, vmax)

    def _on_colormap_curve_changed(self, value: float) -> None:
        """Apply a manual endpoint-stretch curve to the active 2D colormap."""
        self.colormap_curve_strength = max(0.0, float(value))
        self.update_histogram()
        self.update_image()

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
        
        display_grid = self._get_display_grid()

        # IMPORTANT: grid.T creates a VIEW, not a copy!
        # Make a copy immediately to avoid accidentally modifying the source grid.
        image_data = grid_to_image_data(
            display_grid,
            orientation=getattr(self, "orientation", "default"),
            copy=True,
        )
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
            lut = get_lookup_table(cmap_name, 256, curve_strength=self.colormap_curve_strength)
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
    @classmethod
    def _compute_display_stride(cls, grid_shape: tuple[int, int]) -> int:
        """Return a 2D preview stride that keeps redraw cost bounded.

        The tab still stores and processes the full-resolution grid. Only the
        interactive 2D display is decimated for very large scans so that
        histogram threshold drags and repeated redraws remain responsive.
        """
        if len(grid_shape) != 2:
            return 1

        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        if rows <= 0 or cols <= 0:
            return 1

        total_pixels = rows * cols
        if total_pixels <= cls.MAX_DISPLAY_PIXELS:
            return 1
        return max(1, int(np.ceil(np.sqrt(total_pixels / float(cls.MAX_DISPLAY_PIXELS)))))

    def _get_display_grid(self) -> np.ndarray:
        """Return the grid used for interactive 2D drawing.

        Very large scans are shown through a regularly decimated preview while
        all processing and exports continue to use the original full grid.
        """
        stride = self._compute_display_stride(self.grid.shape)
        if stride <= 1:
            return self.grid
        return self.grid[::stride, ::stride]

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
        image_item.setRect(
            build_image_rect(
                self.grid.shape,
                dx=dx,
                dy=dy,
                x0=x0,
                y0=y0,
                orientation=getattr(self, "orientation", "default"),
            )
        )

    def physical_to_indices(self, x_phys: float, y_phys: float) -> tuple[int, int]:
        """Convert physical coordinates to nearest grid indices."""
        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        return orientation_physical_to_indices(
            x_phys,
            y_phys,
            dx=dx,
            dy=dy,
            x0=x0,
            y0=y0,
            orientation=getattr(self, "orientation", "default"),
        )

    def indices_to_physical(self, x_idx: int, y_idx: int) -> tuple[float, float]:
        """Convert grid indices to physical coordinates at pixel centers."""
        dx = self.dx if self.dx not in (None, 0) else 1.0
        dy = self.dy if self.dy not in (None, 0) else 1.0
        x0 = self.xi[0] if self.xi is not None and len(self.xi) else 0.0
        y0 = self.yi[0] if self.yi is not None and len(self.yi) else 0.0
        return orientation_indices_to_physical(
            x_idx,
            y_idx,
            dx=dx,
            dy=dy,
            x0=x0,
            y0=y0,
            orientation=getattr(self, "orientation", "default"),
        )
    
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

    def set_colormap_curve_strength(self, strength: float) -> None:
        """Set the manual endpoint-stretch strength for the 2D colormap."""
        resolved_strength = max(0.0, float(strength))
        self.colormap_curve_strength = resolved_strength
        self.colormap_curve_spin.blockSignals(True)
        self.colormap_curve_spin.setValue(resolved_strength)
        self.colormap_curve_spin.blockSignals(False)
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

        combo_precision = QtWidgets.QComboBox(dialog)
        combo_precision.addItem("Auto", userData=None)
        for decimals in range(0, 7):
            combo_precision.addItem(f"{decimals} decimals", userData=decimals)
        combo_precision.setCurrentIndex(1)

        spin_font_size = QtWidgets.QSpinBox(dialog)
        spin_font_size.setRange(8, 24)
        spin_font_size.setValue(12)

        form.addRow("Source:", combo_source)
        form.addRow("Width [px]:", spin_width)
        form.addRow("Height [px]:", spin_height)
        form.addRow("Precision:", combo_precision)
        form.addRow("Font size [pt]:", spin_font_size)
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
            decimals=combo_precision.currentData(),
            font_size=spin_font_size.value(),
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
        image_data, color_vmin, color_vmax, _scale_vmin, _scale_vmax = self._build_display_array(source)
        return self._display_array_to_qimage(image_data, color_vmin, color_vmax, transparent_background)

    def build_export_colorbar(
        self,
        source: str = "full",
        width: int = 220,
        height: int = 900,
        transparent_background: bool = True,
        include_histogram: bool = True,
        decimals: int | None = 0,
        font_size: int | None = None,
    ) -> QtGui.QImage:
        """Build a standalone colorbar image for the current 2D display.

        Args:
            source: ``"full"`` or ``"viewport"`` to match the exported image scope.
            width: Output image width in pixels.
            height: Output image height in pixels.
            transparent_background: If True, use an alpha-0 canvas.
            include_histogram: If True, draw the data histogram beside the bar.
            decimals: Number of decimal places for exported labels. ``None``
                keeps automatic compact formatting.
            font_size: Optional explicit label font size in points.

        Returns:
            QImage containing the colorbar, labels, and optional histogram.
        """
        width = max(32, int(width))
        height = max(64, int(height))
        image_data, color_vmin, color_vmax, scale_vmin, scale_vmax = self._build_display_array(source)
        values = self._get_histogram_values(source, scale_vmin, scale_vmax)
        renderer = ExportColorbarRenderer(
            ColorbarRenderConfig(
                width=width,
                height=height,
                title_text=self.unit,
                gradient_stops=self._build_colorbar_gradient_stops(),
                color_vmin=color_vmin,
                color_vmax=color_vmax,
                scale_vmin=scale_vmin,
                scale_vmax=scale_vmax,
                hist_values=values,
                transparent_background=transparent_background,
                include_histogram=include_histogram,
                decimals=decimals,
                font_size=font_size,
                title_alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
            )
        )
        return renderer.render()

    def _build_display_array(self, source: str) -> tuple[np.ndarray, float, float, float, float]:
        """Return export-ready image data plus coloring and colorbar ranges.

        The interactive 2D view uses ``grid.T`` because ``pyqtgraph.ImageView``
        expects the first axis to represent X. PNG export goes through
        ``QImage``, which expects the standard raster order
        ``image[row, column] == image[y, x]``. Export therefore keeps the raw
        grid orientation so the saved image matches the visible scan instead of
        being transposed.

        The exported raster keeps the same color mapping as the 2D image view.
        The exported colorbar can span a wider numeric range when visible
        outliers are clipped to endpoint colors, so the scale reflects all
        displayed values while the plateau colors still match the image.
        """
        grid = self._extract_export_grid(source)
        color_vmin, color_vmax, scale_vmin, scale_vmax = self._resolve_export_ranges(grid)

        image_data = grid.copy()
        invalid_mask = np.isnan(image_data)
        if self.hide_below_range:
            image_data[image_data < color_vmin] = np.nan
        else:
            image_data[image_data < color_vmin] = color_vmin
        if self.hide_above_range:
            image_data[image_data > color_vmax] = np.nan
        else:
            image_data[image_data > color_vmax] = color_vmax
        image_data[invalid_mask] = np.nan
        return image_data, color_vmin, color_vmax, scale_vmin, scale_vmax

    def _resolve_export_ranges(self, grid: np.ndarray) -> tuple[float, float, float, float]:
        """Resolve image-color and colorbar ranges for the current export settings.

        The image itself always uses the interactive threshold range for color
        sampling. The colorbar range expands to the full finite data extent
        whenever low or high outliers stay visible, while values outside the
        threshold range keep the endpoint colors exactly as they do in the
        exported raster.
        """
        threshold_vmin, threshold_vmax = self.histogram_manager.get_threshold_range()
        data_min, data_max = self.histogram_manager.get_data_range()
        finite_values = grid[np.isfinite(grid)]
        if finite_values.size > 0:
            data_min = float(np.min(finite_values))
            data_max = float(np.max(finite_values))
        elif data_min is not None and data_max is not None:
            data_min = float(data_min)
            data_max = float(data_max)
        else:
            data_min = 0.0
            data_max = 1.0

        if threshold_vmin is None or threshold_vmax is None:
            threshold_vmin = data_min
            threshold_vmax = data_max

        color_vmin = float(threshold_vmin)
        color_vmax = float(threshold_vmax)
        if color_vmax <= color_vmin:
            color_vmax = color_vmin + 1e-9

        scale_vmin = data_min if not self.hide_below_range else color_vmin
        scale_vmax = data_max if not self.hide_above_range else color_vmax
        if scale_vmax <= scale_vmin:
            scale_vmax = scale_vmin + 1e-9

        return color_vmin, color_vmax, float(scale_vmin), float(scale_vmax)

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
            lut = get_lookup_table(
                self.get_colormap_name(),
                256,
                curve_strength=self.colormap_curve_strength,
            )
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

    def _get_histogram_values(self, source: str, scale_vmin: float, scale_vmax: float) -> np.ndarray:
        """Return finite source values contributing to the exported colorbar histogram.

        Histogram export follows the numeric colorbar scale rather than the
        already-clamped image raster. This keeps visible outliers represented in
        the histogram whenever they are still drawn in the exported image.
        """
        grid = self._extract_export_grid(source)
        values = np.asarray(grid[np.isfinite(grid)], dtype=np.float32)
        if values.size < 1:
            return values
        return values[(values >= scale_vmin) & (values <= scale_vmax)]

    def _build_colorbar_gradient_stops(self) -> list[tuple[float, QtGui.QColor]]:
        """Build gradient stops for the shared exported colorbar renderer."""
        if not self.is_colormap or self.current_colormap is None:
            return [
                (0.0, QtGui.QColor(0, 0, 0)),
                (1.0, QtGui.QColor(255, 255, 255)),
            ]

        return [
            (float(position), QtGui.QColor(*rgba))
            for position, rgba in get_gradient_stops(
                self.get_colormap_name(),
                samples=64,
                curve_strength=self.colormap_curve_strength,
            )
        ]

    def _build_colorbar_gradient(
        self,
        bar_rect: QtCore.QRect,
        color_vmin: float,
        color_vmax: float,
        scale_vmin: float,
        scale_vmax: float,
    ) -> QtGui.QLinearGradient:
        """Return a vertical gradient matching the active 2D colormap.

        When visible outliers extend beyond the threshold range, the colorbar
        shows flat endpoint colors below ``color_vmin`` and above
        ``color_vmax`` to mirror the clipped image rendering.
        """
        gradient = QtGui.QLinearGradient(
            float(bar_rect.left()),
            float(bar_rect.bottom()),
            float(bar_rect.left()),
            float(bar_rect.top()),
        )
        span = max(scale_vmax - scale_vmin, 1e-9)
        lower_fraction = float(np.clip((color_vmin - scale_vmin) / span, 0.0, 1.0))
        upper_fraction = float(np.clip((color_vmax - scale_vmin) / span, 0.0, 1.0))

        if not self.is_colormap or self.current_colormap is None:
            self._populate_grayscale_gradient(gradient, lower_fraction, upper_fraction)
            return gradient

        cmap = get_colormap(self.get_colormap_name())
        if cmap is None:
            self._populate_grayscale_gradient(gradient, lower_fraction, upper_fraction)
            return gradient

        positions, colors = cmap.getStops(mode="byte")
        gradient.setColorAt(
            0.0,
            QtGui.QColor(int(colors[0][0]), int(colors[0][1]), int(colors[0][2]), int(colors[0][3])),
        )
        if lower_fraction > 0.0:
            gradient.setColorAt(
                lower_fraction,
                QtGui.QColor(int(colors[0][0]), int(colors[0][1]), int(colors[0][2]), int(colors[0][3])),
            )

        active_span = max(upper_fraction - lower_fraction, 1e-9)
        for position, rgba in zip(positions, colors):
            mapped_position = lower_fraction + float(position) * active_span
            gradient.setColorAt(
                float(np.clip(mapped_position, 0.0, 1.0)),
                QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
            )
        if upper_fraction < 1.0:
            gradient.setColorAt(
                upper_fraction,
                QtGui.QColor(int(colors[-1][0]), int(colors[-1][1]), int(colors[-1][2]), int(colors[-1][3])),
            )
        gradient.setColorAt(
            1.0,
            QtGui.QColor(int(colors[-1][0]), int(colors[-1][1]), int(colors[-1][2]), int(colors[-1][3])),
        )
        return gradient

    @staticmethod
    def _populate_grayscale_gradient(
        gradient: QtGui.QLinearGradient,
        lower_fraction: float,
        upper_fraction: float,
    ) -> None:
        """Populate a grayscale gradient with flat endpoint plateaus."""
        black = QtGui.QColor(0, 0, 0)
        white = QtGui.QColor(255, 255, 255)
        gradient.setColorAt(0.0, black)
        if lower_fraction > 0.0:
            gradient.setColorAt(lower_fraction, black)
        gradient.setColorAt(upper_fraction, white)
        gradient.setColorAt(1.0, white)

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
        left_label_left: int,
        left_label_width: int,
        right_label_left: int,
        image_width: int,
        scale_vmin: float,
        scale_vmax: float,
        color_vmin: float,
        color_vmax: float,
        text_color: QtGui.QColor,
        tick_color: QtGui.QColor,
        label_font_size: int = 12,
        decimals: int | None = None,
    ) -> None:
        """Draw left-side regular ticks and right-side reference labels."""
        painter.setPen(text_color)
        label_font = QtGui.QFont()
        label_font.setPointSize(int(label_font_size))
        painter.setFont(label_font)
        label_metrics = QtGui.QFontMetricsF(label_font)

        tick_layout = build_colorbar_tick_layout(scale_vmin, scale_vmax, target_count=6.0, minor_subdivisions=3)
        major_ticks = tick_layout["major"]
        minor_ticks = tick_layout["minor"]

        for tick in major_ticks:
            value = float(tick["value"])
            fraction = float(tick["fraction"])
            is_zero = bool(tick["is_zero"])
            y = bar_rect.bottom() - fraction * bar_rect.height()
            tick_length = 12 if is_zero else 10
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.left() - 1, int(y), bar_rect.left() - tick_length, int(y))
            font = QtGui.QFont(label_font)
            font.setBold(is_zero)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(text_color, 1.0))
            text_rect = build_tick_label_rect(
                left_label_left,
                left_label_width - 6,
                y,
                label_metrics,
                placement="above",
            )
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                format_colorbar_value(value, decimals=decimals, trim_trailing_zeros=True),
            )
        painter.setFont(label_font)

        for tick in minor_ticks:
            fraction = float(tick["fraction"])
            minor_y = bar_rect.bottom() - fraction * bar_rect.height()
            painter.setPen(QtGui.QPen(tick_color, 1.0))
            painter.drawLine(bar_rect.left() - 1, int(minor_y), bar_rect.left() - 6, int(minor_y))

        self._draw_colorbar_reference_labels(
            painter,
            bar_rect,
            right_label_left,
            image_width,
            scale_vmin,
            scale_vmax,
            color_vmin,
            color_vmax,
            text_color,
            label_font,
            tick_color,
            decimals=decimals,
        )

    def _build_export_colorbar_tick_layout(self, vmin: float, vmax: float) -> dict[str, list[dict[str, float | bool]]]:
        """Build regular major and minor colorbar ticks with an explicit zero tick.

        Args:
            vmin: Lower end of the displayed data range.
            vmax: Upper end of the displayed data range.

        Returns:
            Dictionary with ``major`` and ``minor`` tick definitions.
        """
        return build_colorbar_tick_layout(vmin, vmax, target_count=6.0, minor_subdivisions=3)

    @staticmethod
    def _select_export_tick_step(vmin: float, vmax: float, target_count: float = 6.0) -> float:
        """Choose a rounded major-tick step for colorbar exports."""
        extent = max(float(vmax) - float(vmin), 1e-9)
        raw_step = extent / max(float(target_count), 1.0)
        exponent = np.floor(np.log10(raw_step))
        base = 10.0 ** exponent
        normalized = raw_step / base
        if normalized <= 1.0:
            nice = 1.0
        elif normalized <= 2.0:
            nice = 2.0
        elif normalized <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * base)

    @staticmethod
    def _build_ticks_for_step(vmin: float, vmax: float, step: float) -> list[dict[str, float | bool]]:
        """Build major ticks for a given regular step and include zero if visible."""
        if step <= 0:
            return []
        tolerance = max(1e-9, abs(step) * 1e-6)
        start_index = int(np.ceil((vmin - tolerance) / step))
        end_index = int(np.floor((vmax + tolerance) / step))
        ticks: list[dict[str, float | bool]] = []
        for index in range(start_index, end_index + 1):
            value = float(index * step)
            if value < vmin - tolerance or value > vmax + tolerance:
                continue
            ticks.append(
                {
                    "value": value,
                    "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
                    "is_zero": abs(value) <= tolerance,
                }
            )

        if vmin <= 0.0 <= vmax and not any(bool(tick["is_zero"]) for tick in ticks):
            ticks.append(
                {
                    "value": 0.0,
                    "fraction": (0.0 - vmin) / max(vmax - vmin, 1e-9),
                    "is_zero": True,
                }
            )

        if not ticks:
            midpoint = 0.5 * (vmin + vmax)
            ticks = [
                {"value": float(vmax), "fraction": 1.0, "is_zero": abs(vmax) <= tolerance},
                {"value": float(midpoint), "fraction": 0.5, "is_zero": abs(midpoint) <= tolerance},
                {"value": float(vmin), "fraction": 0.0, "is_zero": abs(vmin) <= tolerance},
            ]

        ticks.sort(key=lambda item: float(item["fraction"]))
        return ticks

    @staticmethod
    def _build_minor_ticks(vmin: float, vmax: float, major_step: float, subdivisions: int) -> list[dict[str, float]]:
        """Build minor ticks between neighboring major tick values."""
        if major_step <= 0 or subdivisions < 1:
            return []
        tolerance = max(1e-9, abs(major_step) * 1e-6)
        step = major_step / float(subdivisions + 1)
        start_index = int(np.ceil((vmin - tolerance) / step))
        end_index = int(np.floor((vmax + tolerance) / step))
        minor_ticks: list[dict[str, float]] = []
        for index in range(start_index, end_index + 1):
            value = float(index * step)
            if value < vmin - tolerance or value > vmax + tolerance:
                continue
            if abs((value / major_step) - round(value / major_step)) <= 1e-6:
                continue
            minor_ticks.append(
                {
                    "value": value,
                    "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
                }
            )
        return minor_ticks

    @staticmethod
    def _format_export_value(
        value: float,
        decimals: int | None = None,
        trim_trailing_zeros: bool = False,
    ) -> str:
        """Format numeric labels for export annotations.

        Args:
            value: Numeric value to format.
            decimals: Fixed decimal count. ``None`` keeps automatic compact
                formatting with scientific notation for very large or small
                magnitudes.
            trim_trailing_zeros: If True, remove trailing zeroes and a dangling
                decimal separator from fixed-decimal formatting.
        """
        return format_colorbar_value(
            value,
            decimals=decimals,
            trim_trailing_zeros=trim_trailing_zeros,
        )

    def _draw_colorbar_reference_labels(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        label_left: int,
        image_width: int,
        scale_vmin: float,
        scale_vmax: float,
        color_vmin: float,
        color_vmax: float,
        text_color: QtGui.QColor,
        label_font: QtGui.QFont,
        tick_color: QtGui.QColor,
        decimals: int | None = None,
    ) -> None:
        """Draw right-side labels for visible data limits and threshold limits."""
        if scale_vmax <= scale_vmin:
            return

        label_specs = self._build_colorbar_reference_label_specs(
            scale_vmin=scale_vmin,
            scale_vmax=scale_vmax,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
            decimals=decimals,
        )
        if not label_specs:
            return

        sorted_specs = sorted(
            label_specs,
            key=lambda item: float(bar_rect.bottom()) - float(item["fraction"]) * bar_rect.height(),
        )
        target_y_positions = [
            float(bar_rect.bottom()) - float(spec["fraction"]) * bar_rect.height()
            for spec in sorted_specs
        ]
        placed_y_positions = self._layout_colorbar_reference_label_positions(
            bar_rect,
            sorted_specs,
            min_gap=max(float(QtGui.QFontMetricsF(label_font).height()) * 0.9, 12.0),
            font_height=float(QtGui.QFontMetricsF(label_font).height()),
        )

        painter.setFont(label_font)
        for spec, tick_y, text_y in zip(sorted_specs, target_y_positions, placed_y_positions):
            tick_length = 10
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.right() + 1, int(tick_y), bar_rect.right() + tick_length, int(tick_y))

            text_rect = build_tick_label_rect(
                label_left,
                image_width - label_left - 8,
                text_y,
                QtGui.QFontMetricsF(label_font),
                placement="center",
            )
            painter.setFont(label_font)
            painter.setPen(QtGui.QPen(text_color, 1.0))
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                str(spec["text"]),
            )
        painter.setFont(label_font)

    def _build_colorbar_reference_label_specs(
        self,
        scale_vmin: float,
        scale_vmax: float,
        color_vmin: float,
        color_vmax: float,
        decimals: int | None = None,
    ) -> list[dict[str, float | str | bool]]:
        """Build right-side labels for data limits and threshold limits.

        The exported colorbar always marks the visible data limits. Threshold
        limits are added when they differ from the visible minimum or maximum,
        which makes clipped outlier plateaus explicit without crowding the
        regular left-side tick scale.
        """
        return build_reference_label_specs(
            scale_vmin=scale_vmin,
            scale_vmax=scale_vmax,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
            decimals=decimals,
        )

    @staticmethod
    def _layout_colorbar_reference_label_positions(
        bar_rect: QtCore.QRect,
        sorted_specs: list[dict[str, float | str | bool]],
        min_gap: float = 18.0,
        font_height: float = 24.0,
    ) -> list[float]:
        """Lay out right-side labels from top to bottom with a minimum gap."""
        return layout_reference_label_positions(
            bar_rect,
            sorted_specs,
            font_height=font_height,
            min_gap=min_gap,
        )

    @staticmethod
    def _get_colorbar_font_sizes(
        width: int,
        height: int,
        font_size: int | None = None,
    ) -> tuple[int, int]:
        """Return title and label font sizes for the exported colorbar.

        Args:
            width: Export image width in pixels.
            height: Export image height in pixels.
            font_size: Optional explicit label font size in points.
        """
        return get_colorbar_font_sizes(width, height, font_size=font_size)

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
    
    def fill_holes(self, parent=None, mask: np.ndarray = None):
        """Fill holes in scan data using interpolation.
        
        Args:
            parent (QWidget): Parent widget for error messages
            mask (np.ndarray, optional): Boolean mask limiting fill to active ROI.
        """
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return

        tst = np.isnan(self.grid)
        if mask is not None:
            tst = tst & mask

        if not np.any(tst):
            return
        self.grid = fill_holes(self.grid, mask=mask)
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
