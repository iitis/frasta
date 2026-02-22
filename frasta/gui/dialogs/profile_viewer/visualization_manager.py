"""Visualization management for profile viewer.

Handles 3D visualization, image view sizing, volume calculations, and statistics display.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF

from ...viewers import show_3d_viewer

import logging
logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization features including 3D view and statistics.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize visualization manager.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
    
    # ==========================================================================
    # 3D Visualization
    # ==========================================================================
    
    def show_3d_view(self):
        """Open 3D viewer showing both scans and profile line."""
        viewbox = self.parent.image_view.getView()
        x_range, y_range = viewbox.viewRange()
        
        # Convert ranges to image indices
        x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1]))
        
        # Ensure within image bounds
        shape = self.parent.reference_grid_smooth.shape
        x_min = max(0, x_min)
        x_max = min(shape[1] - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(shape[0] - 1, y_max)
        
        # Extract grid fragments
        ref = self.parent.reference_grid_smooth[y_min:y_max + 1, x_min:x_max + 1]
        adj = self.parent.adjusted_grid_corrected[y_min:y_max + 1, x_min:x_max + 1]
        
        logger.debug(f"ref0: {self.parent.reference_grid_smooth.shape}, adj0: {self.parent.adjusted_grid_corrected.shape}")
        logger.debug(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
        logger.debug(f"ref min: {np.nanmin(ref)}, ref max: {np.nanmax(ref)}, ref shape: {ref.shape}")
        logger.debug(f"ref NaN count: {np.isnan(ref).sum()}")
        logger.debug(f"adj min: {np.nanmin(adj)}, adj max: {np.nanmax(adj)}, adj shape: {adj.shape}")
        logger.debug(f"adj NaN count: {np.isnan(adj).sum()}")
        
        # Prepare profile line (limited to fragment, with NaNs to break line)
        if hasattr(self.parent, 'rr_full') and hasattr(self.parent, 'cc_full'):
            line_points = [
                (int(col - x_min), int(row - y_min))
                for col, row in zip(self.parent.cc_full, self.parent.rr_full)
                if x_min <= col <= x_max and y_min <= row <= y_max
            ]
            if len(line_points) < 2:
                line_points = None
        else:
            line_points = None
        
        show_3d_viewer(
            reference_grid=ref,
            adjusted_grid=adj,
            line_points=line_points,
            separation=self.parent.separation,
            show_controls=True,
            pixel_size_x=self.parent.ref_pixel_um.x(),
            pixel_size_y=self.parent.ref_pixel_um.y()
        )
    
    def show_preview(self, fragment, title="Region preview"):
        """Show preview window with image fragment.
        
        Args:
            fragment (np.ndarray): Image data to display.
            title (str): Window title.
        """
        if getattr(self.parent, "_preview_win", None) is None:
            self.parent._preview_win = pg.ImageView()
            self.parent._preview_win.setWindowTitle(title)
            self.parent._preview_win.show()
        
        self.parent._preview_win.setImage(fragment)
        self.parent._preview_win.raise_()
        self.parent._preview_win.activateWindow()
    
    # ==========================================================================
    # View Management
    # ==========================================================================
    
    def resize_image_view(self, shape):
        """Resize image view widget based on aspect ratio.
        
        Args:
            shape (tuple): Image shape (height, width).
        """
        height, width = shape
        aspect = width / height
        base = 500
        
        if aspect >= 1.0:
            w = base
            h = int(base / aspect)
        else:
            h = base
            w = int(base * aspect)
        
        self.parent.image_view.setFixedSize(w, h)
        self.parent.image_view.update()
        self.parent.updateGeometry()
    
    def get_viewbox_ranges_int(self, shape=None, overflow=False):
        """Get current viewbox range as integer pixel coordinates.
        
        Args:
            shape (tuple, optional): Image shape for clamping (height, width).
            overflow (bool): If True, use floor/ceil for range boundaries.
        
        Returns:
            tuple: (x_min, x_max, y_min, y_max) as integers.
        """
        viewbox = self.parent.image_view.getView()
        x_range, y_range = viewbox.viewRange()
        
        min_range = viewbox.mapToParent(QPointF(x_range[0], y_range[0]))
        max_range = viewbox.mapToParent(QPointF(x_range[1], y_range[1]))
        
        x_range = [min_range.x(), max_range.x()]
        y_range = [min_range.y(), max_range.y()]
        
        logger.debug(f"ViewBox x_range: {x_range}, y_range: {y_range}")
        
        if overflow:
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1])) - 1
            y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1])) - 1
        else:
            x_min, x_max = int(np.ceil(x_range[0])), int(np.floor(x_range[1])) - 1
            y_min, y_max = int(np.ceil(y_range[0])), int(np.floor(y_range[1])) - 1
        
        if shape is not None:
            x_min = max(0, x_min)
            x_max = min(shape[1] - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(shape[0] - 1, y_max)
        
        return x_min, x_max, y_min, y_max
    
    def on_range_changed(self, viewbox, ranges):
        """Handle viewbox range change event - update statistics.
        
        Args:
            viewbox: PyQtGraph viewbox.
            ranges: New range values.
        """
        self.update_volume_info()
    
    # ==========================================================================
    # Statistics and Volume Calculations
    # ==========================================================================
    
    def update_volume_info(self):
        """Calculate and display contact area and volume statistics for current view."""
        if self.parent.binary_contact is None:
            return
        
        x_min, x_max, y_min, y_max = self.get_viewbox_ranges_int(
            shape=self.parent.binary_contact.shape
        )
        
        px_um = self.parent.ref_pixel_um.x()
        py_um = self.parent.ref_pixel_um.y()
        pixel_area_um2 = px_um * py_um
        
        # Get binary fragment in current view
        fragment = self.parent.binary_contact[y_min:y_max+1, x_min:x_max+1]
        
        # Calculate contact area
        white_count = np.count_nonzero(fragment)
        white_area_um2 = pixel_area_um2 * white_count
        white_area_mm2 = white_area_um2 * 1e-6
        
        # Calculate volume
        ref = self.parent.reference_grid_smooth[y_min:y_max+1, x_min:x_max+1]
        adj = self.parent.adjusted_grid_corrected[y_min:y_max+1, x_min:x_max+1]
        diff = ref - (adj + self.parent.separation)
        
        diff_masked = np.where(fragment, diff, 0)
        
        volume_um3 = np.abs(np.sum(diff_masked)) * pixel_area_um2
        volume_mm3 = volume_um3 * 1e-9
        
        # Display in status bar
        self.parent.statusBar().showMessage(
            f"White fields in view: {white_count}, "
            f"area: {white_area_um2:.4f}μm² ({white_area_mm2}mm²), "
            f"volume: {volume_um3:.4f}μm³ ({volume_mm3:.4f}mm³)"
        )
