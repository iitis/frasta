"""Interactive handler for scan tab mouse events.

Handles mouse clicks for zero point selection, tilt correction, and seed point marking.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter

from ...processing import fit_plane_local_median_filter

import logging
logger = logging.getLogger(__name__)


class InteractiveHandler:
    """Handles interactive mouse events and modes."""
    
    def __init__(self, parent_tab):
        """Initialize interactive handler.
        
        Args:
            parent_tab: Reference to parent ScanTab instance
        """
        self.parent_tab = parent_tab
        self.zero_point_mode = False
        self.tilt_mode = False
        self.seed_points = []
        
        # Parameters for zero point calculation
        self.zero_window_size = 15
        self.zero_sigma = 2.0
    
    def set_zero_point_mode(self):
        """Enable zero point selection mode."""
        self.zero_point_mode = True
    
    def set_tilt_mode(self):
        """Enable tilt correction mode."""
        self.tilt_mode = True
    
    def handle_mouse_click(self, event):
        """Handle mouse click events on image view.
        
        Args:
            event: QGraphicsSceneMouseEvent containing click position
        """
        grid = self.parent_tab.grid
        if grid is None:
            return
        
        # Get click coordinates
        vb = self.parent_tab.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        
        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
            return
        
        # Handle different modes
        if self.zero_point_mode:
            self._handle_zero_point_click(x, y)
        elif self.tilt_mode:
            self._handle_tilt_click(x, y)
        elif event.modifiers() & QtCore.Qt.ShiftModifier:
            self._handle_seed_point_click(x, y, vb)
    
    def _handle_zero_point_click(self, x: int, y: int):
        """Handle click in zero point mode.
        
        Args:
            x (int): X coordinate of click
            y (int): Y coordinate of click
        """
        value = self._get_zero_point_value(x, y)
        if np.isnan(value):
            QtWidgets.QMessageBox.warning(
                self.parent_tab, "No data available", 
                "The selected point contains no value (NaN)."
            )
            self.zero_point_mode = False
            return
        
        # Get old threshold values
        hist_mgr = self.parent_tab.histogram_manager
        vmin, vmax = hist_mgr.get_threshold_range()
        
        # Shift entire scan in Z axis
        self.parent_tab.grid = self.parent_tab.grid - value
        self.parent_tab.update_image()
        self.zero_point_mode = False
        
        # Adjust threshold lines
        min_val = vmin - value
        max_val = vmax - value
        
        self.parent_tab.update_histogram()
        self.parent_tab.histogram_manager.set_threshold_values(min_val, max_val)
    
    def _handle_tilt_click(self, x: int, y: int):
        """Handle click in tilt correction mode.
        
        Args:
            x (int): X coordinate of click
            y (int): Y coordinate of click
        """
        self.tilt_mode = False
        try:
            # Fit plane to local window around click
            a, b, c = fit_plane_local_median_filter(
                self.parent_tab.grid, x, y, 
                window_size=500, outlier_threshold=300.0
            )
            
            # Create matrix of same size as grid
            rows, cols = self.parent_tab.grid.shape
            yy, xx = np.mgrid[0:rows, 0:cols]
            plane = a * xx + b * yy + c
            
            # Apply correction (note: adding, not subtracting!)
            self.parent_tab.grid = self.parent_tab.grid + plane
            self.parent_tab.update_image()
            self.parent_tab.update_histogram()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self.parent_tab, "Error", 
                f"Failed to fit plane: {str(e)}"
            )
    
    def _handle_seed_point_click(self, x: int, y: int, view_box):
        """Handle shift+click for seed point marking.
        
        Args:
            x (int): X coordinate of click
            y (int): Y coordinate of click
            view_box: PyQtGraph ViewBox to add marker
        """
        self.seed_points.append((y, x))
        scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
        view_box.addItem(scatter)
    
    def _get_zero_point_value(self, x: int, y: int) -> float:
        """Calculate robust zero point value from local window.
        
        This function returns the mean of non-outlier values within a window 
        centered at (x, y), or the median if all values are outliers or missing.
        
        Args:
            x (int): X-coordinate of window center
            y (int): Y-coordinate of window center
            
        Returns:
            float: Calculated zero point value, or NaN if no valid data
        """
        s = self.zero_window_size // 2
        grid = self.parent_tab.grid
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)
        window = grid[ymin:ymax, xmin:xmax]
        
        # Values without NaN
        vals = window[~np.isnan(window)]
        if len(vals) == 0:
            return np.nan
        
        # Reject outliers (e.g. 2 sigma from median)
        median = np.median(vals)
        std = np.std(vals)
        non_outliers = vals[np.abs(vals - median) < self.zero_sigma * std]
        
        # If no values remain after rejection - take median
        return median if len(non_outliers) == 0 else np.mean(non_outliers)
    
    def clear_seed_points(self):
        """Clear all seed points."""
        self.seed_points = []
    
    def get_seed_points(self) -> list:
        """Get list of seed points.
        
        Returns:
            list: List of (y, x) seed point coordinates
        """
        return self.seed_points
