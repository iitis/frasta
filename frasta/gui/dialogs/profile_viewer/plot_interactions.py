"""Plot interaction handling for profile viewer.

Manages mouse interactions with profile plot, including:
- Cursor tracking and display
- Profile point annotation (heights, angles)
- Linear fit visualization
- Point saving (Ctrl+click)
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

import logging
logger = logging.getLogger(__name__)


class PlotInteractions:
    """Handles interactive plot features and user input on profile plot.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize plot interactions handler.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
    
    # ==========================================================================
    # Mouse Event Handlers
    # ==========================================================================
    
    def on_plot_click(self, event):
        """Handle plot click events - save point on Ctrl+click.
        
        Args:
            event: PyQtGraph mouse click event.
        """
        if event.modifiers() == QtCore.Qt.ControlModifier:
            self._handle_ctrl_click(event)
    
    def _handle_ctrl_click(self, event):
        """Handle Ctrl+click to save a profile point.
        
        Args:
            event: PyQtGraph mouse click event.
        """
        pos = event.scenePos()
        if not self.parent.plot_widget.sceneBoundingRect().contains(pos):
            return
        
        mouse_point = self.parent.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_pos = mouse_point.x()
        
        if not (self.parent.positions_line[0] <= x_pos <= self.parent.positions_line[-1]):
            return
        
        idx = np.argmin(np.abs(self.parent.positions_line - x_pos))
        if hasattr(self.parent, 'rr') and hasattr(self.parent, 'cc'):
            self._save_profile_point(idx)
    
    def _save_profile_point(self, idx):
        """Save a profile point with its coordinates and values.
        
        Args:
            idx (int): Index of point in profile arrays.
        """
        y_img = self.parent.rr[idx]
        x_img = self.parent.cc[idx]
        ref_val = self.parent.reference_profile[idx]
        adj_val = self.parent.adjusted_profile[idx]
        pos_mm = self.parent.positions_line[idx]
        
        self.parent.saved_points.append({
            'profile_idx': idx,
            'x_img': int(x_img),
            'y_img': int(y_img),
            'x_pos_mm': float(pos_mm),
            'ref_val': float(ref_val),
            'adj_val': float(adj_val),
        })
        
        # Add visual marker on image
        marker = pg.ScatterPlotItem(
            [x_img], [y_img], size=12, 
            pen=pg.mkPen('g', width=2), 
            brush=pg.mkBrush(0, 255, 255, 120), 
            symbol='+'
        )
        self.parent.image_view.getView().addItem(marker)
        self.parent.saved_point_markers.append(marker)
        
        logger.debug("Saved point:", self.parent.saved_points[-1])
    
    def on_mouse_move(self, pos):
        """Handle mouse movement over plot - show cursor, annotations, fits.
        
        Args:
            pos: Mouse position in scene coordinates.
        """
        if not self.parent.plot_widget.sceneBoundingRect().contains(pos):
            return
        
        mouse_point = self.parent.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_pos = mouse_point.x()
        
        self._clear_cursor_and_annotations()
        self._draw_cursor_line(x_pos)
        
        positions_line = self.parent.positions_line
        if positions_line[0] <= x_pos <= positions_line[-1]:
            idx = np.argmin(np.abs(positions_line - x_pos))
            self._update_image_marker(idx)
            self._draw_annotations_and_fit_lines(x_pos, idx)
        else:
            self._clear_fit_lines_and_marker()
    
    def print_saved_points(self):
        """Print all saved points to logger (for debugging)."""
        for i, pt in enumerate(self.parent.saved_points):
            logger.debug(f"{i+1}: {pt}")
    
    # ==========================================================================
    # Visual Elements Drawing
    # ==========================================================================
    
    def _clear_cursor_and_annotations(self):
        """Clear all cursor lines and annotations from plot."""
        for item in self.parent.cursor_lines + self.parent.annotations:
            self.parent.plot_widget.removeItem(item)
        self.parent.cursor_lines.clear()
        self.parent.annotations.clear()
    
    def _draw_cursor_line(self, x_pos):
        """Draw vertical cursor line at mouse position.
        
        Args:
            x_pos (float): X position in plot coordinates (mm).
        """
        vline = pg.InfiniteLine(
            pos=x_pos, angle=90, 
            pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine)
        )
        self.parent.plot_widget.addItem(vline)
        self.parent.cursor_lines.append(vline)
    
    def _update_image_marker(self, idx):
        """Update marker position on image view at profile index.
        
        Args:
            idx (int): Index in profile arrays.
        """
        if hasattr(self.parent, 'rr') and hasattr(self.parent, 'cc'):
            y_img = self.parent.rr[idx]
            x_img = self.parent.cc[idx]
            view = self.parent.image_view.getView()
            
            if self.parent.image_marker is not None:
                view.removeItem(self.parent.image_marker)
            
            self.parent.image_marker = pg.ScatterPlotItem(
                [x_img], [y_img], size=14, 
                pen=pg.mkPen('m', width=2), 
                brush=pg.mkBrush(255, 0, 255, 100)
            )
            view.addItem(self.parent.image_marker)
    
    def _draw_annotations_and_fit_lines(self, x_pos, idx):
        """Draw height difference annotation and linear fit lines.
        
        Args:
            x_pos (float): X position in plot coordinates (mm).
            idx (int): Index in profile arrays.
        """
        height_diff = self.parent.reference_profile[idx] - self.parent.adjusted_profile[idx]
        window_mm = self.parent.spinbox_window_mm.value()
        pixel_size_mm = self.parent.ref_pixel_um.x() / 1000.0
        window_size = max(1, int(round(window_mm / pixel_size_mm)))
        
        start = max(0, idx - window_size)
        end = min(len(self.parent.positions_line), idx + window_size + 1)
        
        # Fit lines and calculate angles
        slope_ref, angle_ref, reg_ref = self.parent.profile_analyzer.fit_profile(
            self.parent.positions_line[start:end], 
            self.parent.reference_profile[start:end]
        )
        slope_adj, angle_adj, reg_adj = self.parent.profile_analyzer.fit_profile(
            self.parent.positions_line[start:end], 
            self.parent.adjusted_profile[start:end]
        )
        delta_angle = angle_ref - angle_adj
        
        # Draw text annotations
        self._draw_diff_and_angle_text(height_diff, angle_ref, angle_adj, delta_angle)
        
        # Draw fit lines
        self._draw_fit_lines(x_pos, slope_ref, reg_ref, slope_adj, reg_adj, idx, window_mm)
    
    def _draw_diff_and_angle_text(self, height_diff, angle_ref, angle_adj, delta_angle):
        """Draw text annotations for height difference and angles.
        
        Args:
            height_diff (float): Height difference in micrometers.
            angle_ref (float): Reference profile angle in degrees.
            angle_adj (float): Adjusted profile angle in degrees.
            delta_angle (float): Angle difference in degrees.
        """
        vb = self.parent.plot_widget.getPlotItem().vb
        x_min, x_max = vb.viewRange()[0]
        y_min, y_max = vb.viewRange()[1]
        
        # Height difference text
        text1 = pg.TextItem(f"DIFF: {height_diff:.2f} μm", color='r', anchor=(0, 1))
        text1.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.05 * (y_max - y_min))
        self.parent.plot_widget.addItem(text1)
        self.parent.annotations.append(text1)
        
        # Angle information text
        text2 = pg.TextItem(
            f"ANGLE\nref: {angle_ref:.1f}°\nadj: {angle_adj:.1f}°\n  Δ: {delta_angle:.1f}°", 
            color='y', anchor=(0, 1)
        )
        text2.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.2 * (y_max - y_min))
        self.parent.plot_widget.addItem(text2)
        self.parent.annotations.append(text2)
    
    def _draw_fit_lines(self, x_pos, slope_ref, reg_ref, slope_adj, reg_adj, idx, window_mm):
        """Draw linear fit lines over profile data.
        
        Args:
            x_pos (float): Current cursor X position (mm).
            slope_ref (float): Reference profile slope.
            reg_ref: Reference profile regression model.
            slope_adj (float): Adjusted profile slope.
            reg_adj: Adjusted profile regression model.
            idx (int): Current profile index.
            window_mm (float): Fitting window size (mm).
        """
        vb = self.parent.plot_widget.getPlotItem().vb
        line_half_width_mm = window_mm / 2.0
        x0 = x_pos - line_half_width_mm
        x1 = x_pos + line_half_width_mm
        
        # Clear previous fit lines
        for item in self.parent.mytest:
            vb.removeItem(item)
        self.parent.mytest.clear()
        
        # Reference fit line
        a = slope_ref
        if self.parent.checkbox_snap.isChecked():
            y_at_cursor = self.parent.reference_profile[idx] / 1000.0
            b = y_at_cursor - a * x_pos
        else:
            b = reg_ref.intercept_[0]
        y0 = a * x0 + b
        y1 = a * x1 + b
        
        line_ref = pg.PlotDataItem(
            [x0, x1], [y0 * 1000, y1 * 1000], 
            pen=pg.mkPen('y', width=2)
        )
        vb.addItem(line_ref, ignoreBounds=True)
        self.parent.annotations.append(line_ref)
        self.parent.mytest.append(line_ref)
        
        # Adjusted fit line
        a = slope_adj
        if self.parent.checkbox_snap.isChecked():
            y_at_cursor_adj = self.parent.adjusted_profile[idx] / 1000.0
            b_adj = y_at_cursor_adj - a * x_pos
        else:
            b_adj = reg_adj.intercept_[0]
        y0 = a * x0 + b_adj
        y1 = a * x1 + b_adj
        
        line_adj = pg.PlotDataItem(
            [x0, x1], [y0 * 1000, y1 * 1000], 
            pen=pg.mkPen('y', width=2)
        )
        vb.addItem(line_adj, ignoreBounds=True)
        self.parent.annotations.append(line_adj)
        self.parent.mytest.append(line_adj)
    
    def _clear_fit_lines_and_marker(self):
        """Clear fit lines and image marker."""
        vb = self.parent.plot_widget.getPlotItem().vb
        for item in self.parent.mytest:
            vb.removeItem(item)
        self.parent.mytest.clear()
        
        view = self.parent.image_view.getView()
        if self.parent.image_marker is not None:
            view.removeItem(self.parent.image_marker)
            self.parent.image_marker = None
