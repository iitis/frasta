"""Histogram manager for scan tab.

Manages histogram display and threshold line controls for contrast adjustment.
"""

import numpy as np
import pyqtgraph as pg
from ..widgets import ResponsiveInfiniteLine

import logging
logger = logging.getLogger(__name__)


class HistogramManager:
    """Manages histogram display and threshold controls."""
    
    def __init__(self, hist_widget, update_callback):
        """Initialize histogram manager.
        
        Args:
            hist_widget (pg.PlotWidget): Histogram display widget
            update_callback (callable): Callback function when threshold changes
        """
        self.hist_widget = hist_widget
        self.update_callback = update_callback
        self.hist_plot = None
        self.hist_min_line = None
        self.hist_max_line = None
        self._updating_histogram = False
    
    def update_histogram(self, grid: np.ndarray, was_data_negated: bool = False):
        """Update histogram display with new data.
        
        Args:
            grid (np.ndarray): Grid data to display
            was_data_negated (bool): Whether data was recently inverted
        """
        logger.debug(f"update_histogram called: grid is None? {grid is None}")
        if grid is None:
            logger.warning("update_histogram: grid is None!")
            return
        
        data = grid[~np.isnan(grid)]
        if data.size == 0:
            logger.warning("update_histogram: no valid data (all NaN)")
            self.hist_widget.clear()
            return

        # Get data range
        vmin = float(np.min(data))
        vmax = float(np.max(data))

        # Remember old positions (if they exist)
        old_min_line = self.hist_min_line
        old_max_line = self.hist_max_line

        if was_data_negated:
            # Flip threshold positions for inverted data
            min_val = np.clip(-old_max_line.value(), vmin, vmax) if old_max_line else vmin
            max_val = np.clip(-old_min_line.value(), vmin, vmax) if old_min_line else vmax
        else:
            # Try to preserve old line positions, but only if they make sense for new data
            if old_min_line is not None and old_max_line is not None:
                old_min = old_min_line.value()
                old_max = old_max_line.value()
                # Check if old values are reasonable for new data range
                if vmin <= old_min <= vmax and vmin <= old_max <= vmax:
                    min_val = old_min
                    max_val = old_max
                else:
                    # Old values don't make sense for new data, use full range
                    min_val = vmin
                    max_val = vmax
                    logger.warning(f"update_histogram: old threshold outside new data range, resetting")
            else:
                min_val = vmin
                max_val = vmax

        # Create histogram
        y, x = np.histogram(data, bins=1024)
        self.hist_widget.clear()
        self.hist_plot = self.hist_widget.plot(
            x, y, stepMode="center", fillLevel=0, brush=(150, 150, 150, 150)
        )

        # Block callbacks while setting up histogram lines to avoid recursive calls
        self._updating_histogram = True
        
        # Create threshold lines
        self.hist_min_line = ResponsiveInfiniteLine(
            update_callback=self._on_threshold_changed, 
            angle=90, movable=True, 
            pen=pg.mkPen('b', width=2), 
            hoverPen=pg.mkPen('y', width=2)
        )
        self.hist_max_line = ResponsiveInfiniteLine(
            update_callback=self._on_threshold_changed, 
            angle=90, movable=True, 
            pen=pg.mkPen('r', width=2), 
            hoverPen=pg.mkPen('y', width=2)
        )
        self.hist_widget.addItem(self.hist_min_line)
        self.hist_widget.addItem(self.hist_max_line)
        self.hist_min_line.setValue(min_val)
        self.hist_max_line.setValue(max_val)
        
        self._updating_histogram = False
        
        logger.debug(f"update_histogram: final histogram lines set to [{min_val:.2f}, {max_val:.2f}]")
    
    def _on_threshold_changed(self, value):
        """Handle threshold line movement.
        
        Args:
            value (float): New threshold value
        """
        # Block recursive calls during histogram setup
        if self._updating_histogram:
            logger.debug(f"Threshold update blocked during histogram setup: {value}")
            return
            
        logger.debug(f"Threshold updated: {value}")
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_callback(vmin, vmax)
    
    def get_threshold_range(self) -> tuple[float, float]:
        """Get current threshold range.
        
        Returns:
            tuple: (vmin, vmax) threshold values
        """
        if self.hist_min_line is not None and self.hist_max_line is not None:
            vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
            vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
            return vmin, vmax
        return None, None
    
    def set_threshold_values(self, vmin: float, vmax: float):
        """Set threshold line values.
        
        Args:
            vmin (float): Minimum threshold
            vmax (float): Maximum threshold
        """
        if self.hist_min_line is not None and self.hist_max_line is not None:
            self.hist_min_line.setValue(vmin)
            self.hist_max_line.setValue(vmax)
