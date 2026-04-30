"""Histogram manager for scan tab.

Manages histogram display and threshold line controls for contrast adjustment.
"""

import numpy as np
import pyqtgraph as pg
from ..widgets import ResponsiveInfiniteLine
from ...utils import get_brushes_for_values

import logging
logger = logging.getLogger(__name__)


class HistogramManager:
    """Manages histogram display and threshold controls."""

    HISTOGRAM_BINS = 512
    
    def __init__(self, hist_widget, update_callback):
        """Initialize histogram manager.
        
        Args:
            hist_widget (pg.PlotWidget): Histogram display widget
            update_callback (callable): Callback function when threshold changes
        """
        self.hist_widget = hist_widget
        self.update_callback = update_callback
        self.hist_plot = None
        self.hist_bars = None
        self.hist_min_line = None
        self.hist_max_line = None
        self._updating_histogram = False
        self.current_colormap_name = "Gray"
        self._hist_centers = None
    
    def update_histogram(
        self,
        grid: np.ndarray,
        was_data_negated: bool = False,
        colormap_name: str = "Gray",
    ):
        """Update histogram display with new data.
        
        Args:
            grid (np.ndarray): Grid data to display
            was_data_negated (bool): Whether data was recently inverted
            colormap_name (str): Active display colormap for matching histogram fill
        """
        logger.debug(f"update_histogram called: grid is None? {grid is None}")
        if grid is None:
            logger.warning("update_histogram: grid is None!")
            return
        self.current_colormap_name = colormap_name
        
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
        old_x_range = self._get_visible_x_range() if self.hist_bars is not None else None

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
        # Keep the displayed histogram moderately dense; higher bin counts make
        # the per-bin colormap rendering noticeably heavier during interaction.
        y, x = np.histogram(data, bins=self.HISTOGRAM_BINS)
        centers = 0.5 * (x[:-1] + x[1:])
        widths = np.diff(x)
        self._hist_centers = centers
        self.hist_widget.clear()
        view_box = self.hist_widget.getViewBox()
        if hasattr(view_box, "set_data_bounds"):
            view_box.set_data_bounds(float(x[0]), float(x[-1]))
        hist_brushes = self._build_histogram_brushes(min_val, max_val)
        self.hist_bars = pg.BarGraphItem(
            x=centers,
            height=y,
            width=widths,
            brushes=hist_brushes,
            pen=pg.mkPen(255, 255, 255, 50),
        )
        self.hist_widget.addItem(self.hist_bars)
        self.hist_plot = self.hist_widget.plot(
            centers,
            y,
            pen=pg.mkPen(255, 255, 255, 110),
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
        self._apply_histogram_coloring(min_val, max_val)
        if old_x_range is None:
            self.hist_widget.setXRange(float(x[0]), float(x[-1]), padding=0.0)
        else:
            restored_x0, restored_x1 = old_x_range
            if hasattr(view_box, "_clamp_x_range"):
                restored_x0, restored_x1 = view_box._clamp_x_range(restored_x0, restored_x1)
            self.hist_widget.setXRange(float(restored_x0), float(restored_x1), padding=0.0)
        
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
        self._apply_histogram_coloring(vmin, vmax)
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
            self._apply_histogram_coloring(vmin, vmax)

    def _build_histogram_brushes(self, vmin: float, vmax: float):
        """Create per-bin brushes using the active display range.

        Bins outside the selected range are drawn in a neutral tone, while bins
        inside the range use colors sampled from the active colormap.
        """
        if self._hist_centers is None:
            return []

        centers = self._hist_centers
        if self.current_colormap_name in ("Gray", "None", "", None):
            return get_brushes_for_values("Gray", np.zeros_like(centers))

        brushes = []
        span = vmax - vmin
        active_mask = (centers >= vmin) & (centers <= vmax)
        if span > 0.0 and np.any(active_mask):
            normalized = (centers[active_mask] - vmin) / span
            colored_brushes = get_brushes_for_values(self.current_colormap_name, normalized)
            color_iter = iter(colored_brushes)
            for is_active in active_mask:
                if is_active:
                    brushes.append(next(color_iter))
                else:
                    brushes.append(pg.mkBrush(230, 230, 230, 140))
        else:
            brushes = [pg.mkBrush(230, 230, 230, 140) for _ in centers]
        return brushes

    def _apply_histogram_coloring(self, vmin: float, vmax: float):
        """Recolor histogram bars to match the current display range."""
        if self.hist_bars is None:
            return
        self.hist_bars.setOpts(brushes=self._build_histogram_brushes(vmin, vmax))

    def _get_visible_x_range(self):
        """Return the current visible histogram x-range, if available."""
        try:
            view_box = self.hist_widget.getViewBox()
        except AttributeError:
            return None

        if view_box is None or not hasattr(view_box, "viewRange"):
            return None

        try:
            (x0, x1), _ = view_box.viewRange()
        except Exception:
            return None
        return float(x0), float(x1)
