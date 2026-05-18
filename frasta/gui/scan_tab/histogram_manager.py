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
    THRESHOLD_LINE_WIDTH = 3
    THRESHOLD_HOVER_WIDTH = 5
    
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
        self._data_min = None
        self._data_max = None
        self.hide_below_range = True
        self.hide_above_range = True
        self.colormap_curve_strength = 0.0
    
    def update_histogram(
        self,
        grid: np.ndarray,
        was_data_negated: bool = False,
        colormap_name: str = "Gray",
        colormap_curve_strength: float = 0.0,
    ):
        """Update histogram display with new data.
        
        Args:
            grid (np.ndarray): Grid data to display
            was_data_negated (bool): Whether data was recently inverted
            colormap_name (str): Active display colormap for matching histogram fill
            colormap_curve_strength (float): Manual endpoint-stretch strength
        """
        logger.debug(f"update_histogram called: grid is None? {grid is None}")
        if grid is None:
            logger.warning("update_histogram: grid is None!")
            return
        self.current_colormap_name = colormap_name
        self.colormap_curve_strength = max(0.0, float(colormap_curve_strength))
        
        data = grid[~np.isnan(grid)]
        if data.size == 0:
            logger.warning("update_histogram: no valid data (all NaN)")
            self.hist_widget.clear()
            return

        # Get data range
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        self._data_min = vmin
        self._data_max = vmax

        # Remember old positions (if they exist)
        old_min_line = self.hist_min_line
        old_max_line = self.hist_max_line
        old_x_range = self._get_visible_x_range() if self.hist_bars is not None else None

        if was_data_negated:
            # Flip threshold positions for inverted data
            min_val = self._clamp_threshold_value(-old_max_line.value()) if old_max_line else vmin
            max_val = self._clamp_threshold_value(-old_min_line.value()) if old_min_line else vmax
        else:
            # Try to preserve old line positions, but only if they make sense for new data
            if old_min_line is not None and old_max_line is not None:
                old_min = old_min_line.value()
                old_max = old_max_line.value()
                # Check if old values are reasonable for new data range
                if vmin <= old_min <= vmax and vmin <= old_max <= vmax:
                    min_val = self._clamp_threshold_value(old_min)
                    max_val = self._clamp_threshold_value(old_max)
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
            pen=pg.mkPen(40, 140, 255, width=self.THRESHOLD_LINE_WIDTH), 
            hoverPen=pg.mkPen(255, 220, 0, width=self.THRESHOLD_HOVER_WIDTH)
        )
        self.hist_max_line = ResponsiveInfiniteLine(
            update_callback=self._on_threshold_changed, 
            angle=90, movable=True, 
            pen=pg.mkPen(255, 90, 90, width=self.THRESHOLD_LINE_WIDTH), 
            hoverPen=pg.mkPen(255, 220, 0, width=self.THRESHOLD_HOVER_WIDTH)
        )
        self.hist_min_line.setBounds((vmin, vmax))
        self.hist_max_line.setBounds((vmin, vmax))
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
        vmin = self._clamp_threshold_value(min(self.hist_min_line.value(), self.hist_max_line.value()))
        vmax = self._clamp_threshold_value(max(self.hist_min_line.value(), self.hist_max_line.value()))
        self._set_line_positions(vmin, vmax)
        self._apply_histogram_coloring(vmin, vmax)
        self.update_callback(vmin, vmax)
    
    def get_threshold_range(self) -> tuple[float, float]:
        """Get current threshold range.
        
        Returns:
            tuple: (vmin, vmax) threshold values
        """
        if self.hist_min_line is not None and self.hist_max_line is not None:
            vmin = self._clamp_threshold_value(min(self.hist_min_line.value(), self.hist_max_line.value()))
            vmax = self._clamp_threshold_value(max(self.hist_min_line.value(), self.hist_max_line.value()))
            return vmin, vmax
        return None, None
    
    def set_threshold_values(self, vmin: float, vmax: float):
        """Set threshold line values.
        
        Args:
            vmin (float): Minimum threshold
            vmax (float): Maximum threshold
        """
        if self.hist_min_line is not None and self.hist_max_line is not None:
            vmin = self._clamp_threshold_value(vmin)
            vmax = self._clamp_threshold_value(vmax)
            vmin, vmax = min(vmin, vmax), max(vmin, vmax)
            self._set_line_positions(vmin, vmax)
            self._apply_histogram_coloring(vmin, vmax)

    def set_out_of_range_visibility(self, hide_below_range: bool, hide_above_range: bool):
        """Set whether values below/above the selected range should be hidden."""
        self.hide_below_range = bool(hide_below_range)
        self.hide_above_range = bool(hide_above_range)
        vmin, vmax = self.get_threshold_range()
        if vmin is not None and vmax is not None:
            self._apply_histogram_coloring(vmin, vmax)

    def get_data_range(self) -> tuple[float, float]:
        """Return the current histogram data range."""
        return self._data_min, self._data_max

    def _clamp_threshold_value(self, value: float) -> float:
        """Clamp a threshold value to the current histogram data range."""
        if self._data_min is None or self._data_max is None:
            return float(value)
        return float(np.clip(value, self._data_min, self._data_max))

    def _set_line_positions(self, vmin: float, vmax: float):
        """Update threshold line positions without changing their order."""
        self._updating_histogram = True
        try:
            self.hist_min_line.setValue(vmin)
            self.hist_max_line.setValue(vmax)
        finally:
            self._updating_histogram = False

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
            colored_brushes = get_brushes_for_values(
                self.current_colormap_name,
                normalized,
                curve_strength=self.colormap_curve_strength,
            )
            color_iter = iter(colored_brushes)
            for center, is_active in zip(centers, active_mask):
                if is_active:
                    brushes.append(next(color_iter))
                elif center < vmin and self.hide_below_range:
                    brushes.append(pg.mkBrush(230, 230, 230, 140))
                elif center > vmax and self.hide_above_range:
                    brushes.append(pg.mkBrush(230, 230, 230, 140))
                else:
                    brushes.append(pg.mkBrush(110, 110, 110, 180))
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
