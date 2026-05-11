"""Custom ViewBox for histogram interactions."""

from __future__ import annotations

import pyqtgraph as pg


class HistogramViewBox(pg.ViewBox):
    """ViewBox with wheel-based horizontal zoom for histogram plots.

    The histogram uses draggable threshold lines, so the background view box
    keeps panning disabled and only supports zooming the visible x-range with
    the mouse wheel.
    """

    ZOOM_IN_FACTOR = 0.8
    MIN_X_RANGE = 1e-9

    def __init__(self, *args, **kwargs):
        """Initialize histogram view box with panning disabled."""
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self._data_x_min = None
        self._data_x_max = None

    def set_data_bounds(self, x_min: float, x_max: float):
        """Set full horizontal data bounds used for clamping pan/zoom."""
        self._data_x_min = float(x_min)
        self._data_x_max = float(x_max)

    def _clamp_x_range(self, x0: float, x1: float):
        """Clamp an x-range to the stored data bounds."""
        if self._data_x_min is None or self._data_x_max is None:
            return x0, x1

        full_min = self._data_x_min
        full_max = self._data_x_max
        full_width = max(full_max - full_min, self.MIN_X_RANGE)
        width = max(x1 - x0, self.MIN_X_RANGE)

        if width >= full_width:
            return full_min, full_max

        if x0 < full_min:
            x1 += full_min - x0
            x0 = full_min
        if x1 > full_max:
            x0 -= x1 - full_max
            x1 = full_max
        return x0, x1

    def wheelEvent(self, ev, axis=None):
        """Zoom the visible x-range around the cursor position."""
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if delta == 0:
            ev.ignore()
            return

        (x0, x1), _ = self.viewRange()
        width = max(x1 - x0, self.MIN_X_RANGE)
        factor = self.ZOOM_IN_FACTOR if delta > 0 else 1.0 / self.ZOOM_IN_FACTOR
        new_width = max(width * factor, self.MIN_X_RANGE)

        if hasattr(ev, "scenePos"):
            anchor_x = self.mapSceneToView(ev.scenePos()).x()
        else:
            anchor_x = 0.5 * (x0 + x1)

        relative = 0.5 if width <= self.MIN_X_RANGE else (anchor_x - x0) / width
        new_x0 = anchor_x - relative * new_width
        new_x1 = new_x0 + new_width
        new_x0, new_x1 = self._clamp_x_range(new_x0, new_x1)

        self.setXRange(new_x0, new_x1, padding=0.0)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        """Pan the histogram horizontally when dragging the background."""
        if ev.button() not in (pg.QtCore.Qt.LeftButton, pg.QtCore.Qt.MiddleButton):
            ev.ignore()
            return

        ev.accept()
        current_x = self.mapSceneToView(ev.scenePos()).x()
        last_x = self.mapSceneToView(ev.lastScenePos()).x()
        dx = current_x - last_x

        (x0, x1), _ = self.viewRange()
        new_x0, new_x1 = self._clamp_x_range(x0 - dx, x1 - dx)
        self.setXRange(new_x0, new_x1, padding=0.0)
