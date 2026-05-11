"""Dockable binary contact map panel for FRASTA analysis.

Adapted from d:/praca/pyDpVision/plugins/frasta/frastaBinaryDock.py.
The original used GridData64 objects from dpVision. This version operates
directly on NumPy arrays with separate dx/dy pixel-size scalars, matching
the Surface interface used in frasta-toolbox.

Signals
-------
profileLineChanged : pyqtSignal(tuple)
    Emitted when the interactive ROI line moves.
    Payload: (c0, r0, c1, r1) in NumPy (row, col) coordinates.
separationChanged : pyqtSignal(float)
    Emitted whenever the separation spinbox changes value.
quickMessage : pyqtSignal(str)
    Short status text suitable for a main-window status bar.
"""

from __future__ import annotations

import json
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sutherland-Cohen line clipper (unchanged from pyDpVision original)
# ---------------------------------------------------------------------------

def _cohen_sutherland_clip(x0, y0, x1, y1, w, h):
    """Clip a line segment to the rectangle [0, w-1] x [0, h-1]."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def out_code(x, y):
        c = INSIDE
        if x < 0:
            c |= LEFT
        elif x > w - 1:
            c |= RIGHT
        if y < 0:
            c |= BOTTOM
        elif y > h - 1:
            c |= TOP
        return c

    oc0 = out_code(x0, y0)
    oc1 = out_code(x1, y1)
    accept = False

    while True:
        if not (oc0 | oc1):
            accept = True
            break
        if oc0 & oc1:
            break
        out = oc0 or oc1
        if out & TOP:
            x = x0 + (x1 - x0) * ((h - 1 - y0) / (y1 - y0))
            y = h - 1
        elif out & BOTTOM:
            x = x0 + (x1 - x0) * ((-y0) / (y1 - y0))
            y = 0
        elif out & RIGHT:
            y = y0 + (y1 - y0) * ((w - 1 - x0) / (x1 - x0))
            x = w - 1
        else:
            y = y0 + (y1 - y0) * ((-x0) / (x1 - x0))
            x = 0
        if out == oc0:
            x0, y0 = x, y
            oc0 = out_code(x0, y0)
        else:
            x1, y1 = x, y
            oc1 = out_code(x1, y1)

    if accept:
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    return None


def _create_image_view() -> pg.ImageView:
    v = pg.ImageView()
    v.ui.histogram.hide()
    v.ui.roiBtn.hide()
    v.ui.menuBtn.hide()
    return v


# ---------------------------------------------------------------------------
# FrastaBinaryDock
# ---------------------------------------------------------------------------

class FrastaBinaryDock(QtWidgets.QDockWidget):
    """Dockable panel: interactive binary contact map.

    The panel shows the binary map (D > separation) and lets the user draw a
    profile line on the map. Changing separation emits ``separationChanged``
    so other panels (profile dock, 3-D viewer) stay in sync.
    """

    profileLineChanged = QtCore.pyqtSignal(tuple)   # (c0, r0, c1, r1)
    separationChanged  = QtCore.pyqtSignal(float)
    quickMessage       = QtCore.pyqtSignal(str)
    show3dRequested    = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("FRASTA – Binary contact map", parent)
        self.setObjectName("FrastaBinaryDock")

        # Data
        self._diff_map: np.ndarray | None = None   # D = A - B, shape (rows, cols)
        self._grid1:    np.ndarray | None = None   # reference height grid
        self._grid2:    np.ndarray | None = None   # adjusted height grid
        self._dx: float = 1.0
        self._dy: float = 1.0
        self.binary_contact: np.ndarray | None = None

        # ROI line state (numpy row/col)
        self._x1 = self._y1 = 0
        self._x2 = self._y2 = 1
        self._line_roi: pg.LineROI | None = None
        self._roi_visible: bool = True
        self._saved_point_markers: list = []
        self._live_cursor_marker = None  # temporary hover marker
        # Map view mode: 'binary' or 'diff'
        self._map_mode: str = 'binary'
        # Diff display levels
        self._diff_lo: float = 0.0
        self._diff_hi: float = 1.0
        self._hist_min_line: pg.InfiniteLine | None = None
        self._hist_max_line: pg.InfiniteLine | None = None
        self._diff_ctrl_updating: bool = False

        self._build_ui()
        self.visibilityChanged.connect(self._on_visibility_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._image_view = _create_image_view()
        self._image_view.setMinimumWidth(300)
        self._image_view.getView().sigRangeChanged.connect(self._on_range_changed)
        layout.addWidget(self._image_view, 1)

        # Map mode toggle
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Map view:"))
        self._combo_map_mode = QtWidgets.QComboBox()
        self._combo_map_mode.addItem("Binary (contact)", userData='binary')
        self._combo_map_mode.addItem("Diff D (colourmap)", userData='diff')
        self._combo_map_mode.currentIndexChanged.connect(self._on_map_mode_changed)
        mode_row.addWidget(self._combo_map_mode)
        mode_row.addWidget(QtWidgets.QLabel("Colors:"))
        self._combo_binary_cmap = QtWidgets.QComboBox()
        self._combo_binary_cmap.setFixedWidth(90)
        self._combo_binary_cmap.addItem("Gray",     userData='gray')
        self._combo_binary_cmap.addItem("Blue/White", userData='blue')
        self._combo_binary_cmap.currentIndexChanged.connect(self._on_binary_cmap_changed)
        mode_row.addWidget(self._combo_binary_cmap)
        mode_row.addStretch(1)

        # Diff colormap + histogram panel (hidden in binary mode)
        self._diff_panel = QtWidgets.QWidget()
        dp_layout = QtWidgets.QVBoxLayout(self._diff_panel)
        dp_layout.setContentsMargins(0, 0, 0, 0)
        dp_layout.setSpacing(2)

        self._hist_widget = pg.PlotWidget()
        self._hist_widget.setFixedHeight(60)
        self._hist_widget.setBackground((20, 20, 20))
        self._hist_widget.getAxis('left').hide()
        self._hist_widget.getAxis('bottom').setStyle(tickLength=3)
        self._hist_widget.setMouseEnabled(x=True, y=False)
        dp_layout.addWidget(self._hist_widget)

        cr_row = QtWidgets.QHBoxLayout()
        cr_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self._combo_cmap = QtWidgets.QComboBox()
        self._combo_cmap.setFixedWidth(100)
        for _cname in ("CET-R4", "inferno", "viridis", "plasma", "CET-CBC1", "CET-CBC2"):
            self._combo_cmap.addItem(_cname)
        self._combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)
        cr_row.addWidget(self._combo_cmap)
        cr_row.addWidget(QtWidgets.QLabel("Center:"))
        self._spinbox_diff_center = QtWidgets.QDoubleSpinBox()
        self._spinbox_diff_center.setRange(-1e6, 1e6)
        self._spinbox_diff_center.setDecimals(2)
        self._spinbox_diff_center.setSingleStep(1.0)
        self._spinbox_diff_center.setFixedWidth(80)
        self._spinbox_diff_center.valueChanged.connect(self._on_diff_center_changed)
        cr_row.addWidget(self._spinbox_diff_center)
        cr_row.addWidget(QtWidgets.QLabel("Range:"))
        self._spinbox_diff_range = QtWidgets.QDoubleSpinBox()
        self._spinbox_diff_range.setRange(0.001, 1e6)
        self._spinbox_diff_range.setDecimals(2)
        self._spinbox_diff_range.setSingleStep(1.0)
        self._spinbox_diff_range.setFixedWidth(80)
        self._spinbox_diff_range.valueChanged.connect(self._on_diff_range_changed)
        cr_row.addWidget(self._spinbox_diff_range)
        _btn_auto = QtWidgets.QPushButton("Auto")
        _btn_auto.setFixedWidth(42)
        _btn_auto.setToolTip("Reset to full data range")
        _btn_auto.clicked.connect(self._diff_auto_range)
        cr_row.addWidget(_btn_auto)
        cr_row.addStretch(1)
        dp_layout.addLayout(cr_row)

        self._diff_panel.setVisible(False)

        sep_row = QtWidgets.QHBoxLayout()
        sep_row.addWidget(QtWidgets.QLabel("Separation [µm]:"))
        self._spinbox = QtWidgets.QDoubleSpinBox()
        self._spinbox.setRange(-10000.0, 10000.0)
        self._spinbox.setDecimals(2)
        self._spinbox.setSingleStep(0.5)
        self._spinbox.setValue(0.0)
        self._spinbox.valueChanged.connect(self._on_separation_changed)
        sep_row.addWidget(self._spinbox)
        self._sep_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sep_slider.setRange(0, 10000)
        self._sep_slider.setValue(5000)
        self._sep_slider.setToolTip("Drag to change separation (coarse)")
        self._sep_slider.valueChanged.connect(self._on_slider_changed)
        sep_row.addWidget(self._sep_slider, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_3d = QtWidgets.QPushButton("3D view")
        self._btn_3d.setToolTip("Open 3D viewer for the current scan pair")
        self._btn_3d.clicked.connect(self.show3dRequested)
        btn_row.addWidget(self._btn_3d)
        self._btn_export = QtWidgets.QPushButton("Export\u2026")
        self._btn_export.setToolTip("Export binary map (NPZ) or statistics (CSV/JSON)")
        self._btn_export.clicked.connect(self._show_export_menu)
        btn_row.addWidget(self._btn_export)
        btn_row.addStretch(1)

        self._stats_label = QtWidgets.QLabel("")
        self._stats_label.setWordWrap(True)

        layout.addLayout(mode_row)
        layout.addWidget(self._diff_panel)
        layout.addLayout(sep_row)
        layout.addLayout(btn_row)
        layout.addWidget(self._stats_label)

    # ------------------------------------------------------------------
    # Separation property
    # ------------------------------------------------------------------

    @property
    def separation(self) -> float:
        return self._spinbox.value()

    @separation.setter
    def separation(self, value: float) -> None:
        self._spinbox.setValue(float(value))

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    def set_data(
        self,
        diff_map: np.ndarray,
        grid1: np.ndarray | None = None,
        grid2: np.ndarray | None = None,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> None:
        """Load a new difference map (and optionally source grids).

        Parameters
        ----------
        diff_map:
            D = A - B, shape (rows, cols), float, NaN for invalid pixels.
        grid1, grid2:
            Optional source height grids used for volume computation.
        dx, dy:
            Pixel size in µm.
        """
        h, w = diff_map.shape
        self._diff_map = diff_map.astype(float)
        self._grid1 = grid1.astype(float)[:h, :w] if grid1 is not None else None
        self._grid2 = grid2.astype(float)[:h, :w] if grid2 is not None else None
        self._dx = float(dx)
        self._dy = float(dy)

        self._x1, self._y1 = 0, 0
        self._x2, self._y2 = w - 1, h - 1

        self._update_spinbox_range()
        self._init_diff_state()
        self._redraw_roi()
        self._update_plot()

    # ------------------------------------------------------------------
    # Separation range auto-calibration
    # ------------------------------------------------------------------

    def _update_spinbox_range(self) -> None:
        """Adapt separation spinbox and slider range to the loaded data span."""
        spans = []
        for g in (self._grid1, self._grid2):
            if g is None:
                continue
            finite = g[np.isfinite(g)]
            if finite.size:
                spans.append(float(np.max(finite) - np.min(finite)))
        if not spans:
            return
        data_span = max(max(spans), 1.0)
        limit = max(5000.0, 2.0 * data_span)
        cur = self._spinbox.value()
        self._spinbox.blockSignals(True)
        self._spinbox.setRange(-limit, limit)
        self._spinbox.setValue(float(np.clip(cur, -limit, limit)))
        self._spinbox.blockSignals(False)
        self._sync_slider_from_spinbox()

    def _sync_slider_from_spinbox(self) -> None:
        """Move slider thumb to match spinbox without emitting signals."""
        lo = self._spinbox.minimum()
        hi = self._spinbox.maximum()
        if hi == lo:
            return
        frac = (self._spinbox.value() - lo) / (hi - lo)
        self._sep_slider.blockSignals(True)
        self._sep_slider.setValue(int(round(frac * 10000)))
        self._sep_slider.blockSignals(False)

    def _on_slider_changed(self, slider_val: int) -> None:
        """Convert slider position to spinbox value."""
        lo = self._spinbox.minimum()
        hi = self._spinbox.maximum()
        val = lo + (slider_val / 10000.0) * (hi - lo)
        self._spinbox.setValue(val)

    # ------------------------------------------------------------------
    # Coordinate helpers  (image-view ↔ numpy)
    # ------------------------------------------------------------------

    def _shape(self):
        return self._diff_map.shape  # (rows, cols)

    def _view_to_numpy(self, x_view: float, y_view: float, clip: bool = True):
        """Map pyqtgraph view coords to numpy (row, col)."""
        h, w = self._shape()
        col = int(round(x_view))
        row = h - 1 - int(round(y_view))
        if clip:
            row = int(np.clip(row, 0, h - 1))
            col = int(np.clip(col, 0, w - 1))
        return row, col

    def _numpy_to_view(self, row: int, col: int):
        h, _w = self._shape()
        return float(col), float(h - 1 - row)

    # ------------------------------------------------------------------
    # ROI line
    # ------------------------------------------------------------------

    def _get_roi_coords(self):
        if self._line_roi is None:
            return None, None, None, None
        handles = self._line_roi.getHandles()
        pt0 = self._line_roi.mapToParent(handles[0].pos())
        pt1 = self._line_roi.mapToParent(handles[1].pos())
        r0, c0 = self._view_to_numpy(pt0.x(), pt0.y(), clip=False)
        r1, c1 = self._view_to_numpy(pt1.x(), pt1.y(), clip=False)
        h, w = self._shape()
        clipped = _cohen_sutherland_clip(c0, r0, c1, r1, w, h)
        if clipped is None:
            return None, None, None, None
        c0c, r0c, c1c, r1c = clipped
        return r0c, c0c, r1c, c1c

    def _redraw_roi(self) -> None:
        if self._line_roi is not None:
            try:
                self._image_view.getView().removeItem(self._line_roi)
            except Exception:
                pass

        self._line_roi = pg.LineROI(
            [self._x1, self._y1],
            [self._x2, self._y2],
            pen=pg.mkPen("r", width=2),
            width=1,
        )
        self._line_roi.handles[2]["type"] = "center"
        self._line_roi.sigRegionChanged.connect(self._emit_profile_line)
        self._image_view.getView().addItem(self._line_roi)
        self._line_roi.setZValue(10)
        self._line_roi.setVisible(self._roi_visible)
        self._emit_profile_line()

    def _emit_profile_line(self) -> None:
        r0, c0, r1, c1 = self._get_roi_coords()
        if r0 is None:
            return
        self.profileLineChanged.emit((c0, r0, c1, r1))

    # ------------------------------------------------------------------
    # Contact map rendering
    # ------------------------------------------------------------------

    def _on_map_mode_changed(self, _idx: int) -> None:
        self._map_mode = self._combo_map_mode.currentData()
        self._diff_panel.setVisible(self._map_mode == 'diff')
        self._combo_binary_cmap.setVisible(self._map_mode == 'binary')
        self._update_plot()

    def _on_binary_cmap_changed(self, _idx: int) -> None:
        if self._map_mode == 'binary':
            self._update_plot()

    def _on_cmap_changed(self, _idx: int) -> None:
        if self._map_mode == 'diff':
            self._update_plot()

    def _on_diff_center_changed(self, _val: float) -> None:
        if self._diff_ctrl_updating:
            return
        half = self._spinbox_diff_range.value() / 2.0
        c = self._spinbox_diff_center.value()
        self._diff_lo = c - half
        self._diff_hi = c + half
        self._sync_hist_lines()
        if self._map_mode == 'diff':
            self._apply_diff_levels()

    def _on_diff_range_changed(self, _val: float) -> None:
        if self._diff_ctrl_updating:
            return
        half = self._spinbox_diff_range.value() / 2.0
        c = self._spinbox_diff_center.value()
        self._diff_lo = c - half
        self._diff_hi = c + half
        self._sync_hist_lines()
        if self._map_mode == 'diff':
            self._apply_diff_levels()

    def _on_hist_line_changed(self) -> None:
        if self._diff_ctrl_updating or self._hist_min_line is None or self._hist_max_line is None:
            return
        lo = self._hist_min_line.value()
        hi = self._hist_max_line.value()
        if lo > hi:
            lo, hi = hi, lo
        self._diff_lo = lo
        self._diff_hi = hi
        self._sync_spinboxes_from_lo_hi()
        if self._map_mode == 'diff':
            self._apply_diff_levels()

    def _sync_spinboxes_from_lo_hi(self) -> None:
        self._diff_ctrl_updating = True
        try:
            center = (self._diff_lo + self._diff_hi) / 2.0
            rng = max(0.001, self._diff_hi - self._diff_lo)
            self._spinbox_diff_center.setValue(center)
            self._spinbox_diff_range.setValue(rng)
        finally:
            self._diff_ctrl_updating = False

    def _sync_hist_lines(self) -> None:
        if self._hist_min_line is not None:
            self._hist_min_line.blockSignals(True)
            self._hist_min_line.setValue(self._diff_lo)
            self._hist_min_line.blockSignals(False)
        if self._hist_max_line is not None:
            self._hist_max_line.blockSignals(True)
            self._hist_max_line.setValue(self._diff_hi)
            self._hist_max_line.blockSignals(False)

    def _apply_diff_levels(self) -> None:
        lo, hi = self._diff_lo, self._diff_hi
        if hi == lo:
            hi = lo + 1.0
        self._image_view.setLevels(lo, hi)

    def _diff_auto_range(self) -> None:
        if self._diff_map is None:
            return
        valid = self._diff_map[np.isfinite(self._diff_map)]
        if valid.size == 0:
            return
        self._diff_lo = float(np.min(valid))
        self._diff_hi = float(np.max(valid))
        self._sync_spinboxes_from_lo_hi()
        self._sync_hist_lines()
        if self._map_mode == 'diff':
            self._apply_diff_levels()

    def _init_diff_state(self) -> None:
        """Set diff display range from data percentiles and rebuild histogram."""
        if self._diff_map is None:
            return
        valid = self._diff_map[np.isfinite(self._diff_map)]
        if valid.size == 0:
            return
        # Find center and range from the dense part of the histogram:
        # keep only bins with count >= median bin count (upper 50% by density),
        # then take their value range as the display window.
        n_bins = min(256, max(64, int(valid.size ** 0.4)))
        counts, edges = np.histogram(valid, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        threshold = np.median(counts)
        dense = centers[counts >= threshold]
        if dense.size >= 2:
            lo = float(dense[0])
            hi = float(dense[-1])
            margin = max((hi - lo) * 0.05, 1.0)
            lo -= margin
            hi += margin
        else:
            lo, hi = float(np.min(valid)), float(np.max(valid))
        self._diff_lo = lo
        self._diff_hi = hi
        self._sync_spinboxes_from_lo_hi()
        self._update_diff_histogram()

    def _update_diff_histogram(self) -> None:
        """Rebuild histogram bars and draggable threshold lines."""
        self._hist_widget.clear()
        self._hist_min_line = None
        self._hist_max_line = None
        if self._diff_map is None:
            return
        valid = self._diff_map[np.isfinite(self._diff_map)]
        if valid.size == 0:
            return
        n_bins = min(256, max(64, int(valid.size ** 0.4)))
        y, x = np.histogram(valid, bins=n_bins)
        centers = 0.5 * (x[:-1] + x[1:])
        widths = np.diff(x)
        bars = pg.BarGraphItem(
            x=centers, height=y, width=widths,
            brush=pg.mkBrush(100, 150, 220, 160),
            pen=pg.mkPen(None),
        )
        self._hist_widget.addItem(bars)
        lo_init = float(np.clip(self._diff_lo, x[0], x[-1]))
        hi_init = float(np.clip(self._diff_hi, x[0], x[-1]))
        self._hist_min_line = pg.InfiniteLine(
            pos=lo_init, angle=90, movable=True,
            pen=pg.mkPen('#5af', width=2),
            hoverPen=pg.mkPen('#ff0', width=2),
        )
        self._hist_max_line = pg.InfiniteLine(
            pos=hi_init, angle=90, movable=True,
            pen=pg.mkPen('#f55', width=2),
            hoverPen=pg.mkPen('#ff0', width=2),
        )
        self._hist_min_line.sigPositionChanged.connect(self._on_hist_line_changed)
        self._hist_max_line.sigPositionChanged.connect(self._on_hist_line_changed)
        self._hist_widget.addItem(self._hist_min_line)
        self._hist_widget.addItem(self._hist_max_line)
        self._hist_widget.setXRange(float(x[0]), float(x[-1]), padding=0.02)

    def _update_plot(self) -> None:
        if self._diff_map is None:
            return
        valid = np.isfinite(self._diff_map)
        binary = (self._diff_map > self.separation) & valid
        self.binary_contact = binary

        if self._map_mode == 'diff':
            arr = self._diff_map.T.copy()
            arr[~valid.T] = np.nan
            self._image_view.setImage(arr, autoRange=False, autoLevels=False)
            self._apply_diff_levels()
            cmap_name = self._combo_cmap.currentText()
            cmap = None
            for _source in ('colorcet', None):
                try:
                    cmap = pg.colormap.get(cmap_name, source=_source) if _source else pg.colormap.get(cmap_name)
                except Exception:
                    cmap = None
                if cmap is not None:
                    break
            if cmap is None:
                cmap = pg.colormap.get('inferno')
            self._image_view.setColorMap(cmap)
        else:
            arr = binary.astype(np.uint8).T
            self._image_view.setImage(arr, autoRange=False, autoLevels=False)
            self._image_view.setLevels(0, 1)
            bcmap_key = self._combo_binary_cmap.currentData()
            if bcmap_key == 'blue':
                # white=open (0), blue=contact (1)
                lut = np.zeros((256, 3), dtype=np.uint8)
                lut[:, 0] = np.linspace(255, 30, 256).astype(np.uint8)
                lut[:, 1] = np.linspace(255, 100, 256).astype(np.uint8)
                lut[:, 2] = np.linspace(255, 200, 256).astype(np.uint8)
                self._image_view.setColorMap(pg.ColorMap(pos=np.array([0.0, 1.0]),
                                                         color=np.array([[255,255,255,255],[30,100,200,255]], dtype=np.uint8)))
            else:
                self._image_view.setColorMap(pg.ColorMap(pos=np.array([0.0, 1.0]),
                                                         color=np.array([[0,0,0,255],[255,255,255,255]], dtype=np.uint8)))

        self._update_stats()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_stats(self) -> None:
        if self.binary_contact is None:
            return

        vb = self._image_view.getView()
        (x0, x1), (y0, y1) = vb.viewRange()

        r0, c0 = self._view_to_numpy(x0, y0)
        r1, c1 = self._view_to_numpy(x1, y1)
        r_min, r_max = sorted((r0, r1))
        c_min, c_max = sorted((c0, c1))
        h, w = self.binary_contact.shape
        r_min = int(np.clip(r_min, 0, h - 1))
        r_max = int(np.clip(r_max, 0, h - 1))
        c_min = int(np.clip(c_min, 0, w - 1))
        c_max = int(np.clip(c_max, 0, w - 1))

        pixel_area = self._dx * self._dy
        fragment = self.binary_contact[r_min:r_max + 1, c_min:c_max + 1]
        n_contact = int(np.count_nonzero(fragment))
        n_total = int(np.sum(np.isfinite(self._diff_map[r_min:r_max + 1, c_min:c_max + 1])))
        fraction = (n_contact / n_total * 100.0) if n_total > 0 else 0.0
        area_um2 = pixel_area * n_contact
        area_mm2 = area_um2 * 1e-6

        diff_fragment = self._diff_map[r_min:r_max + 1, c_min:c_max + 1] - self.separation
        diff_masked = np.where(fragment, diff_fragment, 0.0)
        vol_um3 = float(np.abs(np.sum(diff_masked)) * pixel_area)
        vol_mm3 = vol_um3 * 1e-9

        # Mean COD (mean aperture in open region) and D range
        diff_frag = self._diff_map[r_min:r_max + 1, c_min:c_max + 1]
        valid_frag = np.isfinite(diff_frag)
        finite_frag = diff_frag[valid_frag]
        open_mask = valid_frag & ~fragment.astype(bool)
        mean_cod_str = "n/a"
        if np.any(open_mask):
            mean_cod = float(np.mean(diff_frag[open_mask]))
            mean_cod_str = f"{mean_cod:.2f} µm"

        d_range_str = "n/a"
        if finite_frag.size:
            d_range_str = f"{float(np.min(finite_frag)):.2f} … {float(np.max(finite_frag)):.2f} µm"

        msg = (
            f"Contact: {n_contact} px  ({fraction:.1f}%)  "
            f"| area: {area_um2:.1f} µm²  ({area_mm2:.4f} mm²)  "
            f"| vol: {vol_um3:.1f} µm³  ({vol_mm3:.4f} mm³)  "
            f"| mean COD: {mean_cod_str}  | D range: {d_range_str}"
        )
        self._stats_label.setText(msg)
        self.quickMessage.emit(msg)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_separation_changed(self, value: float) -> None:
        self._sync_slider_from_spinbox()
        self.separationChanged.emit(value)
        self._update_plot()

    def _on_range_changed(self, _viewbox, _ranges) -> None:
        self._update_stats()

    @QtCore.pyqtSlot(int, int)
    def on_point_selected(self, row: int, col: int) -> None:
        """Persistently mark a point on the map (Ctrl+click in profile dock)."""
        x_img, y_img = self._numpy_to_view(row, col)
        marker = pg.ScatterPlotItem(
            [x_img], [y_img], size=12,
            pen=pg.mkPen("g", width=2),
            brush=pg.mkBrush(0, 255, 255, 120),
            symbol="+",
        )
        self._image_view.getView().addItem(marker)
        self._saved_point_markers.append(marker)

    def update_live_cursor(self, row: int, col: int) -> None:
        """Move the transient hover cursor on the map (follows mouse in profile dock)."""
        view = self._image_view.getView()
        if self._live_cursor_marker is not None:
            view.removeItem(self._live_cursor_marker)
        x_img, y_img = self._numpy_to_view(row, col)
        self._live_cursor_marker = pg.ScatterPlotItem(
            [x_img], [y_img], size=14,
            pen=pg.mkPen("m", width=2),
            brush=pg.mkBrush(255, 0, 255, 120),
            symbol="o",
        )
        view.addItem(self._live_cursor_marker)

    def clear_point_markers(self) -> None:
        for m in self._saved_point_markers:
            self._image_view.getView().removeItem(m)
        self._saved_point_markers.clear()

    def set_roi_visible(self, visible: bool) -> None:
        """Show or hide the profile ROI line on the map."""
        self._roi_visible = visible
        if self._line_roi is not None:
            self._line_roi.setVisible(visible)

    # ------------------------------------------------------------------
    # Visibility guard
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(bool)
    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._diff_map is None:
            self._stats_label.setText(
                "No data loaded — use Tools \u2192 FRASTA panels to load a scan pair."
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _show_export_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction("Export binary map (NPZ)\u2026", self._export_binary_npz)
        menu.addAction("Export statistics (CSV)\u2026", self._export_stats_csv)
        menu.addAction("Export statistics (JSON)\u2026", self._export_stats_json)
        menu.exec_(QtGui.QCursor.pos())

    def _export_binary_npz(self) -> None:
        if self.binary_contact is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No binary map available.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export binary map", "", "NumPy archive (*.npz)"
        )
        if not fname:
            return
        if not fname.endswith(".npz"):
            fname += ".npz"
        arrays = dict(
            binary_contact=self.binary_contact,
            diff_map=self._diff_map,
            separation=np.float64(self.separation),
            dx=np.float64(self._dx),
            dy=np.float64(self._dy),
        )
        if self._grid1 is not None:
            arrays["ref_grid"] = self._grid1
        if self._grid2 is not None:
            arrays["adj_grid"] = self._grid2
        np.savez_compressed(fname, **arrays)
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{fname}")

    def _export_stats_csv(self) -> None:
        if self.binary_contact is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No data available.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export statistics", "", "CSV files (*.csv)"
        )
        if not fname:
            return
        if not fname.endswith(".csv"):
            fname += ".csv"
        pixel_area = self._dx * self._dy
        valid = np.isfinite(self._diff_map)
        n_contact = int(np.count_nonzero(self.binary_contact))
        n_total = int(np.sum(valid))
        fraction = n_contact / n_total * 100.0 if n_total > 0 else 0.0
        area_um2 = pixel_area * n_contact
        diff_masked = np.where(self.binary_contact, self._diff_map - self.separation, 0.0)
        vol_um3 = float(np.abs(np.sum(diff_masked)) * pixel_area)
        h, w = self._diff_map.shape
        lines = [
            f"# frasta_version,1.0",
            f"# export_date,{datetime.now().isoformat()}",
            f"# scan_shape_rows,{h}",
            f"# scan_shape_cols,{w}",
            "parameter,value,unit",
            f"separation,{self.separation:.6f},\u00b5m",
            f"contact_pixels,{n_contact},px",
            f"total_valid_pixels,{n_total},px",
            f"contact_fraction,{fraction:.4f},%",
            f"contact_area,{area_um2:.4f},\u00b5m\u00b2",
            f"contact_area_mm2,{area_um2 * 1e-6:.8f},mm\u00b2",
            f"contact_volume,{vol_um3:.4f},\u00b5m\u00b3",
            f"contact_volume_mm3,{vol_um3 * 1e-9:.10f},mm\u00b3",
            f"pixel_size_dx,{self._dx:.6f},\u00b5m",
            f"pixel_size_dy,{self._dy:.6f},\u00b5m",
        ]
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{fname}")

    def _export_stats_json(self) -> None:
        if self.binary_contact is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No data available.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export statistics (JSON)", "", "JSON files (*.json)"
        )
        if not fname:
            return
        if not fname.endswith(".json"):
            fname += ".json"
        pixel_area = self._dx * self._dy
        valid = np.isfinite(self._diff_map)
        n_contact = int(np.count_nonzero(self.binary_contact))
        n_total = int(np.sum(valid))
        fraction = n_contact / n_total * 100.0 if n_total > 0 else 0.0
        area_um2 = pixel_area * n_contact
        diff_masked = np.where(self.binary_contact, self._diff_map - self.separation, 0.0)
        vol_um3 = float(np.abs(np.sum(diff_masked)) * pixel_area)
        open_mask = valid & ~self.binary_contact.astype(bool)
        mean_cod = float(np.mean(self._diff_map[open_mask])) if np.any(open_mask) else None
        finite_diff = self._diff_map[valid]
        h, w = self._diff_map.shape
        data = {
            "metadata": {
                "frasta_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "scan_shape": [h, w],
            },
            "separation_um": round(self.separation, 6),
            "n_valid_pixels": n_total,
            "n_contact_pixels": n_contact,
            "contact_fraction_pct": round(fraction, 4),
            "contact_area_um2": round(area_um2, 4),
            "contact_area_mm2": round(area_um2 * 1e-6, 8),
            "contact_volume_um3": round(vol_um3, 4),
            "contact_volume_mm3": round(vol_um3 * 1e-9, 10),
            "mean_cod_open_um": round(mean_cod, 4) if mean_cod is not None else None,
            "diff_min_um": round(float(np.min(finite_diff)), 6) if finite_diff.size else None,
            "diff_max_um": round(float(np.max(finite_diff)), 6) if finite_diff.size else None,
            "pixel_dx_um": self._dx,
            "pixel_dy_um": self._dy,
        }
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{fname}")

    # ------------------------------------------------------------------
    # Session restore helpers  (used by frasta_session.load_session)
    # ------------------------------------------------------------------

    def restore_roi(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Move the profile-line ROI to the given view coordinates and redraw."""
        self._x1, self._y1 = x1, y1
        self._x2, self._y2 = x2, y2
        self._redraw_roi()

    def restore_display_settings(
        self,
        map_mode: str = "binary",
        binary_cmap: str = "gray",
        diff_cmap: str = "CET-R4",
        diff_center: float = 0.0,
        diff_range: float = 1.0,
    ) -> None:
        """Restore colourmap / display settings without triggering side-effects."""
        # Map mode
        idx = self._combo_map_mode.findData(map_mode)
        if idx >= 0:
            self._combo_map_mode.blockSignals(True)
            self._combo_map_mode.setCurrentIndex(idx)
            self._combo_map_mode.blockSignals(False)
            self._map_mode = map_mode
            self._diff_panel.setVisible(map_mode == "diff")
            self._combo_binary_cmap.setVisible(map_mode == "binary")

        # Binary colormap
        idx_b = self._combo_binary_cmap.findData(binary_cmap)
        if idx_b >= 0:
            self._combo_binary_cmap.blockSignals(True)
            self._combo_binary_cmap.setCurrentIndex(idx_b)
            self._combo_binary_cmap.blockSignals(False)

        # Diff colormap
        idx_d = self._combo_cmap.findText(diff_cmap)
        if idx_d >= 0:
            self._combo_cmap.blockSignals(True)
            self._combo_cmap.setCurrentIndex(idx_d)
            self._combo_cmap.blockSignals(False)

        # Diff levels
        self._diff_ctrl_updating = True
        self._spinbox_diff_center.blockSignals(True)
        self._spinbox_diff_center.setValue(diff_center)
        self._spinbox_diff_center.blockSignals(False)
        self._spinbox_diff_range.blockSignals(True)
        self._spinbox_diff_range.setValue(diff_range)
        self._spinbox_diff_range.blockSignals(False)
        self._diff_lo = diff_center - diff_range / 2.0
        self._diff_hi = diff_center + diff_range / 2.0
        if self._hist_min_line is not None:
            self._sync_hist_lines()
        self._diff_ctrl_updating = False

