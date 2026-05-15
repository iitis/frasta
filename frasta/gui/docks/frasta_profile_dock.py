"""Dockable cross-sectional profile panel for FRASTA analysis.

Adapted from d:/praca/pyDpVision/plugins/frasta/frastaProfileDock.py.
Unchanged in logic, only the class name follows frasta-toolbox conventions
and the dpVision import is removed.

Signals
-------
profilePointSelected : pyqtSignal(int)
    Index along the profile array of the Ctrl-clicked point.
"""

from __future__ import annotations

import json
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from math import atan, degrees
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from sklearn.linear_model import LinearRegression

import logging
logger = logging.getLogger(__name__)


class FrastaProfileDock(QtWidgets.QDockWidget):
    """Dockable panel: cross-sectional profile plot.

    Displays two height profiles (reference and adjusted) along the line
    chosen in FrastaBinaryDock, together with the difference curve D and a
    horizontal separation line.
    """

    profilePointSelected = QtCore.pyqtSignal(int)
    cursorMoved           = QtCore.pyqtSignal(int)  # profile index under mouse

    def __init__(self, parent=None):
        super().__init__("FRASTA – Profile", parent)
        self.setObjectName("FrastaProfileDock")

        central = QtWidgets.QWidget()
        self.setWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._plot = pg.PlotWidget()
        layout.addWidget(self._plot)

        win_row = QtWidgets.QHBoxLayout()
        win_row.addWidget(QtWidgets.QLabel("Window size [µm]:"))
        self._spinbox_window = QtWidgets.QDoubleSpinBox()
        self._spinbox_window.setRange(1.0, 5000.0)
        self._spinbox_window.setValue(500.0)
        self._spinbox_window.setSingleStep(1.0)
        self._spinbox_window.setDecimals(0)
        win_row.addWidget(self._spinbox_window)
        self._checkbox_snap = QtWidgets.QCheckBox("Snap to plot")
        self._checkbox_snap.setChecked(True)
        win_row.addWidget(self._checkbox_snap)
        self._btn_roughness = QtWidgets.QPushButton("Roughness\u2026")
        self._btn_roughness.setToolTip("Show roughness parameters Ra, Rq, Rz for both profiles")
        self._btn_roughness.clicked.connect(self._show_roughness_summary)
        win_row.addWidget(self._btn_roughness)
        self._btn_export = QtWidgets.QPushButton("Export\u2026")
        self._btn_export.setToolTip("Export profile data to NPZ file")
        self._btn_export.clicked.connect(self._export_profile_npz)
        win_row.addWidget(self._btn_export)
        win_row.addStretch(1)
        layout.addLayout(win_row)

        # Curve visibility toggles (rebuilt dynamically in set_profiles)
        self._cb_container = QtWidgets.QWidget()
        self._cb_layout = QtWidgets.QHBoxLayout(self._cb_container)
        self._cb_layout.setContentsMargins(0, 0, 0, 0)
        self._cb_layout.setSpacing(12)
        layout.addWidget(self._cb_container)

        # Internal state
        self._separation_line: pg.InfiniteLine | None = None
        self._positions: np.ndarray | None = None
        self._ref_profile: np.ndarray | None = None
        self._adj_profile: np.ndarray | None = None
        self._dist_profile: np.ndarray | None = None
        self._fit_lines: list = []
        self._cursor_lines: list = []
        self._annotations: list = []
        # name → (plot item, hex color string)
        self._curve_items: dict[str, tuple] = {}
        self._mouse_updating: bool = False  # re-entrancy guard

        # Metadata for export (set by FrastaController via set_profile_metadata)
        self._export_dx: float = 1.0
        self._export_dy: float = 1.0
        self._export_separation: float = 0.0
        self._export_endpoints: tuple | None = None  # (c0, r0, c1, r1) pixel coords

        self.visibilityChanged.connect(self._on_visibility_changed)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self._plot.scene().sigMouseClicked.connect(self._on_plot_click)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_profiles(
        self,
        positions: np.ndarray,
        profiles: list,
        dist: np.ndarray,
        separation: float = 0.0,
    ) -> None:
        """Render profile curves.

        Parameters
        ----------
        positions:
            1-D array of distances along the profile line (µm).
        profiles:
            List of (name, array, pen) tuples – typically ref + adj.
        dist:
            Difference curve D = ref - adj (µm).
        separation:
            Current separation value; drawn as a horizontal dashed line.
        """
        self._plot.clear()
        self._plot.setTitle("")
        self._curve_items.clear()

        for name, prof, pen in profiles:
            item = self._plot.plot(positions, prof, pen=pen, name=name)
            color = pen.color().name() if hasattr(pen, "color") else "#ffffff"
            self._curve_items[name] = (item, color)

        self._draw_separation_line(separation)
        self._curve_items["Sep."] = (self._separation_line, "#cc4444")

        d_pen = pg.mkPen("r", width=2)
        d_item = self._plot.plot(positions, dist, pen=d_pen, name="D")
        self._curve_items["D"] = (d_item, d_pen.color().name())

        self._positions = positions
        self._ref_profile = profiles[0][1] if profiles else None
        self._adj_profile = profiles[1][1] if len(profiles) > 1 else None
        self._dist_profile = dist

        self._rebuild_checkboxes()

    def draw_separation_line(self, separation: float) -> None:
        """Update (or draw) the horizontal separation marker."""
        self._draw_separation_line(separation)

    def _rebuild_checkboxes(self) -> None:
        """Clear and recreate visibility checkboxes for all current curves."""
        # Preserve current visibility state before destroying widgets
        saved: dict[str, bool] = {}
        for i in range(self._cb_layout.count()):
            w = self._cb_layout.itemAt(i).widget()
            if isinstance(w, QtWidgets.QCheckBox):
                saved[w.text()] = w.isChecked()

        while self._cb_layout.count():
            child = self._cb_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for name, (item, color) in self._curve_items.items():
            cb = QtWidgets.QCheckBox(name)
            visible = saved.get(name, True)
            cb.setChecked(visible)
            item.setVisible(visible)
            cb.setStyleSheet(f"color: {color}; font-weight: bold;")
            cb.toggled.connect(lambda checked, i=item: i.setVisible(checked))
            self._cb_layout.addWidget(cb)

        self._cb_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _draw_separation_line(self, separation: float) -> None:
        if self._separation_line is not None:
            self._plot.removeItem(self._separation_line)
        self._separation_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#800", width=1, style=QtCore.Qt.DashLine),
        )
        self._plot.addItem(self._separation_line, ignoreBounds=True)
        self._separation_line.setPos(separation)

    def _clear_cursor_and_annotations(self) -> None:
        for item in self._cursor_lines + self._annotations:
            self._plot.removeItem(item)
        self._cursor_lines.clear()
        self._annotations.clear()

    def _clear_fit_lines(self) -> None:
        for item in self._fit_lines:
            self._plot.removeItem(item)
        self._fit_lines.clear()

    def _draw_cursor_line(self, pos: float, angle: int = 90, color: str = "r") -> None:
        ln = pg.InfiniteLine(
            angle=angle,
            pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
        )
        self._plot.addItem(ln, ignoreBounds=True)
        ln.setPos(pos)
        self._cursor_lines.append(ln)

    @staticmethod
    def _fit_profile_segment(x_um: np.ndarray, y_um: np.ndarray):
        X = x_um.reshape(-1, 1)
        Y = y_um.reshape(-1, 1)
        reg = LinearRegression().fit(X, Y)
        slope = float(reg.coef_[0][0])
        angle = degrees(atan(slope))
        return slope, angle, reg

    def _draw_fit_line(
        self,
        x_pos_um: float,
        slope: float,
        reg,
        idx: int,
        window_um: float,
        color: str = "y",
    ) -> None:
        half = window_um / 2.0
        x0, x1 = x_pos_um - half, x_pos_um + half
        if self._checkbox_snap.isChecked() and self._ref_profile is not None:
            y_at_cursor = float(self._ref_profile[idx])
            b = y_at_cursor - slope * x_pos_um
        else:
            b = float(reg.intercept_[0])
        y0 = slope * x0 + b
        y1 = slope * x1 + b
        line = pg.PlotDataItem([x0, x1], [y0, y1], pen=pg.mkPen(color, width=2))
        self._plot.addItem(line, ignoreBounds=True)
        self._fit_lines.append(line)

    def _draw_annotations_and_fit_lines(self, x_pos: float, idx: int) -> None:
        self._clear_fit_lines()
        window_um = self._spinbox_window.value()
        if self._positions is None or len(self._positions) < 2:
            return
        step_um = self._positions[1] - self._positions[0]
        window_size = max(1, int(round(window_um / step_um)))
        start = max(0, idx - window_size)
        end = min(len(self._positions), idx + window_size + 1)

        vb = self._plot.getPlotItem().vb
        x_min, x_max = vb.viewRange()[0]
        y_min, y_max = vb.viewRange()[1]
        offset_y = 0.05 * (y_max - y_min)
        y_text = y_max - offset_y

        if self._ref_profile is not None and self._adj_profile is not None:
            sl_ref, ang_ref, reg_ref = self._fit_profile_segment(
                self._positions[start:end], self._ref_profile[start:end]
            )
            sl_adj, ang_adj, reg_adj = self._fit_profile_segment(
                self._positions[start:end], self._adj_profile[start:end]
            )
            val_ref = float(self._ref_profile[idx])
            val_adj = float(self._adj_profile[idx])
            dh = val_ref - val_adj
            dtheta = ang_adj - ang_ref

            self._draw_fit_line(x_pos, sl_ref, reg_ref, idx, window_um, color="g")
            self._draw_fit_line(x_pos, sl_adj, reg_adj, idx, window_um, color="b")

            for text_str, color in (
                (f"Ref: {val_ref:.1f} µm,  {ang_ref:.1f}°", "g"),
                (f"Adj: {val_adj:.1f} µm,  {ang_adj:.1f}°", "b"),
                (f"Δh: {dh:.1f} µm   Δθ: {dtheta:.1f}°", "y"),
            ):
                t = pg.TextItem(text_str, color=color, anchor=(0, 1))
                t.setPos(x_min + 0.02 * (x_max - x_min), y_text)
                self._plot.addItem(t, ignoreBounds=True)
                self._annotations.append(t)
                y_text -= offset_y

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _on_mouse_move(self, pos) -> None:
        if self._mouse_updating:
            return
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        self._mouse_updating = True
        try:
            mp = self._plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mp.x()
            y_pos = mp.y()
            self._clear_cursor_and_annotations()
            if self._positions is not None and len(self._positions) > 0:
                if self._positions[0] <= x_pos <= self._positions[-1]:
                    self._draw_cursor_line(x_pos, angle=90, color="r")
                    self._draw_cursor_line(y_pos, angle=0, color="b")
                    idx = int(np.argmin(np.abs(self._positions - x_pos)))
                    self._draw_annotations_and_fit_lines(x_pos, idx)
                    self.cursorMoved.emit(idx)
        finally:
            self._mouse_updating = False

    def _on_plot_click(self, event) -> None:
        if event.modifiers() != QtCore.Qt.ControlModifier:
            return
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        mp = self._plot.plotItem.vb.mapSceneToView(pos)
        x_pos = mp.x()
        if self._positions is None or len(self._positions) == 0:
            return
        if not (self._positions[0] <= x_pos <= self._positions[-1]):
            return
        idx = int(np.argmin(np.abs(self._positions - x_pos)))
        self.profilePointSelected.emit(idx)

    # ------------------------------------------------------------------
    # Visibility guard
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(bool)
    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._positions is None:
            self._plot.setTitle(
                "No data — use Tools \u2192 FRASTA panels to load a scan pair.",
                color="#888",
            )

    # ------------------------------------------------------------------
    # Roughness summary
    # ------------------------------------------------------------------

    def _show_roughness_summary(self) -> None:
        if self._ref_profile is None or self._adj_profile is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No profile data available.")
            return
        try:
            from ...processing import profile_roughness_parameters
            ref_m = profile_roughness_parameters(self._ref_profile)
            adj_m = profile_roughness_parameters(self._adj_profile)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Roughness error", str(exc))
            return
        lines = [
            "Profile roughness parameters",
            "",
            "Reference profile:",
        ]
        for name in ("Ra", "Rq", "Rz"):
            lines.append(f"  {name}: {ref_m[name]:.6g} \u00b5m")
        lines.extend(["", "Adjusted profile:"])
        for name in ("Ra", "Rq", "Rz"):
            lines.append(f"  {name}: {adj_m[name]:.6g} \u00b5m")
        QtWidgets.QMessageBox.information(self, "Profile roughness", "\n".join(lines))

    # ------------------------------------------------------------------
    # Profile export
    # ------------------------------------------------------------------

    def set_profile_metadata(
        self,
        dx: float,
        dy: float,
        separation: float,
        endpoints: tuple,
    ) -> None:
        """Store spatial metadata used when exporting profile data.

        Parameters
        ----------
        dx, dy:
            Pixel size in µm (x and y).
        separation:
            Current separation value (µm).
        endpoints:
            (c0, r0, c1, r1) pixel coordinates of the profile line endpoints.
        """
        self._export_dx = float(dx)
        self._export_dy = float(dy)
        self._export_separation = float(separation)
        self._export_endpoints = endpoints

    def _export_profile_npz(self) -> None:
        if self._positions is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No profile data available.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export profile data", "", "NumPy archive (*.npz)"
        )
        if not fname:
            return
        if not fname.endswith(".npz"):
            fname += ".npz"
        arrays = dict(positions_um=self._positions)
        if self._ref_profile is not None:
            arrays["ref_profile_um"] = self._ref_profile
        if self._adj_profile is not None:
            arrays["adj_profile_um"] = self._adj_profile
        if self._dist_profile is not None:
            arrays["diff_profile_um"] = self._dist_profile
        np.savez_compressed(fname, **arrays)

        # JSON sidecar with metadata
        meta = {
            "metadata": {
                "frasta_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "description": "Cross-sectional profile export",
            },
            "spatial": {
                "pixel_dx_um": self._export_dx,
                "pixel_dy_um": self._export_dy,
            },
            "settings": {
                "separation_um": self._export_separation,
            },
            "profile_line": {
                "n_points": int(len(self._positions)),
                "length_um": float(self._positions[-1] - self._positions[0]) if len(self._positions) > 1 else 0.0,
            },
        }
        if self._export_endpoints is not None:
            c0, r0, c1, r1 = self._export_endpoints
            meta["profile_line"]["endpoints_px"] = {
                "start": {"col": int(c0), "row": int(r0)},
                "end": {"col": int(c1), "row": int(r1)},
            }
        json_fname = fname[:-4] + "_meta.json" if fname.endswith(".npz") else fname + "_meta.json"
        with open(json_fname, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)

        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{fname}\n{json_fname}")

    # ------------------------------------------------------------------
    # Session restore helper  (used by frasta_session.load_session)
    # ------------------------------------------------------------------

    def restore_settings(
        self,
        window_size_um: float = 500.0,
        snap_to_plot: bool = True,
        curve_visibility: dict | None = None,
    ) -> None:
        """Restore profile-dock UI settings after a session load."""
        self._spinbox_window.setValue(window_size_um)
        self._checkbox_snap.setChecked(snap_to_plot)

        if curve_visibility:
            # Apply to plot items
            for name, (item, _color) in self._curve_items.items():
                if name in curve_visibility:
                    item.setVisible(curve_visibility[name])
            # Sync checkboxes
            for i in range(self._cb_layout.count()):
                w = self._cb_layout.itemAt(i).widget()
                if isinstance(w, QtWidgets.QCheckBox) and w.text() in curve_visibility:
                    w.blockSignals(True)
                    w.setChecked(curve_visibility[w.text()])
                    w.blockSignals(False)

