"""Dedicated crack-path analysis dialog for aligned fracture surfaces."""

from __future__ import annotations

import json

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...core import Surface
from ...processing.crack_path import analyze_crack_path, sweep_crack_path_thresholds

import logging
logger = logging.getLogger(__name__)


class CrackPathDialog(QtWidgets.QWidget):
    """Interactive crack-path tortuosity dialog linked to a surface pair.

    The dialog evaluates one aligned surface pair for a selected opening
    threshold and crack-path extraction method. It displays the binary open
    region with the extracted path overlay, the resulting tortuosity metrics,
    and a local-curvature plot.
    """

    def __init__(
        self,
        surface_a: Surface,
        surface_b: Surface,
        parent=None,
        title_a: str = "Surface A",
        title_b: str = "Surface B",
        initial_threshold: float | None = None,
    ) -> None:
        """Initialize the crack-path dialog for one aligned surface pair."""
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("Crack-path analysis")
        self.setMinimumSize(980, 700)

        self._surface_a = surface_a
        self._surface_b = surface_b
        self._title_a = title_a
        self._title_b = title_b
        self._dx = float(surface_a.dx)
        self._dy = float(surface_a.dy)

        self._diff = np.asarray(surface_a.height - surface_b.height, dtype=float)
        self._valid = np.isfinite(self._diff)
        valid_vals = self._diff[self._valid]
        if valid_vals.size > 0:
            self._diff_min = float(np.min(valid_vals))
            self._diff_max = float(np.max(valid_vals))
            default_threshold = float(np.median(valid_vals))
        else:
            self._diff_min = -1.0
            self._diff_max = 1.0
            default_threshold = 0.0

        self._threshold = default_threshold if initial_threshold is None else float(initial_threshold)
        self._updating_controls = False
        self._analysis_result: dict[str, object] | None = None
        self._threshold_sweep_result: dict[str, object] | None = None

        self._build_ui()
        self._connect_signals()
        self._refresh_analysis()

    def _build_ui(self) -> None:
        """Create the crack-path widgets and layouts."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Threshold s [µm]:"))
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        span = self._diff_max - self._diff_min
        step = max(0.001, span / 200.0)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setRange(self._diff_min - span, self._diff_max + span)
        self._threshold_spin.setSingleStep(step)
        self._threshold_spin.setValue(self._threshold)
        self._threshold_spin.setMaximumWidth(120)
        top_row.addWidget(self._threshold_spin)

        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setRange(0, 1000)
        self._threshold_slider.setValue(self._spin_to_slider(self._threshold))
        top_row.addWidget(self._threshold_slider, stretch=1)

        top_row.addWidget(QtWidgets.QLabel("Method:"))
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItem("First open pixel", userData="first_open_pixel")
        self._method_combo.addItem("Contour", userData="contour")
        top_row.addWidget(self._method_combo)

        top_row.addWidget(QtWidgets.QLabel("Resample [µm]:"))
        self._resample_spin = QtWidgets.QDoubleSpinBox()
        self._resample_spin.setDecimals(3)
        self._resample_spin.setRange(0.001, 1e6)
        self._resample_spin.setSingleStep(min(self._dx, self._dy))
        self._resample_spin.setValue(min(self._dx, self._dy))
        self._resample_spin.setMaximumWidth(90)
        top_row.addWidget(self._resample_spin)

        top_row.addWidget(QtWidgets.QLabel("Smooth win:"))
        self._smoothing_spin = QtWidgets.QSpinBox()
        self._smoothing_spin.setRange(1, 99)
        self._smoothing_spin.setSingleStep(2)
        self._smoothing_spin.setValue(5)
        self._smoothing_spin.setMaximumWidth(70)
        top_row.addWidget(self._smoothing_spin)

        top_row.addWidget(QtWidgets.QLabel("Propagation axis:"))
        self._propagation_axis_combo = QtWidgets.QComboBox()
        self._propagation_axis_combo.addItem("X", userData="x")
        self._propagation_axis_combo.addItem("Y", userData="y")
        top_row.addWidget(self._propagation_axis_combo)

        top_row.addWidget(QtWidgets.QLabel("Front side:"))
        self._front_side_combo = QtWidgets.QComboBox()
        self._front_side_combo.addItem("Min", userData="min")
        self._front_side_combo.addItem("Max", userData="max")
        top_row.addWidget(self._front_side_combo)

        root.addLayout(top_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        left_group = QtWidgets.QGroupBox(
            f"Open region and extracted path  ({self._title_a} vs {self._title_b})"
        )
        left_layout = QtWidgets.QVBoxLayout(left_group)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._open_view = pg.ImageView(view=pg.PlotItem(enableMenu=False))
        self._open_view.ui.histogram.hide()
        self._open_view.ui.roiBtn.hide()
        self._open_view.ui.menuBtn.hide()
        self._open_lut = np.zeros((256, 4), dtype=np.uint8)
        self._open_lut[0] = [45, 45, 45, 255]
        self._open_lut[1] = [240, 240, 240, 255]
        self._open_view.getImageItem().setLookupTable(self._open_lut)
        self._path_curve = pg.PlotDataItem(
            pen=pg.mkPen(255, 170, 40, width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush(255, 170, 40),
            symbolPen=None,
        )
        self._open_view.getView().addItem(self._path_curve)
        left_layout.addWidget(self._open_view)
        splitter.addWidget(left_group)

        right_group = QtWidgets.QGroupBox("Metrics")
        right_layout = QtWidgets.QVBoxLayout(right_group)
        right_layout.setContentsMargins(8, 6, 8, 6)
        right_layout.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        self._lbl_status = QtWidgets.QLabel("—")
        self._lbl_path_method = QtWidgets.QLabel("—")
        self._lbl_path_length = QtWidgets.QLabel("—")
        self._lbl_path_projection = QtWidgets.QLabel("—")
        self._lbl_tortuosity = QtWidgets.QLabel("—")
        self._lbl_curvature = QtWidgets.QLabel("—")
        form.addRow("Status:", self._lbl_status)
        form.addRow("Method:", self._lbl_path_method)
        form.addRow("Effective length:", self._lbl_path_length)
        form.addRow("Projected length:", self._lbl_path_projection)
        form.addRow("Tortuosity:", self._lbl_tortuosity)
        form.addRow("Mean |curvature|:", self._lbl_curvature)
        right_layout.addLayout(form)

        self._curvature_plot = pg.PlotWidget(plotItem=pg.PlotItem(enableMenu=False))
        self._curvature_plot.setMinimumWidth(340)
        self._curvature_plot.setMinimumHeight(180)
        self._curvature_plot.setLabel("bottom", "Arc length", units="µm")
        self._curvature_plot.setLabel("left", "Curvature", units="1/µm")
        self._curvature_plot.showGrid(x=True, y=True, alpha=0.2)
        self._curvature_curve = self._curvature_plot.plot(
            pen=pg.mkPen(255, 170, 40, width=2)
        )
        right_layout.addWidget(self._curvature_plot, stretch=1)

        self._sweep_plot = pg.PlotWidget(plotItem=pg.PlotItem(enableMenu=False))
        self._sweep_plot.setMinimumWidth(340)
        self._sweep_plot.setMinimumHeight(180)
        self._sweep_plot.setLabel("bottom", "Threshold s", units="µm")
        self._sweep_plot.setLabel("left", "Tortuosity", units="")
        self._sweep_plot.showGrid(x=True, y=True, alpha=0.2)
        self._sweep_curve = self._sweep_plot.plot(
            pen=pg.mkPen(80, 180, 255, width=2),
            symbol="o",
            symbolSize=3,
            symbolBrush=pg.mkBrush(80, 180, 255),
            symbolPen=None,
        )
        self._current_threshold_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(255, 170, 40, width=1, style=QtCore.Qt.DashLine),
        )
        self._sweep_plot.addItem(self._current_threshold_line)
        right_layout.addWidget(self._sweep_plot, stretch=1)

        button_row = QtWidgets.QHBoxLayout()
        self._btn_export_npz = QtWidgets.QPushButton("Export path data (NPZ)…")
        self._btn_export_json = QtWidgets.QPushButton("Export metrics (JSON)…")
        self._btn_close = QtWidgets.QPushButton("Close")
        button_row.addWidget(self._btn_export_npz)
        button_row.addWidget(self._btn_export_json)
        button_row.addStretch(1)
        button_row.addWidget(self._btn_close)
        right_layout.addLayout(button_row)
        splitter.addWidget(right_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    def _connect_signals(self) -> None:
        """Wire widget events to analysis refresh handlers."""
        self._threshold_spin.valueChanged.connect(self._on_spin_changed)
        self._threshold_slider.valueChanged.connect(self._on_slider_changed)
        self._method_combo.currentIndexChanged.connect(self._refresh_analysis)
        self._method_combo.currentIndexChanged.connect(self._update_contour_controls)
        self._resample_spin.valueChanged.connect(self._refresh_analysis)
        self._smoothing_spin.valueChanged.connect(self._refresh_analysis)
        self._propagation_axis_combo.currentIndexChanged.connect(self._refresh_analysis)
        self._front_side_combo.currentIndexChanged.connect(self._refresh_analysis)
        self._btn_export_npz.clicked.connect(self._export_npz)
        self._btn_export_json.clicked.connect(self._export_json)
        self._btn_close.clicked.connect(self.close)
        self._update_contour_controls()

    def set_threshold(self, value: float) -> None:
        """Update the threshold from an external linked dialog."""
        if self._updating_controls and np.isclose(self._threshold, value):
            return
        self._threshold = float(value)
        self._updating_controls = True
        self._threshold_spin.setValue(self._threshold)
        self._threshold_slider.setValue(self._spin_to_slider(self._threshold))
        self._updating_controls = False
        self._refresh_analysis()

    def _spin_to_slider(self, value: float) -> int:
        """Map a physical threshold value to a slider integer [0, 1000]."""
        span = self._diff_max - self._diff_min
        if span == 0:
            return 500
        frac = (value - self._diff_min) / span
        return int(round(np.clip(frac, 0.0, 1.0) * 1000))

    def _slider_to_spin(self, pos: int) -> float:
        """Map a slider integer [0, 1000] to a physical threshold value."""
        span = self._diff_max - self._diff_min
        return self._diff_min + (pos / 1000.0) * span

    def _on_spin_changed(self, value: float) -> None:
        """Handle threshold edits from the spin box."""
        if self._updating_controls:
            return
        self._threshold = float(value)
        self._updating_controls = True
        self._threshold_slider.setValue(self._spin_to_slider(self._threshold))
        self._updating_controls = False
        self._refresh_analysis()

    def _on_slider_changed(self, pos: int) -> None:
        """Handle threshold edits from the coarse slider."""
        if self._updating_controls:
            return
        self._threshold = self._slider_to_spin(pos)
        self._updating_controls = True
        self._threshold_spin.setValue(self._threshold)
        self._updating_controls = False
        self._refresh_analysis()

    def _refresh_analysis(self) -> None:
        """Recompute the selected crack-path analysis and refresh the plots."""
        try:
            result = analyze_crack_path(
                self._surface_a,
                self._surface_b,
                dx=self._dx,
                dy=self._dy,
                separation=self._threshold,
                propagation_axis=self._current_propagation_axis(),
                front_side=self._current_front_side(),
                method=self._current_method(),
                contour_resample_step=float(self._resample_spin.value()),
                contour_smoothing_window=int(self._smoothing_spin.value()),
            )
        except ValueError as exc:
            self._analysis_result = None
            self._threshold_sweep_result = None
            self._path_curve.setData([], [])
            self._curvature_curve.setData([], [])
            self._sweep_curve.setData([], [])
            self._open_view.setImage(np.zeros(self._diff.T.shape, dtype=np.uint8), autoLevels=False, levels=(0, 1))
            self._lbl_status.setText(str(exc))
            self._lbl_path_method.setText(self._current_method())
            self._lbl_path_length.setText("—")
            self._lbl_path_projection.setText("—")
            self._lbl_tortuosity.setText("—")
            self._lbl_curvature.setText("—")
            self._current_threshold_line.setPos(self._threshold)
            return

        self._analysis_result = result
        open_image = result["open_mask"].T.astype(np.uint8)
        self._open_view.setImage(open_image, autoLevels=False, levels=(0, 1))

        path_points = np.asarray(result["path_points"], dtype=float)
        self._path_curve.setData(path_points[:, 0] / self._dx, path_points[:, 1] / self._dy)
        self._curvature_curve.setData(result["arc_length"], result["curvature"])

        abs_curvature = np.abs(np.asarray(result["curvature"], dtype=float))
        self._lbl_status.setText("OK")
        self._lbl_path_method.setText(str(result["path_method"]))
        self._lbl_path_length.setText(f"{float(result['effective_length']):.3f} µm")
        self._lbl_path_projection.setText(f"{float(result['projected_length']):.3f} µm")
        self._lbl_tortuosity.setText(f"{float(result['tortuosity']):.5f}")
        self._lbl_curvature.setText(f"{float(np.mean(abs_curvature)):.6f} 1/µm")
        self._refresh_threshold_sweep()

    def _refresh_threshold_sweep(self) -> None:
        """Recompute and redraw the tortuosity-versus-threshold sweep."""
        thresholds = np.linspace(self._diff_min, self._diff_max, 41, dtype=float)
        self._threshold_sweep_result = sweep_crack_path_thresholds(
            self._surface_a,
            self._surface_b,
            thresholds=thresholds,
            dx=self._dx,
            dy=self._dy,
            propagation_axis=self._current_propagation_axis(),
            front_side=self._current_front_side(),
            method=self._current_method(),
            contour_resample_step=float(self._resample_spin.value()),
            contour_smoothing_window=int(self._smoothing_spin.value()),
        )
        self._sweep_curve.setData(
            self._threshold_sweep_result["thresholds"],
            self._threshold_sweep_result["tortuosity"],
        )
        self._current_threshold_line.setPos(self._threshold)

    def _update_contour_controls(self) -> None:
        """Enable contour-specific controls only for contour extraction."""
        is_contour = self._current_method() == "contour"
        self._resample_spin.setEnabled(is_contour)
        self._smoothing_spin.setEnabled(is_contour)

    def _current_method(self) -> str:
        """Return the selected crack-path extraction method."""
        return str(self._method_combo.currentData() or "first_open_pixel")

    def _current_propagation_axis(self) -> str:
        """Return the currently selected propagation axis."""
        return str(self._propagation_axis_combo.currentData() or "x")

    def _current_front_side(self) -> str:
        """Return the currently selected front side."""
        return str(self._front_side_combo.currentData() or "min")

    def _export_npz(self) -> None:
        """Save the current crack-path analysis arrays to a compressed NPZ."""
        if self._analysis_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export crack-path data (NPZ)", "", "NumPy archive (*.npz)"
        )
        if not path:
            return
        np.savez_compressed(
            path,
            difference_map=self._analysis_result["difference_map"],
            open_mask=self._analysis_result["open_mask"],
            path_points=self._analysis_result["path_points"],
            arc_length=self._analysis_result["arc_length"],
            curvature=self._analysis_result["curvature"],
            tangent_angle=self._analysis_result["tangent_angle"],
            sweep_thresholds=(
                self._threshold_sweep_result["thresholds"]
                if self._threshold_sweep_result is not None
                else np.empty((0,), dtype=float)
            ),
            sweep_tortuosity=(
                self._threshold_sweep_result["tortuosity"]
                if self._threshold_sweep_result is not None
                else np.empty((0,), dtype=float)
            ),
            threshold_um=self._threshold,
            dx_um=self._dx,
            dy_um=self._dy,
            method=self._current_method(),
            propagation_axis=self._current_propagation_axis(),
            front_side=self._current_front_side(),
            contour_resample_step_um=self._analysis_result.get("contour_resample_step"),
            contour_smoothing_window=self._analysis_result.get("contour_smoothing_window"),
        )
        logger.info("Crack-path data exported to %s", path)

    def _export_json(self) -> None:
        """Save the current crack-path metrics to a JSON file."""
        if self._analysis_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export crack-path metrics (JSON)", "", "JSON file (*.json)"
        )
        if not path:
            return

        abs_curvature = np.abs(np.asarray(self._analysis_result["curvature"], dtype=float))
        data = {
            "surface_a": self._title_a,
            "surface_b": self._title_b,
            "threshold_um": round(self._threshold, 6),
            "method": self._current_method(),
            "propagation_axis": self._current_propagation_axis(),
            "front_side": self._current_front_side(),
            "contour_resample_step_um": self._analysis_result.get("contour_resample_step"),
            "contour_smoothing_window": self._analysis_result.get("contour_smoothing_window"),
            "effective_length_um": round(float(self._analysis_result["effective_length"]), 6),
            "projected_length_um": round(float(self._analysis_result["projected_length"]), 6),
            "tortuosity": round(float(self._analysis_result["tortuosity"]), 8),
            "path_point_count": int(len(self._analysis_result["path_points"])),
            "mean_abs_curvature_inv_um": round(float(np.mean(abs_curvature)), 8),
            "max_abs_curvature_inv_um": round(float(np.max(abs_curvature)), 8),
            "threshold_sweep_samples": (
                int(len(self._threshold_sweep_result["thresholds"]))
                if self._threshold_sweep_result is not None
                else 0
            ),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        logger.info("Crack-path metrics exported to %s", path)
