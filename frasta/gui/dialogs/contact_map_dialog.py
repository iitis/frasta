"""Contact map dialog for FRASTA-toolbox.

Provides interactive crack-opening displacement (COD) and contact map analysis
for two aligned fracture surfaces. The user adjusts a threshold s to classify
pixels as in-contact (D < s) or open (D >= s), where D = A - B is the
difference map.

The dialog is independent of OverlayViewer and ProfileViewer. It can be opened
from Tools -> Contact map analysis... with any pair of scans already loaded.
"""

from __future__ import annotations

import json

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...core import Surface
from ...utils import get_colormap

import logging
logger = logging.getLogger(__name__)


class ContactMapDialog(QtWidgets.QWidget):
    """Interactive contact map and COD analysis dialog.

    Displays the continuous difference map D = A - B alongside a binary contact
    map that updates in real time as the user moves the threshold slider.

    Parameters
    ----------
    surface_a, surface_b:
        Aligned fracture surface pair. Both must share the same height-array
        shape.
    parent:
        Optional parent widget.
    title_a, title_b:
        Display names used in labels and export metadata.
    """

    def __init__(
        self,
        surface_a: Surface,
        surface_b: Surface,
        parent=None,
        title_a: str = "Surface A",
        title_b: str = "Surface B",
    ):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("Contact map analysis")
        self.setMinimumSize(960, 640)

        self._surface_a = surface_a
        self._surface_b = surface_b
        self._title_a = title_a
        self._title_b = title_b
        self._dx = float(surface_a.dx)
        self._dy = float(surface_a.dy)

        # Difference map D = A - B  (rows, cols)
        self._diff = surface_a.height - surface_b.height
        self._valid = np.isfinite(self._diff)
        self._n_valid = int(np.sum(self._valid))

        valid_vals = self._diff[self._valid]
        if valid_vals.size > 0:
            self._diff_min = float(np.min(valid_vals))
            self._diff_max = float(np.max(valid_vals))
            self._threshold = float(np.median(valid_vals))
        else:
            self._diff_min = -1.0
            self._diff_max = 1.0
            self._threshold = 0.0

        # Continuous diff colormap (reuse project palette)
        self._diff_cmap = get_colormap("difference")
        self._diff_lut = self._diff_cmap.getLookupTable(0.0, 1.0, 512)

        # Binary contact LUT: 0 = open (light gray), 1 = contact (blue)
        self._binary_lut = np.zeros((256, 4), dtype=np.uint8)
        self._binary_lut[0] = [200, 200, 200, 255]   # open: light gray
        self._binary_lut[1] = [30, 100, 220, 255]    # contact: blue

        # Cached contact array for export
        self._contact_binary: np.ndarray | None = None

        self._updating_controls = False

        self._build_ui()
        self._connect_signals()
        self._update_diff_view()
        self._update_contact_map()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Top row: two image views side by side
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: continuous difference map
        left_group = QtWidgets.QGroupBox(
            f"Difference map  D = {self._title_a} \u2212 {self._title_b}  [\u00b5m]"
        )
        left_layout = QtWidgets.QVBoxLayout(left_group)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._diff_view = pg.ImageView(view=pg.PlotItem())
        self._diff_view.ui.roiBtn.hide()
        self._diff_view.ui.menuBtn.hide()
        self._diff_view.setColorMap(self._diff_cmap)
        self._diff_view.getImageItem().setLookupTable(self._diff_lut)
        left_layout.addWidget(self._diff_view)
        splitter.addWidget(left_group)

        # Right: binary contact map
        right_group = QtWidgets.QGroupBox(
            "Binary contact map  (blue = contact  D < s,  gray = open  D \u2265 s)"
        )
        right_layout = QtWidgets.QVBoxLayout(right_group)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._binary_view = pg.ImageView(view=pg.PlotItem())
        self._binary_view.ui.histogram.hide()
        self._binary_view.ui.roiBtn.hide()
        self._binary_view.ui.menuBtn.hide()
        self._binary_view.getImageItem().setLookupTable(self._binary_lut)
        right_layout.addWidget(self._binary_view)
        splitter.addWidget(right_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # Middle: threshold control row
        ctrl_group = QtWidgets.QGroupBox("Contact threshold")
        ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)
        ctrl_layout.setHorizontalSpacing(8)
        ctrl_layout.setVerticalSpacing(4)

        ctrl_layout.addWidget(QtWidgets.QLabel("Threshold  s  [\u00b5m]:"), 0, 0)

        span = self._diff_max - self._diff_min
        step = max(0.001, span / 200.0)
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setRange(
            self._diff_min - span,
            self._diff_max + span,
        )
        self._threshold_spin.setSingleStep(step)
        self._threshold_spin.setValue(self._threshold)
        self._threshold_spin.setMaximumWidth(120)
        ctrl_layout.addWidget(self._threshold_spin, 0, 1)

        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setRange(0, 1000)
        self._threshold_slider.setValue(self._spin_to_slider(self._threshold))
        ctrl_layout.addWidget(self._threshold_slider, 0, 2)

        ctrl_layout.addWidget(
            QtWidgets.QLabel("pixels with D < s are classified as in-contact"), 0, 3
        )
        ctrl_layout.setColumnStretch(2, 1)
        root.addWidget(ctrl_group)

        # Bottom: statistics + export buttons
        bottom = QtWidgets.QHBoxLayout()

        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QFormLayout(stats_group)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setHorizontalSpacing(12)

        self._lbl_fraction = QtWidgets.QLabel("\u2014")
        self._lbl_area = QtWidgets.QLabel("\u2014")
        self._lbl_mean_cod = QtWidgets.QLabel("\u2014")
        self._lbl_d_range = QtWidgets.QLabel(
            f"min {self._diff_min:.3f}  \u2013  max {self._diff_max:.3f}  \u00b5m"
        )

        stats_layout.addRow("D range:", self._lbl_d_range)
        stats_layout.addRow("Contact fraction:", self._lbl_fraction)
        stats_layout.addRow("Contact area:", self._lbl_area)
        stats_layout.addRow("Mean D (open region):", self._lbl_mean_cod)
        bottom.addWidget(stats_group, stretch=1)

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(6)
        self._btn_export_npz = QtWidgets.QPushButton("Export binary map (NPZ)\u2026")
        self._btn_export_json = QtWidgets.QPushButton("Export statistics (JSON)\u2026")
        self._btn_close = QtWidgets.QPushButton("Close")
        btn_layout.addWidget(self._btn_export_npz)
        btn_layout.addWidget(self._btn_export_json)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self._btn_close)
        bottom.addLayout(btn_layout)
        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._threshold_spin.valueChanged.connect(self._on_spin_changed)
        self._threshold_slider.valueChanged.connect(self._on_slider_changed)
        self._btn_export_npz.clicked.connect(self._export_npz)
        self._btn_export_json.clicked.connect(self._export_json)
        self._btn_close.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _update_diff_view(self) -> None:
        """Display the continuous difference map D."""
        img = self._diff.T.astype(np.float32)
        self._diff_view.setImage(img, autoLevels=True)

    def _update_contact_map(self) -> None:
        """Recompute binary contact map for the current threshold and refresh stats."""
        s = self._threshold
        contact = np.where(self._valid, self._diff < s, 0).astype(np.uint8)
        self._contact_binary = contact

        img = contact.T.copy()
        self._binary_view.setImage(img, autoLevels=False, levels=(0, 1))

        # Statistics
        n_contact = int(np.sum(contact[self._valid]))
        fraction = (n_contact / self._n_valid * 100.0) if self._n_valid > 0 else 0.0

        pixel_area_um2 = self._dx * self._dy
        contact_area_um2 = n_contact * pixel_area_um2
        contact_area_mm2 = contact_area_um2 * 1e-6

        open_mask = self._valid & ~contact.astype(bool)
        if np.any(open_mask):
            mean_cod = float(np.mean(self._diff[open_mask]))
        else:
            mean_cod = float("nan")

        self._lbl_fraction.setText(
            f"{fraction:.2f} %  ({n_contact} / {self._n_valid} px)"
        )
        self._lbl_area.setText(
            f"{contact_area_mm2:.6f} mm\u00b2  ({contact_area_um2:.1f} \u00b5m\u00b2)"
        )
        if np.isfinite(mean_cod):
            self._lbl_mean_cod.setText(f"{mean_cod:.3f} \u00b5m")
        else:
            self._lbl_mean_cod.setText("\u2014 (no open region)")

    # ------------------------------------------------------------------
    # Threshold <-> slider conversion
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_spin_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self._threshold = value
        self._updating_controls = True
        self._threshold_slider.setValue(self._spin_to_slider(value))
        self._updating_controls = False
        self._update_contact_map()

    def _on_slider_changed(self, pos: int) -> None:
        if self._updating_controls:
            return
        value = self._slider_to_spin(pos)
        self._threshold = value
        self._updating_controls = True
        self._threshold_spin.setValue(value)
        self._updating_controls = False
        self._update_contact_map()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_npz(self) -> None:
        """Save the binary contact map and difference map to a compressed NPZ file."""
        if self._contact_binary is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export contact map (NPZ)", "", "NumPy archive (*.npz)"
        )
        if not path:
            return
        np.savez_compressed(
            path,
            binary_contact=self._contact_binary,
            difference_map=self._diff,
            threshold_um=self._threshold,
            dx_um=self._dx,
            dy_um=self._dy,
        )
        logger.info("Contact map exported to %s", path)

    def _export_json(self) -> None:
        """Save contact statistics to a JSON file."""
        if self._contact_binary is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export statistics (JSON)", "", "JSON file (*.json)"
        )
        if not path:
            return

        s = self._threshold
        n_contact = int(np.sum(self._contact_binary[self._valid]))
        fraction = (n_contact / self._n_valid * 100.0) if self._n_valid > 0 else 0.0
        pixel_area_um2 = self._dx * self._dy
        contact_area_um2 = n_contact * pixel_area_um2
        open_mask = self._valid & ~self._contact_binary.astype(bool)
        mean_cod: float | None
        if np.any(open_mask):
            mean_cod = float(np.mean(self._diff[open_mask]))
        else:
            mean_cod = None

        data = {
            "surface_a": self._title_a,
            "surface_b": self._title_b,
            "threshold_um": round(s, 6),
            "n_valid_pixels": self._n_valid,
            "n_contact_pixels": n_contact,
            "contact_fraction_pct": round(fraction, 4),
            "contact_area_um2": round(contact_area_um2, 4),
            "contact_area_mm2": round(contact_area_um2 * 1e-6, 8),
            "mean_cod_open_um": (
                round(mean_cod, 4) if mean_cod is not None else None
            ),
            "pixel_dx_um": self._dx,
            "pixel_dy_um": self._dy,
            "diff_min_um": round(self._diff_min, 6),
            "diff_max_um": round(self._diff_max, 6),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Contact statistics exported to %s", path)
