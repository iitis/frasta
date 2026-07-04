"""Overlay viewer for aligning and comparing two scan datasets.

This module provides an interactive viewer for overlaying two scans, allowing
manual alignment through translation and rotation controls, and displaying
the difference map between aligned scans.
"""

import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from ...core import Surface
from ...utils import get_colormap, get_brushes_for_values
from ..orientation import grid_to_image_data

class OverlayViewer(QtWidgets.QWidget):
    """Interactive widget for overlaying and aligning two scan datasets.
    
    Provides side-by-side views of overlaid scans and their difference map,
    with interactive controls for translation, rotation, and visual comparison
    modes (blinking, transparency).
    
    Attributes:
        scan1_data (Surface): First scan dataset.
        scan2_data (Surface): Second scan dataset.
        img1 (pg.ImageItem): Image item for first scan.
        img2 (pg.ImageItem): Image item for second scan (transformable).
        diff_view (pg.ImageView): View displaying the difference map.
        on_accept (callable, optional): Callback when alignment is accepted.
    """
    
    def __init__(
        self,
        scan1_data: Surface,
        scan2_data: Surface,
        on_accept=None,
        parent=None,
        reference_tab=None,
        moving_tab=None,
    ):
        """Initialize the overlay viewer.
        
        Args:
            scan1_data (Surface): First scan to display (reference).
            scan2_data (Surface): Second scan to display (adjustable).
            on_accept (callable, optional): Callback function called when accepting alignment.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window)

        self.on_accept = on_accept
        self.reference_tab = reference_tab
        self.moving_tab = moving_tab

        self.scan1_data = scan1_data
        self.scan2_data = scan2_data

        self.scan1_raw = scan1_data.height
        self.scan2_raw = scan2_data.height
        self.scan1 = grid_to_image_data(
            self.scan1_raw,
            orientation=getattr(scan1_data, "orientation", "default"),
            copy=True,
        )
        self.scan2 = grid_to_image_data(
            self.scan2_raw,
            orientation=getattr(scan2_data, "orientation", "default"),
            copy=True,
        )

        self._orig_scan1 = self.scan1.copy()
        self._orig_scan2 = self.scan2.copy()


        self._last_diff_image = None   # original difference (before masking)
        self._diff_hist_bars = None
        self._last_auto_diff_levels = None
        self._diff_manual_levels = None
        self._diff_use_auto_levels = True
        self._updating_diff_controls = False
        self._diff_center_slider_limits = (-1.0, 1.0)
        self._diff_half_range_slider_limits = (1e-3, 1.0)
        self._diff_cmap = get_colormap("difference")
        self._diff_lut = self._diff_cmap.getLookupTable(0.0, 1.0, 512)
        self._display_stride = self._choose_preview_stride(
            self.scan1.shape,
            self.scan2.shape,
            max_preview_size=1200,
        )
        self._preview_stride = max(
            self._display_stride,
            self._choose_preview_stride(
                self.scan1.shape,
                self.scan2.shape,
                max_preview_size=640,
            ),
        )
        self._display_scan1 = np.ascontiguousarray(
            self.scan1[::self._display_stride, ::self._display_stride],
            dtype=np.float32,
        )
        self._display_scan2 = np.ascontiguousarray(
            self.scan2[::self._display_stride, ::self._display_stride],
            dtype=np.float32,
        )
        self._preview_scan1 = np.ascontiguousarray(
            self.scan1[::self._preview_stride, ::self._preview_stride],
            dtype=np.float32,
        )
        self._preview_scan2 = np.ascontiguousarray(
            self.scan2[::self._preview_stride, ::self._preview_stride],
            dtype=np.float32,
        )
        self._display_diff_buffer = np.empty_like(self._display_scan1, dtype=np.float32)
        self._preview_diff_buffer = np.empty_like(self._preview_scan1, dtype=np.float32)
        self._full_diff_buffer = np.empty_like(self.scan1, dtype=np.float32)
        self._configure_diff_update_timers()
        self._reference_center_full = self._compute_valid_image_center(self.scan1)
        self._moving_center_full = self._compute_valid_image_center(self.scan2)
        self._reference_center_display = self._compute_valid_image_center(self._display_scan1)
        self._moving_center_display = self._compute_valid_image_center(self._display_scan2)
        self._reference_center_preview = self._compute_valid_image_center(self._preview_scan1)
        self._moving_center_preview = self._compute_valid_image_center(self._preview_scan2)
        self._reference_base_transform = self._build_base_image_transform(
            self.scan1_data,
            self.scan1,
        )
        self._moving_base_transform = self._build_base_image_transform(
            self.scan2_data,
            self.scan2,
        )
        self._display_reference_base_transform = self._build_base_image_transform(
            self.scan1_data,
            self._display_scan1,
            stride=self._display_stride,
        )
        self._display_moving_base_transform = self._build_base_image_transform(
            self.scan2_data,
            self._display_scan2,
            stride=self._display_stride,
        )
        self._preview_reference_base_transform = self._build_base_image_transform(
            self.scan1_data,
            self._preview_scan1,
            stride=self._preview_stride,
        )
        self._preview_moving_base_transform = self._build_base_image_transform(
            self.scan2_data,
            self._preview_scan2,
            stride=self._preview_stride,
        )

        self.create_gui()

        # Images
        self.img1 = pg.ImageItem(self.scan1)
        self.img2 = pg.ImageItem(self.scan2)
        self.img2.setOpacity(0.5)
        self.img1.setTransform(self._reference_base_transform)
        self.img2.setTransform(self._moving_base_transform)

        self.viewbox.addItem(self.img1)
        self.viewbox.addItem(self.img2)
        self._sync_overlay_view_to_reference()

        def safe_minmax(arr):
            arr = arr[np.isfinite(arr)]  # discard NaN, +inf, -inf
            return (0.0, 1.0) if arr.size == 0 else (np.min(arr), np.max(arr))
            # if arr.size == 0:
            #     return 0.0, 1.0  # domyślne wartości, jeśli wszystko było złe
            # return np.min(arr), np.max(arr)

        vmin1_, vmax1_ = safe_minmax(self.scan1)
        vmin2_, vmax2_ = safe_minmax(self.scan2)

        self.vmin1 = scan1_data.vmin if scan1_data.vmin is not None else vmin1_
        self.vmax1 = scan1_data.vmax if scan1_data.vmax is not None else vmax1_
        self.vmin2 = scan2_data.vmin if scan2_data.vmin is not None else vmin2_
        self.vmax2 = scan2_data.vmax if scan2_data.vmax is not None else vmax2_

        self.img1.setLevels((self.vmin1, self.vmax1))
        self.img2.setLevels((self.vmin2, self.vmax2))

        # remember grids as attributes
        self.original_scan1 = self.scan1
        self.original_scan2 = self.scan2

        # Connections
        self.slider_tx.valueChanged.connect(self.updateTransform)
        self.slider_ty.valueChanged.connect(self.updateTransform)
        self.slider_angle.valueChanged.connect(self.updateTransform)
        self.slider_tx.sliderReleased.connect(self.update_difference_map)
        self.slider_ty.sliderReleased.connect(self.update_difference_map)
        self.slider_angle.sliderReleased.connect(self.update_difference_map)

        self.updateTransform()
        self.update_difference_map()

    def closeEvent(self, event):
        p = self.parent()
        if p and hasattr(p, "viewer"):
            p.viewer = None
        super().closeEvent(event)

    def create_gui(self):
        """Build the widget layout and initialize static viewer state."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Main window and scene
        self.view = pg.GraphicsLayoutWidget()
        self.view.setMinimumWidth(0)
        self.viewbox = self.view.addViewBox()
        self.viewbox.setAspectLocked(True)
        self.viewbox.invertY(True)
        self._pivot_vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(255, 200, 0, width=2),
        )
        self._pivot_hline = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(255, 200, 0, width=2),
        )
        self._pivot_marker = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen(255, 200, 0, width=2),
            brush=pg.mkBrush(255, 200, 0, 180),
            symbol="o",
        )
        self._pivot_vline.setZValue(1000)
        self._pivot_hline.setZValue(1000)
        self._pivot_marker.setZValue(1001)
        self.viewbox.addItem(self._pivot_vline)
        self.viewbox.addItem(self._pivot_hline)
        self.viewbox.addItem(self._pivot_marker)

        # Difference image
        self.diff_view = pg.ImageView(view=pg.PlotItem())
        self.diff_view.ui.histogram.hide()
        self.diff_view.ui.roiBtn.hide()
        self.diff_view.ui.menuBtn.hide()
        self.diff_view.setMinimumWidth(0)
        self.diff_view.setColorMap(self._diff_cmap)
        self.diff_view.getImageItem().setLookupTable(self._diff_lut)
        self.diff_view.getView().setAspectLocked(True)
        self.diff_view.getView().invertY(True)

        # Horizontal splitter: overlay view + difference view
        self.preview_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.preview_splitter.setChildrenCollapsible(True)
        self.preview_splitter.addWidget(self.view)
        self.preview_splitter.addWidget(self.diff_view)
        self.preview_splitter.setCollapsible(0, True)
        self.preview_splitter.setCollapsible(1, True)
        self.preview_splitter.setStretchFactor(0, 1)
        self.preview_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.preview_splitter, stretch=1)

        south_widget = QtWidgets.QWidget(self)
        south_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        south_layout = QtWidgets.QHBoxLayout(south_widget)
        south_layout.setContentsMargins(0, 0, 0, 0)
        south_layout.setSpacing(6)

        south_layout.addLayout(self.sliders_layout())

        # Switches
        self.checkbox_visible = QtWidgets.QCheckBox("scan2 is visible")
        self.checkbox_visible.setChecked(True)
        self.checkbox_visible.stateChanged.connect(self.toggleVisibility)

        self.checkbox_trans = QtWidgets.QCheckBox("scan2 translucence")
        self.checkbox_trans.setChecked(True)
        self.checkbox_trans.stateChanged.connect(self.toggleTransparency)

        self.checkbox_blink = QtWidgets.QCheckBox("scan2 blinking")
        self.checkbox_blink.setChecked(False)
        self.checkbox_blink.stateChanged.connect(self.toggleBlinking)

        tool2_layout = QtWidgets.QVBoxLayout()
        tool2_layout.setSpacing(6)

        toggles_row = QtWidgets.QHBoxLayout()
        toggles_row.setSpacing(10)
        toggles_row.addWidget(self.checkbox_visible)
        toggles_row.addWidget(self.checkbox_trans)
        toggles_row.addWidget(self.checkbox_blink)
        toggles_row.addStretch(1)
        tool2_layout.addLayout(toggles_row)
        range_and_actions_row = QtWidgets.QHBoxLayout()
        range_and_actions_row.setSpacing(6)
        range_and_actions_row.addWidget(self._create_difference_controls(), stretch=1)

        # Blink timer
        self.blink_timer = QtCore.QTimer()
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self.blinkToggle)
        self.blink_state = True

        action_buttons_layout = QtWidgets.QVBoxLayout()
        action_buttons_layout.setSpacing(6)

        self.auto_icp_btn = QtWidgets.QPushButton("Auto align (ICP, Experimental)")
        self.auto_icp_btn.clicked.connect(self.apply_auto_icp)
        action_buttons_layout.addWidget(self.auto_icp_btn)

        self.debug_dump_btn = QtWidgets.QPushButton("Save debug bundle")
        self.debug_dump_btn.clicked.connect(self.save_debug_bundle)
        action_buttons_layout.addWidget(self.debug_dump_btn)

        # self.save_button = QtWidgets.QPushButton("Save aligned grids to .h5")
        # self.save_button.clicked.connect(self.saveAlignedScans)
        # tool2_layout.addWidget(self.save_button)

        self.accept_button = QtWidgets.QPushButton("Accept changes")
        self.accept_button.clicked.connect(self.accept_result)
        action_buttons_layout.addWidget(self.accept_button)
        action_buttons_layout.addStretch(1)

        range_and_actions_row.addLayout(action_buttons_layout)
        tool2_layout.addLayout(range_and_actions_row)

        # Zakres min/max
        # self.levels_min = QtWidgets.QDoubleSpinBox()
        # self.levels_min.setDecimals(2)
        # self.levels_min.setPrefix("Min: ")
        # self.levels_min.setRange(-1e6, 1e6)

        # self.levels_max = QtWidgets.QDoubleSpinBox()
        # self.levels_max.setDecimals(2)
        # self.levels_max.setPrefix("Max: ")
        # self.levels_max.setRange(-1e6, 1e6)

        # self.levels_min.valueChanged.connect(self.apply_overlay_mask)
        # self.levels_max.valueChanged.connect(self.apply_overlay_mask)

        # tool2_layout.addWidget(self.levels_min)
        # tool2_layout.addWidget(self.levels_max)

        south_layout.addLayout(tool2_layout)
        south_layout.setAlignment(QtCore.Qt.AlignTop)
        south_widget.adjustSize()

        layout.addWidget(south_widget, stretch=0)

    def _configure_diff_update_timers(self) -> None:
        """Create timers used for throttled preview and deferred full redraws."""
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(40)
        self._preview_timer.timeout.connect(self._update_difference_preview)

        self._full_refresh_timer = QtCore.QTimer(self)
        self._full_refresh_timer.setSingleShot(True)
        self._full_refresh_timer.setInterval(180)
        self._full_refresh_timer.timeout.connect(self.update_difference_map)

    @staticmethod
    def _choose_preview_stride(
        reference_shape: tuple[int, int],
        moving_shape: tuple[int, int],
        max_preview_size: int = 512,
    ) -> int:
        """Choose a stride that keeps the live-preview grid reasonably small."""
        max_dim = max(reference_shape + moving_shape)
        return max(1, int(math.ceil(max_dim / float(max_preview_size))))

    def _create_difference_controls(self) -> QtWidgets.QGroupBox:
        """Create a simplified difference-range control panel."""
        group = QtWidgets.QGroupBox("Difference range")
        group.setMaximumWidth(360)
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)

        center_label = QtWidgets.QLabel("Ctr")
        center_label.setToolTip("Display center for the difference scale.")
        layout.addWidget(center_label, 0, 0)

        self.diff_center_spin = QtWidgets.QDoubleSpinBox()
        self.diff_center_spin.setDecimals(3)
        self.diff_center_spin.setRange(-1e9, 1e9)
        self.diff_center_spin.setSingleStep(1.0)
        self.diff_center_spin.setValue(0.0)
        self.diff_center_spin.setMaximumWidth(95)
        self.diff_center_spin.valueChanged.connect(self._on_diff_center_changed)
        layout.addWidget(self.diff_center_spin, 0, 1)

        self.diff_center_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.diff_center_slider.setRange(0, 1000)
        self.diff_center_slider.valueChanged.connect(self._on_diff_center_slider_changed)
        layout.addWidget(self.diff_center_slider, 0, 2)

        range_label = QtWidgets.QLabel("+/-")
        range_label.setToolTip("Half-range around the current display center.")
        layout.addWidget(range_label, 1, 0)

        self.diff_half_range_spin = QtWidgets.QDoubleSpinBox()
        self.diff_half_range_spin.setDecimals(3)
        self.diff_half_range_spin.setRange(0.001, 1e9)
        self.diff_half_range_spin.setSingleStep(1.0)
        self.diff_half_range_spin.setValue(1.0)
        self.diff_half_range_spin.setMaximumWidth(95)
        self.diff_half_range_spin.valueChanged.connect(self._on_diff_half_range_changed)
        layout.addWidget(self.diff_half_range_spin, 1, 1)

        self.diff_half_range_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.diff_half_range_slider.setRange(0, 1000)
        self.diff_half_range_slider.valueChanged.connect(self._on_diff_half_range_slider_changed)
        layout.addWidget(self.diff_half_range_slider, 1, 2)

        self.diff_auto_btn = QtWidgets.QPushButton("Auto")
        self.diff_auto_btn.setMaximumWidth(95)
        self.diff_auto_btn.clicked.connect(self._reset_diff_levels_to_auto)
        layout.addWidget(self.diff_auto_btn, 2, 1)

        self.diff_advanced_toggle = QtWidgets.QPushButton("Advanced histogram")
        self.diff_advanced_toggle.setCheckable(True)
        self.diff_advanced_toggle.toggled.connect(self._toggle_advanced_histogram)
        layout.addWidget(self.diff_advanced_toggle, 2, 2)

        histogram_item = self.diff_view.ui.histogram.item
        histogram_item.sigLevelChangeFinished.connect(self._on_diff_histogram_level_change_finished)
        return group


    def sliders_layout(self):
        # Suwaki
        self.slider_tx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        #self.slider_tx = QtWidgets.QSpinBox()
        self.slider_tx.setMinimum(-2000)
        self.slider_tx.setMaximum(2000)
        self.slider_tx.setValue(0)
        self.label_tx = QtWidgets.QLabel("X: 0")
        self.label_tx.setMinimumWidth(70)

        self.slider_ty = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        #self.slider_ty = QtWidgets.QSpinBox()
        self.slider_ty.setMinimum(-2000)
        self.slider_ty.setMaximum(2000)
        self.slider_ty.setValue(0)
        self.label_ty = QtWidgets.QLabel("Y: 0")
        self.label_ty.setMinimumWidth(70)

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_angle.setMinimum(-600)
        self.slider_angle.setMaximum(600)
        self.slider_angle.setValue(0)
        self.label_angle = QtWidgets.QLabel("Angle: 0.0°")
        self.label_angle.setMinimumWidth(70)

        tool_layout = QtWidgets.QVBoxLayout()
        
        # Layout suwaków z opisami
        trx_layout = QtWidgets.QHBoxLayout()
        # trx_layout.addWidget(QtWidgets.QLabel("Translate X"))
        trx_layout.addWidget(self.label_tx)
        trx_layout.addWidget(self.slider_tx)
        tool_layout.addLayout(trx_layout)

        try_layout = QtWidgets.QHBoxLayout()
        # try_layout.addWidget(QtWidgets.QLabel("Translate Y"))
        try_layout.addWidget(self.label_ty)
        try_layout.addWidget(self.slider_ty)
        tool_layout.addLayout(try_layout)

        rot_layout = QtWidgets.QHBoxLayout()
        # rot_layout.addWidget(QtWidgets.QLabel("Rotate (deg)"))
        rot_layout.addWidget(self.label_angle)
        rot_layout.addWidget(self.slider_angle)
        tool_layout.addLayout(rot_layout)
        
        return tool_layout


    def toggleVisibility(self, state):
        # Wyłączenie obrazu drugiego bez migotania
        if not self.checkbox_blink.isChecked():
            self.img2.setVisible(state == QtCore.Qt.Checked)

    def toggleTransparency(self, state):
        if state == QtCore.Qt.Checked:
            self.img2.setOpacity(0.5)
        else:
            self.img2.setOpacity(1.0)

    def toggleBlinking(self, state):
        if state == QtCore.Qt.Checked:
            self.blink_state = True
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            self.img2.setVisible(self.checkbox_visible.isChecked())  # przywróć widoczność

    def blinkToggle(self):
        self.blink_state = not self.blink_state
        self.img2.setVisible(self.blink_state)

    def saveAlignedScans(self):
        import h5py
        from PyQt5.QtWidgets import QFileDialog

        # Znajdź wspólny rozmiar
        h = min(self.original_scan1.shape[0], self.original_scan2.shape[0])
        w = min(self.original_scan1.shape[1], self.original_scan2.shape[1])

        aligned1 = self.original_scan1[:h, :w].T
        aligned2 = self.original_scan2[:h, :w].T

        # Dialog zapisu
        path, _ = QFileDialog.getSaveFileName(self, "Save the .h5 file", "aligned.h5", "HDF5 (*.h5)")
        if not path:
            return

        # Zapis do HDF5
        with h5py.File(path, "w") as f:
            f.create_dataset("scan1", data=aligned1)
            f.create_dataset("scan2", data=aligned2)

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to file:\n{path}")



    def accept_result(self):
        tx = float(self.slider_tx.value())
        ty = float(self.slider_ty.value())
        angle = float(self.slider_angle.value()) / 10.0
        scan2_trans = np.empty_like(self.scan1, dtype=np.float64)
        self._render_difference(
            self.scan1,
            self.scan2,
            self._reference_base_transform,
            self._moving_base_transform,
            tx,
            ty,
            angle,
            output_buffer=scan2_trans,
        )

        data1 = Surface(
            height=self.scan1.T,
            dx=self.scan1_data.dx,
            dy=self.scan1_data.dy,
            x0=self.scan1_data.x0,
            y0=self.scan1_data.y0,
            vmin=self.vmin1,
            vmax=self.vmax1
        )

        data2 = Surface(
            height=scan2_trans.T,
            dx=self.scan1_data.dx,
            dy=self.scan1_data.dy,
            x0=self.scan1_data.x0,
            y0=self.scan1_data.y0,
            vmin=self.vmin2,
            vmax=self.vmax2
        )

        if self.on_accept is not None:
            self.on_accept(data1, data2)

        self.close()

    @staticmethod
    def _qrectf_to_dict(rect: QtCore.QRectF) -> dict[str, float]:
        """Convert a QRectF into a JSON-serializable dictionary."""
        return {
            "left": float(rect.left()),
            "top": float(rect.top()),
            "width": float(rect.width()),
            "height": float(rect.height()),
            "right": float(rect.right()),
            "bottom": float(rect.bottom()),
        }

    @staticmethod
    def _qtransform_to_dict(transform: QtGui.QTransform) -> dict[str, float]:
        """Convert a QTransform into a JSON-serializable dictionary."""
        return {
            "m11": float(transform.m11()),
            "m12": float(transform.m12()),
            "m13": float(transform.m13()),
            "m21": float(transform.m21()),
            "m22": float(transform.m22()),
            "m23": float(transform.m23()),
            "m31": float(transform.m31()),
            "m32": float(transform.m32()),
            "m33": float(transform.m33()),
        }

    @staticmethod
    def _surface_debug_dict(surface: Surface, image_data: np.ndarray) -> dict[str, object]:
        """Describe one surface and its current image-space raster."""
        return {
            "grid_shape": list(surface.height.shape),
            "image_shape": list(image_data.shape),
            "dx": float(surface.dx),
            "dy": float(surface.dy),
            "x0": float(surface.x0),
            "y0": float(surface.y0),
            "orientation": str(getattr(surface, "orientation", "default")),
            "unit": str(getattr(surface, "unit", "um")),
            "finite_count": int(np.isfinite(image_data).sum()),
        }

    def _current_moving_transform(self) -> QtGui.QTransform:
        """Build the transform currently applied to the moving overlay."""
        return self._build_image_transform(
            float(self.slider_tx.value()),
            float(self.slider_ty.value()),
            float(self.slider_angle.value()) / 10.0,
            self._moving_base_transform,
            scene_rotation_center=self._reference_scene_rotation_center(
                self.scan1,
                self._reference_base_transform,
            ),
        )

    @staticmethod
    def _viewbox_flag(view_like, key: str, default: bool = False) -> bool:
        """Read one boolean state flag from a ViewBox or PlotItem-backed view."""
        if hasattr(view_like, "state") and isinstance(getattr(view_like, "state"), dict):
            return bool(view_like.state.get(key, default))
        if hasattr(view_like, "getViewBox"):
            view_box = view_like.getViewBox()
            if hasattr(view_box, "state") and isinstance(getattr(view_box, "state"), dict):
                return bool(view_box.state.get(key, default))
        return bool(default)

    def _collect_debug_state(self) -> dict[str, object]:
        """Collect the current overlay geometry and transform state."""
        moving_transform = self._current_moving_transform()
        reference_rect = self._map_image_rect(self.scan1, self._reference_base_transform)
        moving_rect = self._map_image_rect(self.scan2, moving_transform)
        diff_rect = self.diff_view.getView().viewRect()
        overlay_rect = self.viewbox.viewRect()
        diff_image_item = self.diff_view.getImageItem()
        diff_image_transform = diff_image_item.transform()
        diff_scene_rect = diff_image_item.sceneBoundingRect()

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sliders": {
                "tx": float(self.slider_tx.value()),
                "ty": float(self.slider_ty.value()),
                "angle_deg": float(self.slider_angle.value()) / 10.0,
            },
            "reference_surface": self._surface_debug_dict(self.scan1_data, self.scan1),
            "moving_surface": self._surface_debug_dict(self.scan2_data, self.scan2),
            "centers": {
                "reference_full": [float(v) for v in self._reference_center_full],
                "moving_full": [float(v) for v in self._moving_center_full],
                "reference_display": [float(v) for v in self._reference_center_display],
                "moving_display": [float(v) for v in self._moving_center_display],
                "reference_preview": [float(v) for v in self._reference_center_preview],
                "moving_preview": [float(v) for v in self._moving_center_preview],
            },
            "transforms": {
                "reference_base": self._qtransform_to_dict(self._reference_base_transform),
                "moving_base": self._qtransform_to_dict(self._moving_base_transform),
                "moving_current": self._qtransform_to_dict(moving_transform),
                "display_reference_base": self._qtransform_to_dict(self._display_reference_base_transform),
                "display_moving_base": self._qtransform_to_dict(self._display_moving_base_transform),
                "preview_reference_base": self._qtransform_to_dict(self._preview_reference_base_transform),
                "preview_moving_base": self._qtransform_to_dict(self._preview_moving_base_transform),
            },
            "rects": {
                "reference_image_rect": self._qrectf_to_dict(reference_rect),
                "moving_image_rect": self._qrectf_to_dict(moving_rect),
                "overlay_view_rect": self._qrectf_to_dict(overlay_rect),
                "difference_view_rect": self._qrectf_to_dict(diff_rect),
                "difference_item_scene_rect": self._qrectf_to_dict(diff_scene_rect),
            },
            "view_state": {
                "overlay_y_inverted": self._viewbox_flag(self.viewbox, "yInverted", False),
                "difference_y_inverted": self._viewbox_flag(self.diff_view.getView(), "yInverted", False),
                "overlay_aspect_locked": self._viewbox_flag(self.viewbox, "aspectLocked", False),
                "difference_aspect_locked": self._viewbox_flag(self.diff_view.getView(), "aspectLocked", False),
                "display_stride": int(self._display_stride),
                "preview_stride": int(self._preview_stride),
            },
            "graphics_items": {
                "overlay_reference_transform": self._qtransform_to_dict(self.img1.transform()),
                "overlay_moving_transform": self._qtransform_to_dict(self.img2.transform()),
                "difference_item_transform": self._qtransform_to_dict(diff_image_transform),
                "difference_item_pos": [
                    float(diff_image_item.pos().x()),
                    float(diff_image_item.pos().y()),
                ],
            },
        }

    @staticmethod
    def _save_widget_snapshot(widget: QtWidgets.QWidget, target: Path) -> None:
        """Save one widget snapshot as PNG."""
        pixmap = widget.grab()
        pixmap.save(str(target), "PNG")

    def save_debug_bundle(self) -> None:
        """Save the current overlay state, rasters, and screenshots for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = Path.cwd() / f"overlay_debug_{timestamp}"
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder for overlay debug bundle",
            str(default_dir.parent),
        )
        if not target_dir:
            return

        bundle_dir = Path(target_dir) / f"overlay_debug_{timestamp}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        try:
            (bundle_dir / "progress.txt").write_text("bundle_dir_created\n", encoding="utf-8")
            debug_state = self._collect_debug_state()
            (bundle_dir / "state.json").write_text(
                json.dumps(debug_state, indent=2),
                encoding="utf-8",
            )
            (bundle_dir / "progress.txt").write_text("state_json_saved\n", encoding="utf-8")

            display_transformed_scan2 = np.empty_like(self._display_scan1, dtype=np.float32)
            self._render_difference(
                self._display_scan1,
                self._display_scan2,
                self._display_reference_base_transform,
                self._display_moving_base_transform,
                float(self.slider_tx.value()),
                float(self.slider_ty.value()),
                float(self.slider_angle.value()) / 10.0,
                display_transformed_scan2,
            )

            np.savez_compressed(
                bundle_dir / "arrays.npz",
                scan1=self.scan1,
                scan2=self.scan2,
                display_scan1=self._display_scan1,
                display_scan2=self._display_scan2,
                display_transformed_scan2=display_transformed_scan2,
                display_reference_mask=np.isfinite(self._display_scan1),
                display_transformed_mask=np.isfinite(display_transformed_scan2),
                preview_scan1=self._preview_scan1,
                preview_scan2=self._preview_scan2,
                diff_image=(
                    self._last_diff_image
                    if self._last_diff_image is not None
                    else np.empty((0, 0), dtype=np.float32)
                ),
            )
            (bundle_dir / "progress.txt").write_text("arrays_saved\n", encoding="utf-8")

            self._save_widget_snapshot(self.view, bundle_dir / "overlay_view.png")
            self._save_widget_snapshot(self.diff_view, bundle_dir / "difference_view.png")
            (bundle_dir / "progress.txt").write_text("snapshots_saved\n", encoding="utf-8")
        except Exception as exc:
            (bundle_dir / "error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Debug bundle failed",
                f"Failed to save overlay debug bundle:\n{exc}\n\nDetails saved to:\n{bundle_dir / 'error.txt'}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Debug bundle saved",
            f"Saved overlay debug bundle to:\n{bundle_dir}",
        )


    # def accept_result(self):
    #     # pobierz aktualne dopasowane siatki, np. po transformacji
    #     # tutaj self.scan2 to oryginał, a self.img2.image to może być obraz po transformacji (sprawdź!)
    #     # dla uproszczenia zakładamy, że masz przetransformowaną wersję jako self.img2.image (lub inny atrybut)
    #     from scipy.ndimage import affine_transform

    #     qt_transform = self.img2.transform()
    #     m = np.array([
    #         [qt_transform.m11(), qt_transform.m21(), qt_transform.m31()],
    #         [qt_transform.m12(), qt_transform.m22(), qt_transform.m32()],
    #         [0,                 0,                 1]
    #     ])
    #     affine_matrix = np.linalg.inv(m)[0:2, 0:3]

    #     scan2_trans = affine_transform(
    #         self.scan2,
    #         matrix=affine_matrix[:, :2],
    #         offset=affine_matrix[:, 2],
    #         order=1,
    #         mode='constant',
    #         cval=np.nan
    #     )
    #     h = min(self.scan1.shape[0], scan2_trans.shape[0])
    #     w = min(self.scan1.shape[1], scan2_trans.shape[1])
    #     scan1_cropped = self.scan1[:h, :w]
    #     scan2_trans_cropped = scan2_trans[:h, :w]

    #     # Wywołaj callback
    #     if self.on_accept is not None:
    #         data1 = self.scan1_data
    #         data1.grid = scan1_cropped
    #         data1.xi = self.scan1_data.xi[:w],
    #         data1.yi = self.scan1_data.yi[:h],
    #         data1.vmin = self.vmin1
    #         data1.vmax = self.vmax1
    #         data2 = self.scan2_data
    #         data2.grid = scan2_trans_cropped
    #         data2.xi = self.scan1_data.xi[:w],
    #         data2.yi = self.scan1_data.yi[:h],
    #         data2.vmin = self.vmin2
    #         data2.vmax = self.vmax2
    #         self.on_accept(data1, data2)
    #     self.close()

    def _build_image_transform(
        self,
        tx: float,
        ty: float,
        angle: float,
        base_transform: QtGui.QTransform,
        rotation_center: tuple[float, float] | None = None,
        scene_rotation_center: QtCore.QPointF | None = None,
    ) -> QtGui.QTransform:
        """Create a scene transform from one base image transform and manual pose."""
        if scene_rotation_center is not None:
            pivot_x = float(scene_rotation_center.x())
            pivot_y = float(scene_rotation_center.y())
        elif rotation_center is not None:
            center_scene = base_transform.map(
                QtCore.QPointF(float(rotation_center[0]), float(rotation_center[1]))
            )
            pivot_x = float(center_scene.x())
            pivot_y = float(center_scene.y())
        else:
            pivot_x = 0.0
            pivot_y = 0.0

        angle_rad = math.radians(float(angle))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        translation_matrix = np.array(
            [
                [1.0, 0.0, float(tx)],
                [0.0, 1.0, float(ty)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        rotate_about_pivot = np.array(
            [
                [cos_a, -sin_a, pivot_x - cos_a * pivot_x + sin_a * pivot_y],
                [sin_a, cos_a, pivot_y - sin_a * pivot_x - cos_a * pivot_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        transform_matrix = (
            translation_matrix
            @ rotate_about_pivot
            @ self._qtransform_to_matrix(base_transform)
        )
        return self._matrix_to_qtransform(transform_matrix)

    @staticmethod
    def _compute_valid_image_center(image_data: np.ndarray) -> tuple[float, float]:
        """Return the centroid of the finite-data footprint in image coordinates."""
        valid_mask = np.isfinite(image_data)
        if np.any(valid_mask):
            valid_x, valid_y = np.where(valid_mask)
            # Use the sample-footprint centroid instead of the bounding-box center.
            # Irregular scan outlines otherwise feel like they rotate around an edge.
            center_x = float(np.mean(valid_x.astype(np.float64) + 0.5))
            center_y = float(np.mean(valid_y.astype(np.float64) + 0.5))
            return center_x, center_y
        return image_data.shape[0] / 2.0, image_data.shape[1] / 2.0

    @staticmethod
    def _build_base_image_transform(
        surface: Surface,
        image_data: np.ndarray,
        *,
        stride: int = 1,
        normalize_origin: bool = True,
    ) -> QtGui.QTransform:
        """Map one image grid into centered local physical coordinates."""
        effective_dx = float(surface.dx) * max(1, int(stride))
        effective_dy = float(surface.dy) * max(1, int(stride))
        local_width = float(image_data.shape[0])
        local_height = float(image_data.shape[1])
        transform = QtGui.QTransform()
        if normalize_origin:
            center_x, center_y = OverlayViewer._compute_valid_image_center(image_data)
            transform.translate(
                -(center_x * effective_dx),
                -(center_y * effective_dy),
            )
        else:
            transform.translate(
                float(surface.x0) - (effective_dx / 2.0),
                float(surface.y0) - (effective_dy / 2.0),
            )
        transform.scale(effective_dx, effective_dy)
        return transform

    @staticmethod
    def _qtransform_to_matrix(transform: QtGui.QTransform) -> np.ndarray:
        """Convert one Qt transform into a homogeneous 3x3 matrix."""
        return np.array(
            [
                [transform.m11(), transform.m21(), transform.m31()],
                [transform.m12(), transform.m22(), transform.m32()],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _matrix_to_qtransform(matrix: np.ndarray) -> QtGui.QTransform:
        """Convert one homogeneous 3x3 matrix into a Qt transform."""
        return QtGui.QTransform(
            float(matrix[0, 0]),
            float(matrix[1, 0]),
            float(matrix[2, 0]),
            float(matrix[0, 1]),
            float(matrix[1, 1]),
            float(matrix[2, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
            float(matrix[2, 2]),
        )

    @staticmethod
    def _qtransform_to_affine(transform: QtGui.QTransform) -> np.ndarray:
        """Convert a Qt transform into the inverse affine matrix for resampling."""
        matrix_3x3 = OverlayViewer._qtransform_to_matrix(transform)
        return np.linalg.inv(matrix_3x3)[0:2, 0:3]

    @staticmethod
    def _map_image_rect(
        image_data: np.ndarray,
        image_transform: QtGui.QTransform,
    ) -> QtCore.QRectF:
        """Return the scene rectangle covered by a transformed image."""
        return image_transform.mapRect(
            QtCore.QRectF(
                0.0,
                0.0,
                float(image_data.shape[0]),
                float(image_data.shape[1]),
            )
        )

    def _sync_overlay_view_to_reference(self) -> None:
        """Keep the overlay view framed to the reference scan extent."""
        self.viewbox.setRange(
            self._map_image_rect(self.scan1, self._reference_base_transform),
            padding=0.0,
        )
        self._update_rotation_pivot_marker()

    def _reference_scene_rotation_center(
        self,
        reference_scan: np.ndarray,
        reference_base_transform: QtGui.QTransform,
    ) -> QtCore.QPointF:
        """Return the reference-view center used as the manual rotation pivot."""
        if reference_scan is self._preview_scan1:
            reference_center = self._reference_center_preview
        elif reference_scan is self._display_scan1:
            reference_center = self._reference_center_display
        else:
            reference_center = self._reference_center_full
        return reference_base_transform.map(
            QtCore.QPointF(float(reference_center[0]), float(reference_center[1]))
        )

    def _update_rotation_pivot_marker(self) -> None:
        """Update the visible pivot marker used by manual rotation."""
        pivot = self._reference_scene_rotation_center(
            self.scan1,
            self._reference_base_transform,
        )
        self._pivot_vline.setPos(float(pivot.x()))
        self._pivot_hline.setPos(float(pivot.y()))
        self._pivot_marker.setData([float(pivot.x())], [float(pivot.y())])

    def _schedule_difference_updates(self) -> None:
        """Queue a lightweight preview and defer full redraw until interaction settles."""
        self._preview_timer.start()
        if (
            self.slider_tx.isSliderDown()
            or self.slider_ty.isSliderDown()
            or self.slider_angle.isSliderDown()
        ):
            self._full_refresh_timer.stop()
            return
        self._full_refresh_timer.start()

    def _render_difference(
        self,
        reference_scan: np.ndarray,
        moving_scan: np.ndarray,
        reference_base_transform: QtGui.QTransform,
        moving_base_transform: QtGui.QTransform,
        tx: float,
        ty: float,
        angle: float,
        output_buffer: np.ndarray,
    ) -> np.ndarray:
        """Transform the moving scan into the reference frame and return the signed difference."""
        from scipy.ndimage import map_coordinates

        rotation_pivot = self._reference_scene_rotation_center(
            reference_scan,
            reference_base_transform,
        )
        moving_transform = self._build_image_transform(
            tx,
            ty,
            angle,
            moving_base_transform,
            scene_rotation_center=rotation_pivot,
        )
        transform_xy = (
            np.linalg.inv(self._qtransform_to_matrix(moving_transform))
            @ self._qtransform_to_matrix(reference_base_transform)
        )

        ref_axis0, ref_axis1 = np.meshgrid(
            np.arange(reference_scan.shape[0], dtype=np.float64) + 0.5,
            np.arange(reference_scan.shape[1], dtype=np.float64) + 0.5,
            indexing='ij',
        )
        reference_points = np.stack(
            (
                ref_axis0.ravel(),
                ref_axis1.ravel(),
                np.ones(ref_axis0.size, dtype=np.float64),
            ),
            axis=0,
        )
        moving_points = transform_xy @ reference_points
        sampled = map_coordinates(
            moving_scan,
            [moving_points[0] - 0.5, moving_points[1] - 0.5],
            order=1,
            mode='constant',
            cval=np.nan,
        ).reshape(reference_scan.shape)
        np.copyto(output_buffer, sampled, casting='unsafe')

        return reference_scan - output_buffer

    @staticmethod
    def _compute_auto_levels(scan_diff: np.ndarray) -> tuple[float, float]:
        """Return automatic display levels centered on the mean difference."""
        finite_diff = scan_diff[np.isfinite(scan_diff)]
        if finite_diff.size < 1:
            return (-1.0, 1.0)

        center = float(np.mean(finite_diff))
        half_range = float(
            max(
                np.max(np.abs(finite_diff - center)),
                1e-3,
            )
        )
        return (center - half_range, center + half_range)

    @staticmethod
    def _slider_to_value(slider_value: int, limits: tuple[float, float]) -> float:
        """Map a slider position in ``[0, 1000]`` into the given value limits."""
        low, high = limits
        if high <= low:
            return low
        return low + (high - low) * (float(slider_value) / 1000.0)

    @staticmethod
    def _value_to_slider(value: float, limits: tuple[float, float]) -> int:
        """Map a value into the slider domain ``[0, 1000]`` using the given limits."""
        low, high = limits
        if high <= low:
            return 0
        normalized = (float(value) - low) / (high - low)
        return int(np.clip(np.round(normalized * 1000.0), 0, 1000))

    def _update_diff_slider_limits(
        self,
        scan_diff: np.ndarray,
        auto_levels: tuple[float, float],
    ) -> None:
        """Adapt center and half-range slider spans to the current difference data."""
        finite_diff = scan_diff[np.isfinite(scan_diff)]
        if finite_diff.size < 1:
            self._diff_center_slider_limits = (-1.0, 1.0)
            self._diff_half_range_slider_limits = (1e-3, 1.0)
            return

        data_min = float(np.min(finite_diff))
        data_max = float(np.max(finite_diff))
        auto_center = 0.5 * (auto_levels[0] + auto_levels[1])
        auto_half_range = max(0.5 * (auto_levels[1] - auto_levels[0]), 1e-3)
        span = max(data_max - data_min, auto_half_range * 2.0, 1e-3)
        center_margin = max(0.25 * span, auto_half_range)

        self._diff_center_slider_limits = (
            data_min - center_margin,
            data_max + center_margin,
        )
        self._diff_half_range_slider_limits = (
            1e-3,
            max(span * 1.5, auto_half_range * 2.0, 1.0),
        )

    def _sync_diff_controls_from_levels(self, levels: tuple[float, float]) -> None:
        """Mirror the effective display range into the simplified controls."""
        low = float(levels[0])
        high = float(levels[1])
        center = 0.5 * (low + high)
        half_range = max(0.5 * (high - low), 1e-3)
        self._updating_diff_controls = True
        try:
            self.diff_center_spin.setValue(center)
            self.diff_half_range_spin.setValue(max(half_range, self.diff_half_range_spin.minimum()))
            self.diff_center_slider.setValue(
                self._value_to_slider(center, self._diff_center_slider_limits)
            )
            self.diff_half_range_slider.setValue(
                self._value_to_slider(half_range, self._diff_half_range_slider_limits)
            )
        finally:
            self._updating_diff_controls = False

    def _apply_manual_diff_levels(self, levels: tuple[float, float]) -> None:
        """Store manual difference levels and apply them to the current view."""
        self._diff_use_auto_levels = False
        self._diff_manual_levels = (float(levels[0]), float(levels[1]))
        self.diff_view.setLevels(*self._diff_manual_levels)
        self._sync_diff_controls_from_levels(self._diff_manual_levels)

    def _reset_diff_levels_to_auto(self) -> None:
        """Return difference scaling to the current automatically estimated range."""
        self._diff_use_auto_levels = True
        self._diff_manual_levels = None
        if self._last_auto_diff_levels is not None:
            self.diff_view.setLevels(*self._last_auto_diff_levels)
            self._sync_diff_controls_from_levels(self._last_auto_diff_levels)

    def _apply_diff_controls_to_manual_levels(self) -> None:
        """Convert simplified controls into manual image levels."""
        center = float(self.diff_center_spin.value())
        half_range = max(float(self.diff_half_range_spin.value()), self.diff_half_range_spin.minimum())
        self._apply_manual_diff_levels((center - half_range, center + half_range))

    def _on_diff_center_changed(self, value: float) -> None:
        """Apply a new center value chosen from the simplified control panel."""
        if self._updating_diff_controls:
            return
        self._apply_diff_controls_to_manual_levels()

    def _on_diff_half_range_changed(self, value: float) -> None:
        """Apply a new half-range value chosen from the simplified control panel."""
        if self._updating_diff_controls:
            return
        self._apply_diff_controls_to_manual_levels()

    def _on_diff_center_slider_changed(self, slider_value: int) -> None:
        """Update the center spin box from the center slider."""
        if self._updating_diff_controls:
            return
        center = self._slider_to_value(slider_value, self._diff_center_slider_limits)
        self._updating_diff_controls = True
        try:
            self.diff_center_spin.setValue(center)
        finally:
            self._updating_diff_controls = False
        self._apply_diff_controls_to_manual_levels()

    def _on_diff_half_range_slider_changed(self, slider_value: int) -> None:
        """Update the half-range spin box from the half-range slider."""
        if self._updating_diff_controls:
            return
        half_range = self._slider_to_value(slider_value, self._diff_half_range_slider_limits)
        self._updating_diff_controls = True
        try:
            self.diff_half_range_spin.setValue(
                max(half_range, self.diff_half_range_spin.minimum())
            )
        finally:
            self._updating_diff_controls = False
        self._apply_diff_controls_to_manual_levels()

    def _toggle_advanced_histogram(self, checked: bool) -> None:
        """Show or hide the advanced histogram editor for the difference map."""
        self.diff_view.ui.histogram.setVisible(bool(checked))
        self.diff_advanced_toggle.setText(
            "Hide advanced histogram" if checked else "Advanced histogram"
        )

    def _on_diff_histogram_level_change_finished(self, _histogram_item) -> None:
        """Capture manual histogram edits so redraws preserve the chosen range."""
        levels = self.diff_view.getLevels()
        if levels is None:
            return
        self._apply_manual_diff_levels((float(levels[0]), float(levels[1])))

    def _display_difference(
        self,
        scan_diff: np.ndarray,
        max_abs: float,
        *,
        image_transform: QtGui.QTransform,
        update_histogram: bool,
        auto_downsample: bool,
    ) -> None:
        """Push the difference image to the viewer using the current color scale."""
        self._last_diff_image = scan_diff
        target_levels = self._compute_auto_levels(scan_diff)
        self._update_diff_slider_limits(scan_diff, target_levels)
        levels_to_apply = (
            self._diff_manual_levels
            if (not self._diff_use_auto_levels and self._diff_manual_levels is not None)
            else target_levels
        )

        self.diff_view.setImage(
            np.nan_to_num(scan_diff, nan=0.0),
            autoRange=False,
            autoLevels=False,
            levels=levels_to_apply,
            autoHistogramRange=False,
        )
        diff_image_item = self.diff_view.getImageItem()
        diff_image_item.setTransform(image_transform)
        diff_view_box = self.diff_view.getView()
        diff_rect = self._map_image_rect(scan_diff, image_transform)
        diff_view_box.setRange(diff_rect, padding=0.0)
        self._last_auto_diff_levels = target_levels
        if self._diff_use_auto_levels:
            self._sync_diff_controls_from_levels(target_levels)
        if update_histogram:
            self._update_difference_histogram(scan_diff, max_abs)

    def _update_difference_preview(self) -> None:
        """Refresh a smaller live-preview difference map while sliders are moving."""
        tx = float(self.slider_tx.value())
        ty = float(self.slider_ty.value())
        angle = float(self.slider_angle.value()) / 10.0
        scan_diff = self._render_difference(
            self._preview_scan1,
            self._preview_scan2,
            self._preview_reference_base_transform,
            self._preview_moving_base_transform,
            tx,
            ty,
            angle,
            self._preview_diff_buffer,
        )
        max_abs = max(abs(level) for level in self._compute_auto_levels(scan_diff))
        self._display_difference(
            scan_diff,
            max_abs,
            image_transform=self._preview_reference_base_transform,
            update_histogram=False,
            auto_downsample=False,
        )

    def update_difference_map(self):
        """Redraw the signed difference map for the current scan alignment."""
        self._preview_timer.stop()
        self._full_refresh_timer.stop()
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle = float(self.slider_angle.value()) / 10.0
        scan_diff = self._render_difference(
            self.scan1,
            self.scan2,
            self._reference_base_transform,
            self._moving_base_transform,
            float(tx),
            float(ty),
            angle,
            self._full_diff_buffer,
        )
        max_abs = max(abs(level) for level in self._compute_auto_levels(scan_diff))
        self._display_difference(
            scan_diff,
            max_abs,
            image_transform=self._reference_base_transform,
            update_histogram=True,
            auto_downsample=True,
        )
        return

        # QTransform do macierzy 3x3
        qt_transform = self.img2.transform()
        m = np.array([
            [qt_transform.m11(), qt_transform.m21(), qt_transform.m31()],
            [qt_transform.m12(), qt_transform.m22(), qt_transform.m32()],
            [0,                 0,                 1]
        ])

        # Zamień na 2x3 do affine_transform (odwrócona!)
        affine_matrix = np.linalg.inv(m)[0:2, 0:3]

        # Przekształć obraz
        scan2_trans = affine_transform(
            self.scan2,
            matrix=affine_matrix[:, :2],
            offset=affine_matrix[:, 2],
            order=1,
            mode='constant',
            cval=np.nan
        )

        # Zabezpieczenie na różne wymiary
        h = min(self.scan1.shape[0], scan2_trans.shape[0])
        w = min(self.scan1.shape[1], scan2_trans.shape[1])

        scan1_cropped = self.scan1[:h, :w]
        scan2_trans_cropped = scan2_trans[:h, :w]

        scan_diff = scan1_cropped - scan2_trans_cropped
        finite_diff = scan_diff[np.isfinite(scan_diff)]
        if finite_diff.size > 0:
            max_abs = float(np.max(np.abs(finite_diff)))
        else:
            max_abs = 1.0
        if max_abs <= 0.0:
            max_abs = 1.0

        self.diff_view.setImage(
            np.nan_to_num(scan_diff, nan=0.0),
            autoLevels=False,
            levels=(-max_abs, max_abs),
        )

        image_item = self.diff_view.getImageItem()
        diff_cmap = get_colormap("difference")
        image_item.setLookupTable(diff_cmap.getLookupTable(0.0, 1.0, 512))
        self.diff_view.setColorMap(diff_cmap)
        self._update_difference_histogram(scan_diff, max_abs)

    def _update_difference_histogram(self, scan_diff: np.ndarray, max_abs: float):
        """Render difference histogram with individually colored bins."""
        hist_item = self.diff_view.ui.histogram.item
        hist_item.plot.setPen(pg.mkPen(255, 255, 255, 110))
        hist_item.plot.setBrush(None)

        if self._diff_hist_bars is not None:
            hist_item.vb.removeItem(self._diff_hist_bars)
            self._diff_hist_bars = None

        finite_diff = scan_diff[np.isfinite(scan_diff)]
        if finite_diff.size == 0:
            return

        y, x = np.histogram(finite_diff, bins=256, range=(-max_abs, max_abs))
        centers = 0.5 * (x[:-1] + x[1:])
        widths = np.diff(x)
        normalized = (centers + max_abs) / (2.0 * max_abs)
        brushes = get_brushes_for_values("difference", normalized)

        self._diff_hist_bars = pg.BarGraphItem(
            x=centers,
            height=y,
            width=widths,
            brushes=brushes,
            pen=pg.mkPen(255, 255, 255, 40),
        )
        hist_item.vb.addItem(self._diff_hist_bars)

    # def update_levels(self):
    #     if self._last_diff_image is None:
    #         return
    #     vmin = self.levels_min.value()
    #     vmax = self.levels_max.value()
    #     # Maskowanie: wszystko poza zakresem ustaw jako NaN
    #     masked = self._last_diff_image.copy()
    #     mask = (masked < vmin) | (masked > vmax)
    #     masked[mask] = np.nan
    #     self.diff_view.setImage(np.nan_to_num(masked, nan=0), autoLevels=False)

    # def update_overlay_levels(self):
    #     vmin = self.levels_min.value()
    #     vmax = self.levels_max.value()
    #     self.img1.setLevels((vmin, vmax))
    #     self.img2.setLevels((vmin, vmax))

    def apply_overlay_mask(self):
        # vmin = self.levels_min.value()
        # vmax = self.levels_max.value()
        masked1 = self._orig_scan1.copy()
        masked2 = self._orig_scan2.copy()
        masked1[(masked1 < self.vmin1) | (masked1 > self.vmax1)] = np.nan
        masked2[(masked2 < self.vmin2) | (masked2 > self.vmax2)] = np.nan
        self.img1.setImage(masked1, autoLevels=False)
        self.img2.setImage(masked2, autoLevels=False)
        self.img1.setLevels((self.vmin1, self.vmax1))
        self.img2.setLevels((self.vmin2, self.vmax2))


    def updateTransform(self):
        """Apply the current slider transform to the moving overlay and refresh previews."""
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle = float(self.slider_angle.value())/10.0

        self.label_tx.setText(f"X: {tx}")
        self.label_ty.setText(f"Y: {ty}")
        self.label_angle.setText(f"Angle: {angle}°")

        transform = self._build_image_transform(
            tx,
            ty,
            angle,
            self._moving_base_transform,
            scene_rotation_center=self._reference_scene_rotation_center(
                self.scan1,
                self._reference_base_transform,
            ),
        )
        self.img2.setTransform(transform)
        self._sync_overlay_view_to_reference()
        self._schedule_difference_updates()
        return

        transform = QtGui.QTransform()
        transform.translate(tx + cx, ty + cy)  # przesuń do środka
        transform.rotate(angle)                # obrót wokół środka
        transform.translate(-cx, -cy)          # wróć na miejsce

        self.img2.setTransform(transform)

    @staticmethod
    def _crop_to_mask_bounds(grid: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Crop grid to the bounding box defined by an ROI mask."""
        if mask is None or not np.any(mask):
            return None
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        return grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    def _prepare_auto_registration_inputs(self, method: str):
        """Prepare scan fragments for automatic alignment proposal."""
        reference_grid = self.scan1
        moving_grid = self.scan2

        parent = self.parent()
        roi_controller = getattr(parent, "roi_controller", None) if parent is not None else None
        if roi_controller is not None:
            reference_mask = roi_controller.create_mask(
                *reference_grid.shape,
                tab=self.reference_tab,
            )
            moving_mask = roi_controller.create_mask(
                *moving_grid.shape,
                tab=self.moving_tab,
            )
            if isinstance(reference_mask, np.ndarray) and isinstance(moving_mask, np.ndarray):
                reference_roi = self._crop_to_mask_bounds(reference_grid, reference_mask)
                moving_roi = self._crop_to_mask_bounds(moving_grid, moving_mask)
                if reference_roi is not None and moving_roi is not None:
                    reference_grid = reference_roi
                    moving_grid = moving_roi

        if method == "correlation" and reference_grid.shape != moving_grid.shape:
            common_height = min(reference_grid.shape[0], moving_grid.shape[0])
            common_width = min(reference_grid.shape[1], moving_grid.shape[1])
            reference_grid = reference_grid[:common_height, :common_width]
            moving_grid = moving_grid[:common_height, :common_width]

        return reference_grid, moving_grid

    @staticmethod
    def _set_slider_value_with_range(slider: QtWidgets.QSlider, value: int):
        """Expand slider range if needed before assigning a new value."""
        if value < slider.minimum():
            slider.setMinimum(value)
        if value > slider.maximum():
            slider.setMaximum(value)
        slider.setValue(value)

    def _apply_auto_alignment(self, method: str):
        """Estimate alignment parameters and write them into manual controls."""
        from ...processing import auto_register_surfaces

        reference_grid, moving_grid = self._prepare_auto_registration_inputs(method)
        if np.all(np.isnan(reference_grid)) or np.all(np.isnan(moving_grid)):
            QtWidgets.QMessageBox.warning(
                self,
                "Automatic alignment",
                "The selected area does not contain enough valid data.",
            )
            return

        cursor_active = False
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            cursor_active = True
            params = auto_register_surfaces(
                reference_grid,
                moving_grid,
                method=method,
                max_iterations=25,
                refine=False,
            )
            QtWidgets.QApplication.restoreOverrideCursor()
            cursor_active = False

            dy, dx = params.get("translation", (0.0, 0.0))
            rotation = params.get("rotation", 0.0)
            reference_scale_x = abs(self._preview_reference_base_transform.m11())
            reference_scale_y = abs(self._preview_reference_base_transform.m22())
            self._set_slider_value_with_range(
                self.slider_tx,
                int(np.round(dx * reference_scale_x)),
            )
            self._set_slider_value_with_range(
                self.slider_ty,
                int(np.round(dy * reference_scale_y)),
            )
            self._set_slider_value_with_range(self.slider_angle, int(np.round(rotation * 10.0)))

            msg = (
                f"Suggested translation: ({dx:.2f}, {dy:.2f}) px\n"
                f"Suggested rotation: {rotation:.2f}°\n"
                f"RMSE: {params.get('rmse', np.nan):.2f}"
            )
            QtWidgets.QMessageBox.information(self, "Automatic alignment", msg)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Automatic alignment",
                f"Automatic alignment failed:\n{exc}",
            )
        finally:
            if cursor_active:
                QtWidgets.QApplication.restoreOverrideCursor()

    def apply_auto_icp(self):
        """Estimate translation and rotation alignment and update sliders."""
        self._apply_auto_alignment("icp")
