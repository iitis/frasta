"""Overlay viewer for aligning and comparing two scan datasets.

This module provides an interactive viewer for overlaying two scans, allowing
manual alignment through translation and rotation controls, and displaying
the difference map between aligned scans.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from ...core import GridData        

class OverlayViewer(QtWidgets.QWidget):
    """Interactive widget for overlaying and aligning two scan datasets.
    
    Provides side-by-side views of overlaid scans and their difference map,
    with interactive controls for translation, rotation, and visual comparison
    modes (blinking, transparency).
    
    Attributes:
        scan1_data (GridData): First scan dataset.
        scan2_data (GridData): Second scan dataset.
        img1 (pg.ImageItem): Image item for first scan.
        img2 (pg.ImageItem): Image item for second scan (transformable).
        diff_view (pg.ImageView): View displaying the difference map.
        on_accept (callable, optional): Callback when alignment is accepted.
    """
    
    def __init__(self, scan1_data: GridData, scan2_data: GridData, on_accept=None, parent=None):
        """Initialize the overlay viewer.
        
        Args:
            scan1_data (GridData): First scan to display (reference).
            scan2_data (GridData): Second scan to display (adjustable).
            on_accept (callable, optional): Callback function called when accepting alignment.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window)

        self.create_gui()

        self.on_accept = on_accept

        self.scan1_data = scan1_data
        self.scan2_data = scan2_data

        self.scan1 = scan1_data.grid
        self.scan2 = scan2_data.grid

        self._orig_scan1 = self.scan1.copy()
        self._orig_scan2 = self.scan2.copy()


        self._last_diff_image = None   # original difference (before masking)

        # Images
        self.img1 = pg.ImageItem(self.scan1)
        self.img2 = pg.ImageItem(self.scan2)
        self.img2.setOpacity(0.5)

        self.viewbox.addItem(self.img1)
        self.viewbox.addItem(self.img2)
        self.viewbox.autoRange()

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

        self.updateTransform()

    def closeEvent(self, event):
        p = self.parent()
        if p and hasattr(p, "viewer"):
            p.viewer = None
        super().closeEvent(event)

    def create_gui(self):
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Main window and scene
        self.view = pg.GraphicsLayoutWidget()
        self.viewbox = self.view.addViewBox()
        self.viewbox.setAspectLocked(True)

        # Difference image
        self.diff_view = pg.ImageView(view=pg.PlotItem())
        #self.diff_view.ui.histogram.hide()
        self.diff_view.ui.roiBtn.hide()
        self.diff_view.ui.menuBtn.hide()
        self.diff_view.setMinimumWidth(500)

        # Horizontal layout: main view + diff view
        split = QtWidgets.QHBoxLayout()
        split.addWidget(self.view)
        split.addWidget(self.diff_view)
        layout.addLayout(split)

        south_layout = QtWidgets.QHBoxLayout()

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

        tool2_layout.addWidget(self.checkbox_visible)
        tool2_layout.addWidget(self.checkbox_trans)
        tool2_layout.addWidget(self.checkbox_blink)

        # Blink timer
        self.blink_timer = QtCore.QTimer()
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self.blinkToggle)
        self.blink_state = True

        self.update_diff_btn = QtWidgets.QPushButton("Redraw difference view")
        self.update_diff_btn.clicked.connect(self.update_difference_map)
        tool2_layout.addWidget(self.update_diff_btn)

        # self.save_button = QtWidgets.QPushButton("Save aligned grids to .h5")
        # self.save_button.clicked.connect(self.saveAlignedScans)
        # tool2_layout.addWidget(self.save_button)

        self.accept_button = QtWidgets.QPushButton("Accept changes")
        self.accept_button.clicked.connect(self.accept_result)
        tool2_layout.addWidget(self.accept_button)

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

        layout.addLayout(south_layout)


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

        aligned1 = self.original_scan1[:h, :w]
        aligned2 = self.original_scan2[:h, :w]

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
        from scipy.ndimage import affine_transform

        qt_transform = self.img2.transform()
        m = np.array([
            [qt_transform.m11(), qt_transform.m21(), qt_transform.m31()],
            [qt_transform.m12(), qt_transform.m22(), qt_transform.m32()],
            [0,                 0,                 1]
        ])
        affine_matrix = np.linalg.inv(m)[0:2, 0:3]

        scan2_trans = affine_transform(
            self.scan2,
            matrix=affine_matrix[:, :2],
            offset=affine_matrix[:, 2],
            order=1,
            mode='constant',
            cval=np.nan
        )

        h = min(self.scan1.shape[0], scan2_trans.shape[0])
        w = min(self.scan1.shape[1], scan2_trans.shape[1])

        data1 = GridData(
            grid=self.scan1[:h, :w],
            xi=self.scan1_data.xi[:w],
            yi=self.scan1_data.yi[:h],
            px_x=self.scan1_data.px_x,
            px_y=self.scan1_data.px_y,
            vmin=self.vmin1,
            vmax=self.vmax1
        )

        data2 = GridData(
            grid=scan2_trans[:h, :w],
            xi=self.scan1_data.xi[:w],  # uwaga: zakładamy, że scan2 został przetransformowany do układu scan1
            yi=self.scan1_data.yi[:h],
            px_x=self.scan1_data.px_x,
            px_y=self.scan1_data.px_y,
            vmin=self.vmin2,
            vmax=self.vmax2
        )

        if self.on_accept is not None:
            self.on_accept(data1, data2)

        self.close()


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

    def update_difference_map(self):
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()

        from scipy.ndimage import affine_transform

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

        self.diff_view.setImage(np.nan_to_num(scan_diff, nan=0), autoLevels=True)

        image_item = self.diff_view.getImageItem()
        lut = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 512)
        image_item.setLookupTable(lut)

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
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle = float(self.slider_angle.value())/10.0

        self.label_tx.setText(f"X: {tx}")
        self.label_ty.setText(f"Y: {ty}")
        self.label_angle.setText(f"Angle: {angle}°")

        h, w = self.original_scan2.shape
        cx, cy = w / 2, h / 2

        transform = QtGui.QTransform()
        transform.translate(tx + cx, ty + cy)  # przesuń do środka
        transform.rotate(angle)                # obrót wokół środka
        transform.translate(-cx, -cy)          # wróć na miejsce

        self.img2.setTransform(transform)


