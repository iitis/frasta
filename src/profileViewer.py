import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from skimage.draw import line
from PyQt5.QtCore import QPointF
from math import atan, degrees
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression

from .grid3DViewer import show_3d_viewer
from .helpers import remove_relative_offset, remove_relative_tilt

import logging
logger = logging.getLogger(__name__)

def create_image_view():
    view = pg.ImageView()
    view.ui.histogram.hide()
    view.ui.roiBtn.hide()
    view.ui.menuBtn.hide()
    # view.getView().setBackgroundColor('w')
    return view

from PyQt5.QtCore import QThread, pyqtSignal

class ProfileWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, sigma):
        super().__init__()
        self.filepath = filepath
        self.sigma = sigma

    def run(self):
        try:
            with h5py.File(self.filepath, "r") as f:
                reference_grid = f["scan1"][:]
                adjusted_grid = f["scan2"][:]
            
            # reference_grid_smooth = gaussian_filter(reference_grid, sigma=self.sigma)
            # adjusted_grid_smooth = gaussian_filter(adjusted_grid, sigma=self.sigma)
            reference_grid_smooth = reference_grid
            adjusted_grid_smooth = adjusted_grid
            
            valid_mask = ~np.isnan(reference_grid_smooth) & ~np.isnan(adjusted_grid_smooth)
            offset_correction = np.nanmean(reference_grid_smooth - adjusted_grid_smooth)
            adjusted_grid_corrected = adjusted_grid_smooth + offset_correction
            # Gotowe, zwróć wszystko w dict
            result = {
                "reference_grid": reference_grid,
                "adjusted_grid": adjusted_grid,
                "reference_grid_smooth": reference_grid_smooth,
                "adjusted_grid_smooth": adjusted_grid_smooth,
                "valid_mask": valid_mask,
                "adjusted_grid_corrected": adjusted_grid_corrected,
            }
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + '\n' + traceback.format_exc())



class ProfileViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive cross-sectional analysis")
        self.setGeometry(100, 100, 1000, 600)

        # --- PARAMETRY, metadane i domyślna ścieżka ---

        # domyslnie w mikrometrach
        self.ref_pixel_um = QPointF(1.0, 1.0)
        self.adj_pixel_um = QPointF(1.0, 1.0)

        self.sigma = 5.0
        self.separation = 0

        self.binary_contact = None
        self._preview_win = None

        menubar = self.menuBar()
        # file_menu = menubar.addMenu('File')
        # self.open_action = QtWidgets.QAction('Open...', self)
        # self.open_action.triggered.connect(self.load_new_data)
        # file_menu.addAction(self.open_action)

        view_menu = menubar.addMenu('View')
        self.open_3d_action = QtWidgets.QAction('Show 3D view', self)
        self.open_3d_action.triggered.connect(self.show_3d_view)
        view_menu.addAction(self.open_3d_action)
        view_menu.addSeparator()
        self.exit_action = QtWidgets.QAction('Exit', self)
        self.exit_action.triggered.connect(self.close)
        view_menu.addAction(self.exit_action)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)

        # Środkowa kolumna – wykres i suwaki
        center_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, 1))
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        center_layout.addWidget(self.plot_widget)

        # Prawa kolumna – binary image
        right_layout = QtWidgets.QVBoxLayout()

        self.image_view = create_image_view() # SnapImageWidget()

        self.image_view.setMinimumWidth(400)
        right_layout.addWidget(self.image_view)
        vb = self.image_view.getView()
        vb.setRange(
            xRange=(0, 1000),
            # yRange=(0, self.reference_grid.shape[0]-1),
            padding=0
        )

        self.image_view.getView().mousePressEvent = self.on_image_click
        self.image_view.getView().mouseReleaseEvent = self.on_image_mouse_release
        self.image_view.getView().mouseMoveEvent = self.on_image_mouse_move
        self.image_view.getView().sigRangeChanged.connect(self.on_range_changed)

        sep_layout = QtWidgets.QHBoxLayout()
        self.spinbox_separation = QtWidgets.QSpinBox()
        self.spinbox_separation.setRange(-1000, 1000)
        self.spinbox_separation.setValue(self.separation)
        self.spinbox_separation.valueChanged.connect(self.update_plot)
        sep_layout.addWidget(QtWidgets.QLabel("Separation:"))
        sep_layout.addWidget(self.spinbox_separation)
        right_layout.addLayout(sep_layout)

        self.spinbox_window_mm = QtWidgets.QDoubleSpinBox()
        self.spinbox_window_mm.setRange(0.001, 5.0)
        self.spinbox_window_mm.setValue(0.5)
        self.spinbox_window_mm.setSingleStep(0.001)
        self.spinbox_window_mm.setDecimals(3)
        self.spinbox_window_mm.valueChanged.connect(self.update_plot)

        self.checkbox_snap = QtWidgets.QCheckBox("Snap to plot")
        self.checkbox_snap.setChecked(True)
        
        win_layout = QtWidgets.QHBoxLayout()
        win_layout.addWidget(QtWidgets.QLabel("Window size [mm]:"))
        win_layout.addWidget(self.spinbox_window_mm)
        win_layout.addWidget(self.checkbox_snap)

        right_layout.addLayout(win_layout)

        self.checkbox_tilt = QtWidgets.QCheckBox("Tilt correction")
        self.checkbox_tilt.setChecked(True)
        self.checkbox_tilt.stateChanged.connect(self.toggle_tilt)

        right_layout.addWidget(self.checkbox_tilt)

        # Dodanie do głównego layoutu
        layout.addLayout(center_layout)
        layout.addLayout(right_layout)

        self.line_drag_active = False
        self.line_drag_which = None  # "start" albo "end"

        self.cursor_lines = []
        self.annotations = []
        self.mytest = []

        self.image_marker = None
        self.saved_points = []
        self.saved_point_markers = []

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # self.statusBar().showMessage("Gotowy")


    def closeEvent(self, event):
        # Powiadom klasę nadrzędną, jeśli trzeba
        if hasattr(self.parent(), '_profile_viewer'):
            self.parent()._profile_viewer = None
        event.accept()

    def load_new_data(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select HDF5 file", "", "HDF5 files (*.h5);;All files (*)")
        if fname:
            self.load_data_from_file(fname)

    def load_data_from_file(self, fname):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.centralWidget().setEnabled(False)
        #self.open_action.setEnabled(False)
        self.statusBar().showMessage("Loading data...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.worker = ProfileWorker(fname, self.sigma)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_error(self, msg):
        self.progress_bar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Error during processing!")
        QtWidgets.QMessageBox.critical(self, "Error", "Error during data processing:\n" + msg)

    def on_worker_finished(self, result):
        self.centralWidget().setEnabled(True)
        #self.open_action.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")

        # Użyj set_data do ustawienia siatek i odświeżenia GUI
        self.set_data(
            result["reference_grid"],
            result["adjusted_grid"],
            self.ref_pixel_um.x(),
            self.ref_pixel_um.y(),
            self.adj_pixel_um.x(),
            self.adj_pixel_um.y()
        )

        QtWidgets.QApplication.restoreOverrideCursor()

    def set_data(self, grid1, grid2, px1_um, py1_um, px2_um, py2_um):
        self.reference_grid = grid1
        self.adjusted_grid = grid2

        self.ref_pixel_um = QPointF(px1_um, py1_um)
        self.adj_pixel_um = QPointF(px2_um, py2_um)

        # self.reference_grid_smooth = gaussian_filter(grid1, self.sigma)
        # self.adjusted_grid_smooth = gaussian_filter(grid2, self.sigma)
        self.reference_grid_smooth = grid1
        self.adjusted_grid_smooth = grid2

        self.valid_mask = ~np.isnan(self.reference_grid_smooth) & ~np.isnan(self.adjusted_grid_smooth)

        self.adjusted_grid_corrected = self.adjusted_grid_smooth + np.nanmean(self.reference_grid_smooth - self.adjusted_grid_smooth)

        if self.checkbox_tilt.isChecked():
            self.adjusted_grid_corrected = remove_relative_tilt(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)

        self.adjusted_grid_corrected = remove_relative_offset(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)

        size_x_mm = self.reference_grid.shape[1] * self.ref_pixel_um.x() / 1000.0
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))

        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Gotowy")

        # Reset ROI i odśwież GUI
        height, width = self.reference_grid_smooth.shape
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = width - 1, height - 1

        self.redraw_roi()
        
        shape = self.update_plot()

        self.resize_image_view(shape)

        vb = self.image_view.getView()
        vb.setAspectLocked(True)

        vb.setLimits( 
            yMin=0, yMax=shape[0]-1,
            xMin=0, xMax=shape[1]-1 
        )
        
        vb.setRange(
            xRange=(0, shape[1]-1),
            yRange=(0, shape[0]-1),
            padding=0
        )

        QtWidgets.QApplication.restoreOverrideCursor()





    def show_preview(self, fragment, title="Region preview"):
        if getattr(self, "_preview_win", None) is None:
            self._preview_win = pg.ImageView()
            self._preview_win.setWindowTitle(title)
            self._preview_win.show()
        self._preview_win.setImage(fragment)
        self._preview_win.raise_()
        self._preview_win.activateWindow()


    def on_image_click(self, event):
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            pos = event.scenePos()
            vb = self.image_view.getView()
            mouse_point = vb.mapSceneToView(pos)
            x_img = int(round(mouse_point.x()))
            y_img = int(round(mouse_point.y()))
            img_shape = self.reference_grid_smooth.shape

            # Po kliknięciu przesuwaj uchwyt [0] natychmiast:
            self.x1 = np.clip(x_img, 0, img_shape[1]-1)
            self.y1 = np.clip(y_img, 0, img_shape[0]-1)
            self.x2 = np.clip(x_img, 0, img_shape[1]-1)
            self.y2 = np.clip(y_img, 0, img_shape[0]-1)
            # Uchwyt [1] zostaje bez zmian (albo podąża za myszą)
            self.redraw_roi()
            self.update_profile_from_roi()
            # Wejdź w tryb "drag" dla drugiego uchwytu
            self.line_drag_active = True
            event.accept()
        else:
            pg.ViewBox.mousePressEvent(self.image_view.getView(), event)

    def on_image_mouse_release(self, event):
        if self.line_drag_active:
            self.line_drag_active = False
            event.accept()
        else:
            pg.ViewBox.mouseReleaseEvent(self.image_view.getView(), event)

    def on_image_mouse_move(self, event):
        if self.line_drag_active:
            pos = event.scenePos()
            vb = self.image_view.getView()
            mouse_point = vb.mapSceneToView(pos)
            x_img = int(round(mouse_point.x()))
            y_img = int(round(mouse_point.y()))
            img_shape = self.reference_grid_smooth.shape
            self.x2 = np.clip(x_img, 0, img_shape[1] - 1)
            self.y2 = np.clip(y_img, 0, img_shape[0] - 1)
            self.redraw_roi()
            self.update_profile_from_roi()
            event.accept()
        else:
            pg.ViewBox.mouseMoveEvent(self.image_view.getView(), event)


    def show_3d_view(self):
        viewbox = self.image_view.getView()
        x_range, y_range = viewbox.viewRange()

        # Zamień zakresy na indeksy obrazka
        x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1]))

        # Upewnij się, że są w granicach obrazka
        shape = self.reference_grid_smooth.shape
        x_min = max(0, x_min)
        x_max = min(shape[1] - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(shape[0] - 1, y_max)

        # Wytnij wycinek z siatek
        ref = self.reference_grid_smooth[y_min:y_max + 1, x_min:x_max + 1]
        adj = self.adjusted_grid_corrected[y_min:y_max + 1, x_min:x_max + 1]

        logger.debug(f"ref0: {self.reference_grid_smooth.shape}, adj0: {self.adjusted_grid_corrected.shape}")
        logger.debug(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
        logger.debug(f"ref min: {np.nanmin(ref)}, ref max: {np.nanmax(ref)}, ref shape: {ref.shape}")
        logger.debug(f"ref NaN count: {np.isnan(ref).sum()}")
        logger.debug(f"adj min: {np.nanmin(adj)}, adj max: {np.nanmax(adj)}, adj shape: {adj.shape}")
        logger.debug(f"adj NaN count: {np.isnan(adj).sum()}")

        # Wyznacz linię profilu (ograniczoną do wycinka)
        if hasattr(self, 'rr') and hasattr(self, 'cc'):
            line_points = [
                (int(col - x_min), int(row - y_min))
                for col, row in zip(self.cc, self.rr)
                if x_min <= col <= x_max and y_min <= row <= y_max
            ]
            if len(line_points) < 2:
                line_points = None
        else:
            line_points = None

        show_3d_viewer(reference_grid=ref,
            adjusted_grid=adj,
            line_points=line_points,
            separation=self.separation,
            show_controls=True)


    def get_viewbox_ranges_int(self, shape=None, overflow=False):
        viewbox = self.image_view.getView()
        x_range, y_range = viewbox.viewRange()

        min_range = viewbox.mapToParent(QPointF(x_range[0],y_range[0]))
        max_range = viewbox.mapToParent(QPointF(x_range[1],y_range[1]))

        x_range = [min_range.x(),max_range.x()]
        y_range = [min_range.y(),max_range.y()]

        logger.debug(f"ViewBox x_range: {x_range}, y_range: {y_range}")

        if overflow:
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))-1
            y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1]))-1
        else:
            x_min, x_max = int(np.ceil(x_range[0])), int(np.floor(x_range[1]))-1
            y_min, y_max = int(np.ceil(y_range[0])), int(np.floor(y_range[1]))-1

        if shape is not None:
            x_min = max(0, x_min)
            x_max = min(shape[1]-1, x_max)
            y_min = max(0, y_min)
            y_max = min(shape[0]-1, y_max)

        # print(f"Image x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

        return x_min, x_max, y_min, y_max            

    def update_volume_info(self):
        if not self.binary_contact is None:
            x_min, x_max, y_min, y_max = self.get_viewbox_ranges_int(shape = self.binary_contact.shape)

            px_um = self.ref_pixel_um.x()
            py_um = self.ref_pixel_um.y()

            pixel_area_um2 = px_um * py_um

            fragment = self.binary_contact[y_min:y_max+1, x_min:x_max+1]
            
            # print(f"fragment.shape: {fragment.shape}")

            white_count = np.count_nonzero(fragment)

            white_area_um2 = pixel_area_um2 * white_count
            white_area_mm2 = white_area_um2 * 1e-6

            ref = self.reference_grid_smooth[y_min:y_max+1, x_min:x_max+1]
            adj = self.adjusted_grid_corrected[y_min:y_max+1, x_min:x_max+1]
            diff = ref - (adj + self.separation)

            diff_masked = np.where(fragment, diff, 0)

            volume_um3 = np.abs(np.sum(diff_masked)) * pixel_area_um2
            volume_mm3 = volume_um3 * 1e-9

            self.statusBar().showMessage(
                f"White fields in view: {white_count}, area: {white_area_um2:.4f}μm² ({white_area_mm2}mm²), volume: {volume_um3:.4f}μm³ ({volume_mm3:.4f}mm³)"
            )


    def on_range_changed(self, viewbox, ranges):
        self.update_volume_info()

    def toggle_tilt(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.centralWidget().setEnabled(False)
        #self.open_action.setEnabled(False)
        offset_correction = np.nanmean(self.reference_grid_smooth - self.adjusted_grid_smooth)
        self.adjusted_grid_corrected = self.adjusted_grid_smooth + offset_correction
        if self.checkbox_tilt.isChecked():
            self.adjusted_grid_corrected = remove_relative_tilt(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)
        self.adjusted_grid_corrected = remove_relative_offset(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)

        self.redraw_roi()
        self.update_plot()

        self.centralWidget().setEnabled(True)
        #self.open_action.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

        
    def resize_image_view(self, shape):
        # shape = (height, width)
        height, width = shape
        aspect = width / height
        # bazowy wymiar, np. 700
        base = 500
        if aspect >= 1.0:
            w = base
            h = int(base / aspect)
        else:
            h = base
            w = int(base * aspect)
        self.image_view.setFixedSize(w, h)
        self.image_view.update()
        self.updateGeometry()

    def update_plot(self):
        # Zaktualizuj widoki obrazów
        self.separation = self.spinbox_separation.value()
        valid_mask = ~np.isnan(self.reference_grid_smooth) & ~np.isnan(self.adjusted_grid_corrected)
        difference = self.reference_grid_smooth - (self.adjusted_grid_corrected + self.separation)
        # binary_contact = (difference <= 0) & valid_mask
        binary_contact = (difference > 0) & valid_mask

        self.image_view.setImage(binary_contact.T.astype(np.uint8), autoRange=False, autoLevels=True)

        self.update_profile_from_roi()

        self.binary_contact = binary_contact

        self.update_volume_info()

        return binary_contact.shape

    def update_roi_markers(self):
        # Usuń stare markery (jeśli są)
        if hasattr(self, "roi_endpoint_markers"):
            for m in self.roi_endpoint_markers:
                self.image_view.getView().removeItem(m)
        self.roi_endpoint_markers = []
        if hasattr(self, "roi_endpoint_labels"):
            for t in self.roi_endpoint_labels:
                self.image_view.getView().removeItem(t)
        self.roi_endpoint_labels = []
        
        # Pobierz BIEŻĄCE pozycje końców ROI w układzie obrazka!
        handle0 = self.line_roi.getHandles()[0]
        handle1 = self.line_roi.getHandles()[1]
        pt0 = self.line_roi.mapToParent(handle0.pos())
        pt1 = self.line_roi.mapToParent(handle1.pos())
        x1, y1 = pt0.x(), pt0.y()
        x2, y2 = pt1.x(), pt1.y()
        
        # Dodaj markery
        marker1 = pg.ScatterPlotItem([x1], [y1], size=18, pen=pg.mkPen('g', width=3), brush=pg.mkBrush(0,255,0,100), symbol='o')
        marker2 = pg.ScatterPlotItem([x2], [y2], size=18, pen=pg.mkPen('r', width=3), brush=pg.mkBrush(255,0,0,100), symbol='x')
        self.image_view.getView().addItem(marker1)
        self.image_view.getView().addItem(marker2)
        self.roi_endpoint_markers = [marker1, marker2]
        
        # Opcjonalnie: etykiety z numerkami
        label1 = pg.TextItem("1", color='g', anchor=(0.5, 1.5))
        label1.setPos(x1, y1)
        label2 = pg.TextItem("2", color='r', anchor=(0.5, 1.5))
        label2.setPos(x2, y2)
        self.image_view.getView().addItem(label1)
        self.image_view.getView().addItem(label2)
        self.roi_endpoint_labels = [label1, label2]

    def redraw_roi(self):
        if hasattr(self, 'line_roi'):
            self.image_view.getView().removeItem(self.line_roi)
        self.line_roi = pg.LineROI([self.x1, self.y1], [self.x2, self.y2], pen=pg.mkPen('r', width=2), width=1)
        self.line_roi.handles[2]['type'] = 'center'
        self.line_roi.sigRegionChanged.connect(self.update_profile_from_roi)
        self.line_roi.sigRegionChanged.connect(self.update_roi_markers)  # <-- dodaj to!
        self.image_view.getView().addItem(self.line_roi)
        self.line_roi.setZValue(10)
        self.update_roi_markers()  # <-- narysuj od razu w dobrym miejscu


    def clamp_roi_to_image(self):
        img_shape = self.reference_grid_smooth.shape  # (rows, cols)
        h1 = self.line_roi.getHandles()[0].pos()
        h2 = self.line_roi.getHandles()[1].pos()
        pos1 = self.line_roi.mapToParent(h1).toPoint()
        pos2 = self.line_roi.mapToParent(h2).toPoint()
        self.x1 = min(max(pos1.x(), 0), img_shape[1] - 1)
        self.y1 = min(max(pos1.y(), 0), img_shape[0] - 1)
        self.x2 = min(max(pos2.x(), 0), img_shape[1] - 1)
        self.y2 = min(max(pos2.y(), 0), img_shape[0] - 1)
        if (pos1.x(), pos1.y(), pos2.x(), pos2.y()) != (self.x1, self.y1, self.x2, self.y2):
            self.redraw_roi()

    def update_profile_from_roi(self):
        self.clamp_roi_to_image()
        rr, cc = line(self.y1, self.x1, self.y2, self.x2)
        rr = np.clip(rr, 0, self.reference_grid_smooth.shape[0] - 1)
        cc = np.clip(cc, 0, self.reference_grid_smooth.shape[1] - 1)
        
        profile_ref = self.reference_grid_smooth[rr, cc]
        profile_adj = (self.adjusted_grid_corrected + self.separation)[rr, cc]
        valid_profile_mask = ~np.isnan(profile_ref) & ~np.isnan(profile_adj)
        
        self.rr = rr[valid_profile_mask]
        self.cc = cc[valid_profile_mask]
        profile_ref = profile_ref[valid_profile_mask]
        profile_adj = profile_adj[valid_profile_mask]
        
        positions_line = np.arange(len(rr))[valid_profile_mask] * self.ref_pixel_um.x() / 1000.0
        
        self.positions_line = positions_line
        self.reference_profile = profile_ref
        self.adjusted_profile = profile_adj
        
        self.plot_widget.clear()
        self.plot_widget.plot(positions_line, self.reference_profile, pen=pg.mkPen('g', width=2))
        self.plot_widget.plot(positions_line, self.adjusted_profile, pen=pg.mkPen('b', width=2))

    def print_saved_points(self):
        for i, pt in enumerate(self.saved_points):
            logger.debug(f"{i+1}: {pt}")

    def on_plot_click(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            self._handle_ctrl_click(event)

    def _handle_ctrl_click(self, event):
        pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_pos = mouse_point.x()
        if not (self.positions_line[0] <= x_pos <= self.positions_line[-1]):
            return
        idx = np.argmin(np.abs(self.positions_line - x_pos))
        if hasattr(self, 'rr') and hasattr(self, 'cc'):
            self._save_profile_point(idx)

    def _save_profile_point(self, idx):
        y_img = self.rr[idx]
        x_img = self.cc[idx]
        ref_val = self.reference_profile[idx]
        adj_val = self.adjusted_profile[idx]
        pos_mm = self.positions_line[idx]
        self.saved_points.append({
            'profile_idx': idx,
            'x_img': int(x_img),
            'y_img': int(y_img),
            'x_pos_mm': float(pos_mm),
            'ref_val': float(ref_val),
            'adj_val': float(adj_val),
        })
        marker = pg.ScatterPlotItem([x_img], [y_img], size=12, pen=pg.mkPen('g', width=2), brush=pg.mkBrush(0, 255, 255, 120), symbol='+')
        self.image_view.getView().addItem(marker)
        self.saved_point_markers.append(marker)
        logger.debug("Saved point:", self.saved_points[-1])


    def on_mouse_move(self, pos):
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_pos = mouse_point.x()
        self._clear_cursor_and_annotations()
        self._draw_cursor_line(x_pos)
        positions_line = self.positions_line
        if positions_line[0] <= x_pos <= positions_line[-1]:
            idx = np.argmin(np.abs(positions_line - x_pos))
            self._update_image_marker(idx)
            self._draw_annotations_and_fit_lines(x_pos, idx)
        else:
            self._clear_fit_lines_and_marker()

    def _clear_cursor_and_annotations(self):
        for item in self.cursor_lines + self.annotations:
            self.plot_widget.removeItem(item)
        self.cursor_lines.clear()
        self.annotations.clear()

    def _draw_cursor_line(self, x_pos):
        vline = pg.InfiniteLine(pos=x_pos, angle=90, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(vline)
        self.cursor_lines.append(vline)

    def _update_image_marker(self, idx):
        if hasattr(self, 'rr') and hasattr(self, 'cc'):
            y_img = self.rr[idx]
            x_img = self.cc[idx]
            view = self.image_view.getView()
            if self.image_marker is not None:
                view.removeItem(self.image_marker)
            self.image_marker = pg.ScatterPlotItem([x_img], [y_img], size=14, pen=pg.mkPen('m', width=2), brush=pg.mkBrush(255, 0, 255, 100))
            view.addItem(self.image_marker)

    def _draw_annotations_and_fit_lines(self, x_pos, idx):
        height_diff = self.reference_profile[idx] - self.adjusted_profile[idx]
        window_mm = self.spinbox_window_mm.value()
        pixel_size_mm = self.ref_pixel_um.x() / 1000.0
        window_size = max(1, int(round(window_mm / pixel_size_mm)))
        start = max(0, idx - window_size)
        end = min(len(self.positions_line), idx + window_size + 1)

        # Fit lines and angles
        slope_ref, angle_ref, reg_ref = self._fit_profile(self.positions_line[start:end], self.reference_profile[start:end])
        slope_adj, angle_adj, reg_adj = self._fit_profile(self.positions_line[start:end], self.adjusted_profile[start:end])
        delta_angle = angle_ref - angle_adj

        # Draw text annotations
        self._draw_diff_and_angle_text(height_diff, angle_ref, angle_adj, delta_angle)

        # Draw fit lines
        self._draw_fit_lines(x_pos, slope_ref, reg_ref, slope_adj, reg_adj, idx, window_mm)

    def _fit_profile(self, x, y):
        x_fit = x.reshape(-1, 1)
        y_fit = y.reshape(-1, 1) / 1000.0
        reg = LinearRegression().fit(x_fit, y_fit)
        slope = reg.coef_[0][0]
        angle = degrees(atan(slope))
        return slope, angle, reg

    def _draw_diff_and_angle_text(self, height_diff, angle_ref, angle_adj, delta_angle):
        text1 = pg.TextItem(f"DIFF: {height_diff:.2f} μm", color='r', anchor=(0, 1))
        vb = self.plot_widget.getPlotItem().vb
        x_min, x_max = vb.viewRange()[0]
        y_min, y_max = vb.viewRange()[1]
        text1.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.05 * (y_max - y_min))
        self.plot_widget.addItem(text1)
        self.annotations.append(text1)
        text2 = pg.TextItem(f"ANGLE\nref: {angle_ref:.1f}°\nadj: {angle_adj:.1f}°\n  Δ: {delta_angle:.1f}°", color='y', anchor=(0, 1))
        text2.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.2 * (y_max - y_min))
        self.plot_widget.addItem(text2)
        self.annotations.append(text2)

    def _draw_fit_lines(self, x_pos, slope_ref, reg_ref, slope_adj, reg_adj, idx, window_mm):
        vb = self.plot_widget.getPlotItem().vb
        line_half_width_mm = window_mm / 2.0
        x0 = x_pos - line_half_width_mm
        x1 = x_pos + line_half_width_mm

        # Reference fit line
        a = slope_ref
        if self.checkbox_snap.isChecked():
            y_at_cursor = self.reference_profile[idx] / 1000.0
            b = y_at_cursor - a * x_pos
        else:
            b = reg_ref.intercept_[0]
        y0 = a * x0 + b
        y1 = a * x1 + b
        for item in self.mytest:
            vb.removeItem(item)
        self.mytest.clear()
        line_ref = pg.PlotDataItem([x0, x1], [y0 * 1000, y1 * 1000], pen=pg.mkPen('y', width=2))
        vb.addItem(line_ref, ignoreBounds=True)
        self.annotations.append(line_ref)
        self.mytest.append(line_ref)

        # Adjusted fit line
        a = slope_adj
        if self.checkbox_snap.isChecked():
            y_at_cursor_adj = self.adjusted_profile[idx] / 1000.0
            b_adj = y_at_cursor_adj - a * x_pos
        else:
            b_adj = reg_adj.intercept_[0]
        y0 = a * x0 + b_adj
        y1 = a * x1 + b_adj
        line_adj = pg.PlotDataItem([x0, x1], [y0 * 1000, y1 * 1000], pen=pg.mkPen('y', width=2))
        vb.addItem(line_adj, ignoreBounds=True)
        self.annotations.append(line_adj)
        self.mytest.append(line_adj)

    def _clear_fit_lines_and_marker(self):
        vb = self.plot_widget.getPlotItem().vb
        for item in self.mytest:
            vb.removeItem(item)
        self.mytest.clear()
        view = self.image_view.getView()
        if self.image_marker is not None:
            view.removeItem(self.image_marker)
            self.image_marker = None

