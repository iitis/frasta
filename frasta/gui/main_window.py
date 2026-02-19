"""Main GUI module for FRASTA-toolbox application.

This module provides the main window and core functionality for loading, viewing,
processing, and analyzing 2D scan data. It includes features for:
- Loading scan data from various formats (CSV, NPZ, H5)
- Multi-tab interface for managing multiple scans
- ROI-based masking (circular and rectangular)
- Data processing (hole filling, Gaussian smoothing, offset/tilt correction)
- 3D visualization and profile analysis
- Scan comparison and overlay views
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from functools import partial
from PyQt5.QtSvg import QSvgRenderer


import sys
import os

from .dialogs import (ProfileViewer, OverlayViewer, AboutDialog,
                      FilterDialog, MorphologyDialog, TransformDialog, RegistrationDialog)
from .scan_tab import ScanTab
from ..core import Surface
from .viewers import show_3d_viewer
from ..utils import resource_path
from ..io import load_csv_data, load_npz_data, load_h5_data, load_stl_data, save_npz, save_h5, save_stl, suggest_units
from .workers import GridWorker

import logging
logger = logging.getLogger(__name__)

def svg_icon(path, size=24):
    renderer = QSvgRenderer(path)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    renderer.render(p)
    p.end()

    return QIcon(pm)

class MainWindow(QtWidgets.QMainWindow):
    """Main application window for the scan loader and hole filler tool.

    Provides a multi-tab interface for loading, viewing, processing, and saving 2D scan data. 
    Supports region-of-interest masking, 3D visualization, scan comparison, and profile analysis.
    """

    def __init__(self):
        """Initializes the main window and sets up the user interface.

        Sets up the tab widget, recent files, actions, menus, toolbar, and shared ROI for scan management.
        """
        super().__init__()
        self.setWindowTitle("FRASTA-toolbox")
        self.setGeometry(100, 100, 1000, 600)

        self.recent_files = []
        self.max_recent_files = 10
        self.settings = QtCore.QSettings("IITiS PAN", "FRASTA-toolbox")
        self.load_recent_files()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)

        self.create_actions()
        self.connect_actions()
        self.create_menubar()
        self.create_toolbar()

        self.shared_circle_roi = None  # będzie przechowywać jedną instancję CircleROI
        self.shared_rectangle_roi = None  # będzie przechowywać jedną instancję RectROI

        self.worker = None
        self.thread = None

    def _is_roi_valid_and_visible(self, roi):
        """Safely checks if a Qt ROI object is valid and visible.
        
        Args:
            roi: ROI object to check (CircleROI or RectROI)
            
        Returns:
            bool: True if ROI exists and is visible, False otherwise
        """
        if roi is None:
            return False
        try:
            return roi.isVisible()
        except RuntimeError:
            # Object has been deleted by Qt
            return False

    def _is_roi_deleted(self, roi):
        """Checks if a Qt ROI object has been deleted.
        
        Args:
            roi: ROI object to check
            
        Returns:
            bool: True if ROI has been deleted, False otherwise
        """
        if roi is None:
            return True
        try:
            # Try to access any property to check if object is valid
            _ = roi.isVisible()
            return False
        except RuntimeError:
            # Object has been deleted by Qt
            return True

        self._global_3d_viewer = None

        self.tabs.currentChanged.connect(self.move_roi_to_current_tab)


    def create_mask(self, h, w):
        """Creates a boolean mask for the currently active ROI (circle or rectangle).

        Determines which ROI is visible and generates the corresponding mask for the given shape.

        Args:
            h (int): Height of the mask (number of rows).
            w (int): Width of the mask (number of columns).

        Returns:
            np.ndarray or None: Boolean mask with True inside the ROI, or None if no ROI is active.
        """
        circle_visible = self._is_roi_valid_and_visible(self.shared_circle_roi)
        rect_visible = self._is_roi_valid_and_visible(self.shared_rectangle_roi)

        mask = None
        if circle_visible:
            pos = self.shared_circle_roi.pos()
            size = self.shared_circle_roi.size()
            cx = pos.x() + size[0]/2
            cy = pos.y() + size[1]/2
            r = size[0]/2
            mask = self.create_circle_mask((h, w), (cx, cy), r)
        elif rect_visible:
            pos = self.shared_rectangle_roi.pos()
            size = self.shared_rectangle_roi.size()
            cx = pos.x() + size[0]/2
            cy = pos.y() + size[1]/2
            width = size[0]
            height = size[1]
            mask = self.create_rectangle_mask((h, w), (cx, cy), width, height)
        return mask

    def apply_roi_mask(self, inside):
        """Applies a mask to the current tab's grid based on the active ROI.

        Generates a mask from the visible ROI and deletes values inside or outside the mask, depending on the 'inside' flag.

        Args:
            inside (bool): If True, deletes values inside the mask; if False, deletes values outside the mask.
        """
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            return

        h, w = tab.grid.shape

        mask = self.create_mask(h,w)

        if mask is None:
            return
        
        if inside:
            tab.delete_unmasked(~mask)
        else:
            tab.delete_unmasked(mask)

    def del_inside_mask(self):
        self.apply_roi_mask(True)

    def del_outside_mask(self):
        self.apply_roi_mask(False)

    def move_roi_to_current_tab(self, idx):
        """Moves the shared ROI (circle or rectangle) to the currently selected tab.

        Ensures that only the active ROI is visible on the current tab and removed from all others.

        Args:
            idx (int): Index of the newly selected tab.
        """
        # Move circle ROI if it exists and is visible
        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.image_view.getView().removeItem(self.shared_circle_roi)
            tab = self.tabs.widget(idx)
            tab.image_view.getView().addItem(self.shared_circle_roi)
            self.shared_circle_roi.show()

        # Move rectangle ROI if it exists and is visible
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.image_view.getView().removeItem(self.shared_rectangle_roi)
            tab = self.tabs.widget(idx)
            tab.image_view.getView().addItem(self.shared_rectangle_roi)
            self.shared_rectangle_roi.show()

    def show_circle_roi(self):
        """Shows or hides the shared circular ROI on the current tab.

        Ensures only the circular ROI is visible, hiding any rectangle ROI if present.
        """
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            return

        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            self.shared_circle_roi.setVisible(False)
            return

        # Hide rectangle ROI if present
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            self.shared_rectangle_roi.setVisible(False)

        if self._is_roi_deleted(self.shared_circle_roi):
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_circle_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_circle_roi.setZValue(100)

        try:
            if self.shared_circle_roi not in tab.image_view.getView().allChildren():
                tab.image_view.getView().addItem(self.shared_circle_roi)
        except RuntimeError:
            # ROI was deleted, recreate it
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_circle_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_circle_roi.setZValue(100)
            tab.image_view.getView().addItem(self.shared_circle_roi)
        self.shared_circle_roi.show()

    def show_rectangle_roi(self):
        """Shows or hides the shared rectangle ROI on the current tab.

        Ensures only the rectangle ROI is visible, hiding any circular ROI if present.
        """
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            return

        # Hide rectangle ROI if already visible, then return
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            self.shared_rectangle_roi.setVisible(False)
            return

        # Hide circle ROI if present and visible
        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            self.shared_circle_roi.setVisible(False)

        # Create rectangle ROI if it does not exist or was deleted
        if self._is_roi_deleted(self.shared_rectangle_roi):
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_rectangle_roi = pg.RectROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_rectangle_roi.setZValue(100)

        # Add rectangle ROI to the current tab if not already present
        try:
            if self.shared_rectangle_roi not in tab.image_view.getView().allChildren():
                tab.image_view.getView().addItem(self.shared_rectangle_roi)
        except RuntimeError:
            # ROI was deleted, recreate it
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_rectangle_roi = pg.RectROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_rectangle_roi.setZValue(100)
            tab.image_view.getView().addItem(self.shared_rectangle_roi)
        self.shared_rectangle_roi.show()

    def close_tab(self, index):
        widget = self.tabs.widget(index)
        if widget is not None:
            self.tabs.removeTab(index)
            widget.deleteLater()

    def create_actions(self):
        self.actions = { 
            "open": QtWidgets.QAction("Open...", self),
            "save_scan": QtWidgets.QAction("Save current scan...", self),
            "save_multi": QtWidgets.QAction("Save multiple scans...", self),
            "fill": QtWidgets.QAction("Fill holes", self),
            "repair": QtWidgets.QAction("Remove holes and outliers", self),
            "flipUD": QtWidgets.QAction("Flip Up/Down", self),
            "flipLR": QtWidgets.QAction("Flip Left/Right", self),
            "rot90": QtWidgets.QAction("Rotate 90-Left", self),
            "inverse": QtWidgets.QAction("Inverse Z", self),
            "zero": QtWidgets.QAction("Set zero point", self),
            "tilt": QtWidgets.QAction("Set tilt", self),
            "colormap": QtWidgets.QAction("Toggle colormap", self),
            "view3d":  QtWidgets.QAction("View 3d...", self),
            "compare": QtWidgets.QAction("Scan positioning...", self),
            "profile": QtWidgets.QAction("Profile analysis...", self),
            "about": QtWidgets.QAction("About...", self),
            "exit": QtWidgets.QAction("Exit", self),
            # Advanced processing actions
            "filter": QtWidgets.QAction("Advanced Filtering...", self),
            "morphology": QtWidgets.QAction("Morphology && Leveling...", self),
            "transform": QtWidgets.QAction("Geometric Transforms...", self),
            "register": QtWidgets.QAction("Auto-Register Surfaces...", self),
        }

        # create_actions
        self.actions["del_outside"] = QtWidgets.QAction("outside of the mask", self)
        self.actions["del_inside"] = QtWidgets.QAction("inside of the mask", self)
        self.actions["show_mask"] = QtWidgets.QAction("Show/hide the circle mask", self)
        self.actions["show_rmask"] = QtWidgets.QAction("Show/hide the rectangle mask", self)

        self.actions["open"].setIcon(QIcon(resource_path("icons/icons8-open-file1-50.png")))
        self.actions["save_scan"].setIcon(QIcon(resource_path("icons/icons8-save1-50.png")))
        # self.actions["save_scan"].setIcon(svg_icon("icons/save.svg", 20))
        self.actions["save_multi"].setIcon(QIcon(resource_path("icons/icons8-save2-50.png")))
        self.actions["repair"].setIcon(QIcon(resource_path("icons/icons8-job-50.png")))
        self.actions["flipUD"].setIcon(QIcon(resource_path("icons/flipUD.png")))
        self.actions["flipLR"].setIcon(QIcon(resource_path("icons/flipLR.png")))
        self.actions["rot90"].setIcon(QIcon(resource_path("icons/icons8-rotate-left-50.png")))
        self.actions["inverse"].setIcon(QIcon(resource_path("icons/icons8-invert-50.png")))
        self.actions["zero"].setIcon(QIcon(resource_path("icons/icons8-eyedropper-50.png")))
        self.actions["tilt"].setIcon(QIcon(resource_path("icons/icons8-tilt-64.png")))
        self.actions["colormap"].setIcon(QIcon(resource_path("icons/icons8-color-palette-50.png")))
        self.actions["view3d"].setIcon(QIcon(resource_path("icons/icons8-3d-80.png")))
        self.actions["compare"].setIcon(QIcon(resource_path("icons/icons8-compare-50.png")))
        self.actions["profile"].setIcon(QIcon(resource_path("icons/icons8-graph-50.png")))
        self.actions["about"].setIcon(QIcon(resource_path("icons/icons8-about-50.png")))
        self.actions["exit"].setIcon(QIcon(resource_path("icons/icons8-exit-50.png")))

        self.actions["filter"].setIcon(QIcon(resource_path("icons/icons8-filter-50.png")))
        self.actions["morphology"].setIcon(QIcon(resource_path("icons/icons8-filter2-50.png")))
        self.actions["transform"].setIcon(QIcon(resource_path("icons/icons8-transform-64.png")))

        self.actions["colormap"].setCheckable(True)
        self.actions["colormap"].setChecked(False)
        
        # Set tooltips for advanced processing actions
        self.actions["filter"].setToolTip("Apply advanced filtering (bilateral, median, morphological)")
        self.actions["morphology"].setToolTip("Level surface and remove polynomial forms")
        self.actions["transform"].setToolTip("Rotate, rescale, or crop grid")
        self.actions["register"].setToolTip("Automatically align two surfaces")

    def connect_actions(self):
        self.actions["open"].triggered.connect(self.open_file)
        self.actions["save_scan"].triggered.connect(self.save_single_scan)
        self.actions["save_multi"].triggered.connect(self.save_multiple_scans)
        self.actions["fill"].triggered.connect(self.fill_holes)
        self.actions["repair"].triggered.connect(self.repair_grid)
        self.actions["flipUD"].triggered.connect(self.flipUD_scan)
        self.actions["flipLR"].triggered.connect(self.flipLR_scan)
        self.actions["rot90"].triggered.connect(self.scan_rot90)
        self.actions["inverse"].triggered.connect(self.invert_scan)
        self.actions["zero"].triggered.connect(self.set_zero_point_mode)
        self.actions["tilt"].triggered.connect(self.set_tilt_mode)
        self.actions["colormap"].triggered.connect(self.toggle_colormap_current_tab)
        self.actions["view3d"].triggered.connect(self.view3d)
        self.actions["compare"].triggered.connect(self.compare_scans)
        self.actions["profile"].triggered.connect(self.start_profile_analysis)
        self.actions["about"].triggered.connect(self.show_about_dialog)
        self.actions["exit"].triggered.connect(self.close)

        # connect_actions
        self.actions["del_outside"].triggered.connect(self.del_outside_mask)
        self.actions["del_inside"].triggered.connect(self.del_inside_mask)
        self.actions["show_mask"].triggered.connect(self.show_circle_roi)
        self.actions["show_rmask"].triggered.connect(self.show_rectangle_roi)
        
        # Advanced processing connections
        self.actions["filter"].triggered.connect(self.apply_advanced_filter)
        self.actions["morphology"].triggered.connect(self.apply_morphology)
        self.actions["transform"].triggered.connect(self.apply_transform)
        self.actions["register"].triggered.connect(self.auto_register_surfaces)

    def create_menubar(self):
        menubar = self.menuBar()

        menu_structure = [
            ("&File", [
                "open",
                "save_scan",
                "save_multi",
                ("recent_menu", []),
                "separator",
                "exit"
            ]),
            ("&Edit", [
                "show_mask","show_rmask",
                ("delete", [
                    "del_outside",
                    "del_inside"
                ])
            ]),
            ("Scan &Actions", [
                "fill", "repair", "flipUD", "flipLR", "rot90", "inverse", "zero", "colormap"
            ]),
            ("&Processing", [
                "filter", "morphology", "transform", "separator", "register"
            ]),
            ("&Tools", [
                "compare", "profile"
            ]),
            ("&Help", [
                "about"
            ])
        ]

        # Tworzymy recent_menu przed budowaniem menu
        self.recent_menu = QtWidgets.QMenu("Recent files", self)
        self.update_recent_files_menu()

        def add_menu_items(menu, items):
            for item in items:
                if item == "separator":
                    menu.addSeparator()
                elif isinstance(item, tuple):
                    submenu_name, subitems = item
                    if submenu_name == "recent_menu":
                        menu.addMenu(self.recent_menu)
                    else:
                        submenu = QtWidgets.QMenu(submenu_name, self)
                        add_menu_items(submenu, subitems)
                        menu.addMenu(submenu)
                else:
                    menu.addAction(self.actions[item])

        for menu_name, items in menu_structure:
            menu = menubar.addMenu(menu_name)
            add_menu_items(menu, items)


    def create_toolbar(self):
        self.toolbar = self.addToolBar("Tools")
        self.toolbar.addAction(self.actions["open"])
        self.toolbar.addAction(self.actions["save_scan"])
        self.toolbar.addAction(self.actions["save_multi"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["repair"])
        self.toolbar.addAction(self.actions["flipUD"])
        self.toolbar.addAction(self.actions["flipLR"])
        self.toolbar.addAction(self.actions["rot90"])
        self.toolbar.addAction(self.actions["inverse"])
        self.toolbar.addAction(self.actions["zero"])
        self.toolbar.addAction(self.actions["tilt"])
        self.toolbar.addAction(self.actions["colormap"])
        self.toolbar.addSeparator()
        # Advanced processing toolbar
        self.toolbar.addAction(self.actions["filter"])
        self.toolbar.addAction(self.actions["morphology"])
        self.toolbar.addAction(self.actions["transform"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["view3d"])
        self.toolbar.addAction(self.actions["compare"])
        self.toolbar.addAction(self.actions["profile"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["about"])
        self.toolbar.addAction(self.actions["exit"])

        self.toolbar.setStyleSheet("QToolButton { color: #222; }")


    def create_circle_mask(self, shape, center, radius):
        """Creates a boolean mask for a circle within a 2D array.

        Generates a mask where points inside the specified circle are True and others are False.

        Args:
            shape (tuple): Shape of the output mask (height, width).
            center (tuple): (x, y) coordinates of the circle center.
            radius (float): Radius of the circle.

        Returns:
            np.ndarray: Boolean mask with True inside the circle.
        """
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist <= radius


    def create_rectangle_mask(self, shape, center, width, height):
        """
        Creates a boolean mask for a rectangle within a 2D array.

        Args:
            shape (tuple): Shape of the output mask (height, width).
            center (tuple): (x, y) coordinates of the rectangle center.
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.

        Returns:
            np.ndarray: Boolean mask with True inside the rectangle.
        """
        Y, X = np.ogrid[:shape[0], :shape[1]]
        x0 = center[0] - width / 2
        x1 = center[0] + width / 2
        y0 = center[1] - height / 2
        y1 = center[1] + height / 2
        return (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)

    def view3d(self):
        if tab := self.current_tab():
            # Przekaż rozmiary pikseli dla prawidłowych proporcji
            dx = getattr(tab, 'dx', 1.0)
            dy = getattr(tab, 'dy', 1.0)
            show_3d_viewer(tab.grid, show_controls=False, pixel_size_x=dx, pixel_size_y=dy)


    def toggle_colormap_current_tab(self):
        if tab := self.current_tab():
            tab.toggle_colormap()


    def repair_grid(self):
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            return
        h, w = tab.grid.shape
        mask = self.create_mask(h, w)
        tab.repair_grid(mask=mask)

    def set_zero_point_mode(self):
        if tab := self.current_tab():
            tab.set_zero_point_mode()

    def set_tilt_mode(self):
        if tab := self.current_tab():
            tab.set_tilt_mode()

    def show_about_dialog(self):
        dlg = AboutDialog(self)
        dlg.exec_()

    def closeEvent(self, event):
        self.settings.setValue("recentFiles", self.recent_files)
        event.accept()

    def add_to_recent_files(self, path):
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_menu()
        self.settings.setValue("recentFiles", self.recent_files)

    def load_recent_files(self):
        self.recent_files = self.settings.value("recentFiles", [], type=list)
        self.max_recent_files = 10

    def update_recent_files_menu(self):
        self.recent_menu.clear()
        if not self.recent_files:
            action = QtWidgets.QAction("No recent files", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        for path in self.recent_files:
            action = QtWidgets.QAction(path, self)
            action.triggered.connect(lambda checked, p=path: self.open_file_from_recent(p))
            self.recent_menu.addAction(action)

    def current_tab(self):
        return self.tabs.currentWidget()

    def load_csv(self, fname, tab):
        # Zapytaj użytkownika o jednostki z sugerowanym wyborem
        units = self._ask_for_units(fname)
        if units is None:
            # Użytkownik anulował
            return
        
        units_xy, units_z = units
        
        dlg = QtWidgets.QProgressDialog("Wczytywanie i gridowanie...", None, 0, 100, self)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        self.worker = GridWorker(fname, units_xy=units_xy, units_z=units_z)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(dlg.setValue)
        self.worker.finished.connect(lambda *args: tab.set_data(*args))
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.process)
        self.thread.start()
        dlg.exec_()
    
    def _ask_for_units(self, fname):
        """Ask user about XY and Z coordinate units with suggested choices based on data sample.
        
        Args:
            fname (str): Path to CSV file.
            
        Returns:
            tuple: (units_xy, units_z) where each is 'mm' or 'um', or None if cancelled.
        """
        # Get suggested units from io module
        suggested_xy, suggested_z = suggest_units(fname)
        
        # Dialog z pytaniem
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select coordinate units")
        layout = QtWidgets.QVBoxLayout()
        
        label = QtWidgets.QLabel(
            "Select the units for X, Y and Z coordinates in the file:\n"
            "(All coordinates will be converted to micrometers internally)"
        )
        layout.addWidget(label)
        
        # Grupa dla XY
        group_xy = QtWidgets.QGroupBox("X and Y coordinates")
        layout_xy = QtWidgets.QVBoxLayout()
        radio_xy_mm = QtWidgets.QRadioButton("Millimeters (mm)")
        radio_xy_um = QtWidgets.QRadioButton("Micrometers (μm)")
        
        if suggested_xy == 'mm':
            radio_xy_mm.setChecked(True)
            radio_xy_mm.setText("Millimeters (mm) [suggested]")
        else:
            radio_xy_um.setChecked(True)
            radio_xy_um.setText("Micrometers (μm) [suggested]")
        
        layout_xy.addWidget(radio_xy_mm)
        layout_xy.addWidget(radio_xy_um)
        group_xy.setLayout(layout_xy)
        layout.addWidget(group_xy)
        
        # Grupa dla Z
        group_z = QtWidgets.QGroupBox("Z coordinate (height)")
        layout_z = QtWidgets.QVBoxLayout()
        radio_z_mm = QtWidgets.QRadioButton("Millimeters (mm)")
        radio_z_um = QtWidgets.QRadioButton("Micrometers (μm)")
        
        if suggested_z == 'mm':
            radio_z_mm.setChecked(True)
            radio_z_mm.setText("Millimeters (mm) [suggested]")
        else:
            radio_z_um.setChecked(True)
            radio_z_um.setText("Micrometers (μm) [suggested]")
        
        layout_z.addWidget(radio_z_mm)
        layout_z.addWidget(radio_z_um)
        group_z.setLayout(layout_z)
        layout.addWidget(group_z)
        
        # Przyciski OK/Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            units_xy = 'mm' if radio_xy_mm.isChecked() else 'um'
            units_z = 'mm' if radio_z_mm.isChecked() else 'um'
            return (units_xy, units_z)
        else:
            return None

    def load_npz(self, fname):
        try:
            scans = load_npz_data(fname)
            for name, grid, xi, yi, dx, dy in scans:
                tab = ScanTab()
                self.tabs.addTab(tab, name)
                self.tabs.setCurrentWidget(tab)
                tab.set_data(grid, xi, yi, dx, dy)
            self.add_to_recent_files(fname)
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Format error", str(e))
            return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading:\n{e}")
            return False

    def load_stl(self, fname, tab):
        """Load STL file and convert to height map grid.
        
        Args:
            fname (str): Path to STL file.
            tab (ScanTab): Tab widget to load data into.
        """
        # Ask user for resolution
        resolution, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "STL Resolution",
            "Enter desired pixel resolution in micrometers (μm):\n(Leave as 0 for automatic resolution)",
            value=0.0,
            min=0.0,
            max=1000.0,
            decimals=2
        )
        
        if not ok:
            return
        
        if resolution == 0.0:
            resolution = None
        
        # Create progress dialog
        dlg = QtWidgets.QProgressDialog("Loading STL file...", None, 0, 100, self)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()
        
        try:
            grid, xi, yi, dx, dy = load_stl_data(
                fname,
                resolution=resolution,
                progress_callback=dlg.setValue
            )
            tab.set_data(grid, xi, yi, dx, dy)
            dlg.setValue(100)
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load STL file:\n{e}")
            # Remove the tab if loading failed
            idx = self.tabs.indexOf(tab)
            if idx >= 0:
                self.tabs.removeTab(idx)
    
    def load_h5(self, fname):
        try:
            scans = load_h5_data(fname)
            for name, grid, xi, yi, dx, dy in scans:
                tab = ScanTab()
                self.tabs.addTab(tab, str(name))
                self.tabs.setCurrentWidget(tab)
                tab.set_data(grid, xi, yi, dx, dy)
            self.add_to_recent_files(fname)
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Format error", str(e))
            return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while opening HDF5 file:\n{e}")
            return False


    def create_tab_and_load(self, fname):
        if fname.endswith('.csv') or fname.endswith('.dat') or fname.endswith('.txt'):
            tab = ScanTab()
            self.tabs.addTab(tab, fname.split('/')[-1])
            self.tabs.setCurrentWidget(tab)
            self.load_csv(fname, tab)
            self.add_to_recent_files(fname)
        elif fname.endswith('.npz'):
            self.load_npz(fname)
        elif fname.endswith('.h5'):
            self.load_h5(fname)
        elif fname.endswith('.stl'):
            tab = ScanTab()
            self.tabs.addTab(tab, fname.split('/')[-1])
            self.tabs.setCurrentWidget(tab)
            self.load_stl(fname, tab)
            self.add_to_recent_files(fname)
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            # self.tabs.removeTab(self.tabs.indexOf(tab))
            return

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "All supported (*.csv *.dat *.txt *.npz *.h5 *.stl);;CSV/DAT/TXT (*.csv *.dat *.txt);;NPZ (*.npz);;HDF5 (*.h5);;STL (*.stl)")
        if not fname:
            return
        self.create_tab_and_load(fname)

    def open_file_from_recent(self, path):
        if not QtCore.QFile.exists(path):
            QtWidgets.QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
            self.recent_files.remove(path)
            self.update_recent_files_menu()
            return
        self.create_tab_and_load(path)

    # format: tabs = [('name0', tab0), ('name1', tab1), ...]
    def save_tabs(self, tabs=None):
        if tabs is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data to save.")
            return

        fname, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Scan", "", "NPZ (*.npz);;HDF5 (*.h5);;STL (*.stl)"
        )
        if not fname:
            return

        if selected_filter.startswith("NPZ") and not fname.endswith(".npz"):
            fname += ".npz"
        elif selected_filter.startswith("HDF5") and not fname.endswith(".h5"):
            fname += ".h5"
        elif selected_filter.startswith("STL") and not fname.endswith(".stl"):
            fname += ".stl"

        try:
            # Prepare scans data: list of (name, Surface)
            scans = []
            for name, tab in tabs:
                surface = tab.getGridData()
                scans.append((name, surface))
            
            if fname.endswith(".npz"):
                save_npz(fname, scans)
            elif fname.endswith(".h5"):
                save_h5(fname, scans)
            elif fname.endswith(".stl"):
                # STL can only save a single scan
                if len(scans) > 1:
                    QtWidgets.QMessageBox.warning(
                        self, "Multiple scans",
                        "STL format can only save a single scan.\nOnly the first scan will be saved."
                    )
                name, surface = scans[0]
                save_stl(fname, surface, binary=True)

            QtWidgets.QMessageBox.information(self, "Saved", f"Scan saved to: {fname}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while saving:\n{e}")


    def save_single_scan(self):
        tab = self.current_tab()
        if not tab or not hasattr(tab, "grid") or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No scan in current tab.")
            return

        self.save_tabs([("nowyskan", tab)])
        

    def save_multiple_scans(self):
        if self.tabs.count() == 0:
            QtWidgets.QMessageBox.warning(self, "No scans", "No scan tabs are open.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save selected scans")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Select scans to save and specify dataset names:"))

        checkboxes = []
        lineedits = []
        for i in range(self.tabs.count()):
            row = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QCheckBox(self.tabs.tabText(i))
            cb.setChecked(True)
            le = QtWidgets.QLineEdit(self.tabs.tabText(i).replace(" ", "_"))
            row.addWidget(cb)
            row.addWidget(le)
            layout.addLayout(row)
            checkboxes.append(cb)
            lineedits.append(le)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        tabs = []
        for i, cb in enumerate(checkboxes):
            if cb.isChecked():
                dataset_name = lineedits[i].text().strip()
                if not dataset_name:
                    QtWidgets.QMessageBox.warning(self, "Invalid name", "Each scan must have a dataset name!")
                    return
                tab = self.tabs.widget(i)
                if not hasattr(tab, "grid") or tab.grid is None:
                    QtWidgets.QMessageBox.warning(self, "No data", f"Tab '{cb.text()}' has no scan data.")
                    return
                tabs.append((dataset_name, tab))

        if not tabs:
            QtWidgets.QMessageBox.warning(self, "Nothing to save", "No scans selected.")
            return

        self.save_tabs(tabs)


    def flipUD_scan(self):
        if tab := self.current_tab():
            tab.flip_scan(direction='UD', parent=self)

    def flipLR_scan(self):
        if tab := self.current_tab():
            tab.flip_scan(direction='LR', parent=self)

    def scan_rot90(self):
        if tab := self.current_tab():
            tab.scan_rot90(parent=self)


    def invert_scan(self):
        if tab := self.current_tab():
            tab.invert_scan(parent=self)

    def fill_holes(self):
        if tab := self.current_tab():
            tab.fill_holes(self)

    def compare_scans(self):
        if self.tabs.count() < 2:
            QtWidgets.QMessageBox.warning(self, "Za mało skanów", "Musisz mieć przynajmniej 2 skany!")
            return

        def receive_aligned_grids(scan1_aligned_data : Surface, scan2_aligned_data : Surface, idx1=None, idx2=None):
            b = idx1 is not None and idx2 is not None
            if b:
                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Dopasowanie skanów")
                msg.setText("Jak chcesz zapisać dopasowanie?")
                btn1 = msg.addButton("Jako nowe zakładki", QtWidgets.QMessageBox.AcceptRole)
                btn2 = msg.addButton("Nadpisz istniejące", QtWidgets.QMessageBox.ActionRole)
                msg.addButton("Anuluj", QtWidgets.QMessageBox.RejectRole)
                msg.exec_()

            if not b or msg.clickedButton() == btn1:
                tab1 = ScanTab()
                tab2 = ScanTab()
                self.tabs.addTab(tab1, "Dopasowany ref")
                self.tabs.addTab(tab2, "Dopasowany scan2")
            elif msg.clickedButton() == btn2:
                tab1 = self.tabs.widget(idx1)
                tab2 = self.tabs.widget(idx2)

            tab1.setGridData(scan1_aligned_data)
            tab2.setGridData(scan2_aligned_data)


        # Dialog wyboru zakładek
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Wybierz skany do porównania")
        layout = QtWidgets.QVBoxLayout(dialog)
        label1 = QtWidgets.QLabel("Referencyjny skan:")
        label2 = QtWidgets.QLabel("Skan do dopasowania:")
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Anuluj")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(ok_btn)
        hl.addWidget(cancel_btn)
        layout.addWidget(label1)
        layout.addWidget(cb1)
        layout.addWidget(label2)
        layout.addWidget(cb2)
        layout.addLayout(hl)

        def accept():
            if cb1.currentIndex() == cb2.currentIndex():
                QtWidgets.QMessageBox.warning(dialog, "Błąd", "Wybierz dwa różne skany!")
                return
            dialog.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        tab1 = self.tabs.widget(idx1)
        tab2 = self.tabs.widget(idx2)

        # if getattr(self, "viewer", None):
        #     self.viewer.close()  # lub .hide() jeśli chcesz zachować stan
        #     self.viewer = None

        self.viewer = OverlayViewer( 
            tab1.getGridData(), 
            tab2.getGridData(),
            on_accept=partial(receive_aligned_grids, idx1=idx1, idx2=idx2),
            parent=self
        )

        self.viewer.setWindowTitle(f"Comparison: {names[idx1]} vs {names[idx2]}")
        self.viewer.show()


    def start_profile_analysis(self):
        if self.tabs.count() < 2:
            QtWidgets.QMessageBox.warning(self, "Not enough scans", "You need at least two scans!")
            return

        # Dialog wyboru dwóch zakładek
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select scans for profile analysis")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Select two scans:"))
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        layout.addWidget(QtWidgets.QLabel("Reference scan:"))
        layout.addWidget(cb1)
        layout.addWidget(QtWidgets.QLabel("Scan for comparison:"))
        layout.addWidget(cb2)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        if idx1 == idx2:
            QtWidgets.QMessageBox.warning(self, "Error", "Select two different scans!")
            return

        tab1 = self.tabs.widget(idx1)
        tab2 = self.tabs.widget(idx2)
        grid1 = tab1.grid  # Use grid, not masked (masked is transposed and thresholded)
        grid2 = tab2.grid

        if grid1.shape != grid2.shape:
            h = min(grid1.shape[0], grid2.shape[0])
            w = min(grid1.shape[1], grid2.shape[1])
            reply = QtWidgets.QMessageBox.question(
                self, "Different sizes",
                f"The scans vary in size:\n"
                f"{grid1.shape} vs {grid2.shape}\n"
                f"Crop both to a common area {h}x{w} and continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            grid1 = grid1[:h, :w]
            grid2 = grid2[:h, :w]

        # -- TYLKO JEDNO OKNO --
        if getattr(self, "_profile_viewer", None) is None:
            self._profile_viewer = ProfileViewer(parent=self)

        self._profile_viewer.set_data(
            grid1, grid2,
            tab1.dx, tab1.dy,
            tab2.dx, tab2.dy
        )
        self._profile_viewer.show()
        self._profile_viewer.raise_()
        self._profile_viewer.activateWindow()


    # ========== Advanced Processing Methods ==========
    
    def apply_advanced_filter(self):
        """Apply advanced filtering to current scan."""
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Please load a scan first!")
            return
        
        dialog = FilterDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        filter_type, params = dialog.get_filter_config()
        
        # Import processing functions
        from ..processing import (
            bilateral_filter, median_filter_nan_aware,
            morphological_opening, morphological_closing,
            robust_gaussian_filter
        )
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            if filter_type == "bilateral":
                result = bilateral_filter(
                    tab.grid, 
                    sigma_spatial=params['sigma_spatial'],
                    sigma_range=params['sigma_range'],
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0
                )
            elif filter_type == "median":
                result = median_filter_nan_aware(
                    tab.grid, 
                    size=params['size'],
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0
                )
            elif filter_type == "opening":
                result = morphological_opening(
                    tab.grid, 
                    size=params['size'],
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0
                )
            elif filter_type == "closing":
                result = morphological_closing(
                    tab.grid, 
                    size=params['size'],
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0
                )
            elif filter_type == "robust_gaussian":
                result = robust_gaussian_filter(
                    tab.grid,
                    sigma=params['sigma'],
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0,
                    iterations=params['max_iterations'],
                    threshold=params['outlier_threshold']
                )
            
            tab.grid = result
            tab.update_histogram()
            tab.update_image()  # This will properly set masked from grid
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Filter applied successfully!\nShape: {result.shape}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Failed to apply filter:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    
    def apply_morphology(self):
        """Apply morphology/leveling operations to current scan."""
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Please load a scan first!")
            return
        
        dialog = MorphologyDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        op_type, params = dialog.get_operation_config()
        
        from ..processing import (
            level_by_plane, remove_polynomial_form, threshold_grid
        )
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            if op_type == "level_ls":
                result = level_by_plane(tab.grid, method='least_squares')
            elif op_type == "level_robust":
                # Call fit_plane_robust directly to use residual_threshold
                from ..processing.morphology import fit_plane_robust
                plane, coeffs, inliers = fit_plane_robust(
                    tab.grid,
                    residual_threshold=params.get('residual_threshold', 10.0)
                )
                result = tab.grid - plane
            elif op_type == "polynomial":
                result = remove_polynomial_form(tab.grid, order=params['order'])
            elif op_type == "threshold":
                result = threshold_grid(
                    tab.grid,
                    low=params['lower'],
                    high=params['upper']
                )
            
            tab.grid = result
            tab.update_histogram()
            tab.update_image()  # This will properly set masked from grid
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Operation applied successfully!"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Failed to apply operation:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    
    def apply_transform(self):
        """Apply geometric transform to current scan."""
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Please load a scan first!")
            return
        
        dialog = TransformDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        transform_type, params = dialog.get_transform_config()
        
        from ..processing import rotate_grid, rescale_grid, crop_to_valid_region
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Ensure coordinate arrays exist
            if not hasattr(tab, 'xi') or tab.xi is None:
                h, w = tab.grid.shape
                tab.xi = np.arange(w) * (tab.dx or 1.0)
                tab.yi = np.arange(h) * (tab.dy or 1.0)
            
            if transform_type == "rotate":
                result, new_xi, new_yi, new_dx, new_dy = rotate_grid(
                    tab.grid,
                    angle_degrees=params['angle'],
                    xi=tab.xi,
                    yi=tab.yi,
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0,
                    order=params.get('order', 3)
                )
                tab.xi = new_xi
                tab.yi = new_yi
                tab.dx = new_dx
                tab.dy = new_dy
            elif transform_type == "rescale":
                result, new_xi, new_yi, new_dx, new_dy = rescale_grid(
                    tab.grid,
                    scale_factor=params['scale'],
                    xi=tab.xi,
                    yi=tab.yi,
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0,
                    order=params.get('order', 3)
                )
                tab.xi = new_xi
                tab.yi = new_yi
                tab.dx = new_dx
                tab.dy = new_dy
            elif transform_type == "crop":
                result, new_xi, new_yi, new_dx, new_dy = crop_to_valid_region(
                    tab.grid,
                    xi=tab.xi,
                    yi=tab.yi,
                    dx=tab.dx or 1.0,
                    dy=tab.dy or 1.0,
                    margin=params.get('margin', 0)
                )
                tab.xi = new_xi
                tab.yi = new_yi
                tab.dx = new_dx
                tab.dy = new_dy
            
            tab.grid = result
            tab.update_histogram()
            tab.update_image()  # This will properly set masked from grid
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Transform applied successfully!\n"
                f"New shape: {result.shape}\n"
                f"Pixel size: {tab.dx:.3f} x {tab.dy:.3f}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Failed to apply transform:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    
    def auto_register_surfaces(self):
        """Automatically register two surfaces."""
        if self.tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self, "Not enough scans", 
                "You need at least two scans for registration!"
            )
            return
        
        # Get scan names
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        
        dialog = RegistrationDialog(names, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        ref_idx, mov_idx, method = dialog.get_registration_config()
        
        if ref_idx == mov_idx:
            QtWidgets.QMessageBox.warning(
                self, "Error", 
                "Please select two different surfaces!"
            )
            return
        
        ref_tab = self.tabs.widget(ref_idx)
        mov_tab = self.tabs.widget(mov_idx)
        
        from ..processing import auto_register_surfaces, apply_registration
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Perform registration
            params = auto_register_surfaces(
                ref_tab.grid, 
                mov_tab.grid,
                method=method
            )
            
            # Auto-fallback: if cross-correlation gives poor RMSE, try ICP
            if method == 'correlation' and params['rmse'] > 500.0:
                logger.warning(f"Cross-correlation RMSE ({params['rmse']:.1f} nm) > 500 nm threshold. Trying ICP...")
                QtWidgets.QMessageBox.information(
                    self,
                    "Registration",
                    f"Cross-correlation RMSE is high ({params['rmse']:.1f} nm).\n\n"
                    f"Automatically switching to ICP method for better alignment."
                )
                params = auto_register_surfaces(
                    ref_tab.grid, 
                    mov_tab.grid,
                    method='icp'
                )
                logger.info(f"ICP result: RMSE={params['rmse']:.1f} nm, translation={params['translation']}")
            
            # Auto-generate coordinate arrays if missing
            h, w = mov_tab.grid.shape
            if not hasattr(mov_tab, 'xi') or mov_tab.xi is None:
                mov_tab.xi = np.arange(w) * (mov_tab.dx or 1.0)
            if not hasattr(mov_tab, 'yi') or mov_tab.yi is None:
                mov_tab.yi = np.arange(h) * (mov_tab.dy or 1.0)
            
            # Apply registration to moving surface
            registered, new_xi, new_yi, new_dx, new_dy = apply_registration(
                mov_tab.grid,
                mov_tab.xi,
                mov_tab.yi,
                mov_tab.dx or 1.0,
                mov_tab.dy or 1.0,
                params['translation'],
                params.get('rotation', 0.0)
            )
            
            # Update moving tab
            mov_tab.grid = registered
            mov_tab.xi = new_xi
            mov_tab.yi = new_yi
            mov_tab.dx = new_dx
            mov_tab.dy = new_dy
            
            # Check if registration resulted in valid data
            valid_data = ~np.isnan(registered)
            num_valid = np.sum(valid_data)
            if num_valid == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Registration Warning",
                    "The registered surface contains no valid data.\n"
                    "This can happen if the translation was too large.\n"
                    "The operation will be undone."
                )
                # Don't apply the registration
                QtWidgets.QApplication.restoreOverrideCursor()
                return
            
            # Update visualization
            mov_tab.update_histogram()
            mov_tab.update_image()  # This will properly set masked from grid
            
            # Show results
            msg = f"Registration completed!\n\n"
            msg += f"Method: {method.upper()}\n"
            if 'translation' in params:
                tx, ty = params['translation']
                msg += f"Translation: ({tx:.1f}, {ty:.1f}) pixels\n"
            if 'rotation' in params:
                msg += f"Rotation: {params['rotation']:.2f}°\n"
            if 'rmse' in params:
                rmse = params['rmse']
                msg += f"RMSE: {rmse:.2f} nm"
                # Add quality assessment
                if rmse < 50:
                    msg += " (excellent)\n"
                elif rmse < 200:
                    msg += " (good)\n"
                elif rmse < 500:
                    msg += " (fair)\n"
                else:
                    msg += " (poor - consider using ICP)\n"
            
            QtWidgets.QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Registration failed:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
