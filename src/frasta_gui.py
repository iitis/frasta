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

import h5py
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from functools import partial

import sys
import os

from .profileViewer import ProfileViewer
from .overlayViewer import OverlayViewer
from .aboutDialog import AboutDialog
from .scanTab import ScanTab
from .gridData import GridData

from .grid3DViewer import show_3d_viewer

import logging
logger = logging.getLogger(__name__)

def resource_path(relative_path):
    """Returns the correct path to resource files in both .exe and .py environments.
    
    Args:
        relative_path (str): Relative path to the resource file.
        
    Returns:
        str: Absolute path to the resource file.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class GridWorker(QtCore.QObject):
    """Background worker for loading and gridding CSV scan data.
    
    Loads large CSV files in chunks, uses specified coordinate units (mm vs μm),
    grids the point cloud data onto a regular 2D array, and emits progress updates.
    
    Signals:
        progress (int): Progress percentage (0-100).
        finished (object, object, object, float, float): Emitted when processing completes,
            returning grid, xi, yi, px_x, px_y.
    """
    
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object, object, object, float, float) # grid, xi, yi, px_x, px_y

    def __init__(self, fname, units_xy='um', units_z='um'):
        """Initialize the grid worker.
        
        Args:
            fname (str): Path to the CSV file to load.
            units_xy (str): Unit for X and Y coordinates: 'mm' or 'um' (micrometers).
            units_z (str): Unit for Z coordinate: 'mm' or 'um' (micrometers).
        """
        super().__init__()
        self.fname = fname
        self.units_xy = units_xy
        self.units_z = units_z

    @QtCore.pyqtSlot()
    def process(self):
        """Process the CSV file and generate a gridded representation.
        
        Reads CSV data in chunks, detects coordinate units, calculates pixel sizes,
        grids the point cloud, and averages duplicate points. Emits progress signals
        during processing and finished signal with results.
        """
        chunk_size = 100_000
        total = sum(1 for _ in open(self.fname, encoding="utf-8"))
        chunks = []

        for i, chunk in enumerate(pd.read_csv(
                self.fname,
                sep=r'[;,\t ]+',
                engine='python',
                header=None,
                names=['x', 'y', 'z'],
                chunksize=chunk_size)):

            chunks.append(chunk)
            self.progress.emit(int(20 + 30 * (i * chunk_size / total)))

        df = pd.concat(chunks, ignore_index=True)
        x, y, z = df['x'].values, df['y'].values, df['z'].values

        # Oblicz typowe kroki w x i y
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        px_x_raw = np.median(dx[dx > 0])
        px_y_raw = np.median(dy[dy > 0])
        typical_step = np.median([px_x_raw, px_y_raw])
        logger.debug(f"typical_step XY: {typical_step}")
        
        # Konwersja XY na podstawie wybranej przez użytkownika jednostki
        if self.units_xy == 'mm':
            logger.info("Przeliczam XY z milimetrów na mikrometry.")
            x *= 1000
            y *= 1000
            # trzeba przeliczyć dx/dy jeszcze raz po skalowaniu
            dx = np.diff(np.sort(np.unique(x)))
            dy = np.diff(np.sort(np.unique(y)))
        else:
            logger.info("Używam XY w mikrometrach - brak konwersji.")
        
        # Konwersja Z niezależnie od XY
        if self.units_z == 'mm':
            logger.info("Przeliczam Z z milimetrów na mikrometry.")
            z *= 1000
        else:
            logger.info("Używam Z w mikrometrach - brak konwersji.")

        px_x = np.median(dx[dx > 0]).round(2)
        px_y = np.median(dy[dy > 0]).round(2)

        logger.debug(f"px_x: {px_x}, px_y: {px_y}")

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid_size_x = int((x_max - x_min) / px_x) + 1
        grid_size_y = int((y_max - y_min) / px_y) + 1

        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
        counts = np.zeros_like(grid, dtype=np.int32)
        N = len(x)
        for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ix = int(round((xi - x_min) / px_x))
            iy = int(round((yi - y_min) / px_y))
            if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
                if np.isnan(grid[iy, ix]):
                    grid[iy, ix] = zi
                else:
                    grid[iy, ix] += zi
                counts[iy, ix] += 1
            if idx % max(1, N//50) == 0:
                self.progress.emit(50 + int(49 * idx / N))

        mask_dup = (counts > 1)
        grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        self.progress.emit(100)
        self.finished.emit(grid, xi_grid, yi_grid, px_x, px_y) #, x, y, z)

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
        circle_visible = self.shared_circle_roi is not None and self.shared_circle_roi.isVisible()
        rect_visible = self.shared_rectangle_roi is not None and self.shared_rectangle_roi.isVisible()

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
        if self.shared_circle_roi is not None and self.shared_circle_roi.isVisible():
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                tab.image_view.getView().removeItem(self.shared_circle_roi)
            tab = self.tabs.widget(idx)
            tab.image_view.getView().addItem(self.shared_circle_roi)
            self.shared_circle_roi.show()

        # Move rectangle ROI if it exists and is visible
        if self.shared_rectangle_roi is not None and self.shared_rectangle_roi.isVisible():
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

        if self.shared_circle_roi is not None and self.shared_circle_roi.isVisible():
            self.shared_circle_roi.setVisible(False)
            return

        # Hide rectangle ROI if present
        if self.shared_rectangle_roi is not None and self.shared_rectangle_roi.isVisible():
            self.shared_rectangle_roi.setVisible(False)

        if self.shared_circle_roi is None:
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_circle_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_circle_roi.setZValue(100)

        if self.shared_circle_roi not in tab.image_view.getView().allChildren():
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
        if self.shared_rectangle_roi is not None and self.shared_rectangle_roi.isVisible():
            self.shared_rectangle_roi.setVisible(False)
            return

        # Hide circle ROI if present and visible
        if self.shared_circle_roi is not None and self.shared_circle_roi.isVisible():
            self.shared_circle_roi.setVisible(False)

        # Create rectangle ROI if it does not exist
        if self.shared_rectangle_roi is None:
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_rectangle_roi = pg.RectROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_rectangle_roi.setZValue(100)

        # Add rectangle ROI to the current tab if not already present
        if self.shared_rectangle_roi not in tab.image_view.getView().allChildren():
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
        }

        # create_actions
        self.actions["del_outside"] = QtWidgets.QAction("outside of the mask", self)
        self.actions["del_inside"] = QtWidgets.QAction("inside of the mask", self)
        self.actions["show_mask"] = QtWidgets.QAction("Show/hide the circle mask", self)
        self.actions["show_rmask"] = QtWidgets.QAction("Show/hide the rectangle mask", self)



        self.actions["open"].setIcon(QIcon(resource_path("icons/icons8-open-file1-50.png")))
        self.actions["save_scan"].setIcon(QIcon(resource_path("icons/icons8-save1-50.png")))
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

        self.actions["colormap"].setCheckable(True)
        self.actions["colormap"].setChecked(False)

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
        self.toolbar.addAction(self.actions["view3d"])
        self.toolbar.addAction(self.actions["compare"])
        self.toolbar.addAction(self.actions["profile"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["about"])
        self.toolbar.addAction(self.actions["exit"])

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
            px_x = getattr(tab, 'px_x', 1.0)
            px_y = getattr(tab, 'px_y', 1.0)
            show_3d_viewer(tab.grid, show_controls=False, pixel_size_x=px_x, pixel_size_y=px_y)


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
        # Wczytaj próbkę danych (pierwsze 5000 wierszy) do heurystyki
        try:
            sample = pd.read_csv(
                fname,
                sep=r'[;,\t ]+',
                engine='python',
                header=None,
                names=['x', 'y', 'z'],
                nrows=5000
            )
            x, y, z = sample['x'].values, sample['y'].values, sample['z'].values
            
            # Heurystyka dla XY
            dx = np.diff(np.sort(np.unique(x)))
            dy = np.diff(np.sort(np.unique(y)))
            px_x_raw = np.median(dx[dx > 0])
            px_y_raw = np.median(dy[dy > 0])
            typical_step_xy = np.median([px_x_raw, px_y_raw])
            
            # Heurystyka dla Z - sprawdź zakres wartości
            z_range = np.nanmax(z) - np.nanmin(z)
            
            # Jeśli typical_step < 0.1, prawdopodobnie mm dla XY
            # Jeśli zakres Z < 1, prawdopodobnie mm dla Z
            suggested_xy = 'mm' if typical_step_xy < 0.1 else 'um'
            suggested_z = 'mm' if z_range < 1 else 'um'
            
            logger.debug(f"Detected typical step XY: {typical_step_xy}, suggesting: {suggested_xy}")
            logger.debug(f"Detected Z range: {z_range}, suggesting: {suggested_z}")
        except Exception as e:
            logger.warning(f"Could not analyze data sample: {e}")
            suggested_xy = 'um'
            suggested_z = 'um'
        
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
        data = np.load(fname)
        if 'frasta_info' in data:
            cnt = data['frasta_cnt']
            for i in range(cnt):
                try:
                    name = str(data[f"name_{i:02}"])
                    grid = data[f"grid_{i:02}"]
                    xi = data[f"xi_{i:02}"]
                    yi = data[f"yi_{i:02}"]
                    px_x = data[f"px_{i:02}"]
                    px_y = data[f"py_{i:02}"]

                    tab = ScanTab()
                    self.tabs.addTab(tab, name)
                    self.tabs.setCurrentWidget(tab)
                    tab.set_data(grid, xi, yi, px_x, px_y)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading:\n{e}")

            self.add_to_recent_files(fname)
            return True
        else:
            QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid data.")
            # self.tabs.removeTab(self.tabs.indexOf(tab))
            return False

    def load_h5(self, fname):
        try:
            with h5py.File(fname, 'r') as f:
                if 'frasta_info' not in f.attrs:
                    QtWidgets.QMessageBox.warning(self, "Format error", "HDF5 does not contain a grid data.")
                    return False

                cnt = f.attrs.get('frasta_cnt', 0)
                for i in range(cnt):
                    try:
                        group_name = f"tab_{i:02}"
                        if group_name not in f:
                            continue

                        group = f[group_name]
                        name = group["name"][()].decode("utf-8")
                        grid = group["grid"][:]
                        xi = group["xi"][:]
                        yi = group["yi"][:]
                        px_x = group["px_x"][:]
                        px_y = group["px_y"][:]

                        tab = ScanTab()
                        self.tabs.addTab(tab, str(name))
                        self.tabs.setCurrentWidget(tab)
                        tab.set_data(grid, xi, yi, px_x, px_y)

                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading tab {i}:\n{e}")

            self.add_to_recent_files(fname)
            return True

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
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            # self.tabs.removeTab(self.tabs.indexOf(tab))
            return

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPZ, H5 (*.csv *.dat *.txt *.npz *.h5)")
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
            self, "Save Scan", "", "NPZ (*.npz);;HDF5 (*.h5)"
        )
        if not fname:
            return

        if selected_filter.startswith("NPZ") and not fname.endswith(".npz"):
            fname += ".npz"
        elif selected_filter.startswith("HDF5") and not fname.endswith(".h5"):
            fname += ".h5"

        i = 0
        try:
            if fname.endswith(".npz"):
                to_save = {
                    'frasta_info': "grid_data",
                }
                for name, tab in tabs:
                    to_save[f"name_{i:02}"] = name
                    to_save[f"grid_{i:02}"] = tab.grid
                    to_save[f"xi_{i:02}"] = tab.xi
                    to_save[f"yi_{i:02}"] = tab.yi
                    to_save[f"px_{i:02}"] = tab.px_x
                    to_save[f"py_{i:02}"] = tab.px_y
                    i += 1
                to_save['frasta_cnt'] = i

                np.savez_compressed(fname, **to_save)

            elif fname.endswith(".h5"):
                with h5py.File(fname, 'w') as f:
                    f.attrs['frasta_info'] = "grid_data"
                    for name, tab in tabs:
                        group = f.create_group(f"tab_{i:02}")
                        group.create_dataset("name", data=np.bytes_(name))
                        group.create_dataset("grid", data=tab.grid, compression="gzip", compression_opts=9)
                        group.create_dataset("xi", data=tab.xi, compression="gzip", compression_opts=9)
                        group.create_dataset("yi", data=tab.yi, compression="gzip", compression_opts=9)
                        group.create_dataset("px_x", data=np.atleast_1d(tab.px_x))
                        group.create_dataset("px_y", data=np.atleast_1d(tab.px_y))
                        i += 1
                    f.attrs['frasta_cnt'] = i

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

        def receive_aligned_grids(scan1_aligned_data : GridData, scan2_aligned_data : GridData, idx1=None, idx2=None):
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
        grid1 = tab1.masked
        grid2 = tab2.masked

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
            tab1.px_x, tab1.px_y,
            tab2.px_x, tab2.px_y
        )
        self._profile_viewer.show()
        self._profile_viewer.raise_()
        self._profile_viewer.activateWindow()
