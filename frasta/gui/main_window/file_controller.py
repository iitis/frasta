"""File controller for main window.

Handles all file operations including:
- Opening files (CSV, NPZ, H5, STL, AL3D, PLUX, DICOM)
- Saving files (single and multiple scans)
- Recent files management
- Unit conversion dialogs
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize, Qt

from ..scan_tab import ScanTab
from ...core import Surface
from ...io import (
    load_alicona_al3d,
    load_csv_data,
    load_dicom_series,
    load_digital_surf_sur,
    load_h5_data,
    load_keyence_zag,
    load_npz_data,
    load_sensofar_plux,
    load_stl_data,
    save_h5,
    save_npz,
    save_stl,
    suggest_units,
)
from ..workers import GridWorker

import logging
logger = logging.getLogger(__name__)

_OPEN_FILE_FILTER = (
    "All supported (*.csv *.dat *.txt *.npz *.h5 *.stl *.al3d *.plux *.dcm *.dicom *.sur *.spro *.ssur *.zag);;"
    "CSV/DAT/TXT (*.csv *.dat *.txt);;"
    "NPZ (*.npz);;"
    "HDF5 (*.h5);;"
    "STL (*.stl);;"
    "Alicona AL3D (*.al3d);;"
    "Sensofar PLUX (*.plux);;"
    "DICOM (*.dcm *.dicom);;"
    "Digital Surf SUR (*.sur *.spro *.ssur);;"
    "Keyence ZAG (*.zag)"
)


class FileController:
    """Controller for file operations."""
    
    def __init__(self, main_window):
        """Initialize file controller.
        
        Args:
            main_window: Reference to MainWindow instance
        """
        self.main_window = main_window
        self.recent_files = []
        self.max_recent_files = 10
        self.settings = QtCore.QSettings("IITiS PAN", "FRASTA-toolbox")
        self.worker = None
        self.thread = None
        
        self.load_recent_files()
    
    def add_to_recent_files(self, path: str):
        """Add a file path to recent files list.
        
        Args:
            path (str): File path to add
        """
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.main_window.menu_builder.update_recent_files_menu()
        self.settings.setValue("recentFiles", self.recent_files)
    
    def load_recent_files(self):
        """Load recent files list from settings."""
        self.recent_files = self.settings.value("recentFiles", [], type=list)
        self.max_recent_files = 10
    
    def _ask_for_units(self, fname: str) -> tuple[str, str] | None:
        """Ask user about XY and Z coordinate units with suggested choices based on data sample.
        
        Args:
            fname (str): Path to CSV file.
            
        Returns:
            tuple: (units_xy, units_z) where each is 'mm' or 'um', or None if cancelled.
        """
        # Get suggested units from io module
        suggested_xy, suggested_z = suggest_units(fname)
        
        # Dialog z pytaniem
        dialog = QtWidgets.QDialog(self.main_window)
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
    
    def load_csv(self, fname: str, tab: ScanTab):
        """Load CSV file with worker thread.
        
        Args:
            fname (str): Path to CSV file
            tab (ScanTab): Tab to load data into
        """
        # Zapytaj użytkownika o jednostki z sugerowanym wyborem
        units = self._ask_for_units(fname)
        if units is None:
            # Użytkownik anulował
            return
        
        units_xy, units_z = units
        
        dlg = QtWidgets.QProgressDialog("Loading and gridding...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        self.worker = GridWorker(fname, units_xy=units_xy, units_z=units_z)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        target_tab = tab
        target_tabs = self.main_window.tabs

        def handle_success(surface):
            """Apply loaded data and persist file history after success."""
            target_tab.set_surface(surface)
            self.add_to_recent_files(fname)

        def handle_error(message: str):
            """Close the failed placeholder tab and report the loading error."""
            idx = target_tabs.indexOf(target_tab)
            if idx >= 0:
                target_tabs.removeTab(idx)
            target_tab.deleteLater()
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Error",
                f"Error while loading CSV file:\n{message}",
            )

        self.worker.progress.connect(dlg.setValue)
        self.worker.finished.connect(handle_success)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(dlg.cancel)
        self.worker.error.connect(handle_error)
        self.thread.finished.connect(dlg.close)
        self.thread.started.connect(self.worker.process)
        self.thread.start()
        dlg.exec_()
    
    def load_npz(self, fname: str) -> bool:
        """Load NPZ file.
        
        Args:
            fname (str): Path to NPZ file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            surfaces = load_npz_data(fname)
            tabs = self.main_window.tabs
            for surface in surfaces:
                name = surface.metadata.get("name", "Scan")
                tab = ScanTab()
                tabs.addTab(tab, name)
                tabs.setCurrentWidget(tab)
                tab.set_surface(surface)
            self.add_to_recent_files(fname)
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.main_window, "Format error", str(e))
            return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Error while loading:\n{e}")
            return False
    
    def load_h5(self, fname: str) -> bool:
        """Load HDF5 file.
        
        Args:
            fname (str): Path to HDF5 file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            surfaces = load_h5_data(fname)
            tabs = self.main_window.tabs
            for surface in surfaces:
                name = surface.metadata.get("name", "Scan")
                tab = ScanTab()
                tabs.addTab(tab, str(name))
                tabs.setCurrentWidget(tab)
                tab.set_surface(surface)
            self.add_to_recent_files(fname)
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.main_window, "Format error", str(e))
            return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Error while opening HDF5 file:\n{e}")
            return False
    
    def load_stl(self, fname: str, tab: ScanTab) -> None:
        """Load STL file and convert to height map grid.
        
        Args:
            fname (str): Path to STL file.
            tab (ScanTab): Tab widget to load data into.
        """
        # Ask user for resolution
        resolution, ok = QtWidgets.QInputDialog.getDouble(
            self.main_window,
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
        dlg = QtWidgets.QProgressDialog("Loading STL file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()
        
        try:
            surface = load_stl_data(
                fname,
                resolution=resolution,
                progress_callback=dlg.setValue
            )
            tab.set_surface(surface)
            dlg.setValue(100)
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load STL file:\n{e}")
            # Remove the tab if loading failed
            tabs = self.main_window.tabs
            idx = tabs.indexOf(tab)
            if idx >= 0:
                tabs.removeTab(idx)

    def load_al3d(self, fname: str, tab: ScanTab) -> None:
        """Load an Alicona AL3D surface file into a tab.

        Args:
            fname (str): Path to the AL3D file.
            tab (ScanTab): Tab widget to receive the loaded surface.
        """

        dlg = QtWidgets.QProgressDialog("Loading Alicona AL3D file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()

        try:
            surface = load_alicona_al3d(fname, progress_callback=dlg.setValue)
            tab.set_surface(surface)
            dlg.setValue(100)
            self.add_to_recent_files(fname)
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load AL3D file:\n{e}")
            tabs = self.main_window.tabs
            idx = tabs.indexOf(tab)
            if idx >= 0:
                tabs.removeTab(idx)
            tab.deleteLater()

    def load_plux(self, fname: str) -> bool:
        """Load one Sensofar PLUX archive and open each height layer in a tab.

        Args:
            fname (str): Path to the PLUX file.

        Returns:
            bool: True if at least one layer was loaded successfully.
        """

        dlg = QtWidgets.QProgressDialog("Loading Sensofar PLUX file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()

        try:
            surfaces = load_sensofar_plux(fname, progress_callback=dlg.setValue)
            tabs = self.main_window.tabs
            for surface in surfaces:
                name = surface.metadata.get("name", "Scan")
                tab = ScanTab()
                tabs.addTab(tab, str(name))
                tabs.setCurrentWidget(tab)
                tab.set_surface(surface)
            dlg.setValue(100)
            self.add_to_recent_files(fname)
            return True
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load PLUX file:\n{e}")
            return False

    def load_dicom(self, fname: str) -> bool:
        """Load one DICOM slice or same-series directory subset into tabs.

        Args:
            fname (str): Path to one DICOM file from the target series.

        Returns:
            bool: True if at least one slice or frame was loaded successfully.
        """

        dlg = QtWidgets.QProgressDialog("Loading DICOM file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()

        try:
            surfaces = load_dicom_series(fname, progress_callback=dlg.setValue)
            tabs = self.main_window.tabs
            for surface in surfaces:
                name = surface.metadata.get("name", "DICOM")
                tab = ScanTab()
                tabs.addTab(tab, str(name))
                tabs.setCurrentWidget(tab)
                tab.set_surface(surface)
            dlg.setValue(100)
            self.add_to_recent_files(fname)
            return True
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load DICOM file:\n{e}")
            return False

    def load_sur(self, fname: str, tab: ScanTab) -> None:
        """Load one Digital Surf surface file into a tab.

        Args:
            fname (str): Path to the SUR-family file.
            tab (ScanTab): Tab widget to receive the loaded surface.
        """

        dlg = QtWidgets.QProgressDialog("Loading Digital Surf file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()

        try:
            surface = load_digital_surf_sur(fname, progress_callback=dlg.setValue)
            tab.set_surface(surface)
            dlg.setValue(100)
            self.add_to_recent_files(fname)
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load SUR file:\n{e}")
            tabs = self.main_window.tabs
            idx = tabs.indexOf(tab)
            if idx >= 0:
                tabs.removeTab(idx)
            tab.deleteLater()

    def load_zag(self, fname: str) -> bool:
        """Load one Keyence ZAG archive and open each measurement in a tab.

        Args:
            fname (str): Path to the ZAG file.

        Returns:
            bool: True if at least one measurement was loaded successfully.
        """

        dlg = QtWidgets.QProgressDialog("Loading Keyence ZAG file...", None, 0, 100, self.main_window)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setCancelButton(None)
        dlg.setValue(0)
        dlg.show()

        try:
            surfaces = load_keyence_zag(fname)
            tabs = self.main_window.tabs
            for surface in surfaces:
                name = surface.metadata.get("name", "ZAG")
                tab = ScanTab()
                tabs.addTab(tab, str(name))
                tabs.setCurrentWidget(tab)
                tab.set_surface(surface)
            dlg.setValue(100)
            self.add_to_recent_files(fname)
            return True
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load ZAG file:\n{e}")
            return False
    
    def create_tab_and_load(self, fname: str):
        """Create a new tab and load file into it.
        
        Args:
            fname (str): Path to file to load
        """
        tabs = self.main_window.tabs
        suffix = fname.lower()

        if suffix.endswith('.csv') or suffix.endswith('.dat') or suffix.endswith('.txt'):
            tab = ScanTab()
            tabs.addTab(tab, fname.split('/')[-1])
            tabs.setCurrentWidget(tab)
            self.load_csv(fname, tab)
        elif suffix.endswith('.npz'):
            self.load_npz(fname)
        elif suffix.endswith('.h5'):
            self.load_h5(fname)
        elif suffix.endswith('.stl'):
            tab = ScanTab()
            tabs.addTab(tab, fname.split('/')[-1])
            tabs.setCurrentWidget(tab)
            self.load_stl(fname, tab)
            self.add_to_recent_files(fname)
        elif suffix.endswith('.al3d'):
            tab = ScanTab()
            tabs.addTab(tab, fname.split('/')[-1])
            tabs.setCurrentWidget(tab)
            self.load_al3d(fname, tab)
        elif suffix.endswith('.plux'):
            self.load_plux(fname)
        elif suffix.endswith('.dcm') or suffix.endswith('.dicom'):
            self.load_dicom(fname)
        elif suffix.endswith('.sur') or suffix.endswith('.spro') or suffix.endswith('.ssur'):
            tab = ScanTab()
            tabs.addTab(tab, fname.split('/')[-1])
            tabs.setCurrentWidget(tab)
            self.load_sur(fname, tab)
        elif suffix.endswith('.zag'):
            self.load_zag(fname)
        else:
            QtWidgets.QMessageBox.warning(self.main_window, "Unknown format", "Unsupported file type.")
            return
    
    def open_file(self):
        """Show file dialog and open selected file."""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, 
            "Open file", 
            "", 
            _OPEN_FILE_FILTER,
        )
        if not fname:
            return
        self.create_tab_and_load(fname)
    
    def open_file_from_recent(self, path: str):
        """Open file from recent files list.
        
        Args:
            path (str): Path to file to open
        """
        if not QtCore.QFile.exists(path):
            QtWidgets.QMessageBox.warning(self.main_window, "File not found", f"File not found:\n{path}")
            self.recent_files.remove(path)
            self.main_window.menu_builder.update_recent_files_menu()
            return
        self.create_tab_and_load(path)
    
    def save_tabs(self, tabs: list[tuple[str, ScanTab]] = None) -> None:
        """Save tabs to file.
        
        Args:
            tabs: List of (name, tab) tuples to save
        """
        if tabs is None:
            QtWidgets.QMessageBox.warning(self.main_window, "Warning", "No data to save.")
            return

        fname, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window, "Save Scan", "", "NPZ (*.npz);;HDF5 (*.h5);;STL (*.stl)"
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
                surface = tab.get_surface()
                scans.append((name, surface))
            
            if fname.endswith(".npz"):
                save_npz(fname, scans)
            elif fname.endswith(".h5"):
                save_h5(fname, scans)
            elif fname.endswith(".stl"):
                # STL can only save a single scan
                if len(scans) > 1:
                    QtWidgets.QMessageBox.warning(
                        self.main_window, "Multiple scans",
                        "STL format can only save a single scan.\nOnly the first scan will be saved."
                    )
                name, surface = scans[0]
                save_stl(fname, surface, binary=True)

            QtWidgets.QMessageBox.information(self.main_window, "Saved", f"Scan saved to: {fname}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Error while saving:\n{e}")
    
    def save_single_scan(self):
        """Save current scan to file."""
        tab = self.main_window.current_tab()
        if not tab or not hasattr(tab, "grid") or tab.grid is None:
            QtWidgets.QMessageBox.warning(self.main_window, "No data", "No scan in current tab.")
            return

        self.save_tabs([("nowyskan", tab)])
    
    def save_multiple_scans(self):
        """Save multiple scans to single file."""
        tabs_widget = self.main_window.tabs
        if tabs_widget.count() == 0:
            QtWidgets.QMessageBox.warning(self.main_window, "No scans", "No scan tabs are open.")
            return

        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Save selected scans")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Select scans to save and specify dataset names:"))

        checkboxes = []
        lineedits = []
        for i in range(tabs_widget.count()):
            row = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QCheckBox(tabs_widget.tabText(i))
            cb.setChecked(True)
            le = QtWidgets.QLineEdit(tabs_widget.tabText(i).replace(" ", "_"))
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
                    QtWidgets.QMessageBox.warning(self.main_window, "Invalid name", "Each scan must have a dataset name!")
                    return
                tab = tabs_widget.widget(i)
                if not hasattr(tab, "grid") or tab.grid is None:
                    QtWidgets.QMessageBox.warning(self.main_window, "No data", f"Tab '{cb.text()}' has no scan data.")
                    return
                tabs.append((dataset_name, tab))

        if not tabs:
            QtWidgets.QMessageBox.warning(self.main_window, "Nothing to save", "No scans selected.")
            return

        self.save_tabs(tabs)
