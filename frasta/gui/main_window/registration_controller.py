"""Registration controller for main window.

Handles scan comparison and registration operations including:
- Scan overlay and comparison
- Automatic surface registration (cross-correlation, ICP)
- Profile analysis between two scans
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from functools import partial

from ..dialogs import ProfileViewer, OverlayViewer, RegistrationDialog
from ..scan_tab import ScanTab
from ...core import Surface

import logging
logger = logging.getLogger(__name__)


class RegistrationController:
    """Controller for scan registration and comparison operations."""
    
    def __init__(self, main_window):
        """Initialize registration controller.
        
        Args:
            main_window: Reference to MainWindow instance
        """
        self.main_window = main_window
        self.viewer = None
        self._profile_viewer = None
    
    def compare_scans(self):
        """Open overlay viewer for comparing two scans."""
        tabs = self.main_window.tabs
        if tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Not enough scans",
                "At least 2 scans are required!"
            )
            return

        def receive_aligned_grids(scan1_aligned_data: Surface, scan2_aligned_data: Surface, idx1=None, idx2=None):
            b = idx1 is not None and idx2 is not None
            if b:
                msg = QtWidgets.QMessageBox(self.main_window)
                msg.setWindowTitle("Scan alignment")
                msg.setText("How would you like to save the alignment?")
                btn1 = msg.addButton("As new tabs", QtWidgets.QMessageBox.AcceptRole)
                btn2 = msg.addButton("Overwrite existing", QtWidgets.QMessageBox.ActionRole)
                msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
                msg.exec_()

            if not b or msg.clickedButton() == btn1:
                tab1 = ScanTab()
                tab2 = ScanTab()
                tabs.addTab(tab1, "Aligned ref")
                tabs.addTab(tab2, "Aligned scan2")
            elif msg.clickedButton() == btn2:
                tab1 = tabs.widget(idx1)
                tab2 = tabs.widget(idx2)

            tab1.set_surface(scan1_aligned_data)
            tab2.set_surface(scan2_aligned_data)

        # Dialog wyboru zakładek
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Select scans for comparison")
        layout = QtWidgets.QVBoxLayout(dialog)
        label1 = QtWidgets.QLabel("Reference scan:")
        label2 = QtWidgets.QLabel("Scan to align:")
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [tabs.tabText(i) for i in range(tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
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
                QtWidgets.QMessageBox.warning(dialog, "Error", "Please select two different scans!")
                return
            dialog.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        tab1 = tabs.widget(idx1)
        tab2 = tabs.widget(idx2)

        self.viewer = OverlayViewer(
            tab1.get_surface(),
            tab2.get_surface(),
            on_accept=partial(receive_aligned_grids, idx1=idx1, idx2=idx2),
            parent=self.main_window
        )

        self.viewer.setWindowTitle(f"Comparison: {names[idx1]} vs {names[idx2]}")
        self.viewer.show()
    
    def start_profile_analysis(self):
        """Start profile analysis between two scans."""
        tabs = self.main_window.tabs
        if tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Not enough scans", 
                "You need at least two scans!"
            )
            return

        # Dialog wyboru dwóch zakładek
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Select scans for profile analysis")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Select two scans:"))
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [tabs.tabText(i) for i in range(tabs.count())]
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
            QtWidgets.QMessageBox.warning(self.main_window, "Error", "Select two different scans!")
            return

        tab1 = tabs.widget(idx1)
        tab2 = tabs.widget(idx2)
        grid1 = tab1.grid  # Use grid, not masked (masked is transposed and thresholded)
        grid2 = tab2.grid

        if grid1.shape != grid2.shape:
            h = min(grid1.shape[0], grid2.shape[0])
            w = min(grid1.shape[1], grid2.shape[1])
            reply = QtWidgets.QMessageBox.question(
                self.main_window, "Different sizes",
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
            self._profile_viewer = ProfileViewer(parent=self.main_window)

        self._profile_viewer.set_data(
            grid1, grid2,
            tab1.dx, tab1.dy,
            tab2.dx, tab2.dy
        )
        self._profile_viewer.show()
        self._profile_viewer.raise_()
        self._profile_viewer.activateWindow()
    
    def auto_register_surfaces(self):
        """Automatically register two surfaces."""
        tabs = self.main_window.tabs
        if tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Not enough scans",
                "You need at least two scans for registration!"
            )
            return
        
        # Get scan names
        names = [tabs.tabText(i) for i in range(tabs.count())]
        
        dialog = RegistrationDialog(names, self.main_window)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        ref_idx, mov_idx, method = dialog.get_registration_config()
        
        if ref_idx == mov_idx:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Error",
                "Please select two different surfaces!"
            )
            return
        
        ref_tab = tabs.widget(ref_idx)
        mov_tab = tabs.widget(mov_idx)
        
        from ...processing import auto_register_surfaces, apply_registration
        
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
                    self.main_window,
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
                    self.main_window,
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
            
            QtWidgets.QMessageBox.information(self.main_window, "Success", msg)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window, "Error",
                f"Registration failed:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
