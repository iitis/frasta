"""Registration controller for main window.

Handles scan comparison and registration operations including:
- Scan overlay and comparison
- Automatic surface registration (cross-correlation, ICP)
- Profile analysis between two scans
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from functools import partial

from ..dialogs import OverlayViewer, RegistrationDialog, ContactMapDialog
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

    def _initialize_scan_pair_selectors(
        self,
        first_combo: QtWidgets.QComboBox,
        second_combo: QtWidgets.QComboBox,
        tab_count: int,
    ) -> None:
        """Set a consistent default pair of distinct scan selections.

        The current tab becomes the first selection when possible, and the
        second selector points to the next available tab so the dialog never
        opens with two identical scans selected by default.
        """
        if tab_count < 2:
            return

        current_index = self.main_window.tabs.currentIndex()
        if current_index < 0 or current_index >= tab_count:
            current_index = 0
        second_index = (current_index + 1) % tab_count

        first_combo.setCurrentIndex(current_index)
        second_combo.setCurrentIndex(second_index)

    def _crop_to_common_area_if_needed(self, ref_tab, mov_tab, method: str):
        """Optionally crop mismatched grids before automatic registration.

        Args:
            ref_tab: Reference scan tab.
            mov_tab: Moving scan tab.
            method (str): Selected registration method.

        Returns:
            tuple | None: Tuple ``(reference_grid, moving_grid)`` ready for
            registration, or ``None`` when the user cancels.
        """
        reference_grid = ref_tab.grid
        moving_grid = mov_tab.grid

        if reference_grid.shape == moving_grid.shape:
            return reference_grid, moving_grid

        if method != "correlation":
            return reference_grid, moving_grid

        common_height = min(reference_grid.shape[0], moving_grid.shape[0])
        common_width = min(reference_grid.shape[1], moving_grid.shape[1])
        reply = QtWidgets.QMessageBox.question(
            self.main_window,
            "Different sizes",
            f"Cross-correlation requires equal grid sizes:\n"
            f"{reference_grid.shape} vs {moving_grid.shape}\n\n"
            f"Crop both scans to the common area {common_height}x{common_width} and continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return None

        return (
            reference_grid[:common_height, :common_width],
            moving_grid[:common_height, :common_width],
        )

    def _apply_roi_mask_if_needed(self, reference_grid, moving_grid, ref_tab=None, mov_tab=None):
        """Restrict automatic registration to the active ROI when present.

        Args:
            reference_grid (np.ndarray): Reference grid used for registration.
            moving_grid (np.ndarray): Moving grid used for registration.

        Returns:
            tuple: ROI-masked copies of ``reference_grid`` and ``moving_grid``.
        """
        roi_controller = getattr(self.main_window, "roi_controller", None)
        if roi_controller is None:
            return reference_grid, moving_grid

        reference_mask = roi_controller.create_mask(*reference_grid.shape, tab=ref_tab)
        moving_mask = roi_controller.create_mask(*moving_grid.shape, tab=mov_tab)
        if (
            reference_mask is None or moving_mask is None or
            not isinstance(reference_mask, np.ndarray) or
            not isinstance(moving_mask, np.ndarray)
        ):
            return reference_grid, moving_grid

        reference_masked = np.array(reference_grid, copy=True, dtype=float)
        moving_masked = np.array(moving_grid, copy=True, dtype=float)
        reference_masked[~reference_mask] = np.nan
        moving_masked[~moving_mask] = np.nan
        return reference_masked, moving_masked

    @staticmethod
    def _crop_to_mask_bounds(grid: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Crop a grid to the bounding box of the valid ROI mask."""
        if mask is None or not np.any(mask):
            return None
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        return grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    def _extract_roi_subgrids_if_possible(self, reference_grid, moving_grid, ref_tab=None, mov_tab=None):
        """Crop both grids to the active ROI bounds when ROI is available."""
        roi_controller = getattr(self.main_window, "roi_controller", None)
        if roi_controller is None:
            return reference_grid, moving_grid, False

        reference_mask = roi_controller.create_mask(*reference_grid.shape, tab=ref_tab)
        moving_mask = roi_controller.create_mask(*moving_grid.shape, tab=mov_tab)
        if (
            not isinstance(reference_mask, np.ndarray) or
            not isinstance(moving_mask, np.ndarray)
        ):
            return reference_grid, moving_grid, False

        reference_roi = self._crop_to_mask_bounds(reference_grid, reference_mask)
        moving_roi = self._crop_to_mask_bounds(moving_grid, moving_mask)
        if reference_roi is None or moving_roi is None:
            return reference_grid, moving_grid, False
        return reference_roi, moving_roi, True
    
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
            result_target = "new_tab"
            if b:
                result_target = self.main_window.prompt_result_target(
                    "Scan alignment",
                    "How would you like to save the aligned scans?",
                    overwrite_label="Overwrite source tabs",
                    new_tab_label="Create new tabs",
                )
            if result_target is None:
                return

            if not b or result_target == "new_tab":
                ref_title = f"{tabs.tabText(idx1)} [aligned]" if idx1 is not None else "Aligned ref"
                mov_title = f"{tabs.tabText(idx2)} [aligned]" if idx2 is not None else "Aligned scan"
                tab1 = self.main_window.create_surface_tab(scan1_aligned_data, ref_title)
                tab2 = self.main_window.create_surface_tab(scan2_aligned_data, mov_title)
                if idx1 is not None and idx2 is not None:
                    self.main_window.copy_scan_display_settings(tabs.widget(idx1), tab1)
                    self.main_window.copy_scan_display_settings(tabs.widget(idx2), tab2)
                return

            if result_target == "overwrite":
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
        self._initialize_scan_pair_selectors(cb1, cb2, tabs.count())
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
            parent=self.main_window,
            reference_tab=tab1,
            moving_tab=tab2,
        )

        self.viewer.setWindowTitle(f"Comparison: {names[idx1]} vs {names[idx2]}")
        self.viewer.show()

    def open_contact_map_dialog(self):
        """Open the interactive contact map analysis dialog for two scans."""
        tabs = self.main_window.tabs
        if tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Not enough scans",
                "At least 2 scans are required for contact map analysis."
            )
            return

        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Select scans for contact map analysis")
        layout = QtWidgets.QVBoxLayout(dialog)

        label1 = QtWidgets.QLabel("Surface A (reference):")
        label2 = QtWidgets.QLabel("Surface B (conjugate):")
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [tabs.tabText(i) for i in range(tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        self._initialize_scan_pair_selectors(cb1, cb2, tabs.count())

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
                QtWidgets.QMessageBox.warning(
                    dialog, "Error", "Please select two different scans."
                )
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

        surf_a = tab1.get_surface()
        surf_b = tab2.get_surface()

        if surf_a.height.shape != surf_b.height.shape:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Shape mismatch",
                f"The two scans have different grid shapes:\n"
                f"{surf_a.height.shape} vs {surf_b.height.shape}\n\n"
                "Please align and crop them to the same size before running "
                "contact map analysis."
            )
            return

        self._contact_map_dialog = ContactMapDialog(
            surf_a,
            surf_b,
            parent=self.main_window,
            title_a=names[idx1],
            title_b=names[idx2],
        )
        self._contact_map_dialog.setWindowTitle(
            f"Contact map: {names[idx1]} vs {names[idx2]}"
        )
        self._contact_map_dialog.show()
    
    def open_frasta_docks(self):
        """Load a scan pair into the dockable FRASTA binary + profile panels."""
        tabs = self.main_window.tabs
        if tabs.count() < 2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Not enough scans",
                "At least 2 scans are required for FRASTA panel analysis."
            )
            return

        names = [tabs.tabText(i) for i in range(tabs.count())]
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Select scans for FRASTA panels")
        layout = QtWidgets.QVBoxLayout(dialog)
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        cb1.addItems(names)
        cb2.addItems(names)
        self._initialize_scan_pair_selectors(cb1, cb2, tabs.count())
        layout.addWidget(QtWidgets.QLabel("Surface A (reference):"))
        layout.addWidget(cb1)
        layout.addWidget(QtWidgets.QLabel("Surface B (conjugate):"))
        layout.addWidget(cb2)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        if idx1 == idx2:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Error", "Please select two different scans."
            )
            return

        tab1 = tabs.widget(idx1)
        tab2 = tabs.widget(idx2)
        surf_a = tab1.get_surface()
        surf_b = tab2.get_surface()

        grid_a = surf_a.height
        grid_b = surf_b.height

        if grid_a.shape != grid_b.shape:
            h = min(grid_a.shape[0], grid_b.shape[0])
            w = min(grid_a.shape[1], grid_b.shape[1])
            reply = QtWidgets.QMessageBox.question(
                self.main_window, "Different sizes",
                f"The scans vary in size:\n"
                f"{grid_a.shape} vs {grid_b.shape}\n"
                f"Crop both to a common area {h}×{w} and continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            grid_a = grid_a[:h, :w]
            grid_b = grid_b[:h, :w]

        ctrl = self.main_window.frasta_controller
        ctrl.set_data(grid_a, grid_b, dx=float(surf_a.dx), dy=float(surf_a.dy))
        ctrl.binary_dock.show()
        ctrl.binary_dock.raise_()
        ctrl.profile_dock.show()

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
        
        registration_config = dialog.get_registration_config()
        if len(registration_config) == 5:
            ref_idx, mov_idx, method, refine, stable_region = registration_config
        elif len(registration_config) == 4:
            ref_idx, mov_idx, method, refine = registration_config
            stable_region = False
        else:
            ref_idx, mov_idx, method = registration_config
            refine = False
            stable_region = False
        
        if ref_idx == mov_idx:
            QtWidgets.QMessageBox.warning(
                self.main_window, "Error",
                "Please select two different surfaces!"
            )
            return
        
        ref_tab = tabs.widget(ref_idx)
        mov_tab = tabs.widget(mov_idx)
        reference_grid, moving_grid, used_roi_subgrids = self._extract_roi_subgrids_if_possible(
            ref_tab.grid,
            mov_tab.grid,
            ref_tab=ref_tab,
            mov_tab=mov_tab,
        )
        registration_inputs = self._crop_to_common_area_if_needed(
            type("GridHolder", (), {"grid": reference_grid})(),
            type("GridHolder", (), {"grid": moving_grid})(),
            method,
        )
        if registration_inputs is None:
            return
        reference_grid, moving_grid = registration_inputs
        if not used_roi_subgrids:
            reference_grid, moving_grid = self._apply_roi_mask_if_needed(
                reference_grid,
                moving_grid,
                ref_tab=ref_tab,
                mov_tab=mov_tab,
            )
        if np.all(np.isnan(reference_grid)) or np.all(np.isnan(moving_grid)):
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "ROI registration",
                "The active ROI does not contain enough valid data for registration.",
            )
            return
        
        from ...processing import auto_register_surfaces, apply_registration
        
        cursor_active = False
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            cursor_active = True
            
            # Perform registration
            params = auto_register_surfaces(
                reference_grid,
                moving_grid,
                method=method,
                refine=refine,
                stable_region=stable_region,
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
                    reference_grid,
                    moving_grid,
                    method='icp',
                    refine=refine,
                    stable_region=stable_region,
                )
                logger.info(f"ICP result: RMSE={params['rmse']:.1f} nm, translation={params['translation']}")
            
            # Apply the estimated transform to the full moving surface, not only
            # to the ROI/common-area subset used during parameter estimation.
            h, w = mov_tab.grid.shape
            if not hasattr(mov_tab, 'xi') or mov_tab.xi is None:
                xi = np.arange(w) * (mov_tab.dx or 1.0)
            else:
                xi = mov_tab.xi[:w]
            if not hasattr(mov_tab, 'yi') or mov_tab.yi is None:
                yi = np.arange(h) * (mov_tab.dy or 1.0)
            else:
                yi = mov_tab.yi[:h]
            
            # Apply registration to moving surface
            registered, new_xi, new_yi, new_dx, new_dy = apply_registration(
                mov_tab.grid,
                xi,
                yi,
                mov_tab.dx or 1.0,
                mov_tab.dy or 1.0,
                params['translation'],
                params.get('rotation', 0.0)
            )
            
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
                cursor_active = False
                return

            source_surface = mov_tab.get_surface()
            result_surface = Surface(
                height=registered,
                dx=new_dx,
                dy=new_dy,
                x0=float(new_xi[0]) if len(new_xi) else 0.0,
                y0=float(new_yi[0]) if len(new_yi) else 0.0,
                unit=source_surface.unit,
                metadata={
                    **source_surface.metadata,
                    "last_operation": f"auto-registration ({method})",
                },
                vmin=source_surface.vmin,
                vmax=source_surface.vmax,
            )
            QtWidgets.QApplication.restoreOverrideCursor()
            cursor_active = False
            result_target = self.main_window.prompt_result_target(
                "Registered scan",
                "How would you like to save the registered moving surface?",
                overwrite_label="Overwrite moving scan",
            )
            if result_target is None:
                return
            if result_target == "overwrite":
                mov_tab.set_surface(result_surface)
            else:
                target_tab = self.main_window.create_surface_tab(
                    result_surface,
                    f"{tabs.tabText(mov_idx)} [registered]",
                )
                self.main_window.copy_scan_display_settings(mov_tab, target_tab)
            
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
            if cursor_active:
                QtWidgets.QApplication.restoreOverrideCursor()
