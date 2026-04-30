"""Processing controller for main window.

Handles all data processing operations including:
- Basic operations (flip, rotate, invert, fill holes)
- Advanced filtering (bilateral, median, morphological)
- Morphology operations (leveling, polynomial removal)
- Geometric transforms (rotate, rescale, crop)
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore

from ..dialogs import FilterDialog, MorphologyDialog, TransformDialog

import logging
logger = logging.getLogger(__name__)


class ProcessingController:
    """Controller for data processing operations."""
    
    def __init__(self, main_window):
        """Initialize processing controller.
        
        Args:
            main_window: Reference to MainWindow instance
        """
        self.main_window = main_window
    
    def flipUD_scan(self):
        """Flip current scan vertically."""
        if tab := self.main_window.current_tab():
            tab.flip_scan(direction='UD', parent=self.main_window)
    
    def flipLR_scan(self):
        """Flip current scan horizontally."""
        if tab := self.main_window.current_tab():
            tab.flip_scan(direction='LR', parent=self.main_window)
    
    def scan_rot90(self):
        """Rotate current scan 90 degrees counter-clockwise."""
        if tab := self.main_window.current_tab():
            tab.scan_rot90(parent=self.main_window)
    
    def invert_scan(self):
        """Invert Z values of current scan."""
        if tab := self.main_window.current_tab():
            tab.invert_scan(parent=self.main_window)
    
    def fill_holes(self):
        """Fill NaN holes in current scan."""
        if tab := self.main_window.current_tab():
            tab.fill_holes(self.main_window)
    
    def repair_grid(self):
        """Remove holes and outliers in current scan."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return
        h, w = tab.grid.shape
        mask = self.main_window.roi_controller.create_mask(h, w)
        tab.repair_grid(mask=mask)

    def show_surface_roughness_summary(self):
        """Show minimal roughness parameters for the current scan."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self.main_window, "No data", "Please load a scan first!")
            return

        from ...processing import surface_roughness_parameters

        try:
            metrics = surface_roughness_parameters(tab.get_surface())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.main_window, "Roughness summary", str(exc))
            return

        lines = ["Minimal surface roughness summary", "", "Values are in current height units."]
        for name in ("Sa", "Sq", "Sz"):
            lines.append(f"{name}: {metrics[name]:.6g}")

        QtWidgets.QMessageBox.information(
            self.main_window,
            "Surface roughness summary",
            "\n".join(lines),
        )
    
    def apply_advanced_filter(self):
        """Apply advanced filtering to current scan."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self.main_window, "No data", "Please load a scan first!")
            return
        
        dialog = FilterDialog(self.main_window)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        filter_type, params = dialog.get_filter_config()
        
        # Import processing functions
        from ...processing import (
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
                self.main_window, "Success", 
                f"Filter applied successfully!\nShape: {result.shape}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window, "Error", 
                f"Failed to apply filter:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def apply_morphology(self):
        """Apply morphology/leveling operations to current scan."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self.main_window, "No data", "Please load a scan first!")
            return
        
        dialog = MorphologyDialog(self.main_window)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        op_type, params = dialog.get_operation_config()
        
        from ...processing import (
            level_by_plane, remove_polynomial_form, threshold_grid
        )
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            if op_type == "level_ls":
                result = level_by_plane(tab.grid, method='least_squares')
            elif op_type == "level_robust":
                # Call fit_plane_robust directly to use residual_threshold
                from ...processing.morphology import fit_plane_robust
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
                self.main_window, "Success", 
                f"Operation applied successfully!"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window, "Error", 
                f"Failed to apply operation:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def apply_transform(self):
        """Apply geometric transform to current scan."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(self.main_window, "No data", "Please load a scan first!")
            return
        
        dialog = TransformDialog(self.main_window)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        transform_type, params = dialog.get_transform_config()
        
        from ...processing import rotate_grid, rescale_grid, crop_to_valid_region
        
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
                self.main_window, "Success", 
                f"Transform applied successfully!\n"
                f"New shape: {result.shape}\n"
                f"Pixel size: {tab.dx:.3f} x {tab.dy:.3f}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window, "Error", 
                f"Failed to apply transform:\n{str(e)}"
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
