"""Data management for profile viewer.

Handles loading and saving of profile data, including:
- HDF5 scan data loading
- Profile analysis export/import (JSON + NPZ)
- Worker thread coordination
- Data validation and error handling
"""

import os
import numpy as np
import json
from datetime import datetime
from PyQt5.QtCore import QPointF, Qt
from PyQt5 import QtWidgets, QtCore

from ...workers import ProfileWorker
from ....processing import remove_relative_offset, remove_relative_tilt

import logging
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, saving, and worker coordination for profile viewer.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize data manager.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
    
    # ==========================================================================
    # Loading Methods
    # ==========================================================================
    
    def load_new_data(self):
        """Open file dialog and load HDF5 scan data."""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Select HDF5 file", "", "HDF5 files (*.h5);;All files (*)"
        )
        if fname:
            self.load_data_from_file(fname)
    
    def load_data_from_file(self, fname):
        """Load scan data from HDF5 file using background worker.
        
        Args:
            fname (str): Path to HDF5 file.
        """
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.parent.centralWidget().setEnabled(False)
        self.parent.statusBar().showMessage("Loading data...")
        self.parent.progress_bar.setVisible(True)
        self.parent.progress_bar.setRange(0, 0)
        
        self.parent.worker = ProfileWorker(fname, self.parent.sigma)
        self.parent.worker.finished.connect(self.on_worker_finished)
        self.parent.worker.error.connect(self.on_worker_error)
        self.parent.worker.start()
    
    def on_worker_error(self, msg):
        """Handle worker thread error.
        
        Args:
            msg (str): Error message.
        """
        self.parent.progress_bar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.parent.statusBar().showMessage("Error during processing!")
        QtWidgets.QMessageBox.critical(
            self.parent, "Error", "Error during data processing:\n" + msg
        )
    
    def on_worker_finished(self, result):
        """Handle worker thread completion and set loaded data.
        
        Args:
            result (dict): Dictionary with reference_grid and adjusted_grid.
        """
        self.parent.centralWidget().setEnabled(True)
        self.parent.progress_bar.setVisible(False)
        self.parent.statusBar().showMessage("Ready")
        
        self.set_data(
            result["reference_grid"],
            result["adjusted_grid"],
            self.parent.ref_pixel_um.x(),
            self.parent.ref_pixel_um.y(),
            self.parent.adj_pixel_um.x(),
            self.parent.adj_pixel_um.y()
        )
        
        QtWidgets.QApplication.restoreOverrideCursor()
    
    def set_surfaces(self, surface1, surface2):
        """Set scan data from Surface objects.
        
        Args:
            surface1 (Surface): Reference surface object.
            surface2 (Surface): Adjusted surface object.
        """
        self.set_data(
            surface1.height, 
            surface2.height,
            surface1.dx, 
            surface1.dy,
            surface2.dx, 
            surface2.dy
        )
    
    def set_data(self, grid1, grid2, px1_um, py1_um, px2_um, py2_um):
        """Set scan data from grid arrays and pixel sizes.
        
        Args:
            grid1 (np.ndarray): Reference grid data.
            grid2 (np.ndarray): Adjusted grid data.
            px1_um (float): Reference pixel size in X (micrometers).
            py1_um (float): Reference pixel size in Y (micrometers).
            px2_um (float): Adjusted pixel size in X (micrometers).
            py2_um (float): Adjusted pixel size in Y (micrometers).
        """
        self.parent.reference_grid = grid1
        self.parent.adjusted_grid = grid2
        
        self.parent.ref_pixel_um = QPointF(px1_um, py1_um)
        self.parent.adj_pixel_um = QPointF(px2_um, py2_um)
        
        self.parent.reference_grid_smooth = grid1
        self.parent.adjusted_grid_smooth = grid2
        
        self.parent.valid_mask = ~np.isnan(self.parent.reference_grid_smooth) & ~np.isnan(self.parent.adjusted_grid_smooth)
        
        # Validate overlapping data
        num_valid_points = np.sum(self.parent.valid_mask)
        if num_valid_points == 0:
            self.parent.progress_bar.setVisible(False)
            self.parent.statusBar().showMessage("Error: No valid overlapping data")
            QtWidgets.QMessageBox.critical(
                self.parent, 
                "No Valid Data", 
                "There is no valid overlapping data between the reference and adjusted grids.\n\n"
                "This can happen if:\n"
                "- The scans don't overlap spatially\n"
                "- Both scans contain only NaN values in the overlapping region\n"
                "- The alignment/registration failed\n\n"
                "Please check your data and alignment parameters."
            )
            return
        
        # Warn if very few valid points
        total_points = self.parent.valid_mask.size
        valid_percentage = (num_valid_points / total_points) * 100
        if valid_percentage < 5:
            logger.warning(f"Only {valid_percentage:.1f}% of data points are valid for analysis")
            self.parent.statusBar().showMessage(f"Warning: Low data overlap ({valid_percentage:.1f}%)")
        
        # Apply corrections
        self.parent.adjusted_grid_corrected = self.parent.adjusted_grid_smooth + np.nanmean(
            self.parent.reference_grid_smooth - self.parent.adjusted_grid_smooth
        )
        
        if self.parent.checkbox_tilt.isChecked():
            try:
                self.parent.adjusted_grid_corrected = remove_relative_tilt(
                    self.parent.reference_grid_smooth, 
                    self.parent.adjusted_grid_corrected, 
                    self.parent.valid_mask
                )
            except ValueError as e:
                logger.error(f"Tilt correction failed: {e}")
                QtWidgets.QMessageBox.warning(
                    self.parent,
                    "Tilt Correction Failed",
                    f"Could not apply tilt correction: {e}\n\nContinuing without tilt correction."
                )
                self.parent.checkbox_tilt.setChecked(False)
        
        self.parent.adjusted_grid_corrected = remove_relative_offset(
            self.parent.reference_grid_smooth, 
            self.parent.adjusted_grid_corrected, 
            self.parent.valid_mask
        )
        
        # Update plot range
        size_x_mm = self.parent.reference_grid.shape[1] * self.parent.ref_pixel_um.x() / 1000.0
        self.parent.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        
        self.parent.progress_bar.setVisible(False)
        self.parent.statusBar().showMessage("Ready")
        
        # Reset ROI and refresh GUI
        height, width = self.parent.reference_grid_smooth.shape
        self.parent.x1, self.parent.y1 = 0, 0
        self.parent.x2, self.parent.y2 = width - 1, height - 1
        
        self.parent.roi_handler.redraw_roi()
        shape = self.parent.update_plot()
        self.parent.visualization_manager.resize_image_view(shape)
        
        vb = self.parent.image_view.getView()
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
    
    # ==========================================================================
    # Profile Loading Methods
    # ==========================================================================
    
    def load_profiles(self):
        """Load previously saved profile analysis from JSON and NPZ files.
        
        Restores:
        - Profile data and ROI line
        - Binary contact map (if .npz file exists)
        - All settings and parameters
        - Scan grids and metadata
        """
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 
            "Load profile analysis", 
            "", 
            "JSON files (*.json);;All files (*)"
        )
        if not fname:
            return
        
        try:
            # Load JSON
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if NPZ file with binary map exists
            npz_fname = fname.rsplit('.', 1)[0] + '_binary_map.npz'
            has_npz = os.path.exists(npz_fname)
            
            if not has_npz:
                reply = QtWidgets.QMessageBox.warning(
                    self.parent,
                    "Binary map not found",
                    f"Binary map file not found:\n{npz_fname}\n\nCannot fully restore the analysis without scan grids.\nLoad anyway (profile data only)?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    return
            
            # Load from NPZ if available
            if has_npz:
                self._load_from_npz(npz_fname, data)
            else:
                self._load_profiles_only(data)
            
            QtWidgets.QMessageBox.information(
                self.parent,
                "Loaded",
                f"Profile analysis loaded successfully from:\n{fname}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.parent,
                "Error",
                f"Error loading data:\n{str(e)}"
            )
            logger.exception("Error loading profile data")
    
    def _load_from_npz(self, npz_fname, json_data):
        """Load full analysis including grids from NPZ file.
        
        Args:
            npz_fname (str): Path to .npz file with binary map and grids.
            json_data (dict): Loaded JSON data with metadata and settings.
        """
        # Load NPZ data
        npz_data = np.load(npz_fname)
        
        # Restore grids
        self.parent.reference_grid_smooth = npz_data['reference_grid']
        self.parent.adjusted_grid_corrected = npz_data['adjusted_grid']
        self.parent.binary_contact = npz_data['binary_contact']
        
        # Set metadata
        self.parent.separation = float(npz_data['separation'])
        px_x = float(npz_data['pixel_size_um_x'])
        px_y = float(npz_data['pixel_size_um_y'])
        self.parent.ref_pixel_um = QPointF(px_x, px_y)
        self.parent.adj_pixel_um = QPointF(px_x, px_y)
        
        # For compatibility, set raw versions
        self.parent.reference_grid = self.parent.reference_grid_smooth.copy()
        self.parent.adjusted_grid = self.parent.adjusted_grid_corrected.copy() - self.parent.separation
        self.parent.valid_mask = ~np.isnan(self.parent.reference_grid_smooth) & ~np.isnan(self.parent.adjusted_grid_corrected)
        
        # Restore settings from JSON
        settings = json_data.get('settings', {})
        self.parent.spinbox_separation.setValue(settings.get('separation', self.parent.separation))
        self.parent.sigma = settings.get('sigma', 5.0)
        self.parent.checkbox_tilt.setChecked(settings.get('tilt_correction_enabled', True))
        self.parent.checkbox_snap.setChecked(settings.get('snap_to_plot_enabled', True))
        self.parent.spinbox_window_mm.setValue(settings.get('window_size_mm', 0.5))
        
        # Restore profile line
        profile_line = json_data.get('profile_line', {})
        endpoints = profile_line.get('endpoints', {})
        if endpoints:
            start = endpoints.get('start', {})
            end = endpoints.get('end', {})
            self.parent.x1 = start.get('x', 0)
            self.parent.y1 = start.get('y', 0)
            self.parent.x2 = end.get('x', self.parent.reference_grid_smooth.shape[1] - 1)
            self.parent.y2 = end.get('y', self.parent.reference_grid_smooth.shape[0] - 1)
        
        # Restore profiles from JSON
        profiles = json_data.get('profiles', {})
        self.parent.positions_line = np.array(profiles.get('positions_mm', []))
        self.parent.reference_profile = np.array(profiles.get('reference_heights_um', []))
        self.parent.adjusted_profile = np.array(profiles.get('adjusted_heights_um', []))
        
        # Restore line coordinates
        pixel_coords = profile_line.get('pixel_coordinates', {})
        self.parent.cc = np.array(pixel_coords.get('x', []))
        self.parent.rr = np.array(pixel_coords.get('y', []))
        
        # Restore saved points
        if 'saved_points' in json_data:
            self.parent.saved_points = json_data['saved_points']
            # Restore markers on image
            for marker in self.parent.saved_point_markers:
                self.parent.image_view.getView().removeItem(marker)
            self.parent.saved_point_markers.clear()
            
            import pyqtgraph as pg
            for pt in self.parent.saved_points:
                x_img = pt['x_img']
                y_img = pt['y_img']
                marker = pg.ScatterPlotItem(
                    [x_img], [y_img], size=12, 
                    pen=pg.mkPen('g', width=2), 
                    brush=pg.mkBrush(0, 255, 255, 120), 
                    symbol='+'
                )
                self.parent.image_view.getView().addItem(marker)
                self.parent.saved_point_markers.append(marker)
        
        # Refresh GUI
        self._refresh_gui_after_load()
    
    def _load_profiles_only(self, json_data):
        """Load only profile data without full grids (limited functionality).
        
        Args:
            json_data (dict): Loaded JSON data.
        """
        QtWidgets.QMessageBox.information(
            self.parent,
            "Limited data",
            "Loading profile data only (without binary map).\n\n" +
            f"Profile length: {json_data.get('profile_line', {}).get('length_mm', 0):.3f} mm\n" +
            f"Number of points: {json_data.get('profiles', {}).get('number_of_points', 0)}"
        )
    
    def _refresh_gui_after_load(self):
        """Refresh all GUI elements after loading data."""
        import pyqtgraph as pg
        
        # Set plot range
        size_x_mm = self.parent.reference_grid_smooth.shape[1] * self.parent.ref_pixel_um.x() / 1000.0
        self.parent.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        
        # Redraw ROI
        self.parent.roi_handler.redraw_roi()
        
        # Refresh profile plot
        self.parent.plot_widget.clear()
        self.parent.plot_widget.plot(
            self.parent.positions_line, self.parent.reference_profile, 
            pen=pg.mkPen('g', width=2)
        )
        self.parent.plot_widget.plot(
            self.parent.positions_line, self.parent.adjusted_profile, 
            pen=pg.mkPen('b', width=2)
        )
        
        # Refresh binary image view
        shape = self.parent.binary_contact.shape
        self.parent.image_view.setImage(
            self.parent.binary_contact.T.astype(np.uint8), 
            autoRange=False, autoLevels=True
        )
        
        # Set image view size
        self.parent.visualization_manager.resize_image_view(shape)
        
        # Set limits and range
        vb = self.parent.image_view.getView()
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
        
        # Update statistics
        self.parent.visualization_manager.update_volume_info()
        self.parent.statusBar().showMessage("Profile analysis loaded")
    
    # ==========================================================================
    # Saving Methods
    # ==========================================================================
    
    def save_profiles(self):
        """Save current profiles and analysis data to JSON and optionally NPZ files.
        
        Exports:
        - Profile data (positions, heights for both scans)
        - Binary contact map with statistics (area, volume)
        - ROI line coordinates
        - Scan metadata (pixel sizes, shape, corrections applied)
        - Current view range and settings
        """
        if not hasattr(self.parent, 'reference_profile') or not hasattr(self.parent, 'adjusted_profile'):
            QtWidgets.QMessageBox.warning(self.parent, "No data", "No profile data available to save.")
            return
        
        # Open save dialog
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 
            "Save profile analysis", 
            "", 
            "JSON files (*.json);;All files (*)"
        )
        if not fname:
            return
        
        # Add extension if missing
        if not fname.endswith('.json'):
            fname += '.json'
        
        try:
            # Prepare data for export
            data = self._prepare_analysis_data()
            
            # Save JSON
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Optionally save binary map as NPZ
            if self.parent.binary_contact is not None:
                reply = QtWidgets.QMessageBox.question(
                    self.parent,
                    "Save binary map",
                    "Do you also want to save the binary contact map as .npz file?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                
                if reply == QtWidgets.QMessageBox.Yes:
                    npz_fname = fname.rsplit('.', 1)[0] + '_binary_map.npz'
                    self._save_binary_map_npz(npz_fname)
                    QtWidgets.QMessageBox.information(
                        self.parent, 
                        "Saved", 
                        f"Data saved successfully:\n- {fname}\n- {npz_fname}"
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self.parent, 
                        "Saved", 
                        f"Data saved successfully to:\n{fname}"
                    )
            else:
                QtWidgets.QMessageBox.information(
                    self.parent, 
                    "Saved", 
                    f"Data saved successfully to:\n{fname}"
                )
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.parent, 
                "Error", 
                f"Error saving data:\n{str(e)}"
            )
            logger.exception("Error saving profile data")
    
    def _prepare_analysis_data(self):
        """Prepare complete analysis data dictionary for JSON export.
        
        Returns:
            dict: Comprehensive analysis data including profiles, binary map, 
                  statistics, and metadata.
        """
        # Basic metadata
        data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "frasta_version": "1.0",
                "description": "Cross-sectional profile analysis export"
            },
            "scan_info": {
                "reference_grid_shape": list(self.parent.reference_grid.shape),
                "adjusted_grid_shape": list(self.parent.adjusted_grid.shape),
                "reference_pixel_size_um": {
                    "x": float(self.parent.ref_pixel_um.x()),
                    "y": float(self.parent.ref_pixel_um.y())
                },
                "adjusted_pixel_size_um": {
                    "x": float(self.parent.adj_pixel_um.x()),
                    "y": float(self.parent.adj_pixel_um.y())
                }
            },
            "settings": {
                "separation": float(self.parent.separation),
                "sigma": float(self.parent.sigma),
                "tilt_correction_enabled": bool(self.parent.checkbox_tilt.isChecked()),
                "snap_to_plot_enabled": bool(self.parent.checkbox_snap.isChecked()),
                "window_size_mm": float(self.parent.spinbox_window_mm.value())
            }
        }
        
        # Profile line (ROI) data
        data["profile_line"] = {
            "endpoints": {
                "start": {"x": int(self.parent.x1), "y": int(self.parent.y1)},
                "end": {"x": int(self.parent.x2), "y": int(self.parent.y2)}
            },
            "pixel_coordinates": {
                "x": self.parent.cc.tolist() if hasattr(self.parent, 'cc') else [],
                "y": self.parent.rr.tolist() if hasattr(self.parent, 'rr') else []
            },
            "length_mm": float(self.parent.positions_line[-1] - self.parent.positions_line[0]) if len(self.parent.positions_line) > 0 else 0
        }
        
        # Profile height data
        data["profiles"] = {
            "positions_mm": self.parent.positions_line.tolist(),
            "reference_heights_um": self.parent.reference_profile.tolist(),
            "adjusted_heights_um": self.parent.adjusted_profile.tolist(),
            "height_difference_um": (self.parent.reference_profile - self.parent.adjusted_profile).tolist(),
            "number_of_points": len(self.parent.positions_line)
        }
        
        # Binary contact map and statistics
        if self.parent.binary_contact is not None:
            # Get current view range
            x_min, x_max, y_min, y_max = self.parent.visualization_manager.get_viewbox_ranges_int(
                shape=self.parent.binary_contact.shape
            )
            
            # Fragment of binary map in current view
            fragment = self.parent.binary_contact[y_min:y_max+1, x_min:x_max+1]
            
            # Calculate statistics
            px_um = self.parent.ref_pixel_um.x()
            py_um = self.parent.ref_pixel_um.y()
            pixel_area_um2 = px_um * py_um
            
            white_count = np.count_nonzero(fragment)
            white_area_um2 = pixel_area_um2 * white_count
            white_area_mm2 = white_area_um2 * 1e-6
            
            # Calculate volume
            ref = self.parent.reference_grid_smooth[y_min:y_max+1, x_min:x_max+1]
            adj = self.parent.adjusted_grid_corrected[y_min:y_max+1, x_min:x_max+1]
            diff = ref - (adj + self.parent.separation)
            diff_masked = np.where(fragment, diff, 0)
            
            volume_um3 = np.abs(np.sum(diff_masked)) * pixel_area_um2
            volume_mm3 = volume_um3 * 1e-9
            
            data["binary_contact_map"] = {
                "full_shape": list(self.parent.binary_contact.shape),
                "view_range": {
                    "x_min": int(x_min),
                    "x_max": int(x_max),
                    "y_min": int(y_min),
                    "y_max": int(y_max)
                },
                "statistics": {
                    "contact_pixels": int(white_count),
                    "contact_area_um2": float(white_area_um2),
                    "contact_area_mm2": float(white_area_mm2),
                    "volume_um3": float(volume_um3),
                    "volume_mm3": float(volume_mm3)
                },
                "note": "Full binary map can be saved as separate .npz file"
            }
        
        # Saved points (if user marked any)
        if hasattr(self.parent, 'saved_points') and len(self.parent.saved_points) > 0:
            data["saved_points"] = self.parent.saved_points
        
        return data
    
    def _save_binary_map_npz(self, fname):
        """Save binary contact map and related grids to NPZ file.
        
        Args:
            fname (str): Path to output .npz file.
        """
        save_dict = {
            'binary_contact': self.parent.binary_contact,
            'reference_grid': self.parent.reference_grid_smooth,
            'adjusted_grid': self.parent.adjusted_grid_corrected,
            'separation': self.parent.separation,
            'pixel_size_um_x': self.parent.ref_pixel_um.x(),
            'pixel_size_um_y': self.parent.ref_pixel_um.y()
        }
        
        np.savez_compressed(fname, **save_dict)
        logger.info(f"Binary map saved to {fname}")
