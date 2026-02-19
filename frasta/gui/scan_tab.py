"""Scan tab widget for displaying and editing 2D scan data.

This module provides the ScanTab widget which displays scan data with interactive
histogram controls, supports various editing operations (masking, hole filling,
rotation, flipping), and provides tools for setting zero points and removing tilt.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh

from ..core import Surface
from ..processing import fill_holes, remove_outliers, nan_aware_gaussian
from .widgets import ResponsiveInfiniteLine
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)



class ScanTab(QtWidgets.QWidget):
    """Widget for displaying and interacting with a single scan dataset.
    
    Provides an image view with histogram-based contrast adjustment, interactive
    tools for zero point selection, tilt removal, hole filling, and various
    geometric transformations.
    
    Attributes:
        image_view (pg.ImageView): Main image display widget.
        hist_widget (pg.PlotWidget): Histogram display for contrast adjustment.
        grid (np.ndarray): Current 2D scan data.
        xi (np.ndarray): X-coordinate array.
        yi (np.ndarray): Y-coordinate array.
        dx (float): Pixel size in x-direction.
        dy (float): Pixel size in y-direction.
        zero_point_mode (bool): Flag for zero point selection mode.
        tilt_mode (bool): Flag for tilt correction mode.
    """
    def __init__(self, parent=None):
        """Initialize the scan tab widget.
        
        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        self.image_view.getView().setMenuEnabled(False)

        self.hist_widget = pg.PlotWidget()
        self.hist_widget.setMaximumHeight(120)  # Narrow strip
        self.hist_widget.setMenuEnabled(False)
        self.hist_widget.setMouseEnabled(x=False, y=False)

        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.addWidget(self.image_view, stretch=1)
        hlayout.addWidget(self.hist_widget)

        self.setLayout(hlayout)

        self.zero_point_mode = False
        self.tilt_mode = False

        self.seed_points = []
        self.grid = None
        self.masked = None
        self.xi = None
        self.yi = None
        self.dx = None
        self.dy = None

        self.is_colormap = False
        self.current_colormap = 'gray'  # or None

        self.zero_window_size = 15  # or another odd number
        self.zero_sigma = 2.0       # how many deviations to accept as "not outliers"

        self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)


    def update_histogram(self, was_data_negated=False):
        logger.debug(f"update_histogram called: grid is None? {self.grid is None}")
        if self.grid is None:
            logger.warning("update_histogram: self.grid is None!")
            return
        
        data = self.grid[~np.isnan(self.grid)]
        if data.size == 0:
            logger.warning("update_histogram: no valid data (all NaN)")
            self.hist_widget.clear()
            return

        # Remember old positions (if they exist)
        vmin = float(np.min(data))
        vmax = float(np.max(data))

        old_min_line = getattr(self, 'hist_min_line', None)
        old_max_line = getattr(self, 'hist_max_line', None)

        if was_data_negated:
            min_val = np.clip(-old_max_line.value(), vmin, vmax) if old_max_line else vmin
            max_val = np.clip(-old_min_line.value(), vmin, vmax) if old_min_line else vmax
        else:
            # Try to preserve old line positions, but only if they make sense for new data
            if old_min_line is not None and old_max_line is not None:
                old_min = old_min_line.value()
                old_max = old_max_line.value()
                # Check if old values are reasonable for new data range
                if vmin <= old_min <= vmax and vmin <= old_max <= vmax:
                    min_val = old_min
                    max_val = old_max
                else:
                    # Old values don't make sense for new data, use full range
                    min_val = vmin
                    max_val = vmax
                    logger.warning(f"update_histogram: old threshold outside new data range, resetting")
            else:
                min_val = vmin
                max_val = vmax

        y, x = np.histogram(data, bins=1024)
        self.hist_widget.clear()
        self.hist_plot = self.hist_widget.plot(x, y, stepMode="center", fillLevel=0, brush=(150, 150, 150, 150))

        # Block callbacks while setting up histogram lines to avoid recursive calls
        self._updating_histogram = True
        
        self.hist_min_line = ResponsiveInfiniteLine(update_callback=self.update_histogram_threshold, angle=90, movable=True, pen=pg.mkPen('b', width=2), hoverPen=pg.mkPen('y', width=2))
        self.hist_max_line = ResponsiveInfiniteLine(update_callback=self.update_histogram_threshold, angle=90, movable=True, pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('y', width=2))
        self.hist_widget.addItem(self.hist_min_line)
        self.hist_widget.addItem(self.hist_max_line)
        self.hist_min_line.setValue(min_val)
        self.hist_max_line.setValue(max_val)
        
        self._updating_histogram = False
        
        logger.debug(f"update_histogram: final histogram lines set to [{min_val:.2f}, {max_val:.2f}]")

    def on_hist_line_changed(self):
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_image(vmin, vmax)

    def update_histogram_threshold(self, value):
        # Block recursive calls during histogram setup
        if getattr(self, '_updating_histogram', False):
            logger.debug(f"Threshold update blocked during histogram setup: {value}")
            return
            
        logger.debug(f"Threshold updated: {value}" )
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_image(vmin, vmax)

    def set_zero_point_mode(self):
        self.zero_point_mode = True
        # QtWidgets.QMessageBox.information(self, "Select point", "Click on scan view point to be new zero.")

    def set_tilt_mode(self):
        self.tilt_mode = True
        # QtWidgets.QMessageBox.information(self, "Select point", "Click on scan view point to be new zero.")

    def delete_unmasked(self, mask):
        if self.grid is not None:
            self.grid = np.where(mask, self.grid, np.nan)
            self.update_image()
            self.update_histogram()

    def get_surface(self) -> Surface:
        # Handle grid that might contain NaN
        valid_data = self.grid[~np.isnan(self.grid)]
        if valid_data.size > 0:
            grid_min = float(np.min(valid_data))
            grid_max = float(np.max(valid_data))
        else:
            # All NaN, use dummy values
            grid_min = 0.0
            grid_max = 1.0
        
        # Extract origin from stored xi, yi arrays
        x0 = self.xi[0] if hasattr(self, 'xi') and len(self.xi) > 0 else 0.0
        y0 = self.yi[0] if hasattr(self, 'yi') and len(self.yi) > 0 else 0.0
        
        data = Surface(
            height=self.grid,
            dx=self.dx,
            dy=self.dy,
            x0=x0,
            y0=y0,
            vmin=grid_min,
            vmax=grid_max
        )
        if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
            data.vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
            data.vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        return data
    
    def set_surface(self, data: Surface):
        """Set scan data from a Surface object.
        
        Args:
            data (Surface): Surface object containing scan data.
        """
        self.grid = data.height
        self.xi = data.xi
        self.yi = data.yi
        self.dx = data.dx
        self.dy = data.dy
        logger.debug(f"grid: {self.grid.shape}, xmin: {self.xi[0]}, ymin: {self.yi[0]}, px_x: {self.dx}, px_y: {self.dy}")
        # Update histogram first to set hist_min_line and hist_max_line
        self.update_histogram()
        self.update_image()
        # Then set the histogram line values if provided
        if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
            if data.vmin is not None:
                self.hist_min_line.setValue(data.vmin)
            if data.vmax is not None:
                self.hist_max_line.setValue(data.vmax)


    def grid_to_mesh_vectorized(self, grid, dx=1.0, dy=1.0):
        h, w = grid.shape

        # XY grid
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_coords = x_indices * dx
        y_coords = y_indices * dy
        z_coords = grid

        # All vertices
        vertices = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)

        # Mask of valid points (not NaN)
        valid_mask = ~np.isnan(vertices[:, 2])
        index_map = -np.ones(h * w, dtype=int)
        index_map[valid_mask] = np.arange(np.count_nonzero(valid_mask))

        # Triangle indices
        idx_tl = np.ravel_multi_index((np.arange(h - 1)[:, None], np.arange(w - 1)[None, :]), dims=(h, w))
        idx_tr = idx_tl + 1
        idx_bl = idx_tl + w
        idx_br = idx_bl + 1

        # Flattened and combined
        idx_tl = idx_tl.ravel()
        idx_tr = idx_tr.ravel()
        idx_bl = idx_bl.ravel()
        idx_br = idx_br.ravel()

        # Only where all 4 are valid
        valid_quad = (index_map[idx_tl] >= 0) & (index_map[idx_tr] >= 0) & \
                    (index_map[idx_bl] >= 0) & (index_map[idx_br] >= 0)

        # Two triangles per square
        faces_a = np.stack([index_map[idx_tl], index_map[idx_tr], index_map[idx_br]], axis=1)[valid_quad]
        faces_b = np.stack([index_map[idx_tl], index_map[idx_br], index_map[idx_bl]], axis=1)[valid_quad]
        faces = np.vstack([faces_a, faces_b])

        # Filtered vertices
        vertices = vertices[valid_mask]

        return vertices.astype(np.float32), faces.astype(np.int32)


    def save_as_mesh(self, grid, dx=1.38, dy=1.38):
        v, f = self.grid_to_mesh_vectorized(grid, dx, dy)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.export("mesh_output.obj")


    def flip_scan(self, direction='UD', parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = np.flipud(self.grid) if direction=='UD' else np.fliplr(self.grid)
        self.update_image()

    def scan_rot90(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = np.rot90(self.grid)
        self.update_image()

    def invert_scan(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
        self.grid = -self.grid
        self.update_histogram(was_data_negated=True)
        self.update_image()



    def create_repair_dialog(self, sigma=25, threshold=100):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select actions")
        layout = QtWidgets.QVBoxLayout(dialog)
        ch_sigma = QtWidgets.QLabel("sigma:")
        ed_sigma = QtWidgets.QSpinBox()
        ed_sigma.setRange(0, 100)
        ed_sigma.setValue(sigma)
        ch_thresh = QtWidgets.QLabel("threshold:")
        ed_thresh = QtWidgets.QSpinBox()
        ed_thresh.setRange(0, 10000)
        ed_thresh.setValue(threshold)
        ch_newtab = QtWidgets.QCheckBox("create new tab:")
        lbl_newtab = QtWidgets.QLabel("tab label:")
        ed_label = QtWidgets.QLineEdit("name")
        ch_newtab.setDisabled(True)
        ed_label.setDisabled(True)
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(ok_btn)
        hl.addWidget(cancel_btn)
        fl = QtWidgets.QFormLayout()
        fl.addRow(ch_sigma, ed_sigma)
        fl.addRow(ch_thresh, ed_thresh)
        fl.addWidget(ch_newtab)
        fl.addRow(lbl_newtab, ed_label)
        layout.addLayout(fl)
        layout.addLayout(hl)
        def accept():
            dialog.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dialog.reject)
        return dialog, ed_sigma, ed_thresh


    def repair_grid_processing(self, sigma, threshold, mask=None):
        grid_filled = fill_holes(self.grid, mask=mask)
        grid_smooth = nan_aware_gaussian(grid_filled, sigma, mask=mask)
        grid_cleaned = remove_outliers(grid_filled, grid_smooth, threshold, mask=mask)

        if mask is not None:
            self.grid[mask] = grid_cleaned[mask]
        else:
            self.grid = grid_cleaned


    def repair_grid(self, mask=None):
        dialog, ed_sigma, ed_thresh = self.create_repair_dialog()
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        sigma = ed_sigma.value()
        threshold = ed_thresh.value()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        self.repair_grid_processing(sigma, threshold, mask)

        self.update_image()
        QtWidgets.QApplication.restoreOverrideCursor()


    def fill_holes(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return

        tst = np.isnan(self.grid)
        if not np.any(tst):
            return
        
        for (iy, ix) in self.seed_points:
            if tst[iy, ix]:
                filled = flood(tst, seed_point=(iy, ix))
                tst[filled] = False

        if not np.any(tst):
            return

        logger.debug(f"grid.shape: {self.grid.shape}, xi len: {len(self.xi)}, yi len: {len(self.yi)}")

        grid_x, grid_y = np.meshgrid(self.xi, self.yi)

        logger.debug(f"grid_x.shape: {grid_x.shape}, grid_y.shape: {grid_y.shape}, tst.shape: {tst.shape}")

        interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

        valid = ~np.isnan(self.grid)
        interp_values = griddata(
            (grid_x[valid], grid_y[valid]),
            self.grid[valid],
            interp_points,
            method='nearest'
        )

        self.grid[tst] = interp_values
        self.update_image()

    def get_zero_point_value(self, x, y):
        """Calculates a robust zero point value from a local window around (x, y).

        This function returns the mean of non-outlier values within a window centered at (x, y), or the median if all values are outliers or missing.

        Args:
            x (int): The x-coordinate of the window center.
            y (int): The y-coordinate of the window center.

        Returns:
            float: The calculated zero point value, or NaN if no valid data is present.
        """
        s = self.zero_window_size // 2
        grid = self.grid
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)
        window = grid[ymin:ymax, xmin:xmax]

        # Values without NaN
        vals = window[~np.isnan(window)]
        if len(vals) == 0:
            return np.nan

        # Reject outliers (e.g. 2 sigma from median)
        median = np.median(vals)
        std = np.std(vals)
        non_outliers = vals[np.abs(vals - median) < self.zero_sigma * std]

        # If no values remain after rejection - take median
        return median if len(non_outliers) == 0 else np.mean(non_outliers)
        # if len(non_outliers) == 0:
        #     return median
        # return np.mean(non_outliers)

    def fit_plane_to_grid(self, grid, x, y, s=100):
        """Fits a plane to a local window of the grid centered at (x, y).
        
        This function uses linear regression to fit a plane to the non-NaN values in a square window of size (2*s+1) around the specified point.

        Args:
            grid (np.ndarray): The 2D array representing the grid data.
            x (int): The x-coordinate of the window center.
            y (int): The y-coordinate of the window center.
            s (int, optional): The half-size of the window. Defaults to 100.

        Returns:
            tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

        Raises:
            ValueError: If there are not enough valid data points to fit a plane.
        """
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        # Convert to 1D and reject NaN
        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Not enough valid data to fit the plane")

        A = np.vstack((X, Y)).T
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(A, Z)
        a, b = model.coef_
        c = model.intercept_

        return a, b, c


    def fit_plane_to_grid_robust(self, grid, x, y, s=100):
        """Fits a plane to a local window of the grid using a robust regression method.

        This function applies RANSAC regression to fit a plane to the non-NaN values in a square window of size (2*s+1) around the specified point, making it resistant to outliers.

        Args:
            grid (np.ndarray): The 2D array representing the grid data.
            x (int): The x-coordinate of the window center.
            y (int): The y-coordinate of the window center.
            s (int, optional): The half-size of the window. Defaults to 100.

        Returns:
            tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

        Raises:
            ValueError: If there are not enough valid data points to fit a plane.
        """
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Not enough valid data to fit the plane")

        A = np.vstack((X, Y)).T

        # Here we use RANSAC to be robust against outliers
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        base_model = LinearRegression()
        model = RANSACRegressor(base_model, min_samples=min(10, len(Z)), residual_threshold=200.0, random_state=42)
        model.fit(A, Z)
        a, b = model.estimator_.coef_
        c = model.estimator_.intercept_

        return a, b, c

    def fit_plane_to_grid_median_filter(self, grid, x, y, s=100, outlier_thresh=300.0):
        """Fits a plane to a local window of the grid using a median filter to remove outliers.

        This function fits a plane to the non-NaN values in a square window of size (2*s+1) around the specified point, excluding outliers based on the median absolute deviation.

        Args:
            grid (np.ndarray): The 2D array representing the grid data.
            x (int): The x-coordinate of the window center.
            y (int): The y-coordinate of the window center.
            s (int, optional): The half-size of the window. Defaults to 100.
            outlier_thresh (float, optional): The threshold multiplier for outlier removal. Defaults to 300.0.

        Returns:
            tuple: Coefficients (a, b, c) of the fitted plane z = a*x + b*y + c.

        Raises:
            ValueError: If there are not enough valid data points to fit a plane.
        """
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Not enough valid data to fit the plane")

        # Usuwanie wartości odstających na podstawie mediany
        z_median = np.median(Z)
        mad = np.median(np.abs(Z - z_median))  # Median Absolute Deviation

        epsilon = 1e-8
        if mad < epsilon:
            # MAD is too small, fallback to std-based outlier detection or treat all as inliers
            std = np.std(Z)
            if std < epsilon:
                # Data is nearly constant, treat all as inliers
                robust_mask = np.ones_like(Z, dtype=bool)
            else:
                # Use standard deviation for outlier detection
                robust_mask = np.abs(Z - z_median) < 3 * std
        else:
            robust_mask = np.abs(Z - z_median) < outlier_thresh * mad
        
        X = X[robust_mask]
        Y = Y[robust_mask]
        Z = Z[robust_mask]

        if len(Z) < 10:
            raise ValueError("Too little data after outlier rejection")

        A = np.vstack((X, Y)).T
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(A, Z)
        a, b = model.coef_
        c = model.intercept_

        return a, b, c


    def mouse_clicked(self, event):
        """Handles mouse click events on the image view.

        This function updates the grid or seed points based on the current mode and the location of the mouse click.
        It supports setting a new zero point, applying tilt correction, or adding seed points for hole filling.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event containing the click position and modifiers.
        """
        if self.grid is None:
            return
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            if self.zero_point_mode:
                value = self.get_zero_point_value(x, y)
                if np.isnan(value):
                    QtWidgets.QMessageBox.warning(self, "No data available", "The selected point contains no value (NaN).")
                    self.zero_point_mode = False
                    return
                # Shift entire scan in Z axis
                self.grid = self.grid - value
                self.update_image()
                self.zero_point_mode = False
                # QtWidgets.QMessageBox.information(self, "Success", f"Set new zero point at ({x},{y}) with height {value:.2f}.")

                min_val = self.hist_min_line.value()-value
                max_val = self.hist_max_line.value()-value

                self.update_histogram()
                self.hist_min_line.setValue(min_val)
                self.hist_max_line.setValue(max_val)
                return
            
            elif self.tilt_mode:
                self.tilt_mode = False
                # a, b, c = self.fit_plane_to_grid_robust(self.grid, x, y, s=100)
                a, b, c = self.fit_plane_to_grid_median_filter(self.grid, x, y, s=500, outlier_thresh=300.0)
                # Create matrix of same size as grid:
                rows, cols = self.grid.shape
                yy, xx = np.mgrid[0:rows, 0:cols]
                plane = a * xx + b * yy + c

                # Correction:
                self.grid = self.grid + plane
                self.update_image()
                self.update_histogram()
                return

            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self.seed_points.append((y, x))
                scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
                vb.addItem(scatter)

    def toggle_colormap(self):
        self.is_colormap = not self.is_colormap
        self.update_image()


#    @measure_time
    def update_image(self, vmin=None, vmax=None):
        """Updates the displayed image based on the current grid and value range.

        This function refreshes the image view using the current grid data, applying the selected colormap and masking values outside the specified range.

        Args:
            vmin (float, optional): The minimum value for display range. If None, uses histogram limits or grid minimum.
            vmax (float, optional): The maximum value for display range. If None, uses histogram limits or grid maximum.
        """
        logger.debug(f"update_image called: grid is None? {self.grid is None}")
        if self.grid is None:
            logger.warning("update_image: self.grid is None!")
            return
        
        if self.grid is not None:
            
            if vmin is None or vmax is None:
                if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
                    vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
                    vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
                else:
                    # Handle case where grid might be all NaN
                    valid_data = self.grid[~np.isnan(self.grid)]
                    if valid_data.size > 0:
                        vmin = float(np.min(valid_data))
                        vmax = float(np.max(valid_data))
                    else:
                        # Grid is all NaN, use dummy range
                        logger.warning(f"update_image: grid is all NaN! shape={self.grid.shape}")
                        vmin = 0.0
                        vmax = 1.0

            logger.debug(f"update_image: grid.shape={self.grid.shape}, vmin={vmin}, vmax={vmax}")
            
            # Debug: check actual data range before masking
            valid_grid_data = self.grid[~np.isnan(self.grid)]
            if valid_grid_data.size > 0:
                actual_min = float(np.min(valid_grid_data))
                actual_max = float(np.max(valid_grid_data))
                logger.debug(f"update_image: grid actual range: [{actual_min:.2f}, {actual_max:.2f}]")
                logger.debug(f"update_image: threshold range: [{vmin:.2f}, {vmax:.2f}]")
                
                # Check if threshold range makes sense
                if vmin > actual_max or vmax < actual_min:
                    logger.error(f"update_image: threshold range [{vmin:.2f}, {vmax:.2f}] is outside data range [{actual_min:.2f}, {actual_max:.2f}]!")
                    # Use actual data range instead
                    vmin = actual_min
                    vmax = actual_max
                    logger.warning(f"update_image: using actual data range instead")
            
            # IMPORTANT: grid.T creates a VIEW, not a copy!
            # Make a copy immediately to avoid accidentally modifying self.grid
            self.masked = self.grid.T.copy()
            self.masked[(self.masked < vmin) | (self.masked > vmax)] = np.nan
            
            nan_count = np.isnan(self.masked).sum()
            total_count = self.masked.size
            logger.debug(f"update_image: masked has {nan_count}/{total_count} NaN values")
            
            if np.isnan(self.masked).all():
                logger.warning("update_image: masked is all NaN, using zeros")
                self.masked = np.zeros_like(self.masked)
                
            image_item = self.image_view.getImageItem()
            if self.is_colormap:
                lut = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)
                image_item.setLookupTable(lut)
            else:
                image_item.setLookupTable(None)
            self.image_view.setImage(self.masked, autoLevels=True, autoRange=False)
        self.seed_points = []

    def update_image2(self, vmin=None, vmax=None):
        if self.grid is not None:
            if vmin is None or vmax is None:
                if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
                    vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
                    vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
                else:
                    vmin = float(np.min(self.grid))
                    vmax = float(np.max(self.grid))

            # IMPORTANT: grid.T creates a VIEW, not a copy!
            # Make a copy immediately to avoid accidentally modifying self.grid
            self.masked = self.grid.T.copy()
            self.masked[(self.masked < vmin) | (self.masked > vmax)] = np.nan

            # Special value above vmax
            special = vmax + max(1.0, (vmax - vmin) * 0.01)
            masked_for_vis = self.masked.copy()
            masked_for_vis[np.isnan(masked_for_vis)] = special

            image_item = self.image_view.getImageItem()
            if self.is_colormap:
                lut = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)
                lut = np.vstack([lut, [255, 0, 0, 255]])  # red at the end
                image_item.setLookupTable(lut)
                # Key: levels from vmin to special
                self.image_view.setImage(masked_for_vis, autoLevels=False, levels=(vmin, special), autoRange=False)
            else:
                # In gray version you can't highlight, but you can e.g. make special color: 255
                masked_for_vis_gray = self.masked.copy()
                masked_for_vis_gray[np.isnan(masked_for_vis_gray)] = 255
                self.image_view.setImage(masked_for_vis_gray, autoLevels=True, autoRange=False)

        self.seed_points = []

