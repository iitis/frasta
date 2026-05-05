"""Camera positioning and view management for 3D visualization.

This module handles:
- Automatic camera positioning based on data bounds
- View centering and distance calculation
"""

import numpy as np
from PyQt5 import QtGui
import logging

logger = logging.getLogger(__name__)

SCAN_VIEW_AZIMUTH_DEGREES = 90.0
SCAN_VIEW_ELEVATION_DEGREES = -89.0


class CameraController:
    """Manages camera position and view settings in 3D view."""
    
    def __init__(self, view):
        """Initialize camera controller.
        
        Args:
            view: The GLViewWidget for 3D rendering.
        """
        self.view = view
    
    def center_camera(self, xs, ys, Z_ref, Z_adj, line_points, pixel_size_x=1.0, pixel_size_y=1.0):
        """Recenter the 3D camera based on the current data bounds.
        
        Calculates the center of all displayed data and sets the camera position accordingly.
        
        Args:
            xs (np.ndarray): X-axis coordinates in physical units.
            ys (np.ndarray): Y-axis coordinates in physical units.
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray or None): Processed adjusted grid.
            line_points (list or np.ndarray): Points for the profile line (in pixel indices).
            pixel_size_x (float): Physical size of pixel in X direction (micrometers).
            pixel_size_y (float): Physical size of pixel in Y direction (micrometers).
        """
        all_x = list(xs)
        all_y = list(ys)
        all_z = [np.nanmin(Z_ref), np.nanmax(Z_ref)]
        
        if Z_adj is not None:
            if not np.all(np.isnan(Z_adj)):
                all_z += [np.nanmin(Z_adj), np.nanmax(Z_adj)]
        
        if line_points is not None:
            pts = np.array(line_points)
            # Match the scan-oriented 3D convention used by surfaces/profiles.
            all_x += list(pts[:, 0] * pixel_size_x)
            all_y += list(pts[:, 1] * -pixel_size_y)
        
        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if all_z else 0
        
        # Calculate appropriate camera distance based on scene size
        dx = max(all_x) - min(all_x)
        dy = max(all_y) - min(all_y)
        dz = max(all_z) - min(all_z) if all_z else 0
        max_dimension = max(dx, dy, dz)
        distance = max_dimension * 1.8  # Multiplier ensures entire scene is visible
        
        # Use a scan-oriented starting view: X increases to the right on screen,
        # matching the 2D scan display while still keeping a 3D perspective.
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(xc, yc, zc),
            distance=distance,
            azimuth=SCAN_VIEW_AZIMUTH_DEGREES,
            elevation=SCAN_VIEW_ELEVATION_DEGREES,
        )
