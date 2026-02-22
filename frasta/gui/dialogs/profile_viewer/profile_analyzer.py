"""Profile analysis and calculations.

Handles mathematical analysis of profile data, including:
- Linear regression fitting
- Angle calculations
- Tilt correction
- Height difference computations
"""

import numpy as np
from math import atan, degrees
from sklearn.linear_model import LinearRegression
from PyQt5 import QtWidgets, QtCore

from ....processing import remove_relative_offset, remove_relative_tilt

import logging
logger = logging.getLogger(__name__)


class ProfileAnalyzer:
    """Handles profile analysis calculations and fitting.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize profile analyzer.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
    
    def toggle_tilt(self, state):
        """Toggle tilt correction on/off and recompute adjusted grid.
        
        Args:
            state: Checkbox state (checked/unchecked).
        """
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.parent.centralWidget().setEnabled(False)
        
        # Apply offset correction
        offset_correction = np.nanmean(
            self.parent.reference_grid_smooth - self.parent.adjusted_grid_smooth
        )
        self.parent.adjusted_grid_corrected = self.parent.adjusted_grid_smooth + offset_correction
        
        # Apply tilt correction if enabled
        if self.parent.checkbox_tilt.isChecked():
            self.parent.adjusted_grid_corrected = remove_relative_tilt(
                self.parent.reference_grid_smooth, 
                self.parent.adjusted_grid_corrected, 
                self.parent.valid_mask
            )
        
        # Always apply final offset correction
        self.parent.adjusted_grid_corrected = remove_relative_offset(
            self.parent.reference_grid_smooth, 
            self.parent.adjusted_grid_corrected, 
            self.parent.valid_mask
        )
        
        # Refresh visualization
        self.parent.roi_handler.redraw_roi()
        self.parent.update_plot()
        
        self.parent.centralWidget().setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()
    
    def fit_profile(self, x, y):
        """Fit linear regression to profile segment.
        
        Args:
            x (np.ndarray): X positions (mm).
            y (np.ndarray): Y heights (μm).
        
        Returns:
            tuple: (slope, angle_degrees, regression_model)
        """
        x_fit = x.reshape(-1, 1)
        y_fit = y.reshape(-1, 1) / 1000.0  # Convert μm to mm
        
        reg = LinearRegression().fit(x_fit, y_fit)
        slope = reg.coef_[0][0]
        angle = degrees(atan(slope))
        
        return slope, angle, reg
