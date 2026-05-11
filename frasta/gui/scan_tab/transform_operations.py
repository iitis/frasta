"""Transform operations for scan data.

Handles geometric transformations like flip, rotate, and invert.
"""

import numpy as np
from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)


class TransformOperations:
    """Handles geometric transformation operations on scan data."""
    
    @staticmethod
    def flip_scan(grid: np.ndarray, direction: str = 'UD', parent=None) -> np.ndarray:
        """Flip scan data vertically or horizontally.
        
        Args:
            grid (np.ndarray): Grid data to flip
            direction (str): 'UD' for up/down, 'LR' for left/right
            parent (QWidget): Parent widget for error messages
            
        Returns:
            np.ndarray: Flipped grid data
        """
        if grid is None:
            if parent:
                QtWidgets.QMessageBox.warning(parent, "No data", "Load grid first.")
            return None
        
        if direction == 'UD':
            return np.flipud(grid)
        else:
            return np.fliplr(grid)
    
    @staticmethod
    def rotate_90(grid: np.ndarray, parent=None) -> np.ndarray:
        """Rotate scan 90 degrees counter-clockwise.
        
        Args:
            grid (np.ndarray): Grid data to rotate
            parent (QWidget): Parent widget for error messages
            
        Returns:
            np.ndarray: Rotated grid data
        """
        if grid is None:
            if parent:
                QtWidgets.QMessageBox.warning(parent, "No data", "Load grid first.")
            return None
        
        return np.rot90(grid)
    
    @staticmethod
    def invert_z(grid: np.ndarray, parent=None) -> np.ndarray:
        """Invert Z values (negate height).
        
        Args:
            grid (np.ndarray): Grid data to invert
            parent (QWidget): Parent widget for error messages
            
        Returns:
            np.ndarray: Inverted grid data
        """
        if grid is None:
            if parent:
                QtWidgets.QMessageBox.warning(parent, "No data", "Load grid first.")
            return None
        
        return -grid
    
    @staticmethod
    def delete_unmasked(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Delete data outside mask (set to NaN).
        
        Args:
            grid (np.ndarray): Grid data
            mask (np.ndarray): Boolean mask (True = keep, False = delete)
            
        Returns:
            np.ndarray: Grid with masked values set to NaN
        """
        if grid is None:
            return None
        
        return np.where(mask, grid, np.nan)
