"""ROI (Region of Interest) controller for main window.

Handles circular and rectangular ROI operations including:
- Showing/hiding ROI overlays
- Creating masks from ROI
- Applying masks to delete data inside/outside ROI
- Moving ROI between tabs
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)


class ROIController:
    """Controller for ROI-related operations."""
    
    def __init__(self, main_window):
        """Initialize ROI controller.
        
        Args:
            main_window: Reference to MainWindow instance
        """
        self.main_window = main_window
        self.shared_circle_roi = None
        self.shared_rectangle_roi = None
    
    def _is_roi_valid_and_visible(self, roi) -> bool:
        """Safely checks if a Qt ROI object is valid and visible.
        
        Args:
            roi: ROI object to check (CircleROI or RectROI)
            
        Returns:
            bool: True if ROI exists and is visible, False otherwise
        """
        if roi is None:
            return False
        try:
            return roi.isVisible()
        except RuntimeError:
            # Object has been deleted by Qt
            return False
    
    def _is_roi_deleted(self, roi) -> bool:
        """Checks if a Qt ROI object has been deleted.
        
        Args:
            roi: ROI object to check
            
        Returns:
            bool: True if ROI has been deleted, False otherwise
        """
        if roi is None:
            return True
        try:
            # Try to access any property to check if object is valid
            _ = roi.isVisible()
            return False
        except RuntimeError:
            # Object has been deleted by Qt
            return True
    
    def create_mask(self, h: int, w: int) -> np.ndarray | None:
        """Creates a boolean mask for the currently active ROI (circle or rectangle).

        Determines which ROI is visible and generates the corresponding mask for the given shape.

        Args:
            h (int): Height of the mask (number of rows).
            w (int): Width of the mask (number of columns).

        Returns:
            np.ndarray or None: Boolean mask with True inside the ROI, or None if no ROI is active.
        """
        circle_visible = self._is_roi_valid_and_visible(self.shared_circle_roi)
        rect_visible = self._is_roi_valid_and_visible(self.shared_rectangle_roi)

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
    
    def create_circle_mask(self, shape: tuple[int, int], center: tuple[int, int], radius: float) -> np.ndarray:
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
    
    def create_rectangle_mask(self, shape: tuple[int, int], center: tuple[int, int], width: float, height: float) -> np.ndarray:
        """Creates a boolean mask for a rectangle within a 2D array.

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
    
    def apply_roi_mask(self, inside: bool):
        """Applies a mask to the current tab's grid based on the active ROI.

        Generates a mask from the visible ROI and deletes values inside or outside the mask, depending on the 'inside' flag.

        Args:
            inside (bool): If True, deletes values inside the mask; if False, deletes values outside the mask.
        """
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return

        h, w = tab.grid.shape
        mask = self.create_mask(h, w)

        if mask is None:
            return
        
        if inside:
            tab.delete_unmasked(~mask)
        else:
            tab.delete_unmasked(mask)
    
    def del_inside_mask(self):
        """Delete data inside the active ROI mask."""
        self.apply_roi_mask(True)
    
    def del_outside_mask(self):
        """Delete data outside the active ROI mask."""
        self.apply_roi_mask(False)
    
    def move_roi_to_current_tab(self, idx: int):
        """Moves the shared ROI (circle or rectangle) to the currently selected tab.

        Ensures that only the active ROI is visible on the current tab and removed from all others.

        Args:
            idx (int): Index of the newly selected tab.
        """
        tabs = self.main_window.tabs
        
        # Move circle ROI if it exists and is visible
        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            for i in range(tabs.count()):
                tab = tabs.widget(i)
                tab.image_view.getView().removeItem(self.shared_circle_roi)
            tab = tabs.widget(idx)
            tab.image_view.getView().addItem(self.shared_circle_roi)
            self.shared_circle_roi.show()

        # Move rectangle ROI if it exists and is visible
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            for i in range(tabs.count()):
                tab = tabs.widget(i)
                tab.image_view.getView().removeItem(self.shared_rectangle_roi)
            tab = tabs.widget(idx)
            tab.image_view.getView().addItem(self.shared_rectangle_roi)
            self.shared_rectangle_roi.show()
    
    def show_circle_roi(self):
        """Shows or hides the shared circular ROI on the current tab.

        Ensures only the circular ROI is visible, hiding any rectangle ROI if present.
        """
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return

        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            self.shared_circle_roi.setVisible(False)
            return

        # Hide rectangle ROI if present
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            self.shared_rectangle_roi.setVisible(False)

        if self._is_roi_deleted(self.shared_circle_roi):
            h, w = tab.grid.shape
            self.shared_circle_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_circle_roi.setZValue(100)

        try:
            if self.shared_circle_roi not in tab.image_view.getView().allChildren():
                tab.image_view.getView().addItem(self.shared_circle_roi)
        except RuntimeError:
            # ROI was deleted, recreate it
            h, w = tab.grid.shape
            self.shared_circle_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_circle_roi.setZValue(100)
            tab.image_view.getView().addItem(self.shared_circle_roi)
        self.shared_circle_roi.show()
    
    def show_rectangle_roi(self):
        """Shows or hides the shared rectangle ROI on the current tab.

        Ensures only the rectangle ROI is visible, hiding any circular ROI if present.
        """
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return

        # Hide rectangle ROI if already visible, then return
        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            self.shared_rectangle_roi.setVisible(False)
            return

        # Hide circle ROI if present and visible
        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            self.shared_circle_roi.setVisible(False)

        # Create rectangle ROI if it does not exist or was deleted
        if self._is_roi_deleted(self.shared_rectangle_roi):
            h, w = tab.grid.shape
            self.shared_rectangle_roi = pg.RectROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_rectangle_roi.setZValue(100)

        # Add rectangle ROI to the current tab if not already present
        try:
            if self.shared_rectangle_roi not in tab.image_view.getView().allChildren():
                tab.image_view.getView().addItem(self.shared_rectangle_roi)
        except RuntimeError:
            # ROI was deleted, recreate it
            h, w = tab.grid.shape
            self.shared_rectangle_roi = pg.RectROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_rectangle_roi.setZValue(100)
            tab.image_view.getView().addItem(self.shared_rectangle_roi)
        self.shared_rectangle_roi.show()
