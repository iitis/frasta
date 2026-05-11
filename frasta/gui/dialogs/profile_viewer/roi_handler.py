"""ROI (Region of Interest) handling for profile line placement and adjustment.

Manages interactive profile line drawing, repositioning, and visual markers.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from skimage.draw import line

import logging
logger = logging.getLogger(__name__)


class ROIHandler:
    """Handles ROI line interactions and profile extraction.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize ROI handler.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
    
    # ============================================================================
    # Mouse Event Handlers
    # ==========================================================================
    
    def on_image_click(self, event):
        """Handle mouse click on image view - start ROI line placement.
        
        Args:
            event: PyQtGraph mouse event.
        """
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            pos = event.scenePos()
            vb = self.parent.image_view.getView()
            mouse_point = vb.mapSceneToView(pos)
            x_img = int(round(mouse_point.x()))
            y_img = int(round(mouse_point.y()))
            img_shape = self.parent.reference_grid_smooth.shape
            
            # Move handle [0] immediately on click
            self.parent.x1 = np.clip(x_img, 0, img_shape[1]-1)
            self.parent.y1 = np.clip(y_img, 0, img_shape[0]-1)
            self.parent.x2 = np.clip(x_img, 0, img_shape[1]-1)
            self.parent.y2 = np.clip(y_img, 0, img_shape[0]-1)
            
            self.redraw_roi()
            self.update_profile_from_roi()
            
            # Enter drag mode for second handle
            self.parent.line_drag_active = True
            event.accept()
        else:
            pg.ViewBox.mousePressEvent(self.parent.image_view.getView(), event)
    
    def on_image_mouse_release(self, event):
        """Handle mouse release on image view - end ROI line placement.
        
        Args:
            event: PyQtGraph mouse event.
        """
        if self.parent.line_drag_active:
            self.parent.line_drag_active = False
            event.accept()
        else:
            pg.ViewBox.mouseReleaseEvent(self.parent.image_view.getView(), event)
    
    def on_image_mouse_move(self, event):
        """Handle mouse move on image view - drag second ROI endpoint.
        
        Args:
            event: PyQtGraph mouse event.
        """
        if self.parent.line_drag_active:
            pos = event.scenePos()
            vb = self.parent.image_view.getView()
            mouse_point = vb.mapSceneToView(pos)
            x_img = int(round(mouse_point.x()))
            y_img = int(round(mouse_point.y()))
            img_shape = self.parent.reference_grid_smooth.shape
            
            self.parent.x2 = np.clip(x_img, 0, img_shape[1] - 1)
            self.parent.y2 = np.clip(y_img, 0, img_shape[0] - 1)
            
            self.redraw_roi()
            self.update_profile_from_roi()
            event.accept()
        else:
            pg.ViewBox.mouseMoveEvent(self.parent.image_view.getView(), event)
    
    # ==========================================================================
    # ROI Drawing and Management
    # ==========================================================================
    
    def redraw_roi(self):
        """Redraw ROI line on image view with current endpoints."""
        if hasattr(self.parent, 'line_roi'):
            self.parent.image_view.getView().removeItem(self.parent.line_roi)
        
        self.parent.line_roi = pg.LineROI(
            [self.parent.x1, self.parent.y1], 
            [self.parent.x2, self.parent.y2], 
            pen=pg.mkPen('r', width=2), 
            width=1
        )
        self.parent.line_roi.handles[2]['type'] = 'center'
        self.parent.line_roi.sigRegionChanged.connect(self.update_profile_from_roi)
        self.parent.line_roi.sigRegionChanged.connect(self.update_roi_markers)
        
        self.parent.image_view.getView().addItem(self.parent.line_roi)
        self.parent.line_roi.setZValue(10)
        
        self.update_roi_markers()
    
    def update_roi_markers(self):
        """Update visual markers at ROI endpoints with numbers and colors."""
        # Remove old markers
        if hasattr(self.parent, "roi_endpoint_markers"):
            for m in self.parent.roi_endpoint_markers:
                self.parent.image_view.getView().removeItem(m)
        self.parent.roi_endpoint_markers = []
        
        if hasattr(self.parent, "roi_endpoint_labels"):
            for t in self.parent.roi_endpoint_labels:
                self.parent.image_view.getView().removeItem(t)
        self.parent.roi_endpoint_labels = []
        
        # Get current ROI endpoint positions in image coordinates
        handle0 = self.parent.line_roi.getHandles()[0]
        handle1 = self.parent.line_roi.getHandles()[1]
        pt0 = self.parent.line_roi.mapToParent(handle0.pos())
        pt1 = self.parent.line_roi.mapToParent(handle1.pos())
        x1, y1 = pt0.x(), pt0.y()
        x2, y2 = pt1.x(), pt1.y()
        
        # Add markers
        marker1 = pg.ScatterPlotItem(
            [x1], [y1], size=18, 
            pen=pg.mkPen('g', width=3), 
            brush=pg.mkBrush(0, 255, 0, 100), 
            symbol='o'
        )
        marker2 = pg.ScatterPlotItem(
            [x2], [y2], size=18, 
            pen=pg.mkPen('r', width=3), 
            brush=pg.mkBrush(255, 0, 0, 100), 
            symbol='x'
        )
        
        self.parent.image_view.getView().addItem(marker1)
        self.parent.image_view.getView().addItem(marker2)
        self.parent.roi_endpoint_markers = [marker1, marker2]
        
        # Add labels with numbers
        label1 = pg.TextItem("1", color='g', anchor=(0.5, 1.5))
        label1.setPos(x1, y1)
        label2 = pg.TextItem("2", color='r', anchor=(0.5, 1.5))
        label2.setPos(x2, y2)
        
        self.parent.image_view.getView().addItem(label1)
        self.parent.image_view.getView().addItem(label2)
        self.parent.roi_endpoint_labels = [label1, label2]
    
    def clamp_roi_to_image(self):
        """Clamp ROI endpoints to image boundaries."""
        img_shape = self.parent.reference_grid_smooth.shape  # (rows, cols)
        h1 = self.parent.line_roi.getHandles()[0].pos()
        h2 = self.parent.line_roi.getHandles()[1].pos()
        pos1 = self.parent.line_roi.mapToParent(h1).toPoint()
        pos2 = self.parent.line_roi.mapToParent(h2).toPoint()
        
        self.parent.x1 = min(max(pos1.x(), 0), img_shape[1] - 1)
        self.parent.y1 = min(max(pos1.y(), 0), img_shape[0] - 1)
        self.parent.x2 = min(max(pos2.x(), 0), img_shape[1] - 1)
        self.parent.y2 = min(max(pos2.y(), 0), img_shape[0] - 1)
        
        if (pos1.x(), pos1.y(), pos2.x(), pos2.y()) != (self.parent.x1, self.parent.y1, self.parent.x2, self.parent.y2):
            self.redraw_roi()
    
    # ==========================================================================
    # Profile Extraction
    # ==========================================================================
    
    def update_profile_from_roi(self):
        """Extract profile data along ROI line and update dependent views.

        Besides refreshing the 2D profile plot, this method also notifies the
        experimental 3D viewer so an already-open window can update the profile
        line and section plane while the ROI is dragged.
        """
        self.clamp_roi_to_image()
        
        # Get line coordinates
        rr, cc = line(self.parent.y1, self.parent.x1, self.parent.y2, self.parent.x2)
        rr = np.clip(rr, 0, self.parent.reference_grid_smooth.shape[0] - 1)
        cc = np.clip(cc, 0, self.parent.reference_grid_smooth.shape[1] - 1)
        
        # Preserve full line coordinates (needed for 3D view)
        self.parent.rr_full = rr
        self.parent.cc_full = cc
        
        # Extract profile data
        profile_ref = self.parent.reference_grid_smooth[rr, cc]
        profile_adj = (self.parent.adjusted_grid_corrected + self.parent.separation)[rr, cc]
        
        # Keep NaN points for PyQtGraph to break lines
        valid_profile_mask = ~np.isnan(profile_ref) & ~np.isnan(profile_adj)
        
        # For saving and analysis, keep only valid points
        self.parent.rr = rr[valid_profile_mask]
        self.parent.cc = cc[valid_profile_mask]
        
        # For plotting, keep all points with NaNs
        positions_line = np.arange(len(rr)) * self.parent.ref_pixel_um.x() / 1000.0
        
        # Set NaN where data is invalid
        profile_ref_plot = profile_ref.copy()
        profile_adj_plot = profile_adj.copy()
        profile_ref_plot[~valid_profile_mask] = np.nan
        profile_adj_plot[~valid_profile_mask] = np.nan
        
        # Store valid profiles
        self.parent.positions_line = positions_line[valid_profile_mask]
        self.parent.reference_profile = profile_ref[valid_profile_mask]
        self.parent.adjusted_profile = profile_adj[valid_profile_mask]
        
        # Plot profiles (PyQtGraph will break at NaN)
        self.parent.plot_widget.clear()
        self.parent.plot_widget.plot(
            positions_line, profile_ref_plot, 
            pen=pg.mkPen('g', width=2), 
            connect='finite'
        )
        self.parent.plot_widget.plot(
            positions_line, profile_adj_plot, 
            pen=pg.mkPen('b', width=2), 
            connect='finite'
        )
        self.parent.visualization_manager.sync_live_point_view_profile()
