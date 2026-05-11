"""Visualization management for profile viewer.

Handles 3D visualization, image view sizing, volume calculations, and statistics display.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF

from ...viewers import (
    show_point_3d_viewer,
)

import logging
logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages visualization features including 3D view and statistics.
    
    Attributes:
        parent: Reference to parent ProfileViewer window.
    """
    
    def __init__(self, parent):
        """Initialize visualization manager.
        
        Args:
            parent: ProfileViewer instance.
        """
        self.parent = parent
        self._live_point_viewer = None
        self._live_point_view_bounds = None
    
    # ==========================================================================
    # 3D Visualization
    # ==========================================================================
    
    def show_3d_view(self):
        """Open the default 3D viewer showing both scans and the profile line."""
        self._show_3d_view_with_backend(show_point_3d_viewer)

    def show_3d_point_view(self):
        """Open the default QOpenGLWidget-based 3D viewer for the current profile view."""
        self._show_3d_view_with_backend(show_point_3d_viewer)

    def _show_3d_view_with_backend(self, viewer_callable):
        """Open a 3D viewer backend showing both scans and the profile line.

        Args:
            viewer_callable: Function used to display the chosen 3D backend.
        """
        payload = self._build_current_3d_payload()
        if payload is None:
            return
        ref, adj, line_points, bounds = payload

        logger.debug(f"ref0: {self.parent.reference_grid_smooth.shape}, adj0: {self.parent.adjusted_grid_corrected.shape}")
        logger.debug(
            "3D crop bounds: x_min=%s, x_max=%s, y_min=%s, y_max=%s",
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
        )
        logger.debug(f"ref min: {np.nanmin(ref)}, ref max: {np.nanmax(ref)}, ref shape: {ref.shape}")
        logger.debug(f"ref NaN count: {np.isnan(ref).sum()}")
        logger.debug(f"adj min: {np.nanmin(adj)}, adj max: {np.nanmax(adj)}, adj shape: {adj.shape}")
        logger.debug(f"adj NaN count: {np.isnan(adj).sum()}")

        viewer = viewer_callable(
            reference_grid=ref,
            adjusted_grid=adj,
            line_points=line_points,
            separation=self.parent.separation,
            pixel_size_x=self.parent.ref_pixel_um.x(),
            pixel_size_y=self.parent.ref_pixel_um.y()
        )
        if viewer_callable is show_point_3d_viewer:
            self._live_point_viewer = viewer
            self._live_point_view_bounds = bounds
            if viewer is not None:
                viewer.destroyed.connect(self._clear_live_point_viewer)

    def _build_current_3d_payload(self):
        """Build the grid fragment and local ROI polyline for the current image view."""
        viewbox = self.parent.image_view.getView()
        x_range, y_range = viewbox.viewRange()

        # Convert the visible image fragment into integer pixel bounds.
        x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1]))

        shape = self.parent.reference_grid_smooth.shape
        x_min = max(0, x_min)
        x_max = min(shape[1] - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(shape[0] - 1, y_max)
        if x_min > x_max or y_min > y_max:
            return None

        ref = self.parent.reference_grid_smooth[y_min:y_max + 1, x_min:x_max + 1]
        adj = self.parent.adjusted_grid_corrected[y_min:y_max + 1, x_min:x_max + 1]
        line_points = self._build_line_points_for_bounds((x_min, x_max, y_min, y_max))
        return ref, adj, line_points, (x_min, x_max, y_min, y_max)

    def _build_line_points_for_bounds(self, bounds):
        """Convert the full ROI polyline into local coordinates for a crop."""
        if not (hasattr(self.parent, 'rr_full') and hasattr(self.parent, 'cc_full')):
            return None

        x_min, x_max, y_min, y_max = bounds
        line_points = [
            (int(col - x_min), int(row - y_min))
            for col, row in zip(self.parent.cc_full, self.parent.rr_full)
            if x_min <= col <= x_max and y_min <= row <= y_max
        ]
        if len(line_points) < 2:
            return None
        return line_points

    def sync_live_point_view_profile(self):
        """Update the open experimental 3D viewer with the current ROI line.

        The method reuses the crop bounds captured when the 3D view was opened,
        so dragging the ROI updates only the profile line and section plane
        without rebuilding meshes or resetting the camera.
        """
        if self._live_point_viewer is None or self._live_point_view_bounds is None:
            return
        if not self._live_point_viewer.isVisible():
            return

        line_points = self._build_line_points_for_bounds(self._live_point_view_bounds)
        self._live_point_viewer.update_profile_overlay(
            line_points=line_points,
            separation=self.parent.separation,
        )

    def _clear_live_point_viewer(self, *_args):
        """Forget the cached live experimental 3D-viewer connection."""
        self._live_point_viewer = None
        self._live_point_view_bounds = None
    
    def show_preview(self, fragment, title="Region preview"):
        """Show preview window with image fragment.
        
        Args:
            fragment (np.ndarray): Image data to display.
            title (str): Window title.
        """
        if getattr(self.parent, "_preview_win", None) is None:
            self.parent._preview_win = pg.ImageView()
            self.parent._preview_win.setWindowTitle(title)
            self.parent._preview_win.show()
        
        self.parent._preview_win.setImage(fragment)
        self.parent._preview_win.raise_()
        self.parent._preview_win.activateWindow()
    
    # ==========================================================================
    # View Management
    # ==========================================================================
    
    def resize_image_view(self, shape):
        """Refresh image-view geometry without forcing a fixed widget size.

        The profile viewer now relies on normal Qt layout negotiation instead
        of clamping the FRASTA map to a constant pixel size. Keeping the image
        view flexible prevents the right-hand panel from dominating the window
        while still preserving the map aspect ratio through the ViewBox.

        Args:
            shape (tuple): Image shape (height, width).
        """
        if not shape or len(shape) != 2:
            return

        height, width = shape
        if height <= 0 or width <= 0:
            return

        aspect = float(width) / float(height)
        base_min = 240
        if aspect >= 1.0:
            min_width = base_min
            min_height = max(180, int(round(base_min / aspect)))
        else:
            min_height = base_min
            min_width = max(180, int(round(base_min * aspect)))

        if isinstance(self.parent.image_view, QtWidgets.QWidget):
            self.parent.image_view.setMinimumSize(min_width, min_height)
            self.parent.image_view.updateGeometry()
            return

        # Test compatibility fallback for non-widget mocks.
        if hasattr(self.parent.image_view, "setFixedSize"):
            self.parent.image_view.setFixedSize(min_width, min_height)

    def fit_contact_image_view_to_image(self) -> None:
        """Fit the FRASTA image to the visible view using the item's true bounds.

        The binary FRASTA map is displayed through ``ImageView`` with a
        transposed array, so deriving the visible extents from the source-array
        shape is fragile. Using the actual ``ImageItem`` bounding rectangle keeps
        the whole map visible after data reloads, updates, and widget resizes.
        The view range is expanded to match the current viewport aspect ratio,
        so the complete image is always fitted to the shorter viewport side
        instead of clipping the long dimension.
        """
        image_item = self.parent.image_view.getImageItem()
        if image_item is None:
            return

        bounds = image_item.boundingRect()
        if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            return

        self._update_contact_image_boundary(bounds)

        view_box = self.parent.image_view.getView()
        viewport_rect = view_box.sceneBoundingRect()
        if viewport_rect.isNull() or viewport_rect.width() <= 0 or viewport_rect.height() <= 0:
            return

        image_width = float(bounds.width())
        image_height = float(bounds.height())
        viewport_aspect = float(viewport_rect.width()) / float(viewport_rect.height())
        image_aspect = image_width / image_height

        x_center = float(bounds.left()) + 0.5 * image_width
        y_center = float(bounds.top()) + 0.5 * image_height
        target_width = image_width
        target_height = image_height

        # Expand the shorter data axis so the full image fits inside the
        # current viewport while preserving the locked aspect ratio.
        if viewport_aspect >= image_aspect:
            target_width = image_height * viewport_aspect
        else:
            target_height = image_width / viewport_aspect

        x_half = 0.5 * target_width
        y_half = 0.5 * target_height
        view_box.setAspectLocked(True)
        view_box.setLimits(
            xMin=float(bounds.left()) - image_width,
            xMax=float(bounds.right()) + image_width,
            yMin=float(bounds.top()) - image_height,
            yMax=float(bounds.bottom()) + image_height,
        )
        view_box.setRange(
            xRange=(x_center - x_half, x_center + x_half),
            yRange=(y_center - y_half, y_center + y_half),
            padding=0,
        )

    def _update_contact_image_boundary(self, bounds) -> None:
        """Draw or refresh an outline showing the actual FRASTA image extent.

        When the viewport aspect ratio leaves free space around the fitted
        image, this outline makes the true image bounds visible instead of
        letting the map fade into the surrounding background.

        Args:
            bounds: Bounding rectangle of the current ``ImageItem``.
        """
        x0 = float(bounds.left())
        x1 = float(bounds.right())
        y0 = float(bounds.top())
        y1 = float(bounds.bottom())
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]

        if self.parent.image_boundary_item is None:
            self.parent.image_boundary_item = pg.PlotCurveItem(
                xs,
                ys,
                pen=pg.mkPen((180, 180, 180), width=1),
            )
            self.parent.image_view.getView().addItem(
                self.parent.image_boundary_item,
                ignoreBounds=True,
            )
            self.parent.image_boundary_item.setZValue(5)
            return

        self.parent.image_boundary_item.setData(xs, ys)
    
    def get_viewbox_ranges_int(self, shape=None, overflow=False):
        """Get current viewbox range as integer pixel coordinates.
        
        Args:
            shape (tuple, optional): Image shape for clamping (height, width).
            overflow (bool): If True, use floor/ceil for range boundaries.
        
        Returns:
            tuple: (x_min, x_max, y_min, y_max) as integers.
        """
        viewbox = self.parent.image_view.getView()
        x_range, y_range = viewbox.viewRange()
        
        min_range = viewbox.mapToParent(QPointF(x_range[0], y_range[0]))
        max_range = viewbox.mapToParent(QPointF(x_range[1], y_range[1]))
        
        x_range = [min_range.x(), max_range.x()]
        y_range = [min_range.y(), max_range.y()]
        
        logger.debug(f"ViewBox x_range: {x_range}, y_range: {y_range}")
        
        if overflow:
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1])) - 1
            y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1])) - 1
        else:
            x_min, x_max = int(np.ceil(x_range[0])), int(np.floor(x_range[1])) - 1
            y_min, y_max = int(np.ceil(y_range[0])), int(np.floor(y_range[1])) - 1
        
        if shape is not None:
            x_min = max(0, x_min)
            x_max = min(shape[1] - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(shape[0] - 1, y_max)
        
        return x_min, x_max, y_min, y_max
    
    def on_range_changed(self, viewbox, ranges):
        """Handle viewbox range change event - update statistics.
        
        Args:
            viewbox: PyQtGraph viewbox.
            ranges: New range values.
        """
        self.update_volume_info()
    
    # ==========================================================================
    # Statistics and Volume Calculations
    # ==========================================================================
    
    def update_volume_info(self):
        """Calculate and display contact area and volume statistics for current view."""
        if self.parent.binary_contact is None:
            return
        
        x_min, x_max, y_min, y_max = self.get_viewbox_ranges_int(
            shape=self.parent.binary_contact.shape
        )
        
        px_um = self.parent.ref_pixel_um.x()
        py_um = self.parent.ref_pixel_um.y()
        pixel_area_um2 = px_um * py_um
        
        # Get binary fragment in current view
        fragment = self.parent.binary_contact[y_min:y_max+1, x_min:x_max+1]
        
        # Calculate contact area
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
        
        # Display in status bar
        self.parent.statusBar().showMessage(
            f"White fields in view: {white_count}, "
            f"area: {white_area_um2:.4f}μm² ({white_area_mm2}mm²), "
            f"volume: {volume_um3:.4f}μm³ ({volume_mm3:.4f}mm³)"
        )
