"""Profile line and cross-section plane management for 3D visualization.

This module handles:
- Profile line rendering with gap detection at NaN values
- Cross-section plane generation
- Visibility toggles for profiles and planes
"""

import numpy as np
import pyqtgraph.opengl as gl
import logging

logger = logging.getLogger(__name__)


class ProfileManager:
    """Manages profile lines and cross-section planes in 3D view."""
    
    def __init__(self, view):
        """Initialize profile manager.
        
        Args:
            view: The GLViewWidget for 3D rendering.
        """
        self.view = view
        
        # Profile line items
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None
    
    def add_profile_and_plane(self, reference_grid, adjusted_grid, line_points, 
                             separation, z_min, z_max, pixel_size_x=1.0, pixel_size_y=1.0):
        """Add a cross-section plane and profile lines to the 3D view.
        
        Draws the cross-section plane and profile lines for both reference and adjusted grids,
        if available. Handles coordinate conversion from pixel indices to physical units.
        
        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray or None): The adjusted grid data.
            line_points (list or np.ndarray): Points for the profile line (in pixel indices).
            separation (float): Vertical separation between surfaces.
            z_min (float): Minimum Z value for the plane (fallback).
            z_max (float): Maximum Z value for the plane (fallback).
            pixel_size_x (float): Physical size of pixel in X direction (micrometers).
            pixel_size_y (float): Physical size of pixel in Y direction (micrometers).
        """
        if line_points is None or len(line_points) < 2:
            return
        
        pts = np.array(line_points)
        h, w = reference_grid.shape
        
        # Validate points are within bounds
        valid_mask = (
            (pts[:, 1] >= 0) & (pts[:, 1] < h) &
            (pts[:, 0] >= 0) & (pts[:, 0] < w)
        )
        if not np.all(valid_mask):
            logger.warning("[Grid3DViewer] Some profile points are out of bounds and will be ignored.")
            pts = pts[valid_mask]
            if len(pts) < 2:
                return  # Not enough points to plot a line
        
        # Convert points from pixel indices to physical units (micrometers)
        pts_physical = pts.copy().astype(np.float32)
        pts_physical[:, 0] *= pixel_size_x  # X in micrometers
        pts_physical[:, 1] *= pixel_size_y  # Y in micrometers
        
        try:
            # Use original indices to fetch data from grid
            ref_prof = reference_grid[pts[:, 1], pts[:, 0]]
            adj_prof = adjusted_grid[pts[:, 1], pts[:, 0]] + separation if adjusted_grid is not None else None
            
            # Calculate Z range based on actual profile values
            z_values = [ref_prof]
            if adj_prof is not None:
                z_values.append(adj_prof)
            z_all = np.concatenate(z_values)
            z_valid = z_all[np.isfinite(z_all)]
            
            if len(z_valid) > 0:
                profile_z_min = np.min(z_valid)
                profile_z_max = np.max(z_valid)
                # Add 10% margin
                z_range = profile_z_max - profile_z_min
                margin = max(0.1 * z_range, 1.0)  # Minimum 1 micrometer margin
                profile_z_min -= margin
                profile_z_max += margin
            else:
                # Fallback if no valid data
                profile_z_min = z_min
                profile_z_max = z_max
            
            # Use physical points to draw plane with adjusted range
            self.cross_plane_item = self.add_cross_section_plane(pts_physical, profile_z_min, profile_z_max)
            
            # Find continuous segments (without NaN) and draw them separately
            self.add_profile_line_segments(pts_physical, ref_prof, color=(0, 0.4, 0, 1))
            
            if adjusted_grid is not None:
                self.add_profile_line_segments(pts_physical, adj_prof, color=(0, 0, 1, 1), is_adjusted=True)
        except Exception as e:
            logger.critical(f"[Grid3DViewer] Failed to plot profile lines: {e}")
    
    def add_profile_line_segments(self, pts_physical, z_values, color, is_adjusted=False):
        """Add profile line as separate segments, breaking at NaN values.
        
        Args:
            pts_physical (np.ndarray): XY coordinates in physical units.
            z_values (np.ndarray): Z values for the profile.
            color (tuple): RGBA color for the line.
            is_adjusted (bool): If True, stores in adj_profile_line_item.
        """
        # Find continuous segments (without NaN)
        valid = np.isfinite(z_values)
        
        # Find beginnings and ends of segments
        segments = []
        start = None
        for i in range(len(valid)):
            if valid[i] and start is None:
                start = i
            elif not valid[i] and start is not None:
                if i - start > 1:  # At least 2 points
                    segments.append((start, i))
                start = None
        # Add last segment if exists
        if start is not None and len(valid) - start > 1:
            segments.append((start, len(valid)))
        
        # Draw each segment separately
        items = []
        for start, end in segments:
            pos = np.column_stack((pts_physical[start:end, 0],
                                  pts_physical[start:end, 1],
                                  z_values[start:end]))
            line_item = gl.GLLinePlotItem(pos=pos, color=color, width=2)
            self.view.addItem(line_item)
            items.append(line_item)
        
        # Keep references (for first segment or all)
        if items:
            if is_adjusted:
                self.adj_profile_line_item = items[0] if len(items) == 1 else items
            else:
                self.ref_profile_line_item = items[0] if len(items) == 1 else items
    
    def add_cross_section_plane(self, pts, z_min, z_max):
        """Add a translucent cross-section plane to the 3D view.
        
        Creates and displays a rectangular plane between two points at the specified z-range.
        
        Args:
            pts (np.ndarray): Array of two (x, y) points defining the plane's endpoints.
            z_min (float): Minimum z-value for the plane.
            z_max (float): Maximum z-value for the plane.
        
        Returns:
            GLMeshItem: The mesh item representing the cross-section plane.
        """
        p0, p1 = pts[0], pts[-1]
        rect = np.array([
            [p0[0], p0[1], z_min],
            [p1[0], p1[1], z_min],
            [p1[0], p1[1], z_max],
            [p0[0], p0[1], z_max],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        # Transparency with depth testing
        color = np.array([0.5, 0.5, 0.7, 0.25])
        mesh = gl.GLMeshItem(vertexes=rect, faces=faces, faceColors=np.tile(color, (2, 1)),
                            glOptions='translucent', smooth=False, drawEdges=False)
        # Important: don't use setDepthValue, let depth testing work naturally
        self.view.addItem(mesh)
        return mesh
    
    def toggle_profile_line(self, state):
        """Toggle visibility of profile lines.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked).
        """
        visible = bool(state)
        
        # Handle single objects or lists
        for item_attr in [self.ref_profile_line_item, self.adj_profile_line_item]:
            if item_attr is None:
                continue
            if isinstance(item_attr, list):
                for it in item_attr:
                    it.setVisible(visible)
            else:
                item_attr.setVisible(visible)
    
    def toggle_cross_plane(self, state):
        """Toggle visibility of cross-section plane.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked).
        """
        if self.cross_plane_item:
            self.cross_plane_item.setVisible(bool(state))
    
    def remove_existing_items(self):
        """Remove all profile line and plane items from the view."""
        # Profile line items can be single objects or lists
        for item_attr in [self.ref_profile_line_item, self.adj_profile_line_item]:
            if item_attr is None:
                continue
            if isinstance(item_attr, list):
                for it in item_attr:
                    try:
                        self.view.removeItem(it)
                    except Exception:
                        pass
            else:
                try:
                    self.view.removeItem(item_attr)
                except Exception:
                    pass
        
        # Cross-section plane
        if self.cross_plane_item:
            try:
                self.view.removeItem(self.cross_plane_item)
            except Exception:
                pass
        
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None
