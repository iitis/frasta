"""Surface rendering and geometry creation for 3D visualization.

This module handles:
- Surface data preparation and downsampling
- Geometry generation (vertices, faces, normals)
- Multiple rendering modes (surface, wireframe, mesh)
- LOD surface placement and styling
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ....utils import get_colormap

import logging

logger = logging.getLogger(__name__)


class SurfaceRenderer:
    """Handles 3D surface rendering with multiple modes and LOD support."""
    
    def __init__(self, view, lod_manager, colormap_manager):
        """Initialize surface renderer.
        
        Args:
            view: The GLViewWidget for 3D rendering.
            lod_manager: LODManager instance for LOD control.
            colormap_manager: ColormapManager for range/colormap control.
        """
        self.view = view
        self.lod_manager = lod_manager
        self.colormap_manager = colormap_manager
        
        # Surface items
        self.surface_ref_item = None
        self.surface_adj_item = None
    
    def prepare_reference_surface(self, reference_grid, max_points=512, clip_abs=1e6,
                                  dx=1.0, dy=1.0, x0=0.0, y0=0.0):
        """Prepare a downsampled reference surface from a 2D grid.
        
        Args:
            reference_grid (np.ndarray): 2D array representing the reference surface grid.
            max_points (int, optional): Maximum number of points along each axis after downsampling.
            clip_abs (float, optional): Absolute value threshold for outlier clipping.
            dx (float, optional): Step size along the x-axis in physical units.
            dy (float, optional): Step size along the y-axis in physical units.
            x0 (float, optional): Origin offset along the x-axis.
            y0 (float, optional): Origin offset along the y-axis.
        
        Returns:
            tuple: (xs, ys, Z, xs_idx, ys_idx)
                - xs: 1D array of x-axis coordinates in physical units.
                - ys: 1D array of y-axis coordinates in physical units.
                - Z: 2D array of downsampled and masked surface values.
                - xs_idx: 1D array of selected x indices.
                - ys_idx: 1D array of selected y indices.
        """
        logger.debug("prepare_reference_surface() - start")
        logger.debug(f"reference_grid.shape: {reference_grid.shape}")
        
        h0, w0 = reference_grid.shape
        step = max(1, min(h0, w0) // max_points)
        
        # Indices (INT) for downsampling
        ys_idx = np.arange(0, h0, step, dtype=np.int32)
        xs_idx = np.arange(0, w0, step, dtype=np.int32)
        
        # Grid at target resolution
        Z = reference_grid[np.ix_(ys_idx, xs_idx)].astype(np.float32, copy=True)
        
        # Mask NaN / outliers (single mask = faster)
        mask = ~np.isfinite(Z) | (np.abs(Z) > clip_abs)
        if mask.any():
            Z[mask] = np.nan
        
        # Axes in units (FLOAT) - for GLSurfacePlotItem
        xs = x0 + dx * xs_idx.astype(np.float32)
        ys = y0 + dy * ys_idx.astype(np.float32)
        
        logger.debug("prepare_reference_surface() - end")
        return xs, ys, Z, xs_idx, ys_idx
    
    def prepare_adjusted_surface(self, adjusted_grid, ys_idx, xs_idx, separation, Z_ref, clip_abs=1e6):
        """Prepare adjusted surface with the same shape as Z_ref.
        
        Args:
            adjusted_grid (np.ndarray): 2D array representing the adjusted surface grid.
            ys_idx (np.ndarray): Y indices used for reference surface.
            xs_idx (np.ndarray): X indices used for reference surface.
            separation (float): Vertical separation between surfaces.
            Z_ref (np.ndarray): Reference surface Z values (for shape matching).
            clip_abs (float, optional): Absolute value threshold for outlier clipping.
        
        Returns:
            np.ndarray: Z_adj with the same shape as Z_ref.
        """
        if adjusted_grid is not None:
            Z_adj = adjusted_grid[np.ix_(ys_idx, xs_idx)].astype(np.float32) + separation
            mask = ~np.isfinite(Z_adj) | (np.abs(Z_adj) > clip_abs)
            if mask.any():
                Z_adj[mask] = np.nan
        else:
            Z_adj = np.full_like(Z_ref, np.nan, dtype=np.float32)
        return Z_adj
    
    def place_surface(self, item_attr, xs, ys, Z, mode, color, colormap, which):
        """Place or update a surface in the 3D view.
        
        Args:
            item_attr (str): Attribute name ('surface_ref_item' or 'surface_adj_item').
            xs (np.ndarray): X coordinates.
            ys (np.ndarray): Y coordinates.
            Z (np.ndarray): Z height values.
            mode (str): Rendering mode ('surface', 'wireframe', or 'mesh').
            color (tuple): Base color (r, g, b, a).
            colormap (str or None): Colormap name or None.
            which (str): 'ref' or 'adj'.
        """
        logger.debug(f"place_surface({which}) - start")
        
        if Z.shape != (len(ys), len(xs)) or np.all(np.isnan(Z)):
            return
        
        if mode == 'mesh':
            # Traditional mesh mode - no LOD
            old = getattr(self, item_attr, None)
            if old is not None:
                try:
                    self.view.removeItem(old)
                except Exception:
                    pass
            
            item = (self.make_voxel_mesh(Z, xs=xs, ys=ys, color=color)
                    if colormap is None
                    else self.make_voxel_mesh(Z, xs=xs, ys=ys, colormap=colormap))
            setattr(self, item_attr, item)
            if item is not None:
                self.view.addItem(item)
            
            # Hide LOD for this channel
            lod = self.lod_manager.get_lod(which)
            if lod:
                lod.set_visible(False)
        else:
            # LOD for surface/wireframe
            lod = self.lod_manager.ensure_lod(which)
            lod.set_visible(True)
            lod.set_data(xs, ys, Z)
            lo, hi = self.colormap_manager.get_lo_hi_for(which, Z)
            lod.update_style(mode=mode, colormap=colormap, base_color=color, lo=lo, hi=hi)
            
            # For compatibility: store LOD manager in this attribute
            setattr(self, item_attr, lod)
        
        logger.debug(f"place_surface({which}) - end")
    
    def add_reference_surface(self, xs, ys, Z, colormap='Metrology'):
        """Add or update the reference surface.
        
        Args:
            xs (np.ndarray): X coordinates.
            ys (np.ndarray): Y coordinates.
            Z (np.ndarray): Z height values.
            colormap (str): Colormap name.
        """
        kolor = (0, 1, 0, 1)
        self.place_surface('surface_ref_item', xs, ys, Z, 
                          self.ref_surface_mode, kolor, colormap, 'ref')
    
    def add_adjusted_surface(self, xs, ys, Z, colormap='Metrology'):
        """Add or update the adjusted surface.
        
        Args:
            xs (np.ndarray): X coordinates.
            ys (np.ndarray): Y coordinates.
            Z (np.ndarray): Z height values.
            colormap (str): Colormap name.
        """
        kolor = (0.2, 0.3, 1, 1)
        self.place_surface('surface_adj_item', xs, ys, Z,
                          self.adj_surface_mode, kolor, colormap, 'adj')
    
    def create_verts_grid(self, Z, xs, ys, cols, rows):
        """Create a grid of 3D vertices for a voxel mesh.
        
        For each grid point, computes the vertex position by averaging the heights
        of neighboring cells and assigning the appropriate x/y coordinate.
        
        Args:
            Z (np.ndarray): 2D array of height values.
            xs (np.ndarray): X coordinates (length = Z.shape[1]).
            ys (np.ndarray): Y coordinates (length = Z.shape[0]).
            cols (int): Number of columns in the grid.
            rows (int): Number of rows in the grid.
        
        Returns:
            np.ndarray: 3D array of vertex positions with shape (rows+1, cols+1, 3).
        """
        verts_grid = np.zeros((rows + 1, cols + 1, 3), dtype=np.float32)
        for i in range(rows + 1):
            for j in range(cols + 1):
                zs = []
                for di in [0, -1]:
                    for dj in [0, -1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(Z[ni, nj]):
                            zs.append(Z[ni, nj])
                z = np.mean(zs) if zs else 0.0
                x = xs[j-1] if j > 0 else xs[0]
                y = ys[i-1] if i > 0 else ys[0]
                verts_grid[i, j] = [x, y, z]
        return verts_grid
    
    def calculate_normals(self, verts_grid, Z, cols, rows):
        """Calculate vertex normals for a voxel mesh.
        
        Computes normals for each vertex by estimating the local surface gradient
        using neighboring vertices, skipping locations where the underlying grid value is NaN.
        
        Args:
            verts_grid (np.ndarray): 3D array of vertex positions with shape (rows+1, cols+1, 3).
            Z (np.ndarray): 2D array of height values.
            cols (int): Number of columns in the grid.
            rows (int): Number of rows in the grid.
        
        Returns:
            np.ndarray: Flattened array of vertex normals with shape ((rows+1)*(cols+1), 3).
        """
        normals_grid = np.zeros_like(verts_grid)
        for i in range(1, rows):
            for j in range(1, cols):
                if np.isnan(Z[i-1, j-1]):
                    continue
                dzdx = (verts_grid[i, j, 2] - verts_grid[i, j-1, 2]) / (verts_grid[i, j, 0] - verts_grid[i, j-1, 0] + 1e-8)
                dzdy = (verts_grid[i, j, 2] - verts_grid[i-1, j, 2]) / (verts_grid[i, j, 1] - verts_grid[i-1, j, 1] + 1e-8)
                n = np.array([-dzdx, -dzdy, 1.0])
                n /= np.linalg.norm(n)
                normals_grid[i, j] = n
        return normals_grid.reshape(-1, 3)
    
    def make_voxel_mesh(self, Z, xs=None, ys=None, color=(0.0, 0.7, 0.0, 1.0), colormap=None):
        """Create an optimized 3D voxel mesh from a 2D grid.
        
        Args:
            Z (np.ndarray): 2D array of height values.
            xs (np.ndarray, optional): X coordinates (length = Z.shape[1]).
            ys (np.ndarray, optional): Y coordinates (length = Z.shape[0]).
            color (tuple, optional): Base color (r, g, b, a).
            colormap (str, optional): Colormap name or None.
        
        Returns:
            GLMeshItem: The generated 3D mesh item, or None if no valid data.
        """
        rows, cols = Z.shape
        if xs is None:
            xs = np.arange(cols)
        if ys is None:
            ys = np.arange(rows)
        
        verts_grid = self.create_verts_grid(Z, xs, ys, cols, rows)
        
        verts = verts_grid.reshape(-1, 3)
        idx = lambda i, j: i * (cols + 1) + j
        
        faces = []
        
        # Top surface
        for i in range(rows):
            for j in range(cols):
                if np.isnan(Z[i, j]):
                    continue
                v00 = idx(i, j)
                v10 = idx(i + 1, j)
                v11 = idx(i + 1, j + 1)
                v01 = idx(i, j + 1)
                faces.extend(([v00, v10, v11], [v00, v11, v01]))
        
        if not len(faces):
            return None
        
        # Map height to colors
        if colormap is None:
            vertex_colors = np.tile(color, (verts.shape[0], 1))
        elif colormap in ('RG', 'B&W'):
            logger.warning("You can't set custom colormaps in 'Mesh' mode, using solid color instead")
            vertex_colors = np.tile(color, (verts.shape[0], 1))
        else:
            z_vals = verts[:, 2]
            z_min, z_max = np.nanmin(z_vals), np.nanmax(z_vals)
            normed = (z_vals - z_min) / (z_max - z_min + 1e-8)
            cmap = get_colormap(colormap)
            vertex_colors = cmap.map(normed, mode='float')
        
        return gl.GLMeshItem(
            vertexes=verts,
            faces=np.array(faces),
            vertexColors=vertex_colors,
            shader='shaded',
            smooth=True,
            drawEdges=False
        )
    
    def compute_z_limits(self, Z_ref, Z_adj, has_adjusted):
        """Compute the minimum and maximum Z values for the 3D view.
        
        Args:
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray): Processed adjusted grid.
            has_adjusted (bool): Whether the adjusted grid is present.
        
        Returns:
            tuple: (z_min, z_max) representing the Z-axis limits.
        """
        if has_adjusted:
            z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
            z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        else:
            z_min = np.nanmin(Z_ref)
            z_max = np.nanmax(Z_ref)
        return z_min, z_max
    
    def remove_existing_items(self):
        """Remove all existing 3D items from the view.
        
        Cleans up surface items and destroys any LOD surface managers.
        """
        # Old single items or item lists
        for item in [self.surface_ref_item, self.surface_adj_item]:
            if item and not hasattr(item, 'set_visible'):  # Not a LODSurface
                try:
                    self.view.removeItem(item)
                except Exception:
                    pass
        
        # LODs
        self.lod_manager.destroy_all()
        
        self.surface_ref_item = None
        self.surface_adj_item = None
    
    def toggle_surface_ref(self, state):
        """Toggle visibility of reference surface.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked).
        """
        obj = self.surface_ref_item
        if hasattr(obj, 'set_visible'):  # LODSurface
            obj.set_visible(bool(state))
        elif obj:
            obj.setVisible(bool(state))
    
    def toggle_surface_adj(self, state):
        """Toggle visibility of adjusted surface.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked).
        """
        obj = self.surface_adj_item
        if hasattr(obj, 'set_visible'):  # LODSurface
            obj.set_visible(bool(state))
        elif obj:
            obj.setVisible(bool(state))
    
    # Store mode for use by place_surface
    ref_surface_mode = 'surface'
    adj_surface_mode = 'surface'
