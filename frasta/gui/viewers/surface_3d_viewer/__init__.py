"""3D surface-viewer package used by the active GUI backend.

This package provides the point or mesh-based 3D viewer used by the current
GUI, together with shared geometry and colormap helpers.
"""

from .colormap_manager import ColormapManager
from .point_3d_viewer import Point3DViewer, show_point_3d_viewer, close_point_3d_viewer

__all__ = [
    'Point3DViewer',
    'show_point_3d_viewer',
    'close_point_3d_viewer',
    'ColormapManager',
]
