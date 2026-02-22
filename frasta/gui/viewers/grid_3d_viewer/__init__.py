"""Grid 3D Viewer submodule - 3D visualization widget with specialized managers.

This package provides:
- Grid3DViewer: Main 3D visualization widget
- show_3d_viewer: Convenience function for displaying 3D data

Specialized managers for internal use:
- LODManager: Level-of-detail rendering management
- ColormapManager: Colormap and value range control
- SurfaceRenderer: Surface geometry and rendering
- ProfileManager: Profile lines and cross-section planes
- CameraController: Camera positioning and view management
"""

from .grid_3d_viewer import Grid3DViewer, show_3d_viewer
from .lod_manager import LODManager
from .colormap_manager import ColormapManager
from .surface_renderer import SurfaceRenderer
from .profile_manager import ProfileManager
from .camera_controller import CameraController

__all__ = [
    'Grid3DViewer',
    'show_3d_viewer',
    'LODManager',
    'ColormapManager',
    'SurfaceRenderer',
    'ProfileManager',
    'CameraController',
]
