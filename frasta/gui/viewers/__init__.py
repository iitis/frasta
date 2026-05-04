"""3D viewers and visualization widgets."""

from __future__ import annotations

import os

from .grid_3d_viewer import Grid3DViewer, show_3d_viewer
from .limited_gl_view import LimitedGLView
from .lod_surface import LODSurface
from .point_3d_viewer import Point3DViewer, show_point_3d_viewer


def legacy_3d_viewer_enabled() -> bool:
    """Return whether the legacy pyqtgraph 3D viewer should be exposed in the UI.

    The default GUI path uses the newer QOpenGLWidget-based backend. The legacy
    backend remains available for fallback testing when the
    ``FRASTA_ENABLE_LEGACY_3D_VIEWER`` environment variable is set to a truthy
    value such as ``1``, ``true``, or ``yes``.
    """
    value = os.environ.get("FRASTA_ENABLE_LEGACY_3D_VIEWER", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}

__all__ = [
    'Grid3DViewer',
    'show_3d_viewer',
    'Point3DViewer',
    'show_point_3d_viewer',
    'legacy_3d_viewer_enabled',
    'LimitedGLView',
    'LODSurface'
]
