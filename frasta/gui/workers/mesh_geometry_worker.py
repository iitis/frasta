"""Background worker for experimental mesh geometry generation.

This worker builds mesh geometry for the experimental QOpenGLWidget-based
viewer in a background thread so the GUI remains responsive while the mesh is
prepared.
"""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class MeshGeometryWorker(QThread):
    """Build one mesh geometry request in a background thread.

    Signals:
        finished_geometry: Emitted with ``(request_id, which, stride, geometry)``.
        failed_geometry: Emitted with ``(request_id, which, stride, message)``.
    """

    finished_geometry = pyqtSignal(int, str, int, object)
    failed_geometry = pyqtSignal(int, str, int, str)

    def __init__(
        self,
        request_id: int,
        which: str,
        stride: int,
        grid,
        dx: float,
        dy: float,
        z_offset: float,
        base_positions=None,
        valid_mask=None,
    ) -> None:
        """Initialize the worker with one mesh-generation request.

        Args:
            request_id: Monotonic request token used to discard stale results.
            which: Cloud identifier, typically ``"ref"`` or ``"adj"``.
            stride: Sampling stride for the generated mesh.
            grid: Source height map.
            dx: Pixel spacing in X.
            dy: Pixel spacing in Y.
            z_offset: Additional Z offset for the mesh.
            base_positions: Optional precomputed point positions reused by the
                mesh builder for the same stride.
            valid_mask: Optional sampled validity mask matching the reused
                point positions.
        """
        super().__init__()
        self.request_id = request_id
        self.which = which
        self.stride = int(stride)
        self.grid = grid
        self.dx = float(dx)
        self.dy = float(dy)
        self.z_offset = float(z_offset)
        self.base_positions = base_positions
        self.valid_mask = valid_mask

    def run(self) -> None:
        """Build mesh geometry and emit the result or a formatted error."""
        try:
            if self.isInterruptionRequested():
                return

            from ..viewers.surface_3d_viewer.point_cloud_geometry import (
                build_mesh_geometry_from_grid,
            )

            geometry = build_mesh_geometry_from_grid(
                self.grid,
                dx=self.dx,
                dy=self.dy,
                z_offset=self.z_offset,
                stride=self.stride,
                cancel_check=self.isInterruptionRequested,
                base_positions=self.base_positions,
                valid_mask=self.valid_mask,
            )
            if self.isInterruptionRequested():
                return
            self.finished_geometry.emit(
                self.request_id,
                self.which,
                self.stride,
                geometry,
            )
        except InterruptedError:
            return
        except Exception as exc:
            self.failed_geometry.emit(
                self.request_id,
                self.which,
                self.stride,
                f"{exc}\n{traceback.format_exc()}",
            )
