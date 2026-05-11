"""Controller coordinating FrastaBinaryDock and FrastaProfileDock.

Adapted from d:/praca/pyDpVision/plugins/frasta/frastaController.py.
The original used dpVision's GridData64. This version accepts Surface
objects or plain NumPy arrays with dx/dy pixel sizes.

Wiring
------
binary_dock.profileLineChanged  -> controller -> profile_dock.set_profiles()
binary_dock.separationChanged   -> controller -> profile_dock + 3-D viewer
profile_dock.profilePointSelected -> controller -> binary_dock.on_point_selected()
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from skimage.draw import line as skimage_line

from .frasta_binary_dock import FrastaBinaryDock
from .frasta_profile_dock import FrastaProfileDock
from ..viewers import show_point_3d_viewer

import logging
logger = logging.getLogger(__name__)


class FrastaController(QtCore.QObject):
    """Mediator between FRASTA dock widgets.

    Creates and owns both docks. Connect ``separation_changed`` or
    ``point_selected`` to additional subscribers (e.g. a 3-D viewer).
    """

    # Re-exported for convenience
    separation_changed = QtCore.pyqtSignal(float)  # mirrors binary_dock.separationChanged
    point_selected     = QtCore.pyqtSignal(int, int)  # (row, col) on binary map

    def __init__(self, parent=None):
        super().__init__(parent)

        self.binary_dock   = FrastaBinaryDock()
        self.profile_dock  = FrastaProfileDock()

        # Internal profile state
        self._current_rr: np.ndarray | None = None
        self._current_cc: np.ndarray | None = None
        self._3d_viewer = None  # reference to open Point3DViewer, if any

        # Wire docks together
        self.binary_dock.profileLineChanged.connect(self._on_profile_line_changed)
        self.binary_dock.separationChanged.connect(self._on_separation_changed)
        self.profile_dock.profilePointSelected.connect(self._on_profile_point_selected)

        # ROI line visible only when profile dock is visible
        self.profile_dock.visibilityChanged.connect(self.binary_dock.set_roi_visible)
        self.binary_dock.set_roi_visible(False)  # hidden by default

        # Live cursor: profile mouse hover → binary map + 3D viewer
        self.profile_dock.cursorMoved.connect(self._on_profile_cursor_moved)

        # 3D viewer
        self.binary_dock.show3dRequested.connect(self._show_3d_view)

        # Forward point marker to binary dock
        self.point_selected.connect(self.binary_dock.on_point_selected)

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    def set_data(
        self,
        grid1: np.ndarray,
        grid2: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> None:
        """Load a pair of aligned height grids.

        Parameters
        ----------
        grid1, grid2:
            Reference and adjusted height maps, shape (rows, cols), float.
        dx, dy:
            Pixel size in µm.
        """
        diff = grid1 - grid2
        self.binary_dock.set_data(diff, grid1, grid2, dx=dx, dy=dy)

    def set_surfaces(self, surface_a, surface_b) -> None:
        """Convenience wrapper accepting Surface objects."""
        dx = float(surface_a.dx)
        dy = float(surface_a.dy)
        self.set_data(surface_a.height, surface_b.height, dx=dx, dy=dy)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(tuple)
    def _on_profile_line_changed(self, coords: tuple) -> None:
        """Re-extract profiles when the ROI line moves."""
        c0, r0, c1, r1 = coords
        rr, cc = skimage_line(int(r0), int(c0), int(r1), int(c1))
        self._current_rr = rr
        self._current_cc = cc
        self._refresh_profile()
        if self._3d_viewer is not None:
            self._3d_viewer.update_profile_overlay(
                line_points=self._build_line_points(),
            )

    @QtCore.pyqtSlot(float)
    def _on_separation_changed(self, value: float) -> None:
        """Propagate separation change to profile dock and external listeners."""
        self._refresh_profile()
        self._sync_3d_separation(value)
        self.separation_changed.emit(value)

    @QtCore.pyqtSlot(int)
    def _on_profile_point_selected(self, idx: int) -> None:
        """Convert profile index to grid (row, col) and emit point_selected."""
        if self._current_rr is None or self._current_cc is None:
            return
        if idx < 0 or idx >= len(self._current_rr):
            return
        r = int(self._current_rr[idx])
        c = int(self._current_cc[idx])
        self.point_selected.emit(r, c)

    @QtCore.pyqtSlot(int)
    def _on_profile_cursor_moved(self, idx: int) -> None:
        """Move the live hover cursor on the binary map and optionally in 3D."""
        if self._current_rr is None or self._current_cc is None:
            return
        if idx < 0 or idx >= len(self._current_rr):
            return
        r = int(self._current_rr[idx])
        c = int(self._current_cc[idx])
        self.binary_dock.update_live_cursor(r, c)
        if self._3d_viewer is not None:
            self._3d_viewer.highlight_profile_point(idx)

    # ------------------------------------------------------------------
    # Profile extraction
    # ------------------------------------------------------------------

    def _refresh_profile(self) -> None:
        """Extract profiles along the current ROI line and update the profile dock."""
        rr = self._current_rr
        cc = self._current_cc
        if rr is None or cc is None:
            return

        diff_map = self.binary_dock._diff_map
        grid1    = self.binary_dock._grid1
        grid2    = self.binary_dock._grid2
        sep      = self.binary_dock.separation
        dx       = self.binary_dock._dx
        dy       = self.binary_dock._dy

        if diff_map is None:
            return

        # Sample the difference map along the line
        prof_dist = diff_map[rr, cc]
        valid = np.isfinite(prof_dist)

        profiles = []
        if grid1 is not None and grid2 is not None:
            prof1 = grid1[rr, cc]
            prof2 = grid2[rr, cc] + sep
            valid &= np.isfinite(prof1) & np.isfinite(prof2)
            profiles.append(("Ref", prof1[valid], pg.mkPen("g", width=2)))
            profiles.append(("Adj", prof2[valid], pg.mkPen("b", width=2)))

        # Physical positions along the line (µm)
        xs = rr * dy
        ys = cc * dx
        xs_v = xs[valid]
        ys_v = ys[valid]
        if len(xs_v) < 2:
            return
        positions = np.hypot(xs_v - xs_v[0], ys_v - ys_v[0])
        prof_dist_v = prof_dist[valid]

        self.profile_dock.set_profiles(
            positions,
            profiles,
            prof_dist_v,
            separation=sep,
        )

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def _show_3d_view(self) -> None:
        """Open the 3D viewer with the current scan pair and profile line."""
        grid1 = self.binary_dock._grid1
        grid2 = self.binary_dock._grid2
        if grid1 is None or grid2 is None:
            return

        sep = self.binary_dock.separation
        dx  = self.binary_dock._dx
        dy  = self.binary_dock._dy

        line_points = self._build_line_points()

        # Pass grid2 without +sep — the viewer applies _separation internally
        viewer = show_point_3d_viewer(
            reference_grid=grid1,
            adjusted_grid=grid2,
            line_points=line_points,
            separation=sep,
            pixel_size_x=dx,
            pixel_size_y=dy,
        )
        self._3d_viewer = viewer
        if viewer is not None:
            viewer.destroyed.connect(self._on_3d_viewer_destroyed)

    def _sync_3d_separation(self, sep: float) -> None:
        """Push updated separation to the 3D viewer if it is open."""
        viewer = self._3d_viewer
        if viewer is None:
            return
        # Lightweight path: only shift adj geometry, camera untouched.
        # Also refresh the profile-line overlay so adj z-values are updated.
        viewer.update_separation(sep)
        viewer.update_profile_overlay(
            line_points=self._build_line_points(),
            separation=sep,
        )

    def _build_line_points(self):
        if self._current_rr is None or self._current_cc is None:
            return None
        return list(zip(self._current_cc.tolist(), self._current_rr.tolist()))

    @QtCore.pyqtSlot()
    def _on_3d_viewer_destroyed(self) -> None:
        self._3d_viewer = None

