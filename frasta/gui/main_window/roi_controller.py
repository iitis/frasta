"""ROI (Region of Interest) controller for the main window.

Handles circular, rectangular, and polygonal ROI operations, including:
- showing and hiding ROI overlays,
- creating masks from ROI geometry,
- applying ROI masks to the current scan,
- switching between shared and per-scan ROI behavior.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage.draw import polygon2mask

from ..dialogs import ROIDialog

logger = logging.getLogger(__name__)


class FixedPolygonROI(pg.PolyLineROI):
    """PolyLineROI variant with stable segment interaction for polygon editing.

    The upstream ``pyqtgraph.PolyLineROI`` builds each edge as a separate
    ``LineSegmentROI``. Those segments still react to modifier-driven drag
    gestures such as rotate and scale, which makes a polygon edge appear to
    move independently from the rest of the ROI. This subclass disables those
    per-segment transforms and also ignores edge clicks that would otherwise
    insert extra vertices by accident.
    """

    def addSegment(self, h1, h2, index=None):
        """Create a non-transformable segment between two polygon handles."""
        seg = pg.graphicsItems.ROI._PolyLineSegment(
            handles=(h1, h2),
            pen=self.pen,
            hoverPen=self.hoverPen,
            parent=self,
            movable=False,
            rotatable=False,
            resizable=False,
            antialias=self._antialias,
        )
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        seg.sigClicked.connect(self.segmentClicked)
        seg.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        seg.setZValue(self.zValue() + 1)
        seg.translatable = False
        seg.rotatable = False
        seg.resizable = False
        for handle in seg.handles:
            handle["item"].setDeletable(True)
            handle["item"].setAcceptedMouseButtons(
                handle["item"].acceptedMouseButtons() | QtCore.Qt.MouseButton.LeftButton
            )

    def segmentClicked(self, segment, ev=None, pos=None):
        """Add a vertex on plain edge clicks while ignoring modifier gestures."""
        if ev is not None:
            modifiers = ev.modifiers()
            if modifiers & (
                QtCore.Qt.KeyboardModifier.ShiftModifier
                | QtCore.Qt.KeyboardModifier.AltModifier
                | QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                return None
        return super().segmentClicked(segment, ev=ev, pos=pos)


class RingROI(pg.CircleROI):
    """Circular annulus ROI with a configurable inner radius fraction."""

    def __init__(self, pos, size=None, radius=None, inner_fraction: float = 0.5, **args):
        """Initialize an annular ROI."""
        self._inner_fraction = float(np.clip(inner_fraction, 0.0, 0.999))
        self._min_thickness = 1.0
        self._xi: np.ndarray | None = None
        self._yi: np.ndarray | None = None
        super().__init__(pos, size=size, radius=radius, **args)
        outer_diameter = float(min(self.size()[0], self.size()[1]))
        self._inner_diameter = outer_diameter * self._inner_fraction
        self._inner_handle = self.addFreeHandle(
            [0.5 + self._inner_fraction / 2.0, 0.5],
            name="inner_radius",
        )
        self._inner_handle.setZValue(self.zValue() + 2)

    def _sync_inner_handle(self):
        """Place the inner-radius handle on the right side of the inner circle."""
        for handle_info in self.handles:
            if handle_info.get("name") == "inner_radius":
                handle_info["pos"] = pg.Point(0.5 + self._inner_fraction / 2.0, 0.5)
                if handle_info["item"] in self.childItems():
                    handle_info["item"].setPos(handle_info["pos"] * self.state["size"])
                break

    def set_sampling(self, xi: np.ndarray | None, yi: np.ndarray | None):
        """Attach the current tab sampling so annulus thickness can be validated on-grid."""
        self._xi = None if xi is None else np.asarray(xi, dtype=float)
        self._yi = None if yi is None else np.asarray(yi, dtype=float)

    def _annulus_mask_has_pixels(self, outer_diameter: float, inner_diameter: float) -> bool:
        """Return whether the current annulus rasterizes to at least one pixel on the tab grid."""
        if self._xi is None or self._yi is None or self._xi.size == 0 or self._yi.size == 0:
            return (outer_diameter - inner_diameter) >= self._min_thickness

        center_x = float(self.pos().x()) + outer_diameter / 2.0
        center_y = float(self.pos().y()) + outer_diameter / 2.0
        x_coords = self._xi[np.newaxis, :]
        y_coords = self._yi[:, np.newaxis]
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        outer_radius = outer_diameter / 2.0
        inner_radius = max(inner_diameter / 2.0, 0.0)
        mask = (dist <= outer_radius) & ~(dist <= inner_radius)
        return bool(np.any(mask))

    def _clamp_inner_diameter(self, inner_diameter: float, outer_diameter: float) -> float:
        """Clamp the inner diameter so the annulus remains visible on the pixel grid."""
        max_inner = max(outer_diameter - self._min_thickness, 0.0)
        candidate = min(max(float(inner_diameter), 0.0), max_inner)
        if self._annulus_mask_has_pixels(outer_diameter, candidate):
            return candidate

        low = 0.0
        high = candidate
        for _ in range(24):
            mid = (low + high) / 2.0
            if self._annulus_mask_has_pixels(outer_diameter, mid):
                low = mid
            else:
                high = mid
        return low

    def _clamp_outer_diameter(self, outer_diameter: float, inner_diameter: float) -> float:
        """Clamp the outer diameter so the annulus remains visible on the pixel grid."""
        candidate = max(float(outer_diameter), inner_diameter + self._min_thickness, 1e-6)
        if self._annulus_mask_has_pixels(candidate, inner_diameter):
            return candidate

        step = self._min_thickness
        for _ in range(24):
            candidate += step
            if self._annulus_mask_has_pixels(candidate, inner_diameter):
                return candidate
        return candidate

    def set_min_thickness(self, min_thickness: float):
        """Set the minimum allowed annulus thickness in parent units."""
        self._min_thickness = max(float(min_thickness), 0.0)
        outer_diameter = max(float(min(self.size()[0], self.size()[1])), 1e-9)
        self._inner_diameter = self._clamp_inner_diameter(self._inner_diameter, outer_diameter)
        self._inner_fraction = float(np.clip(self._inner_diameter / outer_diameter, 0.0, 0.999))
        self._sync_inner_handle()
        self.update()

    def set_inner_fraction(self, inner_fraction: float):
        """Set the inner radius as a fraction of the outer radius."""
        self._inner_fraction = float(np.clip(inner_fraction, 0.0, 0.999))
        outer_diameter = float(min(self.size()[0], self.size()[1]))
        self._inner_diameter = self._clamp_inner_diameter(outer_diameter * self._inner_fraction, outer_diameter)
        self._inner_fraction = float(np.clip(self._inner_diameter / max(outer_diameter, 1e-9), 0.0, 0.999))
        self._sync_inner_handle()
        self.update()
        self.stateChanged(finish=True)

    def inner_fraction(self) -> float:
        """Return the inner radius fraction."""
        return self._inner_fraction

    def inner_diameter(self) -> float:
        """Return the absolute inner diameter in parent units."""
        return float(self._inner_diameter)

    def saveState(self):
        """Return the ROI state including annulus thickness."""
        state = super().saveState()
        state["inner_fraction"] = self._inner_fraction
        state["inner_diameter"] = self._inner_diameter
        return state

    def setState(self, state, update=True):
        """Restore the ROI state including annulus thickness."""
        self._inner_fraction = float(np.clip(state.get("inner_fraction", self._inner_fraction), 0.0, 0.999))
        self._inner_diameter = float(state.get("inner_diameter", self._inner_diameter))
        super().setState(state, update=update)
        outer_diameter = max(float(min(self.size()[0], self.size()[1])), 1e-9)
        self._inner_diameter = self._clamp_inner_diameter(self._inner_diameter, outer_diameter)
        self._inner_fraction = float(np.clip(self._inner_diameter / outer_diameter, 0.0, 0.999))
        self._sync_inner_handle()
        self.update()

    def setSize(self, size, center=None, centerLocal=None, snap=False, update=True, finish=True):
        """Resize the outer circle without implicitly changing the inner diameter."""
        size_point = pg.Point(size)
        requested_diameter = float(max(size_point[0], size_point[1]))
        clamped_diameter = self._clamp_outer_diameter(requested_diameter, float(getattr(self, "_inner_diameter", 0.0)))
        size_point = pg.Point(clamped_diameter, clamped_diameter)
        super().setSize(size_point, center=center, centerLocal=centerLocal, snap=snap, update=update, finish=finish)
        outer_diameter = max(float(min(self.size()[0], self.size()[1])), 1e-9)
        self._inner_diameter = self._clamp_inner_diameter(float(getattr(self, "_inner_diameter", 0.0)), outer_diameter)
        self._inner_fraction = float(np.clip(self._inner_diameter / outer_diameter, 0.0, 0.999))
        self._sync_inner_handle()

    def movePoint(self, handle, pos, modifiers=None, finish=True, coords='parent'):
        """Update the inner radius when its dedicated handle is dragged."""
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        index = self.indexOfHandle(handle)
        handle_info = self.handles[index]
        if handle_info.get("name") != "inner_radius":
            if handle_info.get("type") == "s":
                p0 = self.mapToParent(handle_info["pos"] * self.state["size"])
                p1 = pg.Point(pos)
                if coords == "scene":
                    p1 = self.mapSceneToParent(p1)
                elif coords != "parent":
                    raise Exception("New point location must be given in either 'parent' or 'scene' coordinates.")

                c = handle_info["center"]
                cs = c * self.state["size"]
                lp0 = self.mapFromParent(p0) - cs
                lp1 = self.mapFromParent(p1) - cs

                if handle_info["center"][0] == handle_info["pos"][0]:
                    lp1[0] = 0
                if handle_info["center"][1] == handle_info["pos"][1]:
                    lp1[1] = 0

                if self.scaleSnap or (modifiers & QtCore.Qt.KeyboardModifier.ControlModifier):
                    lp1[0] = round(lp1[0] / self.scaleSnapSize) * self.scaleSnapSize
                    lp1[1] = round(lp1[1] / self.scaleSnapSize) * self.scaleSnapSize

                if handle_info["lockAspect"] or (modifiers & QtCore.Qt.KeyboardModifier.AltModifier):
                    lp1 = lp1.proj(lp0)

                hs = handle_info["pos"] - c
                if hs[0] == 0:
                    hs[0] = 1
                if hs[1] == 0:
                    hs[1] = 1
                requested_size = lp1 / hs
                if requested_size[0] == 0:
                    requested_size[0] = self.state["size"][0]
                if requested_size[1] == 0:
                    requested_size[1] = self.state["size"][1]
                if not self.invertible:
                    if requested_size[0] < 0:
                        requested_size[0] = self.state["size"][0]
                    if requested_size[1] < 0:
                        requested_size[1] = self.state["size"][1]
                if self.aspectLocked:
                    requested_size[0] = requested_size[1]

                requested_diameter = float(requested_size[0])
                if requested_diameter < self._clamp_outer_diameter(requested_diameter, self._inner_diameter) - 1e-9:
                    return
            return super().movePoint(handle, pos, modifiers=modifiers, finish=finish, coords=coords)

        if coords == 'scene':
            parent_pos = self.mapSceneToParent(pg.Point(pos))
        elif coords == 'parent':
            parent_pos = pg.Point(pos)
        else:
            raise Exception("New point location must be given in either 'parent' or 'scene' coordinates.")

        local_pos = self.mapFromParent(parent_pos)
        outer_radius = max(float(min(self.state["size"][0], self.state["size"][1])) / 2.0, 1e-9)
        center_local = pg.Point(self.state["size"][0] / 2.0, self.state["size"][1] / 2.0)
        inner_radius = float(np.hypot(local_pos.x() - center_local.x(), local_pos.y() - center_local.y()))
        self._inner_diameter = self._clamp_inner_diameter(inner_radius * 2.0, outer_radius * 2.0)
        inner_radius = self._inner_diameter / 2.0
        self._inner_fraction = float(np.clip(inner_radius / outer_radius, 0.0, 0.999))
        self._sync_inner_handle()
        self.update()
        self.stateChanged(finish=finish)

    def paint(self, p, opt, widget):
        """Draw the outer and inner circular boundaries of the annulus."""
        r = self.boundingRect()

        p.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing,
            self._antialias,
        )
        p.setPen(self.currentPen)
        p.scale(r.width(), r.height())
        normalized_rect = QtCore.QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)
        p.drawEllipse(normalized_rect)

        inner_diameter = max(self._inner_fraction, 0.0)
        if inner_diameter <= 0.0:
            return

        inset = (1.0 - inner_diameter) / 2.0
        inner_rect = QtCore.QRectF(inset, inset, inner_diameter, inner_diameter)
        p.drawEllipse(inner_rect)


class ROIController:
    """Controller for ROI-related operations."""

    def __init__(self, main_window):
        """Initialize the ROI controller.

        Args:
            main_window: Reference to the main window instance.
        """
        self.main_window = main_window
        self.shared_circle_roi = None
        self.shared_ring_roi = None
        self.shared_rectangle_roi = None
        self.shared_polygon_roi = None
        self.mode = "global"
        self.global_roi_state: dict | None = None
        self._tab_roi_states: dict[int, dict] = {}
        self._last_tab = None
        self._last_delete_snapshot: dict | None = None

    def _show_status_message(self, message: str, timeout_ms: int = 5000):
        """Show a transient message in the main-window status bar when available."""
        try:
            self.main_window.statusBar().showMessage(message, timeout_ms)
        except Exception:
            logger.debug("Unable to show ROI status message: %s", message)

    def _set_undo_delete_enabled(self, enabled: bool):
        """Enable or disable the ROI-delete undo action when it exists."""
        action = getattr(getattr(self.main_window, "menu_builder", None), "actions", {}).get("undo_roi_delete")
        if action is not None:
            action.setEnabled(enabled)

    def _clear_last_delete_snapshot(self):
        """Forget the currently stored ROI-delete undo snapshot."""
        self._last_delete_snapshot = None
        self._set_undo_delete_enabled(False)

    def _tab_physical_bounds(self, tab) -> tuple[float, float, float, float]:
        """Return physical bounds of a tab image in native units."""
        dx = tab.dx if tab.dx not in (None, 0) else 1.0
        dy = tab.dy if tab.dy not in (None, 0) else 1.0
        x0 = tab.xi[0] if tab.xi is not None and len(tab.xi) else 0.0
        y0 = tab.yi[0] if tab.yi is not None and len(tab.yi) else 0.0
        x_min = x0 - dx / 2.0
        y_min = y0 - dy / 2.0
        x_max = x_min + tab.grid.shape[1] * dx
        y_max = y_min + tab.grid.shape[0] * dy
        return x_min, x_max, y_min, y_max

    def _tab_key(self, tab) -> int | None:
        """Return a stable dictionary key for a tab widget."""
        if tab is None:
            return None
        return id(tab)

    def _tab_min_roi_thickness(self, tab) -> float:
        """Return the minimum meaningful ROI thickness for the tab grid."""
        if tab is None:
            return 1.0
        dx = abs(tab.dx) if getattr(tab, "dx", None) not in (None, 0) else 1.0
        dy = abs(tab.dy) if getattr(tab, "dy", None) not in (None, 0) else 1.0
        return min(dx, dy)

    def _clone_state(self, state: dict | None) -> dict | None:
        """Return a detached copy of an ROI state dictionary."""
        if state is None:
            return None
        cloned_state = {
            "shape": state["shape"],
            "visible": bool(state.get("visible", True)),
        }
        if state["shape"] == "polygon":
            cloned_state["points"] = tuple(
                (float(point[0]), float(point[1])) for point in state.get("points", ())
            )
            return cloned_state
        if state["shape"] == "ring":
            cloned_state["pos"] = tuple(state["pos"])
            cloned_state["size"] = tuple(state["size"])
            cloned_state["inner_size"] = float(state.get("inner_size", 0.0))
            return cloned_state

        cloned_state["pos"] = tuple(state["pos"])
        cloned_state["size"] = tuple(state["size"])
        return cloned_state

    @staticmethod
    def _polygon_bounds(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return polygon center and bounding-box size in physical coordinates."""
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2 or points_array.shape[0] == 0:
            return (0.0, 0.0), (1.0, 1.0)

        min_x = float(np.min(points_array[:, 0]))
        max_x = float(np.max(points_array[:, 0]))
        min_y = float(np.min(points_array[:, 1]))
        max_y = float(np.max(points_array[:, 1]))
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        size = (max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6))
        return center, size

    @staticmethod
    def _build_polygon_points(center: tuple[float, float], size: tuple[float, float]) -> tuple[tuple[float, float], ...]:
        """Create a simple triangular ROI centered inside the requested bounding box."""
        radius_x = max(float(size[0]) / 2.0, 1e-6)
        radius_y = max(float(size[1]) / 2.0, 1e-6)
        return (
            (float(center[0]), float(center[1] - radius_y)),
            (float(center[0] + radius_x), float(center[1] + radius_y)),
            (float(center[0] - radius_x), float(center[1] + radius_y)),
        )

    def _is_roi_valid_and_visible(self, roi) -> bool:
        """Safely check whether a Qt ROI object exists and is visible."""
        if roi is None:
            return False
        try:
            return roi.isVisible()
        except RuntimeError:
            return False

    def _is_roi_deleted(self, roi) -> bool:
        """Check whether a Qt ROI object has already been deleted."""
        if roi is None:
            return True
        try:
            _ = roi.isVisible()
            return False
        except RuntimeError:
            return True

    def _connect_live_roi_signals(self, roi):
        """Connect signal handlers for a live ROI instance."""
        roi.sigRegionChanged.connect(self._handle_live_roi_changed)

    def _ensure_circle_roi(self, tab):
        """Create the shared live circle ROI if needed."""
        if not self._is_roi_deleted(self.shared_circle_roi):
            return self.shared_circle_roi

        x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
        width_phys = max(x_max - x_min, 1.0)
        height_phys = max(y_max - y_min, 1.0)
        self.shared_circle_roi = pg.CircleROI(
            [x_min + width_phys * 0.375, y_min + height_phys * 0.375],
            [width_phys * 0.25, width_phys * 0.25],
            pen=pg.mkPen("g", width=2),
        )
        self.shared_circle_roi.setZValue(100)
        self._connect_live_roi_signals(self.shared_circle_roi)
        return self.shared_circle_roi

    def _ensure_ring_roi(self, tab):
        """Create the shared live annular ROI if needed."""
        if not self._is_roi_deleted(self.shared_ring_roi):
            self.shared_ring_roi.set_min_thickness(self._tab_min_roi_thickness(tab))
            self.shared_ring_roi.set_sampling(
                None if tab is None else getattr(tab, "xi", None),
                None if tab is None else getattr(tab, "yi", None),
            )
            return self.shared_ring_roi

        x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
        width_phys = max(x_max - x_min, 1.0)
        height_phys = max(y_max - y_min, 1.0)
        diameter = min(width_phys, height_phys) * 0.5
        inner_fraction = 0.5
        self.shared_ring_roi = RingROI(
            [x_min + (width_phys - diameter) * 0.5, y_min + (height_phys - diameter) * 0.5],
            [diameter, diameter],
            inner_fraction=inner_fraction,
            pen=pg.mkPen("g", width=2),
        )
        self.shared_ring_roi.setZValue(100)
        self.shared_ring_roi.set_min_thickness(self._tab_min_roi_thickness(tab))
        self.shared_ring_roi.set_sampling(
            None if tab is None else getattr(tab, "xi", None),
            None if tab is None else getattr(tab, "yi", None),
        )
        self._connect_live_roi_signals(self.shared_ring_roi)
        return self.shared_ring_roi

    def _ensure_rectangle_roi(self, tab):
        """Create the shared live rectangle ROI if needed."""
        if not self._is_roi_deleted(self.shared_rectangle_roi):
            return self.shared_rectangle_roi

        x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
        width_phys = max(x_max - x_min, 1.0)
        height_phys = max(y_max - y_min, 1.0)
        self.shared_rectangle_roi = pg.RectROI(
            [x_min + width_phys * 0.25, y_min + height_phys * 0.25],
            [width_phys * 0.5, height_phys * 0.5],
            pen=pg.mkPen("g", width=2),
        )
        self.shared_rectangle_roi.setZValue(100)
        self._connect_live_roi_signals(self.shared_rectangle_roi)
        return self.shared_rectangle_roi

    def _ensure_polygon_roi(self, tab):
        """Create the shared live polygon ROI if needed."""
        if not self._is_roi_deleted(self.shared_polygon_roi):
            return self.shared_polygon_roi

        x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
        width_phys = max(x_max - x_min, 1.0)
        height_phys = max(y_max - y_min, 1.0)
        center = (x_min + width_phys * 0.5, y_min + height_phys * 0.5)
        size = (width_phys * 0.5, height_phys * 0.5)
        points = self._build_polygon_points(center, size)
        self.shared_polygon_roi = FixedPolygonROI(
            positions=points,
            closed=True,
            pos=[0.0, 0.0],
            pen=pg.mkPen("g", width=2),
            movable=True,
            rotatable=False,
            resizable=False,
        )
        self.shared_polygon_roi.setZValue(100)
        self._connect_live_roi_signals(self.shared_polygon_roi)
        return self.shared_polygon_roi

    def _current_live_state(self) -> dict | None:
        """Capture the currently visible live ROI geometry."""
        if self._is_roi_valid_and_visible(self.shared_polygon_roi):
            roi_state = self.shared_polygon_roi.saveState()
            pos_x, pos_y = roi_state["pos"]
            points = tuple(
                (float(pos_x + point[0]), float(pos_y + point[1]))
                for point in roi_state.get("points", [])
            )
            return {
                "shape": "polygon",
                "points": points,
                "visible": True,
            }

        if self._is_roi_valid_and_visible(self.shared_ring_roi):
            pos = self.shared_ring_roi.pos()
            size = self.shared_ring_roi.size()
            diameter = float(min(size[0], size[1]))
            return {
                "shape": "ring",
                "pos": (float(pos.x()), float(pos.y())),
                "size": (diameter, diameter),
                "inner_size": self.shared_ring_roi.inner_diameter(),
                "visible": True,
            }

        if self._is_roi_valid_and_visible(self.shared_circle_roi):
            pos = self.shared_circle_roi.pos()
            size = self.shared_circle_roi.size()
            return {
                "shape": "circle",
                "pos": (float(pos.x()), float(pos.y())),
                "size": (float(size[0]), float(size[1])),
                "visible": True,
            }

        if self._is_roi_valid_and_visible(self.shared_rectangle_roi):
            pos = self.shared_rectangle_roi.pos()
            size = self.shared_rectangle_roi.size()
            return {
                "shape": "rectangle",
                "pos": (float(pos.x()), float(pos.y())),
                "size": (float(size[0]), float(size[1])),
                "visible": True,
            }

        return None

    def _default_state(self, tab, shape: str, reference_state: dict | None = None) -> dict:
        """Build a reasonable default ROI state for the current tab."""
        x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
        width_phys = max(x_max - x_min, 1.0)
        height_phys = max(y_max - y_min, 1.0)
        if reference_state is not None:
            if reference_state.get("shape") == "polygon":
                polygon_points = tuple(reference_state.get("points", ()))
                if shape == "polygon" and polygon_points:
                    return {
                        "shape": "polygon",
                        "points": polygon_points,
                        "visible": True,
                    }
                (center_x, center_y), ref_size = self._polygon_bounds(polygon_points)
            else:
                ref_pos = reference_state["pos"]
                ref_size = reference_state["size"]
                center_x = ref_pos[0] + ref_size[0] / 2.0
                center_y = ref_pos[1] + ref_size[1] / 2.0
            if shape == "circle":
                diameter = float(min(ref_size[0], ref_size[1]))
                return {
                    "shape": "circle",
                    "pos": (center_x - diameter / 2.0, center_y - diameter / 2.0),
                    "size": (diameter, diameter),
                    "visible": True,
                }
            if shape == "ring":
                diameter = float(min(ref_size[0], ref_size[1]))
                return {
                    "shape": "ring",
                    "pos": (center_x - diameter / 2.0, center_y - diameter / 2.0),
                    "size": (diameter, diameter),
                    "inner_size": diameter * 0.5,
                    "visible": True,
                }
            if shape == "polygon":
                return {
                    "shape": "polygon",
                    "points": self._build_polygon_points((center_x, center_y), tuple(ref_size)),
                    "visible": True,
                }
            return {
                "shape": "rectangle",
                "pos": (center_x - ref_size[0] / 2.0, center_y - ref_size[1] / 2.0),
                "size": tuple(ref_size),
                "visible": True,
            }

        if shape == "circle":
            return {
                "shape": "circle",
                "pos": (x_min + width_phys * 0.375, y_min + height_phys * 0.375),
                "size": (width_phys * 0.25, width_phys * 0.25),
                "visible": True,
            }

        if shape == "ring":
            diameter = min(width_phys, height_phys) * 0.5
            return {
                "shape": "ring",
                "pos": (x_min + (width_phys - diameter) * 0.5, y_min + (height_phys - diameter) * 0.5),
                "size": (diameter, diameter),
                "inner_size": diameter * 0.5,
                "visible": True,
            }

        if shape == "polygon":
            center = (x_min + width_phys * 0.5, y_min + height_phys * 0.5)
            size = (width_phys * 0.5, height_phys * 0.5)
            return {
                "shape": "polygon",
                "points": self._build_polygon_points(center, size),
                "visible": True,
            }

        return {
            "shape": "rectangle",
            "pos": (x_min + width_phys * 0.25, y_min + height_phys * 0.25),
            "size": (width_phys * 0.5, height_phys * 0.5),
            "visible": True,
        }

    def _detach_live_rois(self):
        """Remove live ROI items from all tab views."""
        tabs = self.main_window.tabs
        for roi in (self.shared_circle_roi, self.shared_ring_roi, self.shared_rectangle_roi, self.shared_polygon_roi):
            if self._is_roi_deleted(roi):
                continue
            for index in range(tabs.count()):
                tab = tabs.widget(index)
                try:
                    tab.image_view.getView().removeItem(roi)
                except Exception:
                    continue

    def _hide_live_rois(self):
        """Hide any live ROI objects that still exist."""
        if not self._is_roi_deleted(self.shared_circle_roi):
            self.shared_circle_roi.hide()
        if not self._is_roi_deleted(self.shared_ring_roi):
            self.shared_ring_roi.hide()
        if not self._is_roi_deleted(self.shared_rectangle_roi):
            self.shared_rectangle_roi.hide()
        if not self._is_roi_deleted(self.shared_polygon_roi):
            self.shared_polygon_roi.hide()

    def _apply_state_to_tab(self, tab, state: dict | None):
        """Display the selected ROI state on the requested tab."""
        self._detach_live_rois()
        self._hide_live_rois()

        if tab is None or tab.grid is None or state is None or not state.get("visible", True):
            return

        if state["shape"] == "polygon":
            roi = self._ensure_polygon_roi(tab)
            tab.image_view.getView().addItem(roi)
            roi_state = {
                "pos": (0.0, 0.0),
                "size": (1.0, 1.0),
                "angle": 0.0,
                "closed": True,
                "points": list(state.get("points", ())),
            }
            roi.setState(roi_state)
        elif state["shape"] == "ring":
            roi = self._ensure_ring_roi(tab)
            diameter = float(min(state["size"][0], state["size"][1]))
            inner_fraction = 0.0 if diameter <= 0 else float(np.clip(state.get("inner_size", 0.0) / diameter, 0.0, 0.999))
            roi_state = {
                "pos": tuple(state["pos"]),
                "size": (diameter, diameter),
                "angle": 0.0,
                "inner_fraction": inner_fraction,
                "inner_diameter": float(state.get("inner_size", 0.0)),
            }
            roi.setState(roi_state)
            tab.image_view.getView().addItem(roi)
        elif state["shape"] == "circle":
            roi = self._ensure_circle_roi(tab)
            roi.setPos(*state["pos"])
            roi.setSize(state["size"])
            tab.image_view.getView().addItem(roi)
        else:
            roi = self._ensure_rectangle_roi(tab)
            roi.setPos(*state["pos"])
            roi.setSize(state["size"])
            tab.image_view.getView().addItem(roi)
        roi.show()

    def _get_state_for_tab(self, tab) -> dict | None:
        """Return ROI state for a tab according to the active controller mode."""
        if self.mode == "per_scan":
            return self._clone_state(self._tab_roi_states.get(self._tab_key(tab)))
        return self._clone_state(self.global_roi_state)

    def _set_state_for_tab(self, tab, state: dict | None):
        """Store ROI state for a tab according to the active controller mode."""
        if self.mode == "per_scan":
            key = self._tab_key(tab)
            if key is None:
                return
            if state is None:
                self._tab_roi_states.pop(key, None)
            else:
                self._tab_roi_states[key] = self._clone_state(state)
        else:
            self.global_roi_state = self._clone_state(state)

    def _save_live_state_for_tab(self, tab):
        """Persist the currently visible live ROI geometry."""
        self._set_state_for_tab(tab, self._current_live_state())

    def _handle_live_roi_changed(self):
        """Persist ROI geometry after the user moves or resizes it."""
        current_tab = self.main_window.current_tab()
        self._save_live_state_for_tab(current_tab)

    def initialize_tab_roi_state(self, tab, source_tab=None):
        """Initialize a new tab's ROI state when per-scan mode is active."""
        if self.mode != "per_scan" or tab is None:
            return

        source_state = None
        if source_tab is not None:
            source_state = self._tab_roi_states.get(self._tab_key(source_tab))
        if source_state is None:
            source_state = self.global_roi_state
        if source_state is not None:
            self._tab_roi_states[self._tab_key(tab)] = self._clone_state(source_state)

    def remove_tab_state(self, tab):
        """Remove stored per-scan ROI state for a closing tab."""
        key = self._tab_key(tab)
        if key is not None:
            self._tab_roi_states.pop(key, None)
        if self._last_tab is tab:
            self._last_tab = None
        if self._last_delete_snapshot is not None and self._last_delete_snapshot.get("tab") is tab:
            self._clear_last_delete_snapshot()

    def set_mode(self, mode: str):
        """Switch between shared and per-scan ROI behavior."""
        if mode not in {"global", "per_scan"} or mode == self.mode:
            return

        current_tab = self.main_window.current_tab()
        self._save_live_state_for_tab(current_tab)

        if mode == "per_scan":
            seed_state = self._clone_state(self.global_roi_state)
            self._tab_roi_states.clear()
            if seed_state is not None:
                for index in range(self.main_window.tabs.count()):
                    tab = self.main_window.tabs.widget(index)
                    self._tab_roi_states[self._tab_key(tab)] = self._clone_state(seed_state)
        else:
            new_global_state = self._get_state_for_tab(current_tab)
            self.global_roi_state = self._clone_state(new_global_state)

        self.mode = mode
        self.move_roi_to_current_tab(self.main_window.tabs.currentIndex())

    def set_global_mode(self):
        """Enable the shared ROI mode used by the original workflow."""
        self.set_mode("global")

    def set_per_scan_mode(self):
        """Enable per-scan ROI mode with independent geometry per tab."""
        self.set_mode("per_scan")

    def _state_to_dialog_config(self, state: dict | None) -> dict:
        """Convert internal ROI state into dialog-friendly geometry."""
        if state is None:
            tab = self.main_window.current_tab()
            if tab is None or tab.grid is None:
                center = (0.0, 0.0)
                size = (100.0, 100.0)
            else:
                x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
                width_phys = max(x_max - x_min, 1.0)
                height_phys = max(y_max - y_min, 1.0)
                center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
                size = (width_phys * 0.5, height_phys * 0.5)
            return {
                "mode": self.mode,
                "enabled": False,
                "shape": "circle",
                "center": center,
                "size": size,
                "points": tuple(),
            }

        pos_x, pos_y = state["pos"]
        size_x, size_y = state["size"]
        config = {
            "mode": self.mode,
            "enabled": bool(state.get("visible", True)),
            "shape": state["shape"],
            "center": (pos_x + size_x / 2.0, pos_y + size_y / 2.0),
            "size": (size_x, size_y),
            "points": tuple(state.get("points", ())),
            "inner_size": float(state.get("inner_size", 0.0)),
        }
        return config

    def get_dialog_config(self, tab=None) -> dict:
        """Return the current ROI configuration for editing."""
        if tab is None:
            tab = self.main_window.current_tab()
        state = self._get_state_for_tab(tab)
        if state is not None and state.get("shape") == "polygon":
            center, size = self._polygon_bounds(state.get("points", ()))
            dialog_state = {
                "shape": "polygon",
                "visible": bool(state.get("visible", True)),
                "points": tuple(state.get("points", ())),
                "pos": (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0),
                "size": size,
            }
            return self._state_to_dialog_config(dialog_state)
        return self._state_to_dialog_config(state)

    def apply_dialog_config(self, config: dict, tab=None):
        """Apply a dialog-provided ROI configuration."""
        if tab is None:
            tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return

        self.set_mode(config.get("mode", self.mode))

        if not config.get("enabled", False):
            self._set_state_for_tab(tab, None)
            self._apply_state_to_tab(tab, None)
            self._last_tab = tab
            return

        size_x, size_y = config["size"]
        center_x, center_y = config["center"]
        if config["shape"] == "polygon":
            points = tuple(config.get("points", ()))
            state = {
                "shape": "polygon",
                "points": points if points else self._build_polygon_points((center_x, center_y), (size_x, size_y)),
                "visible": True,
            }
        elif config["shape"] == "ring":
            diameter = float(min(size_x, size_y))
            min_thickness = self._tab_min_roi_thickness(tab)
            inner_size = float(
                np.clip(
                    config.get("inner_size", diameter * 0.5),
                    0.0,
                    max(diameter - min_thickness, 0.0),
                )
            )
            state = {
                "shape": "ring",
                "pos": (center_x - diameter / 2.0, center_y - diameter / 2.0),
                "size": (diameter, diameter),
                "inner_size": inner_size,
                "visible": True,
            }
        else:
            state = {
                "shape": config["shape"],
                "pos": (center_x - size_x / 2.0, center_y - size_y / 2.0),
                "size": (size_x, size_y),
                "visible": True,
            }
        self._set_state_for_tab(tab, state)
        self._apply_state_to_tab(tab, state)
        self._last_tab = tab

    def open_roi_settings_dialog(self):
        """Open a dialog for editing ROI mode, shape, and geometry."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "ROI settings",
                "Please load a scan first!",
            )
            return

        dialog = ROIDialog(
            self.get_dialog_config(tab),
            unit_label=getattr(tab, "unit", "µm"),
            parent=self.main_window,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.apply_dialog_config(dialog.get_roi_config(), tab=tab)

    def create_circle_mask(self, *args) -> np.ndarray:
        """Create a boolean mask for a circular ROI.

        Supported signatures:
        - ``(shape, center, radius)`` for legacy pixel coordinates
        - ``(xi, yi, center, radius)`` for physical coordinates
        """
        if len(args) == 3:
            shape, center, radius = args
            rows, cols = shape
            xi = np.arange(cols, dtype=float)
            yi = np.arange(rows, dtype=float)
        elif len(args) == 4:
            xi, yi, center, radius = args
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(yi, dtype=float)
        else:
            raise TypeError(
                "create_circle_mask expects (shape, center, radius) or (xi, yi, center, radius)"
            )

        x_coords = xi[np.newaxis, :]
        y_coords = yi[:, np.newaxis]
        dist = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)
        return dist <= radius

    def create_rectangle_mask(self, *args) -> np.ndarray:
        """Create a boolean mask for a rectangular ROI.

        Supported signatures:
        - ``(shape, center, width, height)`` for legacy pixel coordinates
        - ``(xi, yi, center, width, height)`` for physical coordinates
        """
        if len(args) == 4:
            shape, center, width, height = args
            rows, cols = shape
            xi = np.arange(cols, dtype=float)
            yi = np.arange(rows, dtype=float)
        elif len(args) == 5:
            xi, yi, center, width, height = args
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(yi, dtype=float)
        else:
            raise TypeError(
                "create_rectangle_mask expects (shape, center, width, height) or (xi, yi, center, width, height)"
            )

        x_coords = xi[np.newaxis, :]
        y_coords = yi[:, np.newaxis]
        x0 = center[0] - width / 2.0
        x1 = center[0] + width / 2.0
        y0 = center[1] - height / 2.0
        y1 = center[1] + height / 2.0
        return (x_coords >= x0) & (x_coords < x1) & (y_coords >= y0) & (y_coords < y1)

    def create_polygon_mask(self, *args) -> np.ndarray:
        """Create a boolean mask for a polygonal ROI.

        Supported signatures:
        - ``(shape, points)`` for legacy pixel coordinates
        - ``(xi, yi, points)`` for physical coordinates
        """
        if len(args) == 2:
            shape, points = args
            rows, cols = shape
            if len(points) < 3:
                return np.zeros((rows, cols), dtype=bool)
            vertices = np.asarray([(float(point[1]), float(point[0])) for point in points], dtype=float)
            return polygon2mask((rows, cols), vertices).astype(bool, copy=False)

        if len(args) == 3:
            xi, yi, points = args
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(yi, dtype=float)
            if len(points) < 3:
                return np.zeros((yi.size, xi.size), dtype=bool)
            point_array = np.asarray(points, dtype=float)
            cols = np.interp(point_array[:, 0], xi, np.arange(xi.size, dtype=float))
            rows = np.interp(point_array[:, 1], yi, np.arange(yi.size, dtype=float))
            vertices = np.column_stack([rows, cols])
            return polygon2mask((yi.size, xi.size), vertices).astype(bool, copy=False)

        raise TypeError("create_polygon_mask expects (shape, points) or (xi, yi, points)")

    def create_ring_mask(self, *args) -> np.ndarray:
        """Create a boolean mask for an annular ROI."""
        if len(args) == 4:
            shape, center, outer_radius, inner_radius = args
            min_thickness = 1.0
            if (outer_radius - inner_radius) < min_thickness:
                rows, cols = shape
                return np.zeros((rows, cols), dtype=bool)
            outer_mask = self.create_circle_mask(shape, center, outer_radius)
            inner_mask = self.create_circle_mask(shape, center, inner_radius)
            return outer_mask & ~inner_mask
        if len(args) == 5:
            xi, yi, center, outer_radius, inner_radius = args
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(yi, dtype=float)
            dx = float(np.min(np.abs(np.diff(xi)))) if xi.size > 1 else 1.0
            dy = float(np.min(np.abs(np.diff(yi)))) if yi.size > 1 else 1.0
            min_thickness = min(dx, dy)
            if (outer_radius - inner_radius) < min_thickness:
                return np.zeros((yi.size, xi.size), dtype=bool)
            outer_mask = self.create_circle_mask(xi, yi, center, outer_radius)
            inner_mask = self.create_circle_mask(xi, yi, center, inner_radius)
            return outer_mask & ~inner_mask
        raise TypeError(
            "create_ring_mask expects (shape, center, outer_radius, inner_radius) or (xi, yi, center, outer_radius, inner_radius)"
        )

    def create_mask(self, h: int, w: int, tab=None) -> np.ndarray | None:
        """Create a boolean mask for the requested tab and grid shape."""
        active_tab = tab if tab is not None else self.main_window.current_tab()
        state = self._get_state_for_tab(active_tab)
        if state is None and active_tab is None:
            state = self._current_live_state()
        if state is None or not state.get("visible", True):
            return None

        if state["shape"] == "polygon":
            points = tuple(state.get("points", ()))
            if active_tab is None or active_tab.xi is None or active_tab.yi is None:
                mask = self.create_polygon_mask((h, w), points)
            else:
                mask = self.create_polygon_mask(active_tab.xi[:w], active_tab.yi[:h], points)
            return mask if np.any(mask) else None

        pos_x, pos_y = state["pos"]
        size_x, size_y = state["size"]
        center_x = pos_x + size_x / 2.0
        center_y = pos_y + size_y / 2.0
        if state["shape"] == "ring":
            outer_radius = min(size_x, size_y) / 2.0
            inner_radius = max(0.0, float(state.get("inner_size", 0.0)) / 2.0)
            if active_tab is None or active_tab.xi is None or active_tab.yi is None:
                mask = self.create_ring_mask((h, w), (center_x, center_y), outer_radius, inner_radius)
            else:
                mask = self.create_ring_mask(active_tab.xi[:w], active_tab.yi[:h], (center_x, center_y), outer_radius, inner_radius)
            return mask if np.any(mask) else None
        if active_tab is None or active_tab.xi is None or active_tab.yi is None:
            if state["shape"] == "circle":
                mask = self.create_circle_mask((h, w), (center_x, center_y), size_x / 2.0)
            else:
                mask = self.create_rectangle_mask((h, w), (center_x, center_y), size_x, size_y)
            return mask if np.any(mask) else None

        if state["shape"] == "circle":
            mask = self.create_circle_mask(active_tab.xi[:w], active_tab.yi[:h], (center_x, center_y), size_x / 2.0)
        else:
            mask = self.create_rectangle_mask(active_tab.xi[:w], active_tab.yi[:h], (center_x, center_y), size_x, size_y)
        return mask if np.any(mask) else None

    def apply_roi_mask(self, inside: bool):
        """Delete data inside or outside the active ROI on the current tab."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            self._show_status_message("ROI delete skipped: no active scan.")
            return

        h, w = tab.grid.shape
        mask = self.create_mask(h, w, tab=tab)
        if mask is None:
            self._show_status_message("ROI delete skipped: no active ROI on the current tab.")
            return

        delete_mask = mask if inside else ~mask
        deleted_points = int(np.count_nonzero(np.isfinite(tab.grid) & delete_mask))
        if deleted_points == 0:
            self._clear_last_delete_snapshot()
            location = "inside" if inside else "outside"
            self._show_status_message(f"ROI delete skipped: no valid points found {location} the ROI.")
            return

        original_grid = np.array(tab.grid, copy=True)
        if inside:
            tab.delete_unmasked(~mask)
        else:
            tab.delete_unmasked(mask)

        self._last_delete_snapshot = {
            "tab": tab,
            "grid": original_grid,
            "deleted_points": deleted_points,
            "inside": inside,
        }
        self._set_undo_delete_enabled(True)
        location = "inside" if inside else "outside"
        self._show_status_message(
            f"Deleted {deleted_points} valid points {location} the ROI. Use Edit -> Undo ROI delete to restore."
        )

    def del_inside_mask(self):
        """Delete data inside the active ROI mask."""
        self.apply_roi_mask(True)

    def del_outside_mask(self):
        """Delete data outside the active ROI mask."""
        self.apply_roi_mask(False)

    def undo_last_roi_delete(self):
        """Restore the grid modified by the most recent ROI-delete operation."""
        snapshot = self._last_delete_snapshot
        if snapshot is None:
            self._show_status_message("Nothing to undo for ROI delete.")
            return

        tab = snapshot.get("tab")
        if tab is None:
            self._clear_last_delete_snapshot()
            self._show_status_message("ROI delete undo is no longer available.")
            return

        tab.grid = np.array(snapshot["grid"], copy=True)
        tab.update_image()
        tab.update_histogram()
        restored_points = int(snapshot.get("deleted_points", 0))
        self._clear_last_delete_snapshot()
        self._show_status_message(f"Restored {restored_points} points from the last ROI delete.")

    def move_roi_to_current_tab(self, idx: int):
        """Refresh the live ROI overlay after a tab change."""
        current_tab = self.main_window.tabs.widget(idx) if idx >= 0 else None

        if self.mode == "per_scan" and self._last_tab is not None and self._last_tab is not current_tab:
            self._save_live_state_for_tab(self._last_tab)
        elif self.mode == "global":
            self.global_roi_state = self._clone_state(self._current_live_state()) or self.global_roi_state

        state = self._get_state_for_tab(current_tab)
        self._apply_state_to_tab(current_tab, state)
        self._last_tab = current_tab

    def _toggle_roi_shape(self, shape: str):
        """Show or hide the requested ROI shape for the current tab or globally."""
        tab = self.main_window.current_tab()
        if tab is None or tab.grid is None:
            return

        current_state = self._get_state_for_tab(tab)
        if current_state is not None and current_state["shape"] == shape and current_state.get("visible", True):
            next_state = None
        else:
            reference_state = current_state or self.global_roi_state
            next_state = self._default_state(tab, shape, reference_state=reference_state)

        self._set_state_for_tab(tab, next_state)
        self._apply_state_to_tab(tab, next_state)
        self._last_tab = tab

    def show_circle_roi(self):
        """Show or hide the circular ROI in the active ROI mode."""
        self._toggle_roi_shape("circle")

    def show_ring_roi(self):
        """Show or hide the annular ROI in the active ROI mode."""
        self._toggle_roi_shape("ring")

    def show_rectangle_roi(self):
        """Show or hide the rectangular ROI in the active ROI mode."""
        self._toggle_roi_shape("rectangle")

    def show_polygon_roi(self):
        """Show or hide the polygon ROI in the active ROI mode."""
        self._toggle_roi_shape("polygon")
