"""ROI (Region of Interest) controller for the main window.

Handles circular and rectangular ROI operations, including:
- showing and hiding ROI overlays,
- creating masks from ROI geometry,
- applying ROI masks to the current scan,
- switching between shared and per-scan ROI behavior.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from ..dialogs import ROIDialog

logger = logging.getLogger(__name__)


class ROIController:
    """Controller for ROI-related operations."""

    def __init__(self, main_window):
        """Initialize the ROI controller.

        Args:
            main_window: Reference to the main window instance.
        """
        self.main_window = main_window
        self.shared_circle_roi = None
        self.shared_rectangle_roi = None
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

    def _clone_state(self, state: dict | None) -> dict | None:
        """Return a detached copy of an ROI state dictionary."""
        if state is None:
            return None
        return {
            "shape": state["shape"],
            "pos": tuple(state["pos"]),
            "size": tuple(state["size"]),
            "visible": bool(state.get("visible", True)),
        }

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

    def _current_live_state(self) -> dict | None:
        """Capture the currently visible live ROI geometry."""
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

        return {
            "shape": "rectangle",
            "pos": (x_min + width_phys * 0.25, y_min + height_phys * 0.25),
            "size": (width_phys * 0.5, height_phys * 0.5),
            "visible": True,
        }

    def _detach_live_rois(self):
        """Remove live ROI items from all tab views."""
        tabs = self.main_window.tabs
        for roi in (self.shared_circle_roi, self.shared_rectangle_roi):
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
        if not self._is_roi_deleted(self.shared_rectangle_roi):
            self.shared_rectangle_roi.hide()

    def _apply_state_to_tab(self, tab, state: dict | None):
        """Display the selected ROI state on the requested tab."""
        self._detach_live_rois()
        self._hide_live_rois()

        if tab is None or tab.grid is None or state is None or not state.get("visible", True):
            return

        if state["shape"] == "circle":
            roi = self._ensure_circle_roi(tab)
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
            else:
                x_min, x_max, y_min, y_max = self._tab_physical_bounds(tab)
                center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
            return {
                "mode": self.mode,
                "enabled": False,
                "shape": "circle",
                "center": center,
                "size": (1.0, 1.0),
            }

        pos_x, pos_y = state["pos"]
        size_x, size_y = state["size"]
        return {
            "mode": self.mode,
            "enabled": bool(state.get("visible", True)),
            "shape": state["shape"],
            "center": (pos_x + size_x / 2.0, pos_y + size_y / 2.0),
            "size": (size_x, size_y),
        }

    def get_dialog_config(self, tab=None) -> dict:
        """Return the current ROI configuration for editing."""
        if tab is None:
            tab = self.main_window.current_tab()
        state = self._get_state_for_tab(tab)
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

    def create_mask(self, h: int, w: int, tab=None) -> np.ndarray | None:
        """Create a boolean mask for the requested tab and grid shape."""
        active_tab = tab if tab is not None else self.main_window.current_tab()
        state = self._get_state_for_tab(active_tab)
        if state is None and active_tab is None:
            state = self._current_live_state()
        if state is None or not state.get("visible", True):
            return None

        pos_x, pos_y = state["pos"]
        size_x, size_y = state["size"]
        center_x = pos_x + size_x / 2.0
        center_y = pos_y + size_y / 2.0
        if active_tab is None or active_tab.xi is None or active_tab.yi is None:
            if state["shape"] == "circle":
                return self.create_circle_mask((h, w), (center_x, center_y), size_x / 2.0)
            return self.create_rectangle_mask((h, w), (center_x, center_y), size_x, size_y)

        if state["shape"] == "circle":
            return self.create_circle_mask(active_tab.xi[:w], active_tab.yi[:h], (center_x, center_y), size_x / 2.0)
        return self.create_rectangle_mask(active_tab.xi[:w], active_tab.yi[:h], (center_x, center_y), size_x, size_y)

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

    def show_rectangle_roi(self):
        """Show or hide the rectangular ROI in the active ROI mode."""
        self._toggle_roi_shape("rectangle")
