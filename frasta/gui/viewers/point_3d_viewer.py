"""Experimental point-based 3D viewer built on QOpenGLWidget.

The widget intentionally mirrors the public entry point of the existing
pyqtgraph-based viewer while keeping the implementation separate for
incremental experiments.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .point_cloud_geometry import (
    build_colormap_lut,
    build_mesh_geometry_from_grid,
    build_point_positions_from_grid,
    compute_progressive_stride_schedule,
)
from .point_cloud_gl_widget import PointCloudGLWidget
from .grid_3d_viewer.colormap_manager import ColormapManager
from ..workers.mesh_geometry_worker import MeshGeometryWorker

import logging

logger = logging.getLogger(__name__)


class Point3DViewer(QtWidgets.QWidget):
    """Experimental viewer for rendering one or two grids as points or meshes."""

    def __init__(self, parent=None):
        """Initialize the point viewer and its controls."""
        super().__init__(parent)
        self.setWindowTitle("3D Point Viewer (Experimental)")
        self.resize(900, 700)

        self.colormap_manager = ColormapManager()
        self.gl_widget = PointCloudGLWidget(self)
        self._ref_grid = None
        self._adj_grid = None
        self._line_points = None
        self._pixel_size_x = 1.0
        self._pixel_size_y = 1.0
        self._separation = 0.0
        self._render_mode = "mesh"
        self._geometry_cache: dict[str, dict[str, dict[int, object]]] = {
            "points": {"ref": {}, "adj": {}},
            "mesh": {"ref": {}, "adj": {}},
        }
        self._mesh_request_id = 0
        self._mesh_workers: dict[tuple[int, str, int], MeshGeometryWorker] = {}
        self._stride_schedule = [1]
        self._stride_schedule_index = 0
        self._refinement_timer = QtCore.QTimer(self)
        self._refinement_timer.setSingleShot(True)
        self._refinement_timer.timeout.connect(self._advance_refinement_stage)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_controls())
        layout.addWidget(self.gl_widget, 1)

    def _build_controls(self) -> QtWidgets.QWidget:
        """Create a control panel visually aligned with the legacy 3D viewer."""
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        ref_row = QtWidgets.QHBoxLayout()
        adj_row = QtWidgets.QHBoxLayout()
        extra_row = QtWidgets.QHBoxLayout()

        self.checkbox_ref = QtWidgets.QCheckBox("Ref points")
        self.checkbox_ref.setChecked(True)
        self.checkbox_adj = QtWidgets.QCheckBox("Adj points")
        self.checkbox_adj.setChecked(True)
        self.combo_render_mode = QtWidgets.QComboBox()
        self.combo_render_mode.addItem("Points", userData="points")
        self.combo_render_mode.addItem("Shaded mesh", userData="mesh")
        self.combo_render_mode.setCurrentIndex(self.combo_render_mode.findData("mesh"))

        self.combo_cmap_ref = QtWidgets.QComboBox()
        self.combo_cmap_adj = QtWidgets.QComboBox()
        for combo in (self.combo_cmap_ref, self.combo_cmap_adj):
            combo.addItems(["None", "RG", "B&W", "Metrology", "viridis", "plasma", "magma"])
            combo.setCurrentText("Metrology")

        self.chk_auto_ref = QtWidgets.QCheckBox("Ref auto range")
        self.chk_auto_ref.setChecked(True)
        self.chk_auto_adj = QtWidgets.QCheckBox("Adj auto range")
        self.chk_auto_adj.setChecked(True)
        self.chk_link_ranges = QtWidgets.QCheckBox("Link ranges")

        self.spin_lo_ref = QtWidgets.QDoubleSpinBox()
        self.spin_hi_ref = QtWidgets.QDoubleSpinBox()
        self.spin_lo_adj = QtWidgets.QDoubleSpinBox()
        self.spin_hi_adj = QtWidgets.QDoubleSpinBox()
        for spinbox in (self.spin_lo_ref, self.spin_hi_ref, self.spin_lo_adj, self.spin_hi_adj):
            spinbox.setDecimals(6)
            spinbox.setRange(-1e12, 1e12)
            spinbox.setEnabled(False)

        self.spin_point_size = QtWidgets.QDoubleSpinBox()
        self.spin_point_size.setDecimals(1)
        self.spin_point_size.setRange(1.0, 20.0)
        self.spin_point_size.setSingleStep(0.5)
        self.spin_point_size.setValue(2.0)
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)

        self.colormap_manager.set_widgets(
            self.spin_lo_ref,
            self.spin_hi_ref,
            self.spin_lo_adj,
            self.spin_hi_adj,
            self.chk_auto_ref,
            self.chk_auto_adj,
            self.chk_link_ranges,
        )

        top_row.addWidget(QtWidgets.QLabel("Render:"))
        top_row.addWidget(self.combo_render_mode)
        top_row.addSpacing(12)
        top_row.addWidget(QtWidgets.QLabel("Point size:"))
        top_row.addWidget(self.spin_point_size)
        top_row.addStretch(1)

        ref_row.addWidget(self.checkbox_ref)
        ref_row.addWidget(QtWidgets.QLabel("Ref cmap:"))
        ref_row.addWidget(self.combo_cmap_ref)
        ref_row.addWidget(QtWidgets.QLabel("Ref lo/hi:"))
        ref_row.addWidget(self.spin_lo_ref)
        ref_row.addWidget(self.spin_hi_ref)
        ref_row.addWidget(self.chk_auto_ref)
        ref_row.addStretch(1)

        adj_row.addWidget(self.checkbox_adj)
        adj_row.addWidget(QtWidgets.QLabel("Adj cmap:"))
        adj_row.addWidget(self.combo_cmap_adj)
        adj_row.addWidget(QtWidgets.QLabel("Adj lo/hi:"))
        adj_row.addWidget(self.spin_lo_adj)
        adj_row.addWidget(self.spin_hi_adj)
        adj_row.addWidget(self.chk_auto_adj)
        adj_row.addWidget(self.chk_link_ranges)
        adj_row.addStretch(1)

        extra_row.addWidget(self.checkbox_line)
        extra_row.addWidget(self.checkbox_plane)
        extra_row.addStretch(1)

        layout.addLayout(top_row)
        layout.addLayout(ref_row)
        layout.addLayout(adj_row)
        layout.addLayout(extra_row)

        self.checkbox_ref.toggled.connect(lambda on: self.gl_widget.set_cloud_visible("ref", on))
        self.checkbox_adj.toggled.connect(lambda on: self.gl_widget.set_cloud_visible("adj", on))
        self.checkbox_line.toggled.connect(self.gl_widget.set_profile_line_visible)
        self.checkbox_plane.toggled.connect(self.gl_widget.set_profile_plane_visible)
        self.combo_render_mode.currentIndexChanged.connect(self._render_mode_changed)
        self.combo_cmap_ref.currentIndexChanged.connect(lambda _: self._refresh_clouds())
        self.combo_cmap_adj.currentIndexChanged.connect(lambda _: self._refresh_clouds())
        self.chk_auto_ref.toggled.connect(self._auto_ref_toggled)
        self.chk_auto_adj.toggled.connect(self._auto_adj_toggled)
        self.chk_link_ranges.toggled.connect(self._link_toggled)
        self.spin_lo_ref.valueChanged.connect(lambda _: self._range_changed("ref"))
        self.spin_hi_ref.valueChanged.connect(lambda _: self._range_changed("ref"))
        self.spin_lo_adj.valueChanged.connect(lambda _: self._range_changed("adj"))
        self.spin_hi_adj.valueChanged.connect(lambda _: self._range_changed("adj"))
        self.spin_point_size.valueChanged.connect(self.gl_widget.set_point_size)
        return panel

    def update_data(
        self,
        reference_grid,
        adjusted_grid=None,
        line_points=None,
        separation=0.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
    ) -> None:
        """Update the rendered point clouds from structured grids."""
        self._refinement_timer.stop()
        self._ref_grid = np.asarray(reference_grid, dtype=np.float32)
        self._adj_grid = None if adjusted_grid is None else np.asarray(adjusted_grid, dtype=np.float32)
        self._line_points = line_points
        self._pixel_size_x = float(pixel_size_x)
        self._pixel_size_y = float(pixel_size_y)
        self._separation = float(separation)
        self._mesh_request_id += 1
        self._geometry_cache = {
            "points": {"ref": {}, "adj": {}},
            "mesh": {"ref": {}, "adj": {}},
        }
        self._reset_refinement_schedule()
        has_adjusted = self._adj_grid is not None
        self.checkbox_adj.setVisible(has_adjusted)
        self.combo_cmap_adj.setVisible(has_adjusted)
        self.spin_lo_adj.setVisible(has_adjusted)
        self.spin_hi_adj.setVisible(has_adjusted)
        self.chk_auto_adj.setVisible(has_adjusted)
        self.chk_link_ranges.setVisible(has_adjusted)
        self._apply_render_mode_to_ui()
        self._refresh_clouds()
        self._refresh_profile_line()
        self.gl_widget.fit_camera_to_scene()
        self._schedule_next_refinement()

    def _refresh_clouds(self) -> None:
        """Rebuild point clouds using the current stride and color settings."""
        if self._ref_grid is None:
            return

        stride = self._current_stride()
        ref_range = self._get_value_range("ref", self._ref_grid)
        self._apply_geometry("ref", stride)
        self.gl_widget.set_cloud_style(
            "ref",
            self._build_lut(self.combo_cmap_ref),
            ref_range,
            visible=self.checkbox_ref.isChecked(),
        )
        self.colormap_manager.set_data_cache((None, None, self._ref_grid), (None, None, self._adj_grid))
        self._sync_range_widgets("ref", self._ref_grid)

        if self._adj_grid is None:
            self.gl_widget.clear_cloud("adj")
            return

        adj_range = self._get_value_range("adj", self._adj_grid)
        self._apply_geometry("adj", stride)
        self.gl_widget.set_cloud_style(
            "adj",
            self._build_lut(self.combo_cmap_adj),
            adj_range,
            visible=self.checkbox_adj.isChecked(),
        )
        self._sync_range_widgets("adj", self._adj_grid)

    def _refresh_profile_line(self) -> None:
        """Convert the profile line into 3D overlays and a cross-section plane."""
        if self._line_points is None or self._ref_grid is None:
            self.gl_widget.set_profile_lines(None, None)
            self.gl_widget.set_profile_plane(None)
            return
        pts = np.asarray(self._line_points, dtype=np.int32)
        if len(pts) < 2:
            self.gl_widget.set_profile_lines(None, None)
            self.gl_widget.set_profile_plane(None)
            return
        h, w = self._ref_grid.shape
        in_bounds = (
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        pts = pts[in_bounds]
        if len(pts) < 2:
            self.gl_widget.set_profile_lines(None, None)
            self.gl_widget.set_profile_plane(None)
            return

        ref_profile = self._ref_grid[pts[:, 1], pts[:, 0]]
        adj_profile = (
            self._adj_grid[pts[:, 1], pts[:, 0]] + self._separation
            if self._adj_grid is not None
            else None
        )

        pts_physical = np.column_stack(
            (
                pts[:, 0].astype(np.float32) * self._pixel_size_x,
                pts[:, 1].astype(np.float32) * -self._pixel_size_y,
            )
        )
        ref_line_positions = self._build_profile_line_positions(pts_physical, ref_profile)
        adj_line_positions = self._build_profile_line_positions(pts_physical, adj_profile)
        self.gl_widget.set_profile_lines(ref_line_positions, adj_line_positions)

        z_values = [ref_profile]
        if adj_profile is not None:
            z_values.append(adj_profile)
        z_all = np.concatenate([values[np.isfinite(values)] for values in z_values if values is not None])
        if len(z_all) < 1:
            self.gl_widget.set_profile_plane(None)
            return

        z_min = float(np.min(z_all))
        z_max = float(np.max(z_all))
        margin = max(0.1 * (z_max - z_min), 1.0)
        z_min -= margin
        z_max += margin
        plane_vertices = np.array(
            [
                [pts_physical[0, 0], pts_physical[0, 1], z_min],
                [pts_physical[-1, 0], pts_physical[-1, 1], z_min],
                [pts_physical[-1, 0], pts_physical[-1, 1], z_max],
                [pts_physical[0, 0], pts_physical[0, 1], z_max],
            ],
            dtype=np.float32,
        )
        self.gl_widget.set_profile_plane(plane_vertices)

    def _build_profile_line_positions(
        self,
        pts_physical: np.ndarray,
        z_values: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build one polyline for a profile overlay, skipping invalid values."""
        if z_values is None:
            return None
        valid = np.isfinite(z_values)
        if np.count_nonzero(valid) < 2:
            return None
        return np.column_stack(
            (
                pts_physical[valid, 0],
                pts_physical[valid, 1],
                z_values[valid].astype(np.float32),
            )
        ).astype(np.float32, copy=False)

    def _selected_colormap(self, combo: QtWidgets.QComboBox) -> str | None:
        """Normalize combo-box text to the renderer colormap API."""
        text = combo.currentText()
        return None if text == "None" else text

    def _build_lut(self, combo: QtWidgets.QComboBox) -> np.ndarray:
        """Build a compact LUT for the currently selected colormap."""
        return build_colormap_lut(self._selected_colormap(combo), size=256)

    def _sync_range_widgets(self, which: str, grid: np.ndarray) -> None:
        """Update range spinboxes from the effective data range."""
        lo, hi = self._get_value_range(which, grid)
        auto = self.colormap_manager.range_ref_auto if which == "ref" else self.colormap_manager.range_adj_auto
        self.colormap_manager.update_range_widgets(which, lo, hi, auto=auto)

    def _get_value_range(self, which: str, grid: np.ndarray) -> tuple[float, float]:
        """Return the current color normalization range for a grid."""
        return self.colormap_manager.get_lo_hi_for(which, grid)

    def _auto_ref_toggled(self, on: bool) -> None:
        """React to reference auto-range changes."""
        if self.colormap_manager.ui_auto_ref_toggled(on):
            self._refresh_clouds()

    def _auto_adj_toggled(self, on: bool) -> None:
        """React to adjusted auto-range changes."""
        if self.colormap_manager.ui_auto_adj_toggled(on):
            self._refresh_clouds()

    def _link_toggled(self, on: bool) -> None:
        """React to linked-range changes."""
        if self.colormap_manager.ui_link_toggled(on):
            self._refresh_clouds()

    def _range_changed(self, which: str) -> None:
        """React to manual range changes."""
        if self.colormap_manager.ui_lohi_changed(which):
            self._refresh_clouds()

    def _apply_geometry(self, which: str, stride: int) -> None:
        """Upload cached geometry for the active render mode."""
        geometry = self._get_or_build_geometry(which, stride)
        if self._render_mode == "mesh":
            if not self._can_present_mesh_stage(stride):
                preview_positions = self._get_or_build_point_preview(which, stride)
                self.gl_widget.set_cloud_data(which, preview_positions)
                return
            positions, normals, indices = geometry
            self.gl_widget.set_mesh_data(which, positions, normals, indices)
        else:
            self.gl_widget.set_cloud_data(which, geometry)

    def _get_or_build_geometry(self, which: str, stride: int):
        """Return cached geometry for a cloud, stride, and render mode."""
        cache = self._geometry_cache[self._render_mode][which]
        if stride in cache:
            return cache[stride]

        if which == "ref":
            grid = self._ref_grid
            z_offset = 0.0
        else:
            grid = self._adj_grid
            z_offset = self._separation

        if grid is None:
            if self._render_mode == "mesh":
                return (
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.uint32),
                )
            return np.empty((0, 3), dtype=np.float32)

        if self._render_mode == "mesh":
            self._ensure_mesh_geometry_worker(which, stride, grid, z_offset)
            return None
        else:
            geometry = build_point_positions_from_grid(
                grid,
                dx=self._pixel_size_x,
                dy=self._pixel_size_y,
                z_offset=z_offset,
                stride=stride,
            )
        cache[stride] = geometry
        return geometry

    def _get_or_build_point_preview(self, which: str, stride: int) -> np.ndarray:
        """Return cached point geometry used as a mesh preview before the mesh is ready."""
        cache = self._geometry_cache["points"][which]
        if stride in cache:
            return cache[stride]

        if which == "ref":
            grid = self._ref_grid
            z_offset = 0.0
        else:
            grid = self._adj_grid
            z_offset = self._separation

        if grid is None:
            return np.empty((0, 3), dtype=np.float32)

        preview_positions = build_point_positions_from_grid(
            grid,
            dx=self._pixel_size_x,
            dy=self._pixel_size_y,
            z_offset=z_offset,
            stride=stride,
        )
        cache[stride] = preview_positions
        return preview_positions

    def _current_stride(self) -> int:
        """Return the stride of the currently displayed refinement stage."""
        if not self._stride_schedule:
            return 1
        index = min(self._stride_schedule_index, len(self._stride_schedule) - 1)
        return self._stride_schedule[index]

    def _schedule_next_refinement(self) -> None:
        """Queue the next refinement stage without blocking the first frame."""
        if self._stride_schedule_index >= len(self._stride_schedule) - 1:
            return
        # Delay slightly so the coarse preview becomes visible first.
        delay_ms = 75 if self._render_mode == "points" else 120
        self._refinement_timer.start(delay_ms)

    def _advance_refinement_stage(self) -> None:
        """Render the next denser stage of the point cloud."""
        if self._stride_schedule_index >= len(self._stride_schedule) - 1:
            return
        self._stride_schedule_index += 1
        self._refresh_clouds()
        self._schedule_next_refinement()

    def _render_mode_changed(self, _index: int) -> None:
        """Switch the experimental backend between points and shaded mesh."""
        self._render_mode = self.combo_render_mode.currentData() or "points"
        self._mesh_request_id += 1
        self.gl_widget.set_render_mode(self._render_mode)
        self._apply_render_mode_to_ui()
        self._refinement_timer.stop()
        self._reset_refinement_schedule()
        self._refresh_clouds()
        self._schedule_next_refinement()

    def _apply_render_mode_to_ui(self) -> None:
        """Synchronize UI controls with the active render mode."""
        is_points = self._render_mode == "points"
        self.spin_point_size.setVisible(is_points)
        self.gl_widget.set_render_mode(self._render_mode)
        self.checkbox_ref.setText("Ref points" if is_points else "Ref mesh")
        self.checkbox_adj.setText("Adj points" if is_points else "Adj mesh")

    def _reset_refinement_schedule(self) -> None:
        """Recompute the stride schedule for the active render mode."""
        if self._ref_grid is None:
            self._stride_schedule = [1]
            self._stride_schedule_index = 0
            return
        target_initial_points = 120_000 if self._render_mode == "points" else 45_000
        if self._render_mode == "mesh":
            min_stride = 4 if self._adj_grid is not None else 2
        else:
            min_stride = 1
        self._stride_schedule = compute_progressive_stride_schedule(
            self._ref_grid.shape,
            cloud_count=2 if self._adj_grid is not None else 1,
            target_initial_points=target_initial_points,
            min_stride=min_stride,
        )
        self._stride_schedule_index = 0

    def _ensure_mesh_geometry_worker(
        self,
        which: str,
        stride: int,
        grid: np.ndarray,
        z_offset: float,
    ) -> None:
        """Start a background mesh-generation request if one is not pending."""
        key = (self._mesh_request_id, which, int(stride))
        if key in self._mesh_workers:
            return

        worker = MeshGeometryWorker(
            request_id=self._mesh_request_id,
            which=which,
            stride=stride,
            grid=grid,
            dx=self._pixel_size_x,
            dy=self._pixel_size_y,
            z_offset=z_offset,
        )
        worker.finished_geometry.connect(self._on_mesh_geometry_ready)
        worker.failed_geometry.connect(self._on_mesh_geometry_failed)
        worker.finished.connect(lambda _key=key: self._mesh_workers.pop(_key, None))
        self._mesh_workers[key] = worker
        worker.start()

    def _on_mesh_geometry_ready(self, request_id: int, which: str, stride: int, geometry) -> None:
        """Store finished mesh geometry and upload it if still relevant."""
        if request_id != self._mesh_request_id:
            return

        self._geometry_cache["mesh"][which][stride] = geometry
        if self._render_mode != "mesh":
            return
        if stride != self._current_stride():
            return
        if not self._can_present_mesh_stage(stride):
            return

        for cloud_name in self._visible_cloud_names():
            positions, normals, indices = self._geometry_cache["mesh"][cloud_name][stride]
            self.gl_widget.set_mesh_data(cloud_name, positions, normals, indices)

    def _on_mesh_geometry_failed(self, request_id: int, which: str, stride: int, message: str) -> None:
        """Log mesh-generation failures without crashing the GUI."""
        if request_id != self._mesh_request_id:
            return
        logger.warning(
            "Experimental mesh generation failed for %s stride=%s:\n%s",
            which,
            stride,
            message,
        )

    def _visible_cloud_names(self) -> list[str]:
        """Return cloud names that should participate in the current scene."""
        names = []
        if self._ref_grid is not None and self.checkbox_ref.isChecked():
            names.append("ref")
        if self._adj_grid is not None and self.checkbox_adj.isChecked():
            names.append("adj")
        return names

    def _can_present_mesh_stage(self, stride: int) -> bool:
        """Return True when all visible clouds have mesh geometry for this stage.

        The mesh path stays in point-preview mode until every visible surface
        has finished building its geometry for the current stride. This avoids
        mixed point/mesh frames during startup and refinement.
        """
        for cloud_name in self._visible_cloud_names():
            if stride not in self._geometry_cache["mesh"][cloud_name]:
                return False
        return True


_global_point_viewer = None


def show_point_3d_viewer(
    reference_grid,
    adjusted_grid=None,
    line_points=None,
    separation=0.0,
    pixel_size_x=1.0,
    pixel_size_y=1.0,
):
    """Display the experimental point-based 3D viewer window."""
    global _global_point_viewer
    if _global_point_viewer is None:
        _global_point_viewer = Point3DViewer()
    _global_point_viewer.update_data(
        reference_grid=reference_grid,
        adjusted_grid=adjusted_grid,
        line_points=line_points,
        separation=separation,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
    )
    _global_point_viewer.show()
    _global_point_viewer.raise_()
    _global_point_viewer.activateWindow()
