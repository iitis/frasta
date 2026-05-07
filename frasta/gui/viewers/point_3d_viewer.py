"""Experimental point-based 3D viewer built on QOpenGLWidget.

The widget intentionally mirrors the public entry point of the existing
pyqtgraph-based viewer while keeping the implementation separate for
incremental experiments.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .point_cloud_geometry import (
    build_colormap_lut,
    build_mesh_geometry_from_grid,
    build_point_positions_from_grid,
    compute_progressive_stride_schedule,
)
from .point_cloud_gl_widget import PointCloudGLWidget
from .grid_3d_viewer.colormap_manager import ColormapManager
from ..workers.mesh_geometry_worker import MeshGeometryWorker
from ...utils import get_colormap

import logging

logger = logging.getLogger(__name__)


class Point3DViewer(QtWidgets.QWidget):
    """Experimental viewer for rendering one or two grids as points or meshes."""

    def __init__(self, parent=None):
        """Initialize the point viewer and its controls."""
        super().__init__(parent)
        self.setWindowTitle("3D Point Viewer (Experimental)")
        self.resize(900, 700)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

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
        layout.addWidget(self._build_controls(), 1)

    def _build_controls(self) -> QtWidgets.QWidget:
        """Create the full experimental 3D-viewer UI around the GL widget."""
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        content_row = QtWidgets.QHBoxLayout()

        self.checkbox_ref = QtWidgets.QCheckBox("Ref points")
        self.checkbox_ref.setChecked(True)
        self.checkbox_adj = QtWidgets.QCheckBox("Adj points")
        self.checkbox_adj.setChecked(True)
        self.combo_render_mode = QtWidgets.QComboBox()
        self.combo_render_mode.addItem("Points", userData="points")
        self.combo_render_mode.addItem("Shaded mesh", userData="mesh")
        self.combo_render_mode.setCurrentIndex(self.combo_render_mode.findData("mesh"))
        self.combo_projection_mode = QtWidgets.QComboBox()
        self.combo_projection_mode.addItem("Perspective", userData="perspective")
        self.combo_projection_mode.addItem("Orthogonal", userData="orthographic")
        self.combo_auto_range_mode = QtWidgets.QComboBox()
        self.combo_auto_range_mode.addItem("Full", userData="full")
        self.combo_auto_range_mode.addItem("Percentile", userData="percentile")
        self.combo_auto_range_mode.setCurrentIndex(self.combo_auto_range_mode.findData("full"))

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
        self.chk_hide_below_ref = QtWidgets.QCheckBox("Ref hide below")
        self.chk_hide_above_ref = QtWidgets.QCheckBox("Ref hide above")
        self.chk_hide_below_adj = QtWidgets.QCheckBox("Adj hide below")
        self.chk_hide_above_adj = QtWidgets.QCheckBox("Adj hide above")

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
        self.button_background = QtWidgets.QPushButton("Background...")
        self.button_background_reset = QtWidgets.QPushButton("Reset Background")
        self.button_screenshot = QtWidgets.QPushButton("Screenshot...")
        self.button_colorbar = QtWidgets.QPushButton("Colorbar...")
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)
        self.checkbox_reference_rect = QtWidgets.QCheckBox("Show Z=0 Rectangle")
        self.checkbox_reference_rect.setChecked(True)

        self.colormap_manager.set_widgets(
            self.spin_lo_ref,
            self.spin_hi_ref,
            self.spin_lo_adj,
            self.spin_hi_adj,
            self.chk_auto_ref,
            self.chk_auto_adj,
            self.chk_link_ranges,
        )

        self.point_size_label = QtWidgets.QLabel("Point size:")

        top_row.addWidget(QtWidgets.QLabel("Render:"))
        top_row.addWidget(self.combo_render_mode)
        top_row.addSpacing(12)
        top_row.addWidget(QtWidgets.QLabel("Projection:"))
        top_row.addWidget(self.combo_projection_mode)
        top_row.addSpacing(12)
        top_row.addWidget(QtWidgets.QLabel("Auto range:"))
        top_row.addWidget(self.combo_auto_range_mode)
        top_row.addSpacing(12)
        top_row.addWidget(self.point_size_label)
        top_row.addWidget(self.spin_point_size)
        top_row.addSpacing(12)
        top_row.addWidget(self.button_background)
        top_row.addWidget(self.button_background_reset)
        top_row.addSpacing(12)
        top_row.addWidget(self.button_screenshot)
        top_row.addWidget(self.button_colorbar)
        top_row.addStretch(1)
        sidebar = QtWidgets.QWidget(panel)
        sidebar.setMinimumWidth(160)
        sidebar.setMaximumWidth(190)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        view_group = QtWidgets.QGroupBox("View", sidebar)
        view_layout = QtWidgets.QVBoxLayout(view_group)
        view_layout.setContentsMargins(10, 12, 10, 10)
        view_layout.setSpacing(6)
        view_layout.addWidget(self.checkbox_ref)
        view_layout.addWidget(self.checkbox_adj)
        view_layout.addWidget(self.checkbox_line)
        view_layout.addWidget(self.checkbox_plane)
        view_layout.addWidget(self.checkbox_reference_rect)
        sidebar_layout.addWidget(view_group)

        reference_group = QtWidgets.QGroupBox("Reference", sidebar)
        reference_layout = QtWidgets.QFormLayout(reference_group)
        reference_layout.setContentsMargins(10, 12, 10, 10)
        reference_layout.setSpacing(6)
        reference_layout.addRow("Colormap:", self.combo_cmap_ref)
        reference_range_widget = QtWidgets.QWidget(reference_group)
        reference_range_layout = QtWidgets.QFormLayout(reference_range_widget)
        reference_range_layout.setContentsMargins(0, 0, 0, 0)
        reference_range_layout.setSpacing(4)
        reference_range_layout.addRow("Min:", self.spin_lo_ref)
        reference_range_layout.addRow("Max:", self.spin_hi_ref)
        reference_layout.addRow("Range:", reference_range_widget)
        reference_layout.addRow("", self.chk_auto_ref)
        reference_layout.addRow("", self.chk_hide_below_ref)
        reference_layout.addRow("", self.chk_hide_above_ref)
        sidebar_layout.addWidget(reference_group)

        self.group_adjusted = QtWidgets.QGroupBox("Adjusted", sidebar)
        adjusted_layout = QtWidgets.QFormLayout(self.group_adjusted)
        adjusted_layout.setContentsMargins(10, 12, 10, 10)
        adjusted_layout.setSpacing(6)
        adjusted_layout.addRow("Colormap:", self.combo_cmap_adj)
        adjusted_range_widget = QtWidgets.QWidget(self.group_adjusted)
        adjusted_range_layout = QtWidgets.QFormLayout(adjusted_range_widget)
        adjusted_range_layout.setContentsMargins(0, 0, 0, 0)
        adjusted_range_layout.setSpacing(4)
        adjusted_range_layout.addRow("Min:", self.spin_lo_adj)
        adjusted_range_layout.addRow("Max:", self.spin_hi_adj)
        adjusted_layout.addRow("Range:", adjusted_range_widget)
        adjusted_layout.addRow("", self.chk_auto_adj)
        adjusted_layout.addRow("", self.chk_link_ranges)
        adjusted_layout.addRow("", self.chk_hide_below_adj)
        adjusted_layout.addRow("", self.chk_hide_above_adj)
        sidebar_layout.addWidget(self.group_adjusted)
        sidebar_layout.addStretch(1)

        content_row.addWidget(sidebar, 0)
        content_row.addWidget(self.gl_widget, 1)

        layout.addLayout(top_row)
        layout.addLayout(content_row, 1)

        self.checkbox_ref.toggled.connect(lambda on: self.gl_widget.set_cloud_visible("ref", on))
        self.checkbox_adj.toggled.connect(lambda on: self.gl_widget.set_cloud_visible("adj", on))
        self.checkbox_line.toggled.connect(self.gl_widget.set_profile_line_visible)
        self.checkbox_plane.toggled.connect(self.gl_widget.set_profile_plane_visible)
        self.checkbox_reference_rect.toggled.connect(self.gl_widget.set_reference_rectangle_visible)
        self.combo_render_mode.currentIndexChanged.connect(self._render_mode_changed)
        self.combo_projection_mode.currentIndexChanged.connect(self._projection_mode_changed)
        self.combo_auto_range_mode.currentIndexChanged.connect(self._auto_range_mode_changed)
        self.button_background.clicked.connect(self._choose_background_color)
        self.button_background_reset.clicked.connect(self.gl_widget.reset_background_color)
        self.combo_cmap_ref.currentIndexChanged.connect(lambda _: self._refresh_clouds())
        self.combo_cmap_adj.currentIndexChanged.connect(lambda _: self._refresh_clouds())
        self.chk_auto_ref.toggled.connect(self._auto_ref_toggled)
        self.chk_auto_adj.toggled.connect(self._auto_adj_toggled)
        self.chk_link_ranges.toggled.connect(self._link_toggled)
        self.chk_hide_below_ref.toggled.connect(lambda _: self._refresh_clouds())
        self.chk_hide_above_ref.toggled.connect(lambda _: self._refresh_clouds())
        self.chk_hide_below_adj.toggled.connect(lambda _: self._refresh_clouds())
        self.chk_hide_above_adj.toggled.connect(lambda _: self._refresh_clouds())
        self.spin_lo_ref.valueChanged.connect(lambda _: self._range_changed("ref"))
        self.spin_hi_ref.valueChanged.connect(lambda _: self._range_changed("ref"))
        self.spin_lo_adj.valueChanged.connect(lambda _: self._range_changed("adj"))
        self.spin_hi_adj.valueChanged.connect(lambda _: self._range_changed("adj"))
        self.spin_point_size.valueChanged.connect(self.gl_widget.set_point_size)
        self.button_screenshot.clicked.connect(self._export_screenshot)
        self.button_colorbar.clicked.connect(self._export_colorbar)
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
        self.group_adjusted.setVisible(has_adjusted)
        
        has_profile = line_points is not None and len(line_points) >= 2
        self.checkbox_line.setVisible(has_profile)
        self.checkbox_plane.setVisible(has_profile)

        self._refresh_reference_rectangle()
        self._apply_render_mode_to_ui()
        self._refresh_clouds()
        self._refresh_profile_line()
        self.gl_widget.fit_camera_to_scene(reset_orientation=True)
        self._schedule_next_refinement()

    def update_profile_overlay(
        self,
        line_points=None,
        separation: float | None = None,
    ) -> None:
        """Refresh only the profile-line and section-plane overlays.

        This lightweight path is used by the profile-analysis window to keep an
        already-open 3D view in sync while the ROI line is being dragged. The
        surface geometry and camera are left untouched.

        Args:
            line_points: Updated profile polyline in local pixel coordinates.
            separation: Optional adjusted-surface offset applied to the profile.
        """
        self._line_points = line_points
        if separation is not None:
            self._separation = float(separation)

        has_profile = line_points is not None and len(line_points) >= 2
        self.checkbox_line.setVisible(has_profile)
        self.checkbox_plane.setVisible(has_profile)
        self._refresh_profile_line()

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
            hide_below_range=self.chk_hide_below_ref.isChecked(),
            hide_above_range=self.chk_hide_above_ref.isChecked(),
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
            hide_below_range=self.chk_hide_below_adj.isChecked(),
            hide_above_range=self.chk_hide_above_adj.isChecked(),
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
        margin = max(0.2 * (z_max - z_min), 1.0)
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

    def _refresh_reference_rectangle(self) -> None:
        """Build a ``Z=0`` rectangle that matches the reference-grid extents."""
        if self._ref_grid is None or self._ref_grid.ndim != 2:
            self.gl_widget.set_reference_rectangle(None)
            self.gl_widget.set_reference_rectangle_annotations(None)
            return

        rows, cols = self._ref_grid.shape
        x_max = max(0.0, (cols - 1) * self._pixel_size_x)
        y_min = min(0.0, -(rows - 1) * self._pixel_size_y)
        x_ticks = self._build_reference_axis_ticks(0.0, x_max)
        y_ticks = self._build_reference_axis_ticks(y_min, 0.0)
        unit_scale, unit_label = self._select_export_axis_unit(max(x_max, abs(y_min)))
        rectangle_positions = self._build_reference_rectangle_positions(
            x_min=0.0,
            x_max=x_max,
            y_min=y_min,
            y_max=0.0,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
        )
        rectangle_annotations = self._build_reference_rectangle_annotations(
            x_min=0.0,
            x_max=x_max,
            y_min=y_min,
            y_max=0.0,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            unit_scale=unit_scale,
            unit_label=unit_label,
        )
        self.gl_widget.set_reference_rectangle(rectangle_positions)
        self.gl_widget.set_reference_rectangle_annotations(rectangle_annotations)

    def _build_reference_rectangle_positions(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x_ticks: list[float],
        y_ticks: list[float],
    ) -> np.ndarray:
        """Build rectangle-edge and tick-mark line segments for the ``Z=0`` guide."""
        segments: list[list[float]] = [
            [x_min, y_max, 0.0], [x_max, y_max, 0.0],
            [x_max, y_max, 0.0], [x_max, y_min, 0.0],
            [x_max, y_min, 0.0], [x_min, y_min, 0.0],
            [x_min, y_min, 0.0], [x_min, y_max, 0.0],
        ]

        x_tick_length = max(abs(y_max - y_min) * 0.015, self._pixel_size_y * 3.0, 0.5)
        y_tick_length = max(abs(x_max - x_min) * 0.015, self._pixel_size_x * 3.0, 0.5)
        for x_value in x_ticks:
            segments.append([x_value, y_max, 0.0])
            segments.append([x_value, y_max + x_tick_length, 0.0])
            segments.append([x_value, y_min, 0.0])
            segments.append([x_value, y_min - x_tick_length, 0.0])

        for y_value in y_ticks:
            segments.append([x_min, y_value, 0.0])
            segments.append([x_min - y_tick_length, y_value, 0.0])
            segments.append([x_max, y_value, 0.0])
            segments.append([x_max + y_tick_length, y_value, 0.0])

        return np.asarray(segments, dtype=np.float32)

    def _build_reference_axis_ticks(self, axis_min: float, axis_max: float) -> list[float]:
        """Build rounded major-tick positions for one reference-rectangle axis."""
        if not np.isfinite(axis_min) or not np.isfinite(axis_max):
            return []

        lower = min(float(axis_min), float(axis_max))
        upper = max(float(axis_min), float(axis_max))
        extent = upper - lower
        if extent <= 1e-9:
            return [lower]

        step = self._select_export_axis_tick_step(extent)
        if step <= 0.0:
            return [lower, upper]

        tick_values: list[float] = []
        start_index = int(np.ceil((lower - 1e-9) / step))
        end_index = int(np.floor((upper + 1e-9) / step))
        for tick_index in range(start_index, end_index + 1):
            value = tick_index * step
            if lower - 1e-9 <= value <= upper + 1e-9:
                tick_values.append(float(value))

        for boundary in (lower, upper):
            if not any(abs(existing - boundary) <= max(1e-6, step * 1e-6) for existing in tick_values):
                tick_values.append(float(boundary))

        tick_values.sort()
        return tick_values

    def _build_reference_rectangle_annotations(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x_ticks: list[float],
        y_ticks: list[float],
        unit_scale: float,
        unit_label: str,
    ) -> dict[str, object]:
        """Build screen-space label metadata for the reference-rectangle ticks."""
        return {
            "frame_world": np.array(
                [
                    [x_min, y_max, 0.0],
                    [x_max, y_max, 0.0],
                    [x_max, y_min, 0.0],
                    [x_min, y_min, 0.0],
                ],
                dtype=np.float32,
            ),
            "center": np.array(
                [
                    0.5 * (x_min + x_max),
                    0.5 * (y_min + y_max),
                    0.0,
                ],
                dtype=np.float32,
            ),
            "x_ticks": [float(tick_value) for tick_value in x_ticks],
            "y_ticks": [float(tick_value) for tick_value in y_ticks],
            "unit_scale": float(unit_scale),
            "unit_label": str(unit_label),
        }

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

    def _projection_mode_changed(self, _index: int) -> None:
        """Switch the camera projection used by the experimental 3D widget."""
        self.gl_widget.set_projection_mode(
            self.combo_projection_mode.currentData() or "perspective"
        )

    def _auto_range_mode_changed(self, _index: int) -> None:
        """Switch how automatic color ranges are computed in the 3D viewer."""
        if self.colormap_manager.set_auto_range_mode(
            self.combo_auto_range_mode.currentData() or "full"
        ):
            self._refresh_clouds()

    def _choose_background_color(self) -> None:
        """Open a color dialog and apply a new viewer background color."""
        color = QtWidgets.QColorDialog.getColor(
            self.gl_widget.get_background_color(),
            self,
            "Choose 3D background color",
        )
        if color.isValid():
            self.gl_widget.set_background_color(color)

    def _apply_render_mode_to_ui(self) -> None:
        """Synchronize UI controls with the active render mode."""
        is_points = self._render_mode == "points"
        self.spin_point_size.setVisible(is_points)
        self.point_size_label.setVisible(is_points)
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

    def _stop_mesh_workers(self) -> None:
        """Stop and forget all active background mesh workers."""
        workers = list(self._mesh_workers.values())
        self._mesh_workers.clear()
        for worker in workers:
            try:
                worker.requestInterruption()
            except Exception:
                pass
            worker.quit()
            worker.wait(250)

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

    def _export_screenshot(self) -> None:
        """Export the current 3D view to a PNG image."""
        current_width = max(1, self.gl_widget.width())
        current_height = max(1, self.gl_widget.height())

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export 3D screenshot")
        form = QtWidgets.QFormLayout(dialog)

        spin_width = QtWidgets.QSpinBox(dialog)
        spin_width.setRange(1, 16384)
        spin_width.setValue(current_width)

        spin_height = QtWidgets.QSpinBox(dialog)
        spin_height.setRange(1, 16384)
        spin_height.setValue(current_height)

        chk_transparent = QtWidgets.QCheckBox("Transparent background", dialog)
        chk_transparent.setChecked(True)

        form.addRow("Width [px]:", spin_width)
        form.addRow("Height [px]:", spin_height)
        form.addRow("", chk_transparent)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 3D screenshot",
            "frasta-3d-view.png",
            "PNG Images (*.png)",
        )
        if not output_path:
            return

        image = self._build_screenshot_image(
            width=spin_width.value(),
            height=spin_height.value(),
            transparent_background=chk_transparent.isChecked(),
        )
        if not image.save(output_path, "PNG"):
            QtWidgets.QMessageBox.warning(
                self,
                "Export failed",
                "Could not save the screenshot to the selected file.",
            )

    def _build_screenshot_image(
        self,
        width: int,
        height: int,
        transparent_background: bool,
    ) -> QtGui.QImage:
        """Build the exported 3D screenshot image.

        Args:
            width: Final image width in pixels.
            height: Final image height in pixels.
            transparent_background: If True, keep the background alpha at 0.

        Returns:
            Final PNG-ready screenshot image.
        """
        width = max(1, int(width))
        height = max(1, int(height))
        return self.gl_widget.render_to_image(
            width=width,
            height=height,
            transparent_background=transparent_background,
        )

    def _draw_export_axes_overlay(
        self,
        axes_overlay: dict[str, object] | None,
        painter: QtGui.QPainter,
        content_width: int,
        content_height: int,
        offset: QtCore.QPointF,
    ) -> None:
        """Draw label-only X/Y annotations around the screenshot."""
        if axes_overlay is None:
            return

        frame_world = np.asarray(axes_overlay["frame_world"], dtype=np.float32)
        projected = self.gl_widget.project_world_points(frame_world, content_width, content_height)
        projected += np.array([[offset.x(), offset.y()]], dtype=np.float32)

        origin_index = int(axes_overlay["origin_index"])
        x_adjacent_index = int(axes_overlay["x_adjacent_index"])
        y_adjacent_index = int(axes_overlay["y_adjacent_index"])
        x_extent = float(axes_overlay["x_extent"])
        y_extent = float(axes_overlay["y_extent"])
        unit_scale = float(axes_overlay["unit_scale"])
        unit_label = str(axes_overlay["unit_label"])
        x_ticks = list(axes_overlay.get("x_ticks", []))
        y_ticks = list(axes_overlay.get("y_ticks", []))
        center = np.mean(projected, axis=0)
        self._draw_export_axis(
            painter,
            start=projected[origin_index],
            end=projected[x_adjacent_index],
            center=center,
            unit_label=unit_label,
            axis_name="X",
            ticks=x_ticks,
        )
        self._draw_export_axis(
            painter,
            start=projected[origin_index],
            end=projected[y_adjacent_index],
            center=center,
            unit_label=unit_label,
            axis_name="Y",
            ticks=y_ticks,
        )

    def _build_export_axes_overlay(self, z_plane: float | None = None) -> dict[str, object] | None:
        """Build export-only X/Y frame geometry anchored below the scene."""
        bounds = self.gl_widget.get_scene_bounds()
        if bounds is None:
            return None

        mins, maxs = bounds
        if z_plane is None:
            # Keep the frame clearly below the visible geometry. Using only the
            # Z span can place the frame too close to wide, relatively flat
            # surfaces, which makes depth precision artifacts more likely.
            scene_span = np.asarray(maxs - mins, dtype=np.float32)
            depth_offset = max(
                1e-3,
                0.02 * float(np.max(scene_span)),
            )
            z_plane = float(mins[2] - depth_offset)
        frame_world = np.array(
            [
                [mins[0], mins[1], z_plane],
                [maxs[0], mins[1], z_plane],
                [maxs[0], maxs[1], z_plane],
                [mins[0], maxs[1], z_plane],
            ],
            dtype=np.float32,
        )
        projected = self.gl_widget.project_world_points(frame_world, max(1, self.gl_widget.width()), max(1, self.gl_widget.height()))
        origin_index = int(np.argmin(projected[:, 0] - projected[:, 1]))
        # The export frame is a closed 4-corner rectangle, so neighbor lookup
        # must wrap explicitly instead of relying on ``origin_index +/- 1``.
        adjacent_indices = {
            0: (1, 3),
            1: (0, 2),
            2: (3, 1),
            3: (0, 2),
        }
        x_adjacent_index, y_adjacent_index = adjacent_indices[origin_index]

        x_extent = abs(float(frame_world[x_adjacent_index, 0] - frame_world[origin_index, 0]))
        y_extent = abs(float(frame_world[y_adjacent_index, 1] - frame_world[origin_index, 1]))
        unit_scale, unit_label = self._select_export_axis_unit(max(x_extent, y_extent))
        x_ticks = self._build_export_axis_ticks(x_extent, unit_scale)
        y_ticks = self._build_export_axis_ticks(y_extent, unit_scale)
        if not x_ticks:
            x_ticks = self._build_fallback_export_axis_ticks()
        if not y_ticks:
            y_ticks = self._build_fallback_export_axis_ticks()

        positions = self._build_export_axes_positions(
            frame_world=frame_world,
            origin_index=origin_index,
            x_adjacent_index=x_adjacent_index,
            y_adjacent_index=y_adjacent_index,
            x_tick_fractions=[float(tick["fraction"]) for tick in x_ticks],
            y_tick_fractions=[float(tick["fraction"]) for tick in y_ticks],
        )
        return {
            "positions": positions,
            "frame_world": frame_world,
            "origin_index": origin_index,
            "x_adjacent_index": x_adjacent_index,
            "y_adjacent_index": y_adjacent_index,
            "x_extent": x_extent,
            "y_extent": y_extent,
            "unit_scale": unit_scale,
            "unit_label": unit_label,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "color": (0.35, 0.35, 0.35, 0.85),
        }

    @staticmethod
    def _build_export_axes_positions(
        frame_world: np.ndarray,
        origin_index: int,
        x_adjacent_index: int,
        y_adjacent_index: int,
        x_tick_fractions: list[float] | None = None,
        y_tick_fractions: list[float] | None = None,
    ) -> np.ndarray:
        """Build 3D line segments for the screenshot X/Y frame and ticks."""
        segments: list[np.ndarray] = []
        frame_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        for start_idx, end_idx in frame_edges:
            segments.append(frame_world[start_idx])
            segments.append(frame_world[end_idx])

        axis_tick_fractions = (
            (x_adjacent_index, x_tick_fractions or []),
            (y_adjacent_index, y_tick_fractions or []),
        )
        for end_index, tick_fractions in axis_tick_fractions:
            start = frame_world[origin_index]
            end = frame_world[end_index]
            edge = end - start
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 1e-6:
                continue
            edge_dir = edge / edge_length
            tick_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            tick_length = max(0.01 * edge_length, 0.75)
            for t in tick_fractions:
                point = start + edge_dir * (edge_length * t)
                segments.append(point)
                segments.append(point + tick_dir * tick_length)

        return np.asarray(segments, dtype=np.float32)

    def _draw_export_axis(
        self,
        painter: QtGui.QPainter,
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        unit_label: str,
        axis_name: str,
        ticks: list[dict[str, float | str]],
    ) -> None:
        """Draw one labeled screenshot axis with ticks and numeric labels."""
        direction = np.asarray(end - start, dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return

        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        midpoint = 0.5 * (start + end)
        if np.linalg.norm((midpoint + normal * 12.0) - center) < np.linalg.norm((midpoint - normal * 12.0) - center):
            normal *= -1.0

        axis_pen = QtGui.QPen(QtGui.QColor(40, 40, 40, 230), 1.4)
        tick_pen = QtGui.QPen(QtGui.QColor(70, 70, 70, 220), 1.0)
        painter.setPen(axis_pen)
        painter.drawLine(
            QtCore.QPointF(float(start[0]), float(start[1])),
            QtCore.QPointF(float(end[0]), float(end[1])),
        )

        label_font = QtGui.QFont()
        label_font.setPointSize(9)
        painter.setFont(label_font)
        for tick in ticks:
            t = float(tick["fraction"])
            tick_point = start + direction * (length * t)
            tick_out = tick_point + normal * 6.0
            painter.setPen(tick_pen)
            painter.drawLine(
                QtCore.QPointF(float(tick_point[0]), float(tick_point[1])),
                QtCore.QPointF(float(tick_out[0]), float(tick_out[1])),
            )

            label = str(tick["label"])
            label_anchor = tick_out + normal * 10.0
            label_rect = QtCore.QRectF(
                float(label_anchor[0] - 28.0),
                float(label_anchor[1] - 10.0),
                56.0,
                20.0,
            )
            painter.setPen(axis_pen)
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, label)

        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        title_anchor = end + normal * 26.0
        title_rect = QtCore.QRectF(
            float(title_anchor[0] - 44.0),
            float(title_anchor[1] - 12.0),
            88.0,
            24.0,
        )
        painter.drawText(title_rect, QtCore.Qt.AlignCenter, f"{axis_name} [{unit_label}]")

    @staticmethod
    def _select_export_axis_unit(max_extent_um: float) -> tuple[float, str]:
        """Choose a compact XY unit for screenshot axis labels."""
        if max_extent_um >= 1000.0:
            return 1000.0, "mm"
        return 1.0, "um"

    def _build_export_axis_ticks(self, extent_um: float, unit_scale: float) -> list[dict[str, float | str]]:
        """Build readable export-axis ticks at rounded steps in the active unit."""
        if not np.isfinite(extent_um) or extent_um <= 1e-9 or unit_scale <= 0.0:
            return []

        extent_units = extent_um / unit_scale
        step = self._select_export_axis_tick_step(extent_units)
        if step <= 0.0:
            return []

        tick_limit = max(1, int(np.floor((extent_units + 1e-9) / step)))
        ticks: list[dict[str, float | str]] = []
        for tick_index in range(1, tick_limit + 1):
            value = tick_index * step
            if value > extent_units + 1e-9:
                break
            ticks.append(
                {
                    "fraction": float((value * unit_scale) / extent_um),
                    "value": float(value),
                    "label": self._format_axis_value(value),
                }
            )
        return ticks

    @staticmethod
    def _select_export_axis_tick_step(extent_units: float) -> float:
        """Choose a rounded major-tick step that keeps the export axis readable."""
        if not np.isfinite(extent_units) or extent_units <= 0.0:
            return 0.0

        target_tick_count = 8.0
        raw_step = extent_units / target_tick_count
        magnitude = 10.0 ** np.floor(np.log10(raw_step)) if raw_step > 0.0 else 1.0
        normalized = raw_step / magnitude
        if normalized <= 1.0:
            nice_factor = 1.0
        elif normalized <= 2.0:
            nice_factor = 2.0
        elif normalized <= 5.0:
            nice_factor = 5.0
        else:
            nice_factor = 10.0
        return nice_factor * magnitude

    @staticmethod
    def _build_fallback_export_axis_ticks() -> list[dict[str, float | str]]:
        """Build a minimal fallback tick layout if rounded ticks cannot be derived."""
        fractions = (0.25, 0.5, 0.75, 1.0)
        return [
            {
                "fraction": float(fraction),
                "value": float(fraction),
                "label": "",
            }
            for fraction in fractions
        ]

    @staticmethod
    def _format_axis_value(value: float) -> str:
        """Format one screenshot axis tick label compactly."""
        if not np.isfinite(value):
            return "nan"
        magnitude = abs(float(value))
        if magnitude >= 1e4 or (0 < magnitude < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _export_colorbar(self) -> None:
        """Export a standalone colorbar image for the current surface styling."""
        has_adjusted = self._adj_grid is not None

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export colorbar")
        form = QtWidgets.QFormLayout(dialog)

        combo_surface = QtWidgets.QComboBox(dialog)
        combo_surface.addItem("Reference", userData="ref")
        if has_adjusted:
            combo_surface.addItem("Adjusted", userData="adj")

        spin_width = QtWidgets.QSpinBox(dialog)
        spin_width.setRange(32, 4096)
        spin_width.setValue(220)

        spin_height = QtWidgets.QSpinBox(dialog)
        spin_height.setRange(64, 4096)
        spin_height.setValue(900)

        chk_transparent = QtWidgets.QCheckBox("Transparent background", dialog)
        chk_transparent.setChecked(True)
        chk_histogram = QtWidgets.QCheckBox("Include histogram", dialog)
        chk_histogram.setChecked(True)

        form.addRow("Surface:", combo_surface)
        form.addRow("Width [px]:", spin_width)
        form.addRow("Height [px]:", spin_height)
        form.addRow("", chk_transparent)
        form.addRow("", chk_histogram)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        which = combo_surface.currentData() or "ref"
        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save colorbar",
            f"frasta-{which}-colorbar.png",
            "PNG Images (*.png)",
        )
        if not output_path:
            return

        image = self._build_colorbar_image(
            which=which,
            width=spin_width.value(),
            height=spin_height.value(),
            transparent_background=chk_transparent.isChecked(),
            include_histogram=chk_histogram.isChecked(),
        )
        if not image.save(output_path, "PNG"):
            QtWidgets.QMessageBox.warning(
                self,
                "Export failed",
                "Could not save the colorbar to the selected file.",
            )

    def _build_colorbar_image(
        self,
        which: str,
        width: int,
        height: int,
        transparent_background: bool,
        include_histogram: bool,
    ) -> QtGui.QImage:
        """Build a standalone colorbar image for one surface.

        Args:
            which: Surface identifier, ``"ref"`` or ``"adj"``.
            width: Output image width in pixels.
            height: Output image height in pixels.
            transparent_background: If True, use an alpha-0 background.

        Returns:
            A rendered QImage containing the colorbar, labels, and optional histogram.
        """
        width = max(32, int(width))
        height = max(64, int(height))

        if which == "adj" and self._adj_grid is not None:
            grid = self._adj_grid
            combo = self.combo_cmap_adj
            title = "Adjusted"
        else:
            which = "ref"
            grid = self._ref_grid
            combo = self.combo_cmap_ref
            title = "Reference"

        lo, hi = self._get_value_range(which, grid)
        cmap_name = combo.currentText()
        hist_values = self._get_colorbar_histogram_values(which, grid, lo, hi)
        unit_label = "μm"

        unit_label = "um"
        image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32_Premultiplied)
        background = QtGui.QColor(0, 0, 0, 0) if transparent_background else QtGui.QColor(255, 255, 255, 255)
        image.fill(background)

        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

            text_color = QtGui.QColor(255, 255, 255) if transparent_background else QtGui.QColor(20, 20, 20)
            border_color = QtGui.QColor(255, 255, 255, 180) if transparent_background else QtGui.QColor(40, 40, 40)
            painter.setPen(text_color)
            title_font_size, label_font_size = self._get_colorbar_font_sizes(width, height)

            title_font = QtGui.QFont()
            title_font.setBold(True)
            title_font.setPointSize(title_font_size)
            painter.setFont(title_font)
            title_rect = QtCore.QRectF(12, 10, width - 24, 28)
            painter.drawText(
                title_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                f"{title} [{unit_label}]",
            )

            label_font = QtGui.QFont()
            label_font.setPointSize(label_font_size)
            painter.setFont(label_font)

            bar_top = 52
            bar_bottom = max(bar_top + 40, height - 24)
            bar_height = bar_bottom - bar_top
            bar_width = max(28, min(40, width // 5))
            bar_left = 20
            bar_rect = QtCore.QRect(bar_left, bar_top, bar_width, bar_height)

            gradient = QtGui.QLinearGradient(
                float(bar_rect.left()),
                float(bar_rect.bottom()),
                float(bar_rect.left()),
                float(bar_rect.top()),
            )
            if cmap_name == "None":
                gradient.setColorAt(0.0, QtGui.QColor.fromRgbF(0.85, 0.85, 0.85, 1.0))
                gradient.setColorAt(1.0, QtGui.QColor.fromRgbF(0.85, 0.85, 0.85, 1.0))
            elif cmap_name == "RG":
                gradient.setColorAt(0.0, QtGui.QColor(255, 0, 0))
                gradient.setColorAt(1.0, QtGui.QColor(0, 255, 0))
            elif cmap_name == "B&W":
                gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0))
                gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255))
            else:
                cmap = get_colormap(None if cmap_name == "None" else cmap_name)
                stops = cmap.getStops(mode="byte")
                for position, rgba in zip(stops[0], stops[1]):
                    gradient.setColorAt(
                        float(position),
                        QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
                    )

            painter.fillRect(bar_rect, gradient)
            painter.setPen(QtGui.QPen(border_color, 1.0))
            painter.drawRect(bar_rect)

            if include_histogram and hist_values.size > 0:
                hist_left = bar_rect.right() + 4
                hist_width = max(20, min(60, width // 4))
                hist_rect = QtCore.QRect(hist_left, bar_top, hist_width, bar_height)
                self._draw_colorbar_histogram(
                    painter,
                    hist_rect,
                    hist_values,
                    lo,
                    hi,
                    QtGui.QColor(180, 180, 180, 170),
                )

            label_left = bar_rect.right() + (max(20, min(60, width // 4)) + 12 if include_histogram else 12)
            self._draw_colorbar_ticks(
                painter,
                bar_rect,
                label_left,
                width,
                lo,
                hi,
                text_color,
                border_color,
                label_font_size=label_font_size,
            )
        finally:
            painter.end()

        return image

    def _get_colorbar_histogram_values(
        self,
        which: str,
        grid: np.ndarray | None,
        lo: float,
        hi: float,
    ) -> np.ndarray:
        """Return displayed finite values contributing to the 3D colorbar histogram."""
        if grid is None:
            return np.empty(0, dtype=np.float32)

        values = np.asarray(grid, dtype=np.float32)
        mask = np.isfinite(values)
        if which == "ref":
            if self.chk_hide_below_ref.isChecked():
                mask &= values >= lo
            if self.chk_hide_above_ref.isChecked():
                mask &= values <= hi
        else:
            if self.chk_hide_below_adj.isChecked():
                mask &= values >= lo
            if self.chk_hide_above_adj.isChecked():
                mask &= values <= hi

        if not np.any(mask):
            return np.empty(0, dtype=np.float32)

        clipped = np.clip(values[mask], lo, hi)
        return clipped[np.isfinite(clipped)].astype(np.float32, copy=False)

    def _draw_colorbar_histogram(
        self,
        painter: QtGui.QPainter,
        hist_rect: QtCore.QRect,
        values: np.ndarray,
        vmin: float,
        vmax: float,
        color: QtGui.QColor,
    ) -> None:
        """Draw a normalized histogram beside the exported 3D colorbar."""
        if values.size < 1 or vmax <= vmin:
            return
        counts, edges = np.histogram(values, bins=min(128, max(16, hist_rect.height() // 4)), range=(vmin, vmax))
        if counts.size < 1 or np.max(counts) <= 0:
            return

        counts = counts.astype(np.float32)
        counts /= float(np.max(counts))
        if counts.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= np.sum(kernel)
            counts = np.convolve(counts, kernel, mode="same")
            counts /= max(float(np.max(counts)), 1e-9)

        path = QtGui.QPainterPath()
        path.moveTo(hist_rect.left(), hist_rect.bottom())
        for idx, count in enumerate(counts):
            y0 = hist_rect.bottom() - (edges[idx] - vmin) / (vmax - vmin) * hist_rect.height()
            y1 = hist_rect.bottom() - (edges[idx + 1] - vmin) / (vmax - vmin) * hist_rect.height()
            x = hist_rect.left() + count * hist_rect.width()
            if idx == 0:
                path.lineTo(x, y0)
            path.lineTo(x, y1)
        path.lineTo(hist_rect.left(), hist_rect.top())
        path.closeSubpath()
        painter.fillPath(path, QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(color.darker(130), 1.0))
        painter.drawPath(path)

    def _draw_colorbar_ticks(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        label_left: int,
        image_width: int,
        vmin: float,
        vmax: float,
        text_color: QtGui.QColor,
        tick_color: QtGui.QColor,
        label_font_size: int = 11,
    ) -> None:
        """Draw major and minor 3D colorbar ticks with a highlighted zero value."""
        painter.setPen(text_color)
        label_font = QtGui.QFont()
        label_font.setPointSize(int(label_font_size))
        painter.setFont(label_font)

        tick_layout = self._build_colorbar_tick_layout(vmin, vmax)
        major_ticks = tick_layout["major"]
        minor_ticks = tick_layout["minor"]

        for tick in major_ticks:
            value = float(tick["value"])
            fraction = float(tick["fraction"])
            is_zero = bool(tick["is_zero"])
            y = bar_rect.bottom() - fraction * bar_rect.height()
            tick_length = 12 if is_zero else 10
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.right() + 1, int(y), bar_rect.right() + tick_length, int(y))
            font = QtGui.QFont(label_font)
            font.setBold(is_zero)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(text_color, 1.0))
            text_rect = QtCore.QRectF(label_left, y - 12, image_width - label_left - 12, 24)
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._format_colorbar_value(value))
        painter.setFont(label_font)

        for tick in minor_ticks:
            fraction = float(tick["fraction"])
            minor_y = bar_rect.bottom() - fraction * bar_rect.height()
            painter.setPen(QtGui.QPen(tick_color, 1.0))
            painter.drawLine(bar_rect.right() + 1, int(minor_y), bar_rect.right() + 6, int(minor_y))

        self._draw_colorbar_edge_labels(
            painter,
            bar_rect,
            label_left,
            image_width,
            vmin,
            vmax,
            major_ticks,
            text_color,
            label_font,
        )

    def _build_colorbar_tick_layout(self, vmin: float, vmax: float) -> dict[str, list[dict[str, float | bool]]]:
        """Build regular major and minor 3D colorbar ticks with explicit zero."""
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return {"major": [], "minor": []}
        if vmax <= vmin:
            vmax = vmin + 1e-9

        major_step = self._select_colorbar_tick_step(vmin, vmax, target_count=6.0)
        major_ticks = self._build_colorbar_ticks_for_step(vmin, vmax, major_step)
        minor_ticks = self._build_colorbar_minor_ticks(vmin, vmax, major_step, subdivisions=2)
        return {"major": major_ticks, "minor": minor_ticks}

    @staticmethod
    def _select_colorbar_tick_step(vmin: float, vmax: float, target_count: float = 6.0) -> float:
        """Choose a rounded major-tick step for exported 3D colorbars."""
        extent = max(float(vmax) - float(vmin), 1e-9)
        raw_step = extent / max(float(target_count), 1.0)
        exponent = np.floor(np.log10(raw_step))
        base = 10.0 ** exponent
        normalized = raw_step / base
        if normalized <= 1.0:
            nice = 1.0
        elif normalized <= 2.0:
            nice = 2.0
        elif normalized <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * base)

    @staticmethod
    def _build_colorbar_ticks_for_step(vmin: float, vmax: float, step: float) -> list[dict[str, float | bool]]:
        """Build regular major colorbar ticks and force zero into view when needed."""
        if step <= 0:
            return []
        tolerance = max(1e-9, abs(step) * 1e-6)
        start_index = int(np.ceil((vmin - tolerance) / step))
        end_index = int(np.floor((vmax + tolerance) / step))
        ticks: list[dict[str, float | bool]] = []
        for index in range(start_index, end_index + 1):
            value = float(index * step)
            if value < vmin - tolerance or value > vmax + tolerance:
                continue
            ticks.append(
                {
                    "value": value,
                    "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
                    "is_zero": abs(value) <= tolerance,
                }
            )

        if vmin <= 0.0 <= vmax and not any(bool(tick["is_zero"]) for tick in ticks):
            ticks.append(
                {
                    "value": 0.0,
                    "fraction": (0.0 - vmin) / max(vmax - vmin, 1e-9),
                    "is_zero": True,
                }
            )

        if not ticks:
            midpoint = 0.5 * (vmin + vmax)
            ticks = [
                {"value": float(vmax), "fraction": 1.0, "is_zero": abs(vmax) <= tolerance},
                {"value": float(midpoint), "fraction": 0.5, "is_zero": abs(midpoint) <= tolerance},
                {"value": float(vmin), "fraction": 0.0, "is_zero": abs(vmin) <= tolerance},
            ]

        ticks.sort(key=lambda item: float(item["fraction"]))
        return ticks

    @staticmethod
    def _build_colorbar_minor_ticks(vmin: float, vmax: float, major_step: float, subdivisions: int) -> list[dict[str, float]]:
        """Build minor 3D colorbar ticks between the major regular ticks."""
        if major_step <= 0 or subdivisions < 1:
            return []
        tolerance = max(1e-9, abs(major_step) * 1e-6)
        step = major_step / float(subdivisions + 1)
        start_index = int(np.ceil((vmin - tolerance) / step))
        end_index = int(np.floor((vmax + tolerance) / step))
        minor_ticks: list[dict[str, float]] = []
        for index in range(start_index, end_index + 1):
            value = float(index * step)
            if value < vmin - tolerance or value > vmax + tolerance:
                continue
            if abs((value / major_step) - round(value / major_step)) <= 1e-6:
                continue
            minor_ticks.append(
                {
                    "value": value,
                    "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
                }
            )
        return minor_ticks

    def _draw_colorbar_edge_labels(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        label_left: int,
        image_width: int,
        vmin: float,
        vmax: float,
        major_ticks: list[dict[str, float | bool]],
        text_color: QtGui.QColor,
        label_font: QtGui.QFont,
    ) -> None:
        """Draw endpoint labels when they do not collide with regular major ticks."""
        if vmax <= vmin:
            return

        major_ys = [
            float(bar_rect.bottom() - float(tick["fraction"]) * bar_rect.height())
            for tick in major_ticks
        ]
        collision_threshold = 18.0
        edge_specs = [
            (float(vmax), float(bar_rect.top())),
            (float(vmin), float(bar_rect.bottom())),
        ]
        painter.setPen(QtGui.QPen(text_color, 1.0))
        painter.setFont(label_font)
        for value, y in edge_specs:
            if any(abs(y - major_y) < collision_threshold for major_y in major_ys):
                continue
            rounded_value = self._round_colorbar_edge_value(value, major_ticks)
            text_rect = QtCore.QRectF(label_left, y - 12, image_width - label_left - 12, 24)
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                self._format_colorbar_value(rounded_value),
            )

    @staticmethod
    def _get_colorbar_font_sizes(width: int, height: int) -> tuple[int, int]:
        """Return title and label font sizes scaled to the export dimensions."""
        scale_base = max(64, min(int(width), int(height)))
        title_size = int(np.clip(round(scale_base * 0.055), 10, 18))
        label_size = int(np.clip(round(scale_base * 0.05), 9, 16))
        return title_size, label_size

    @staticmethod
    def _round_colorbar_edge_value(
        value: float,
        major_ticks: list[dict[str, float | bool]],
    ) -> float:
        """Round an edge label to the precision implied by the major tick step."""
        major_values = sorted(
            float(tick["value"])
            for tick in major_ticks
            if np.isfinite(float(tick["value"]))
        )
        if len(major_values) < 2:
            return float(value)

        step = min(
            abs(b - a)
            for a, b in zip(major_values[:-1], major_values[1:])
            if abs(b - a) > 1e-12
        ) if any(abs(b - a) > 1e-12 for a, b in zip(major_values[:-1], major_values[1:])) else 0.0
        if step <= 0.0:
            return float(value)

        if step >= 1.0:
            return float(round(value))

        decimals = int(np.clip(np.ceil(-np.log10(step)) + 1, 0, 6))
        return float(round(value, decimals))

    @staticmethod
    def _format_colorbar_value(value: float) -> str:
        """Format one colorbar tick label compactly for export."""
        if not np.isfinite(value):
            return "nan"
        magnitude = abs(float(value))
        if magnitude >= 1e4 or (0 < magnitude < 1e-3):
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Release viewer resources when the 3D window is closed."""
        self._refinement_timer.stop()
        self._mesh_request_id += 1
        self._stop_mesh_workers()
        self.gl_widget.release_resources()
        self._geometry_cache = {
            "points": {"ref": {}, "adj": {}},
            "mesh": {"ref": {}, "adj": {}},
        }
        super().closeEvent(event)

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


def _reset_global_point_viewer(*_args) -> None:
    """Forget the cached experimental 3D viewer after it is destroyed."""
    global _global_point_viewer
    _global_point_viewer = None


def show_point_3d_viewer(
    reference_grid,
    adjusted_grid=None,
    line_points=None,
    separation=0.0,
    pixel_size_x=1.0,
    pixel_size_y=1.0,
):
    """Display the experimental point-based 3D viewer window.

    Returns:
        Point3DViewer: Shared experimental viewer instance.
    """
    global _global_point_viewer
    if _global_point_viewer is None:
        _global_point_viewer = Point3DViewer()
        _global_point_viewer.destroyed.connect(_reset_global_point_viewer)
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
    return _global_point_viewer
