"""Experimental QOpenGLWidget-based renderer for large point clouds.

This widget is intentionally decoupled from pyqtgraph.opengl. It provides a
small camera controller and GPU-backed point rendering as a foundation for
future 3D viewer work.
"""

from __future__ import annotations

import math

import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets

from .point_cloud_geometry import compute_bounds


class PointCloudGLWidget(QtWidgets.QOpenGLWidget):
    """Render one or more point clouds with a simple orbit camera."""

    frameSwapped = QtCore.pyqtSignal()
    DEFAULT_CAMERA_AZIMUTH = 90.0
    DEFAULT_CAMERA_ELEVATION = -89.0
    DEFAULT_BACKGROUND_RGBA = (0.08, 0.08, 0.10, 1.0)

    def __init__(self, parent=None):
        """Initialize the OpenGL widget and camera state."""
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setUpdateBehavior(QtWidgets.QOpenGLWidget.NoPartialUpdate)

        self._program = None
        self._mesh_program = None
        self._line_program = None
        self._plane_program = None
        self._clouds: dict[str, dict[str, object]] = {}
        self._ref_profile_positions = np.empty((0, 3), dtype=np.float32)
        self._adj_profile_positions = np.empty((0, 3), dtype=np.float32)
        self._ref_profile_buffer = None
        self._adj_profile_buffer = None
        self._reference_rect_positions = np.empty((0, 3), dtype=np.float32)
        self._reference_rect_buffer = None
        self._reference_rect_annotations = None
        self._plane_vertices = np.empty((0, 3), dtype=np.float32)
        self._plane_indices = np.empty((0, 3), dtype=np.uint32)
        self._plane_vbo = None
        self._plane_ibo = None
        self._show_profile_lines = True
        self._show_profile_plane = True
        self._plane_color = QtGui.QColor.fromRgbF(0.5, 0.5, 0.7, 0.5)
        self._show_reference_rectangle = True
        # Cursor marker (GL_LINES cross drawn by the line shader)
        self._cursor_marker_positions = np.empty((0, 3), dtype=np.float32)
        self._cursor_marker_buffer = None
        self._projection_mode = "perspective"
        self._background_rgba = self.DEFAULT_BACKGROUND_RGBA

        self._camera_center = np.zeros(3, dtype=np.float32)
        self._camera_distance = 100.0
        # Match the 2D scan orientation used in tabs: the default view starts
        # almost top-down, looking against the Z axis like the 2D image view.
        self._camera_azimuth = self.DEFAULT_CAMERA_AZIMUTH
        self._camera_elevation = self.DEFAULT_CAMERA_ELEVATION
        self._point_size = 2.0
        self._render_mode = "points"
        self._last_mouse_pos = QtCore.QPoint()
        self._resize_active = False
        self._resize_debounce_timer = QtCore.QTimer(self)
        self._resize_debounce_timer.setSingleShot(True)
        self._resize_debounce_timer.timeout.connect(self._finish_resize_interaction)

    def set_cloud_data(
        self,
        name: str,
        positions: np.ndarray,
        recenter: bool = False,
    ) -> None:
        """Replace the geometry buffers for a named point cloud.

        Args:
            name: Cloud identifier.
            positions: Point positions with shape ``(N, 3)``.
            recenter: If True, recenter the camera after the update.
        """
        entry = self._clouds.get(name)
        if entry is None:
            entry = {
                "positions": np.empty((0, 3), dtype=np.float32),
                "visible": True,
                "vbo_positions": None,
                "vbo_normals": None,
                "ibo_indices": None,
                "index_count": 0,
                "texture_id": None,
                "value_range": (0.0, 1.0),
            }
            self._clouds[name] = entry

        entry["positions"] = np.ascontiguousarray(positions, dtype=np.float32)
        entry["index_count"] = 0
        self._upload_cloud_buffers(name)
        if recenter:
            self._recenter_camera()
        self.update()

    def set_mesh_data(
        self,
        name: str,
        positions: np.ndarray,
        normals: np.ndarray,
        indices: np.ndarray,
        recenter: bool = False,
    ) -> None:
        """Replace the mesh buffers for a named cloud.

        Args:
            name: Cloud identifier.
            positions: Mesh vertex positions.
            normals: Mesh vertex normals.
            indices: Triangle index buffer.
            recenter: If True, recenter the camera after the update.
        """
        entry = self._clouds.get(name)
        if entry is None:
            self.set_cloud_data(name, positions, recenter=False)
            entry = self._clouds[name]
        entry["positions"] = np.ascontiguousarray(positions, dtype=np.float32)
        entry["normals"] = np.ascontiguousarray(normals, dtype=np.float32)
        entry["indices"] = np.ascontiguousarray(indices, dtype=np.uint32)
        entry["index_count"] = int(indices.size)
        self._upload_cloud_buffers(name)
        if recenter:
            self._recenter_camera()
        self.update()

    def set_cloud_style(
        self,
        name: str,
        colormap_lut: np.ndarray,
        value_range: tuple[float, float],
        visible: bool = True,
        hide_below_range: bool = False,
        hide_above_range: bool = False,
        z_offset: float = 0.0,
    ) -> None:
        """Update the GPU colormap texture and range for a named point cloud.

        Args:
            name: Cloud identifier.
            colormap_lut: Compact RGBA lookup table uploaded to the GPU.
            value_range: Active ``(lo, hi)`` color range.
            visible: Visibility toggle for the cloud.
            hide_below_range: If True, discard fragments below ``lo``.
            hide_above_range: If True, discard fragments above ``hi``.
            z_offset: Uniform Z shift applied in the vertex shader (µm).
        """
        entry = self._clouds.get(name)
        if entry is None:
            self.set_cloud_data(name, np.empty((0, 3), dtype=np.float32))
            entry = self._clouds[name]
        lo, hi = value_range
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-6
        entry["visible"] = bool(visible)
        entry["hide_below_range"] = bool(hide_below_range)
        entry["hide_above_range"] = bool(hide_above_range)
        entry["value_range"] = (float(lo), float(hi))
        entry["z_offset"] = float(z_offset)
        entry["texture_id"] = self._upload_or_replace_texture(
            entry.get("texture_id"),
            colormap_lut,
        )
        self.update()

    def clear_cloud(self, name: str) -> None:
        """Remove a named point cloud from the widget."""
        entry = self._clouds.pop(name, None)
        if entry is not None:
            self._delete_buffer(entry.get("vbo_positions"))
            self._delete_buffer(entry.get("vbo_normals"))
            self._delete_buffer(entry.get("ibo_indices"))
            self._delete_texture(entry.get("texture_id"))
        self.update()

    def set_cloud_visible(self, name: str, visible: bool) -> None:
        """Toggle visibility for a named point cloud."""
        if name in self._clouds:
            self._clouds[name]["visible"] = bool(visible)
            self.update()

    def set_point_size(self, point_size: float) -> None:
        """Update the rasterized point size."""
        self._point_size = max(1.0, float(point_size))
        self.update()

    def set_render_mode(self, render_mode: str) -> None:
        """Switch between point and mesh rendering."""
        self._render_mode = render_mode if render_mode in {"points", "mesh"} else "points"
        self.update()

    def set_projection_mode(self, projection_mode: str) -> None:
        """Switch between perspective and orthographic camera projection."""
        self._projection_mode = (
            projection_mode
            if projection_mode in {"perspective", "orthographic"}
            else "perspective"
        )
        self.update()

    def set_background_color(self, color: QtGui.QColor) -> None:
        """Update the non-transparent background color used by the viewer."""
        if not color.isValid():
            return
        self._background_rgba = (
            float(color.redF()),
            float(color.greenF()),
            float(color.blueF()),
            1.0,
        )
        self.update()

    def reset_background_color(self) -> None:
        """Restore the default background color used by the viewer."""
        self._background_rgba = self.DEFAULT_BACKGROUND_RGBA
        self.update()

    def get_background_color(self) -> QtGui.QColor:
        """Return the current viewer background color as a ``QColor``."""
        return QtGui.QColor.fromRgbF(*self._background_rgba[:3])

    def set_profile_lines(
        self,
        ref_positions: np.ndarray | None = None,
        adj_positions: np.ndarray | None = None,
    ) -> None:
        """Upload optional reference and adjusted profile polylines."""
        self._ref_profile_positions = (
            np.empty((0, 3), dtype=np.float32)
            if ref_positions is None
            else np.ascontiguousarray(ref_positions, dtype=np.float32)
        )
        self._adj_profile_positions = (
            np.empty((0, 3), dtype=np.float32)
            if adj_positions is None
            else np.ascontiguousarray(adj_positions, dtype=np.float32)
        )
        self._upload_profile_buffers()
        self.update()

    def set_reference_rectangle(self, positions: np.ndarray | None) -> None:
        """Upload the optional reference-grid rectangle drawn at ``Z=0``."""
        self._reference_rect_positions = (
            np.empty((0, 3), dtype=np.float32)
            if positions is None
            else np.ascontiguousarray(positions, dtype=np.float32)
        )
        self._upload_reference_rectangle_buffer()
        self.update()

    def set_reference_rectangle_annotations(self, annotations: dict[str, object] | None) -> None:
        """Store optional text annotations for the reference-grid rectangle."""
        self._reference_rect_annotations = annotations
        self.update()

    def set_reference_rectangle_visible(self, visible: bool) -> None:
        """Toggle visibility of the reference-grid rectangle overlay."""
        self._show_reference_rectangle = bool(visible)
        self.update()

    def set_profile_plane(self, vertices: np.ndarray | None, indices: np.ndarray | None = None) -> None:
        """Upload an optional cross-section plane."""
        if vertices is None or len(vertices) == 0:
            self._plane_vertices = np.empty((0, 3), dtype=np.float32)
            self._plane_indices = np.empty((0, 3), dtype=np.uint32)
        else:
            self._plane_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
            if indices is None:
                indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
            self._plane_indices = np.ascontiguousarray(indices, dtype=np.uint32)
        self._upload_plane_buffers()
        self.update()

    def set_profile_line_visible(self, visible: bool) -> None:
        """Toggle profile line visibility."""
        self._show_profile_lines = bool(visible)
        self.update()

    def set_cursor_marker(self, positions: np.ndarray | None) -> None:
        """Upload a GL_LINES cross marker for the profile hover cursor.

        Args:
            positions: Array of shape (N*2, 3) with segment endpoint pairs,
                or None to clear the marker.
        """
        if positions is None or len(positions) == 0:
            self._cursor_marker_positions = np.empty((0, 3), dtype=np.float32)
        else:
            self._cursor_marker_positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._upload_cursor_marker_buffer()
        self.update()

    def set_profile_plane_visible(self, visible: bool) -> None:
        """Toggle plane visibility."""
        self._show_profile_plane = bool(visible)
        self.update()

    def set_profile_plane_color(self, color: QtGui.QColor) -> None:
        """Update the cross-section plane color including alpha."""
        if not color.isValid():
            return
        self._plane_color = QtGui.QColor(color)
        self.update()

    def get_profile_plane_color(self) -> QtGui.QColor:
        """Return the current cross-section plane color including alpha."""
        return QtGui.QColor(self._plane_color)

    def initializeGL(self) -> None:
        """Create shader programs and configure OpenGL state."""
        GL.glClearColor(*self._background_rgba)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

        self._program = self._build_shader_program(_POINT_VERTEX_SHADER, _POINT_FRAGMENT_SHADER)
        self._mesh_program = self._build_shader_program(_MESH_VERTEX_SHADER, _MESH_FRAGMENT_SHADER)
        self._line_program = self._build_shader_program(_LINE_VERTEX_SHADER, _LINE_FRAGMENT_SHADER)
        self._plane_program = self._build_shader_program(_PLANE_VERTEX_SHADER, _PLANE_FRAGMENT_SHADER)
        for name in list(self._clouds):
            self._upload_cloud_buffers(name)
            self._upload_existing_texture(name)
        self._upload_profile_buffers()
        self._upload_reference_rectangle_buffer()
        self._upload_plane_buffers()
        self._upload_cursor_marker_buffer()

    def resizeGL(self, width: int, height: int) -> None:
        """Keep the viewport in sync with the widget size."""
        device_ratio = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        fb_width = max(1, int(round(width * device_ratio)))
        fb_height = max(1, int(round(height * device_ratio)))
        GL.glViewport(0, 0, fb_width, fb_height)

    def paintGL(self) -> None:
        """Draw visible clouds and the optional profile line."""
        if self._resize_active:
            self._render_resize_placeholder(self.width(), self.height(), clear_rgba=self._background_rgba)
        else:
            self._render_scene(self.width(), self.height(), clear_rgba=self._background_rgba)
        self._draw_reference_rectangle_annotations_on_widget()
        self.frameSwapped.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Debounce live-resize updates to reduce OpenGL driver pressure."""
        self._resize_active = True
        self._resize_debounce_timer.start(180)
        super().resizeEvent(event)

    def render_to_image(
        self,
        width: int | None = None,
        height: int | None = None,
        transparent_background: bool = True,
        axes_overlay: dict[str, object] | None = None,
    ) -> QtGui.QImage:
        """Render the current 3D scene to an image using an off-screen FBO.

        Args:
            width: Target output width in pixels. Defaults to current widget width.
            height: Target output height in pixels. Defaults to current widget height.
            transparent_background: If True, clear the off-screen buffer with
                alpha 0 instead of the viewer background color.
            axes_overlay: Optional export-only X/Y frame description rendered
                in 3D with depth testing.

        Returns:
            Rendered image in top-left origin orientation.
        """
        target_width = max(1, int(width or self.width() or 1))
        target_height = max(1, int(height or self.height() or 1))

        self.makeCurrent()
        try:
            fmt = QtGui.QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QtGui.QOpenGLFramebufferObject.CombinedDepthStencil)
            fbo = QtGui.QOpenGLFramebufferObject(target_width, target_height, fmt)
            if not fbo.isValid():
                raise RuntimeError("Failed to create OpenGL framebuffer for screenshot export.")

            clear_rgba = (
                (0.0, 0.0, 0.0, 0.0)
                if transparent_background
                else self._background_rgba
            )
            fbo.bind()
            self._render_scene(
                target_width,
                target_height,
                clear_rgba=clear_rgba,
                axes_overlay=axes_overlay,
            )
            image = fbo.toImage()
            fbo.release()
        finally:
            self.doneCurrent()

        self._draw_reference_rectangle_annotations_on_image(image)
        return image

    def release_resources(self) -> None:
        """Release GPU-side resources owned by this widget.

        The experimental 3D viewer can be opened and closed repeatedly during
        one application session. Explicit cleanup helps avoid leaving textures,
        buffers, and cached overlay objects alive in the OpenGL driver longer
        than necessary.
        """
        context = self.context()
        if context is not None and self.isValid():
            self.makeCurrent()
            try:
                for name in list(self._clouds):
                    self.clear_cloud(name)
                self._delete_buffer(self._ref_profile_buffer)
                self._delete_buffer(self._adj_profile_buffer)
                self._delete_buffer(self._reference_rect_buffer)
                self._delete_buffer(self._plane_vbo)
                self._delete_buffer(self._plane_ibo)
                self._delete_buffer(self._cursor_marker_buffer)
            finally:
                self.doneCurrent()
        else:
            self._clouds.clear()

        self._ref_profile_buffer = None
        self._adj_profile_buffer = None
        self._reference_rect_buffer = None
        self._reference_rect_annotations = None
        self._plane_vbo = None
        self._plane_ibo = None
        self._cursor_marker_buffer = None
        self._cursor_marker_positions = np.empty((0, 3), dtype=np.float32)
        self._ref_profile_positions = np.empty((0, 3), dtype=np.float32)
        self._adj_profile_positions = np.empty((0, 3), dtype=np.float32)
        self._reference_rect_positions = np.empty((0, 3), dtype=np.float32)
        self._plane_vertices = np.empty((0, 3), dtype=np.float32)
        self._plane_indices = np.empty((0, 3), dtype=np.uint32)

        for program_attr in ("_program", "_mesh_program", "_line_program", "_plane_program"):
            program = getattr(self, program_attr)
            if program is not None:
                try:
                    program.removeAllShaders()
                except Exception:
                    pass
            setattr(self, program_attr, None)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Remember the last mouse position for drag interactions."""
        self._last_mouse_pos = event.pos()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Orbit or pan the camera depending on the pressed mouse button."""
        delta = event.pos() - self._last_mouse_pos
        self._last_mouse_pos = event.pos()

        if event.buttons() & QtCore.Qt.LeftButton:
            # Match common 3D-viewer interaction: dragging left rotates the
            # scene left, so the camera azimuth changes in the opposite sign.
            self._camera_azimuth -= delta.x() * 0.5
            self._camera_elevation = float(
                np.clip(self._camera_elevation - delta.y() * 0.35, -89.0, 89.0)
            )
            self.update()
        elif event.buttons() & QtCore.Qt.RightButton:
            self._pan_camera(delta)
            self.update()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Zoom the orbit camera in or out."""
        delta_steps = event.angleDelta().y() / 120.0
        zoom_factor = math.pow(0.85, delta_steps)
        self._camera_distance = max(1e-3, self._camera_distance * zoom_factor)
        self.update()
        event.accept()

    def _draw_clouds(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render all visible clouds using the point shader."""
        if self._program is None:
            return

        self._program.bind()
        self._program.setUniformValue("u_mvp", mvp)
        self._program.setUniformValue("u_point_size", float(self._point_size))
        self._program.setUniformValue("u_colormap", 0)

        pos_loc = self._program.attributeLocation("a_position")
        for entry in self._clouds.values():
            if not entry["visible"] or len(entry["positions"]) == 0:
                continue
            vbo_positions = entry.get("vbo_positions")
            texture_id = entry.get("texture_id")
            if vbo_positions is None or texture_id is None:
                continue
            lo, hi = entry.get("value_range", (0.0, 1.0))
            self._program.setUniformValue("u_z_offset", float(entry.get("z_offset", 0.0)))
            self._program.setUniformValue("u_lo", float(lo))
            self._program.setUniformValue("u_hi", float(hi))
            self._program.setUniformValue(
                "u_hide_below_range",
                1 if entry.get("hide_below_range", False) else 0,
            )
            self._program.setUniformValue(
                "u_hide_above_range",
                1 if entry.get("hide_above_range", False) else 0,
            )
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            vbo_positions.bind()
            self._program.enableAttributeArray(pos_loc)
            self._program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
            GL.glDrawArrays(GL.GL_POINTS, 0, len(entry["positions"]))
            vbo_positions.release()
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._program.disableAttributeArray(pos_loc)
        self._program.release()

    def _draw_profile_line(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render the optional polyline overlay."""
        if self._line_program is None or not self._show_profile_lines:
            return
        if self._ref_profile_buffer is None and self._adj_profile_buffer is None:
            return

        self._line_program.bind()
        self._line_program.setUniformValue("u_mvp", mvp)
        pos_loc = self._line_program.attributeLocation("a_position")
        self._line_program.enableAttributeArray(pos_loc)
        GL.glLineWidth(2.0)

        for buffer_obj, positions, color in (
            (
                self._ref_profile_buffer,
                self._ref_profile_positions,
                (0.0, 0.4, 0.0, 1.0),
            ),
            (
                self._adj_profile_buffer,
                self._adj_profile_positions,
                (0.0, 0.0, 1.0, 1.0),
            ),
        ):
            if buffer_obj is None or len(positions) < 2:
                continue
            self._line_program.setUniformValue(
                "u_color",
                QtGui.QVector4D(*color),
            )
            buffer_obj.bind()
            self._line_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(positions))
            buffer_obj.release()

        self._line_program.disableAttributeArray(pos_loc)
        self._line_program.release()

    def _draw_reference_rectangle_annotations_on_widget(self) -> None:
        """Draw 2D labels for the reference-grid rectangle on the live widget."""
        painter = QtGui.QPainter(self)
        try:
            self._draw_reference_rectangle_annotations(painter, self.width(), self.height())
        finally:
            painter.end()

    def _draw_reference_rectangle_annotations_on_image(self, image: QtGui.QImage) -> None:
        """Draw 2D labels for the reference-grid rectangle on an exported image."""
        painter = QtGui.QPainter(image)
        try:
            self._draw_reference_rectangle_annotations(painter, image.width(), image.height())
        finally:
            painter.end()

    def _draw_reference_rectangle_annotations(
        self,
        painter: QtGui.QPainter,
        target_width: int,
        target_height: int,
    ) -> None:
        """Draw 2D labels for the reference-grid rectangle major ticks."""
        if not self._show_reference_rectangle or not self._reference_rect_annotations:
            return

        frame_world = np.asarray(self._reference_rect_annotations.get("frame_world", ()), dtype=np.float32)
        center_world = np.asarray(self._reference_rect_annotations.get("center", ()), dtype=np.float32)
        x_ticks = list(self._reference_rect_annotations.get("x_ticks", ()))
        y_ticks = list(self._reference_rect_annotations.get("y_ticks", ()))
        unit_scale = float(self._reference_rect_annotations.get("unit_scale", 1.0))
        unit_label = str(self._reference_rect_annotations.get("unit_label", "um"))
        if len(frame_world) != 4 or center_world.size != 3:
            return

        x_edge, y_edge = self._select_reference_rectangle_edges(frame_world)

        x_tick_positions = np.asarray(
            [[float(tick_value), float(frame_world[x_edge[0], 1]), 0.0] for tick_value in x_ticks],
            dtype=np.float32,
        )
        y_tick_positions = np.asarray(
            [[float(frame_world[y_edge[0], 0]), float(tick_value), 0.0] for tick_value in y_ticks],
            dtype=np.float32,
        )
        axis_annotations = (
            {
                "start": frame_world[x_edge[0]],
                "end": frame_world[x_edge[1]],
                "positions": x_tick_positions,
                "labels": [self._format_reference_label(float(tick_value) / unit_scale) for tick_value in x_ticks],
                "title": f"X [{unit_label}]",
            },
            {
                "start": frame_world[y_edge[0]],
                "end": frame_world[y_edge[1]],
                "positions": y_tick_positions,
                "labels": [self._format_reference_label(abs(float(tick_value)) / unit_scale) for tick_value in y_ticks],
                "title": f"Y [{unit_label}]",
            },
        )

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        font = QtGui.QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(185, 185, 185, 220)))

        center_projected = self.project_world_points(
            center_world.reshape(1, 3),
            max(1, target_width),
            max(1, target_height),
        )[0]
        for axis_annotation in axis_annotations:
            self._draw_reference_axis_annotation(
                painter=painter,
                axis_annotation=axis_annotation,
                center_projected=center_projected,
                target_width=target_width,
                target_height=target_height,
            )

    def _select_reference_rectangle_edges(self, frame_world: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """Choose the visually nearest X and Y edges of the rectangle."""
        frame_view = self.transform_world_points_to_view(frame_world)
        x_edges = ((0, 1), (3, 2))
        y_edges = ((0, 3), (1, 2))
        x_edge = max(
            x_edges,
            key=lambda edge: float(np.mean(frame_view[list(edge), 2])),
        )
        y_edge = max(
            y_edges,
            key=lambda edge: float(np.mean(frame_view[list(edge), 2])),
        )
        return x_edge, y_edge

    def _draw_reference_axis_annotation(
        self,
        painter: QtGui.QPainter,
        axis_annotation: dict[str, object],
        center_projected: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> None:
        """Draw one set of reference-rectangle tick labels in screen space."""
        axis_start = np.asarray(axis_annotation.get("start", ()), dtype=np.float32)
        axis_end = np.asarray(axis_annotation.get("end", ()), dtype=np.float32)
        tick_positions = np.asarray(axis_annotation.get("positions", ()), dtype=np.float32)
        tick_labels = list(axis_annotation.get("labels", ()))
        axis_title = str(axis_annotation.get("title", ""))
        if axis_start.size != 3 or axis_end.size != 3 or len(tick_positions) != len(tick_labels):
            return
        if len(tick_positions) == 0:
            return

        axis_projected = self.project_world_points(
            np.vstack((axis_start, axis_end)),
            max(1, target_width),
            max(1, target_height),
        )
        direction = np.asarray(axis_projected[1] - axis_projected[0], dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        midpoint = 0.5 * (axis_projected[0] + axis_projected[1])
        if np.linalg.norm((midpoint + normal * 12.0) - center_projected) < np.linalg.norm((midpoint - normal * 12.0) - center_projected):
            normal *= -1.0

        projected_ticks = self.project_world_points(
            tick_positions,
            max(1, target_width),
            max(1, target_height),
        )
        metrics = QtGui.QFontMetricsF(painter.font())
        title_rect = None
        if axis_title:
            title_width = max(72.0, float(metrics.horizontalAdvance(axis_title)) + 12.0)
            title_height = max(20.0, float(metrics.height()) + 4.0)
            title_anchor = midpoint + normal * 38.0
            title_rect = QtCore.QRectF(
                float(title_anchor[0] - title_width * 0.5),
                float(title_anchor[1] - title_height * 0.5),
                float(title_width),
                float(title_height),
            )

        for tick_point, tick_label in zip(projected_ticks, tick_labels):
            label_width = max(40.0, float(metrics.horizontalAdvance(str(tick_label))) + 10.0)
            label_height = max(18.0, float(metrics.height()) + 2.0)
            tick_along_axis = float(np.dot(tick_point - axis_projected[0], direction))
            edge_margin = min(label_width * 0.3, 12.0)
            clamped_along_axis = float(np.clip(tick_along_axis, edge_margin, length - edge_margin))
            label_offset = 14.0
            label_anchor = axis_projected[0] + direction * clamped_along_axis + normal * label_offset
            label_rect = QtCore.QRectF(
                float(label_anchor[0] - label_width * 0.5),
                float(label_anchor[1] - label_height * 0.5),
                float(label_width),
                float(label_height),
            )
            if title_rect is not None and label_rect.intersects(title_rect):
                continue
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, str(tick_label))

        if title_rect is not None:
            painter.drawText(title_rect, QtCore.Qt.AlignCenter, axis_title)

    @staticmethod
    def _format_reference_label(value: float) -> str:
        """Format one reference-rectangle tick label compactly."""
        if not np.isfinite(value):
            return "nan"
        magnitude = abs(float(value))
        if magnitude >= 1e4 or (0 < magnitude < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _draw_reference_rectangle(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render the reference-grid rectangle that marks the ``Z=0`` plane."""
        if self._line_program is None or not self._show_reference_rectangle:
            return
        if self._reference_rect_buffer is None:
            return
        if len(self._reference_rect_positions) < 2:
            return

        self._line_program.bind()
        self._line_program.setUniformValue("u_mvp", mvp)
        self._line_program.setUniformValue(
            "u_color",
            QtGui.QVector4D(0.45, 0.45, 0.45, 0.9),
        )
        pos_loc = self._line_program.attributeLocation("a_position")
        self._line_program.enableAttributeArray(pos_loc)
        self._reference_rect_buffer.bind()
        self._line_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
        GL.glLineWidth(1.1)
        GL.glDrawArrays(GL.GL_LINES, 0, len(self._reference_rect_positions))
        self._reference_rect_buffer.release()
        self._line_program.disableAttributeArray(pos_loc)
        self._line_program.release()

    def _draw_profile_plane(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render the optional translucent cross-section plane."""
        if self._plane_program is None or not self._show_profile_plane:
            return
        if self._plane_vbo is None or self._plane_ibo is None or len(self._plane_vertices) < 4:
            return

        self._plane_program.bind()
        self._plane_program.setUniformValue("u_mvp", mvp)
        plane_color = self._plane_color
        self._plane_program.setUniformValue(
            "u_color",
            QtGui.QVector4D(
                float(plane_color.redF()),
                float(plane_color.greenF()),
                float(plane_color.blueF()),
                float(plane_color.alphaF()),
            ),
        )
        pos_loc = self._plane_program.attributeLocation("a_position")
        self._plane_vbo.bind()
        self._plane_program.enableAttributeArray(pos_loc)
        self._plane_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
        self._plane_ibo.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, int(self._plane_indices.size), GL.GL_UNSIGNED_INT, None)
        self._plane_ibo.release()
        self._plane_vbo.release()
        self._plane_program.disableAttributeArray(pos_loc)
        self._plane_program.release()

    def _draw_cursor_marker(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render the 3D cursor cross marker at the hovered profile point."""
        if self._line_program is None:
            return
        if self._cursor_marker_buffer is None or len(self._cursor_marker_positions) < 2:
            return

        self._line_program.bind()
        self._line_program.setUniformValue("u_mvp", mvp)
        self._line_program.setUniformValue(
            "u_color",
            QtGui.QVector4D(1.0, 0.9, 0.0, 1.0),  # bright yellow
        )
        pos_loc = self._line_program.attributeLocation("a_position")
        self._line_program.enableAttributeArray(pos_loc)
        self._cursor_marker_buffer.bind()
        self._line_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
        GL.glLineWidth(3.0)
        GL.glDrawArrays(GL.GL_LINES, 0, len(self._cursor_marker_positions))
        self._cursor_marker_buffer.release()
        self._line_program.disableAttributeArray(pos_loc)
        self._line_program.release()

    def _draw_meshes(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render all visible surfaces as shaded triangle meshes."""
        if self._mesh_program is None:
            return

        self._mesh_program.bind()
        self._mesh_program.setUniformValue("u_mvp", mvp)
        self._mesh_program.setUniformValue("u_colormap", 0)
        self._mesh_program.setUniformValue("u_light_dir", QtGui.QVector3D(-0.4, 0.3, 1.0))

        pos_loc = self._mesh_program.attributeLocation("a_position")
        norm_loc = self._mesh_program.attributeLocation("a_normal")
        for entry in self._clouds.values():
            if not entry["visible"] or len(entry["positions"]) == 0 or entry.get("index_count", 0) == 0:
                continue
            vbo_positions = entry.get("vbo_positions")
            vbo_normals = entry.get("vbo_normals")
            ibo_indices = entry.get("ibo_indices")
            texture_id = entry.get("texture_id")
            if any(value is None for value in (vbo_positions, vbo_normals, ibo_indices, texture_id)):
                continue
            lo, hi = entry.get("value_range", (0.0, 1.0))
            self._mesh_program.setUniformValue("u_z_offset", float(entry.get("z_offset", 0.0)))
            self._mesh_program.setUniformValue("u_lo", float(lo))
            self._mesh_program.setUniformValue("u_hi", float(hi))
            self._mesh_program.setUniformValue(
                "u_hide_below_range",
                1 if entry.get("hide_below_range", False) else 0,
            )
            self._mesh_program.setUniformValue(
                "u_hide_above_range",
                1 if entry.get("hide_above_range", False) else 0,
            )
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            vbo_positions.bind()
            self._mesh_program.enableAttributeArray(pos_loc)
            self._mesh_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
            vbo_normals.bind()
            self._mesh_program.enableAttributeArray(norm_loc)
            self._mesh_program.setAttributeBuffer(norm_loc, GL.GL_FLOAT, 0, 3)
            ibo_indices.bind()
            GL.glDrawElements(GL.GL_TRIANGLES, int(entry["index_count"]), GL.GL_UNSIGNED_INT, None)
            ibo_indices.release()
            vbo_normals.release()
            vbo_positions.release()
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._mesh_program.disableAttributeArray(pos_loc)
        self._mesh_program.disableAttributeArray(norm_loc)
        self._mesh_program.release()

    def _build_projection_matrix(self, width: int, height: int) -> QtGui.QMatrix4x4:
        """Create a projection matrix for the given viewport size."""
        aspect_ratio = max(1e-6, float(width) / max(1.0, float(height)))
        projection = QtGui.QMatrix4x4()
        if self._projection_mode == "orthographic":
            ortho_half_height = max(1.0, self._camera_distance * 0.5)
            ortho_half_width = ortho_half_height * aspect_ratio
            projection.ortho(
                -ortho_half_width,
                ortho_half_width,
                -ortho_half_height,
                ortho_half_height,
                0.1,
                max(10.0, self._camera_distance * 20.0),
            )
        else:
            projection.perspective(45.0, aspect_ratio, 0.1, max(10.0, self._camera_distance * 20.0))
        return projection

    def _build_view_matrix(self) -> QtGui.QMatrix4x4:
        """Create the orbit-camera view matrix."""
        azimuth = math.radians(self._camera_azimuth)
        elevation = math.radians(self._camera_elevation)
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        eye = self._camera_center - direction * self._camera_distance
        view = QtGui.QMatrix4x4()
        view.lookAt(
            QtGui.QVector3D(*[float(value) for value in eye]),
            QtGui.QVector3D(*[float(value) for value in self._camera_center]),
            QtGui.QVector3D(0.0, 0.0, 1.0),
        )
        return view

    def _pan_camera(self, delta: QtCore.QPoint) -> None:
        """Pan the camera target in screen space."""
        azimuth = math.radians(self._camera_azimuth)
        elevation = math.radians(self._camera_elevation)
        forward = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        right_norm = np.linalg.norm(right)
        if right_norm > 0:
            right /= right_norm
        up = np.cross(right, forward)
        scale = self._camera_distance * 0.0025
        self._camera_center -= right * (delta.x() * scale)
        self._camera_center += up * (delta.y() * scale)

    def _recenter_camera(self) -> None:
        """Center the camera on the current cloud bounds."""
        point_sets = [entry["positions"] for entry in self._clouds.values()]
        if len(self._ref_profile_positions) > 0:
            point_sets.append(self._ref_profile_positions)
        if len(self._adj_profile_positions) > 0:
            point_sets.append(self._adj_profile_positions)
        mins, maxs = compute_bounds(*point_sets)
        self._camera_center = ((mins + maxs) * 0.5).astype(np.float32, copy=False)
        span = np.max(maxs - mins)
        self._camera_distance = max(10.0, float(span) * 1.8)

    def fit_camera_to_scene(self, reset_orientation: bool = False) -> None:
        """Recenter the camera using the current scene bounds.

        Args:
            reset_orientation: If True, restore the default azimuth and
                elevation before updating the view.
        """
        self._recenter_camera()
        if reset_orientation:
            self._camera_azimuth = self.DEFAULT_CAMERA_AZIMUTH
            self._camera_elevation = self.DEFAULT_CAMERA_ELEVATION
        self.update()

    def get_scene_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the current point-cloud bounds without overlay geometry.

        Returns:
            ``(mins, maxs)`` arrays for the visible scene geometry, or ``None``
            when no cloud positions are available yet.
        """
        point_sets = [
            entry["positions"]
            for entry in self._clouds.values()
            if len(entry.get("positions", ())) > 0
        ]
        if not point_sets:
            return None
        return compute_bounds(*point_sets)

    def project_world_points(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """Project 3D world points into 2D image coordinates.

        Args:
            points: World-space positions with shape ``(N, 3)``.
            width: Target image width in pixels.
            height: Target image height in pixels.

        Returns:
            Array of projected 2D coordinates with shape ``(N, 2)``.
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        projection = self._build_projection_matrix(int(width), int(height))
        view = self._build_view_matrix()
        mvp = projection * view

        projected = np.empty((len(points), 2), dtype=np.float32)
        for idx, point in enumerate(points):
            clip = mvp * QtGui.QVector4D(float(point[0]), float(point[1]), float(point[2]), 1.0)
            w = clip.w() if abs(clip.w()) > 1e-9 else 1e-9
            ndc_x = clip.x() / w
            ndc_y = clip.y() / w
            projected[idx, 0] = float((ndc_x * 0.5 + 0.5) * width)
            projected[idx, 1] = float((1.0 - (ndc_y * 0.5 + 0.5)) * height)
        return projected

    def transform_world_points_to_view(self, points: np.ndarray) -> np.ndarray:
        """Transform world-space positions into camera view space."""
        points = np.ascontiguousarray(points, dtype=np.float32)
        view = self._build_view_matrix()
        transformed = np.empty((len(points), 3), dtype=np.float32)
        for idx, point in enumerate(points):
            view_point = view * QtGui.QVector4D(float(point[0]), float(point[1]), float(point[2]), 1.0)
            transformed[idx, 0] = float(view_point.x())
            transformed[idx, 1] = float(view_point.y())
            transformed[idx, 2] = float(view_point.z())
        return transformed

    def _should_use_point_fallback(self) -> bool:
        """Use point fallback until all visible meshes are ready."""
        visible_entries = [entry for entry in self._clouds.values() if entry.get("visible")]
        if not visible_entries:
            return True
        return not all(entry.get("index_count", 0) > 0 for entry in visible_entries)

    def _upload_cloud_buffers(self, name: str) -> None:
        """Upload cloud arrays to GPU buffers when a context is available."""
        if not self.isValid() or name not in self._clouds:
            return
        entry = self._clouds[name]
        entry["vbo_positions"] = self._create_or_replace_buffer(
            entry.get("vbo_positions"),
            entry["positions"],
        )
        if "normals" in entry:
            entry["vbo_normals"] = self._create_or_replace_buffer(
                entry.get("vbo_normals"),
                entry["normals"],
            )
        if "indices" in entry:
            entry["ibo_indices"] = self._create_or_replace_buffer(
                entry.get("ibo_indices"),
                entry["indices"],
                buffer_type=QtGui.QOpenGLBuffer.IndexBuffer,
            )

    def _upload_existing_texture(self, name: str) -> None:
        """Upload an already stored colormap texture after context creation."""
        if not self.isValid() or name not in self._clouds:
            return
        entry = self._clouds[name]
        texture_id = entry.get("texture_id")
        if isinstance(texture_id, np.ndarray):
            entry["texture_id"] = self._upload_or_replace_texture(None, texture_id)

    def _upload_profile_buffers(self) -> None:
        """Upload the optional profile line buffers."""
        if not self.isValid():
            return
        self._ref_profile_buffer = self._create_or_replace_buffer(
            self._ref_profile_buffer,
            self._ref_profile_positions,
        )
        self._adj_profile_buffer = self._create_or_replace_buffer(
            self._adj_profile_buffer,
            self._adj_profile_positions,
        )

    def _upload_reference_rectangle_buffer(self) -> None:
        """Upload the optional reference-grid rectangle buffer."""
        if not self.isValid():
            return
        self._reference_rect_buffer = self._create_or_replace_buffer(
            self._reference_rect_buffer,
            self._reference_rect_positions,
        )

    def _upload_cursor_marker_buffer(self) -> None:
        """Upload the cursor cross marker buffer."""
        if not self.isValid():
            return
        self._cursor_marker_buffer = self._create_or_replace_buffer(
            self._cursor_marker_buffer,
            self._cursor_marker_positions,
        )

    def _upload_plane_buffers(self) -> None:
        """Upload the optional plane buffers."""
        if not self.isValid():
            return
        self._plane_vbo = self._create_or_replace_buffer(
            self._plane_vbo,
            self._plane_vertices,
        )
        self._plane_ibo = self._create_or_replace_buffer(
            self._plane_ibo,
            self._plane_indices,
            buffer_type=QtGui.QOpenGLBuffer.IndexBuffer,
        )

    def _create_or_replace_buffer(self, buffer_obj, data: np.ndarray, buffer_type=QtGui.QOpenGLBuffer.VertexBuffer):
        """Create or replace a QOpenGLBuffer with contiguous float data."""
        if buffer_obj is None:
            buffer_obj = QtGui.QOpenGLBuffer(buffer_type)
            buffer_obj.create()
        if not buffer_obj.isCreated():
            buffer_obj.create()

        if buffer_type == QtGui.QOpenGLBuffer.IndexBuffer:
            contiguous = np.ascontiguousarray(data, dtype=np.uint32)
        else:
            contiguous = np.ascontiguousarray(data, dtype=np.float32)
        buffer_obj.bind()
        buffer_obj.allocate(contiguous.tobytes(), contiguous.nbytes)
        buffer_obj.release()
        return buffer_obj

    def _delete_buffer(self, buffer_obj) -> None:
        """Destroy a QOpenGLBuffer if it exists."""
        if buffer_obj is not None:
            try:
                buffer_obj.destroy()
            except Exception:
                pass

    def _upload_or_replace_texture(self, texture_id, lut: np.ndarray):
        """Create or update a 2D LUT texture for GPU color mapping."""
        lut_rgba = np.ascontiguousarray(lut, dtype=np.uint8)
        if not self.isValid():
            return lut_rgba

        if texture_id is None or isinstance(texture_id, np.ndarray):
            texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            lut_rgba.shape[0],
            1,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            lut_rgba,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id

    def _delete_texture(self, texture_id) -> None:
        """Delete an OpenGL texture if it exists."""
        if texture_id is None or isinstance(texture_id, np.ndarray):
            return
        try:
            GL.glDeleteTextures([int(texture_id)])
        except Exception:
            pass

    def _build_shader_program(self, vertex_source: str, fragment_source: str):
        """Compile and link a shader program."""
        program = QtGui.QOpenGLShaderProgram(self)
        program.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex, vertex_source)
        program.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment, fragment_source)
        if not program.link():
            raise RuntimeError(program.log())
        return program

    def _render_scene(
        self,
        width: int,
        height: int,
        clear_rgba: tuple[float, float, float, float],
        axes_overlay: dict[str, object] | None = None,
    ) -> None:
        """Render the full scene into the currently bound framebuffer.

        Args:
            width: Active framebuffer width in pixels.
            height: Active framebuffer height in pixels.
            clear_rgba: Background clear color including alpha.
            axes_overlay: Optional export-only X/Y frame description.
        """
        # QPainter-based 2D overlays can modify GL state between frames, so the
        # core depth/blend configuration is restored explicitly for each render.
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glViewport(0, 0, max(1, int(width)), max(1, int(height)))
        GL.glClearColor(*clear_rgba)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        projection = self._build_projection_matrix(width, height)
        view = self._build_view_matrix()
        mvp = projection * view

        if self._render_mode == "mesh" and not self._should_use_point_fallback():
            self._draw_meshes(mvp)
        else:
            self._draw_clouds(mvp)
        if axes_overlay:
            self._draw_export_axes(mvp, axes_overlay)
        self._draw_reference_rectangle(mvp)
        self._draw_profile_plane(mvp)
        self._draw_profile_line(mvp)
        self._draw_cursor_marker(mvp)

    def _render_resize_placeholder(
        self,
        width: int,
        height: int,
        clear_rgba: tuple[float, float, float, float],
    ) -> None:
        """Render only a clean background while the user is actively resizing.

        Continuous resize events can trigger a large number of full-scene
        redraws and FBO reallocations inside QOpenGLWidget. On some Windows
        drivers this can leave the OpenGL context in a corrupted visual state.
        A cheap placeholder render during the resize interaction avoids that
        pressure and restores the real scene once resizing stops briefly.
        """
        GL.glViewport(0, 0, max(1, int(width)), max(1, int(height)))
        GL.glClearColor(*clear_rgba)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def _finish_resize_interaction(self) -> None:
        """Resume normal scene rendering after resize input settles."""
        self._resize_active = False
        self.update()

    def _draw_export_axes(self, mvp: QtGui.QMatrix4x4, axes_overlay: dict[str, object]) -> None:
        """Draw an export-only X/Y frame directly in the 3D scene."""
        if self._line_program is None:
            return

        positions = np.asarray(axes_overlay.get("positions", ()), dtype=np.float32)
        if positions.size == 0:
            return

        color = axes_overlay.get("color", (0.35, 0.35, 0.35, 0.85))
        buffer_obj = self._create_or_replace_buffer(None, positions)
        try:
            self._line_program.bind()
            self._line_program.setUniformValue("u_mvp", mvp)
            self._line_program.setUniformValue("u_color", QtGui.QVector4D(*[float(v) for v in color]))
            pos_loc = self._line_program.attributeLocation("a_position")
            self._line_program.enableAttributeArray(pos_loc)
            buffer_obj.bind()
            self._line_program.setAttributeBuffer(pos_loc, GL.GL_FLOAT, 0, 3)
            GL.glLineWidth(1.2)
            GL.glDrawArrays(GL.GL_LINES, 0, len(positions))
            buffer_obj.release()
            self._line_program.disableAttributeArray(pos_loc)
            self._line_program.release()
        finally:
            self._delete_buffer(buffer_obj)

_POINT_VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
uniform mat4 u_mvp;
uniform float u_point_size;
uniform float u_lo;
uniform float u_hi;
uniform float u_z_offset;
varying float v_t;
varying float v_z;

void main() {
    float z_actual = a_position.z + u_z_offset;
    gl_Position = u_mvp * vec4(a_position.xy, z_actual, 1.0);
    gl_PointSize = u_point_size;
    v_t = clamp((z_actual - u_lo) / max(u_hi - u_lo, 1e-6), 0.0, 1.0);
    v_z = z_actual;
}
"""


_POINT_FRAGMENT_SHADER = """
#version 120
uniform sampler2D u_colormap;
uniform float u_lo;
uniform float u_hi;
uniform int u_hide_below_range;
uniform int u_hide_above_range;
varying float v_t;
varying float v_z;

void main() {
    if (u_hide_below_range != 0 && v_z < u_lo) {
        discard;
    }
    if (u_hide_above_range != 0 && v_z > u_hi) {
        discard;
    }
    gl_FragColor = texture2D(u_colormap, vec2(v_t, 0.5));
}
"""


_LINE_VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""


_LINE_FRAGMENT_SHADER = """
#version 120
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""


_PLANE_VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""


_PLANE_FRAGMENT_SHADER = """
#version 120
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""


_MESH_VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
attribute vec3 a_normal;
uniform mat4 u_mvp;
uniform float u_lo;
uniform float u_hi;
uniform float u_z_offset;
varying float v_t;
varying vec3 v_normal;
varying float v_z;

void main() {
    float z_actual = a_position.z + u_z_offset;
    gl_Position = u_mvp * vec4(a_position.xy, z_actual, 1.0);
    v_t = clamp((z_actual - u_lo) / max(u_hi - u_lo, 1e-6), 0.0, 1.0);
    v_normal = normalize(a_normal);
    v_z = z_actual;
}
"""


_MESH_FRAGMENT_SHADER = """
#version 120
uniform sampler2D u_colormap;
uniform vec3 u_light_dir;
uniform float u_lo;
uniform float u_hi;
uniform int u_hide_below_range;
uniform int u_hide_above_range;
varying float v_t;
varying vec3 v_normal;
varying float v_z;

void main() {
    if (u_hide_below_range != 0 && v_z < u_lo) {
        discard;
    }
    if (u_hide_above_range != 0 && v_z > u_hi) {
        discard;
    }
    vec4 base = texture2D(u_colormap, vec2(v_t, 0.5));
    float diffuse = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
    float lighting = 0.28 + 0.72 * diffuse;
    gl_FragColor = vec4(base.rgb * lighting, base.a);
}
"""
