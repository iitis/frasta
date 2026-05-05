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
        self._plane_vertices = np.empty((0, 3), dtype=np.float32)
        self._plane_indices = np.empty((0, 3), dtype=np.uint32)
        self._plane_vbo = None
        self._plane_ibo = None
        self._show_profile_lines = True
        self._show_profile_plane = True

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
    ) -> None:
        """Update the GPU colormap texture and range for a named point cloud.

        Args:
            name: Cloud identifier.
            colormap_lut: Compact RGBA lookup table uploaded to the GPU.
            value_range: Active ``(lo, hi)`` color range.
            visible: Visibility toggle for the cloud.
            hide_below_range: If True, discard fragments below ``lo``.
            hide_above_range: If True, discard fragments above ``hi``.
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

    def set_profile_plane_visible(self, visible: bool) -> None:
        """Toggle plane visibility."""
        self._show_profile_plane = bool(visible)
        self.update()

    def initializeGL(self) -> None:
        """Create shader programs and configure OpenGL state."""
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
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
        self._upload_plane_buffers()

    def resizeGL(self, width: int, height: int) -> None:
        """Keep the viewport in sync with the widget size."""
        device_ratio = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        fb_width = max(1, int(round(width * device_ratio)))
        fb_height = max(1, int(round(height * device_ratio)))
        GL.glViewport(0, 0, fb_width, fb_height)

    def paintGL(self) -> None:
        """Draw visible clouds and the optional profile line."""
        if self._resize_active:
            self._render_resize_placeholder(self.width(), self.height(), clear_rgba=(0.08, 0.08, 0.10, 1.0))
        else:
            self._render_scene(self.width(), self.height(), clear_rgba=(0.08, 0.08, 0.10, 1.0))
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
                else (0.08, 0.08, 0.10, 1.0)
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
                self._delete_buffer(self._plane_vbo)
                self._delete_buffer(self._plane_ibo)
            finally:
                self.doneCurrent()
        else:
            self._clouds.clear()

        self._ref_profile_buffer = None
        self._adj_profile_buffer = None
        self._plane_vbo = None
        self._plane_ibo = None
        self._ref_profile_positions = np.empty((0, 3), dtype=np.float32)
        self._adj_profile_positions = np.empty((0, 3), dtype=np.float32)
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
            self._camera_azimuth += delta.x() * 0.5
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

    def _draw_profile_plane(self, mvp: QtGui.QMatrix4x4) -> None:
        """Render the optional translucent cross-section plane."""
        if self._plane_program is None or not self._show_profile_plane:
            return
        if self._plane_vbo is None or self._plane_ibo is None or len(self._plane_vertices) < 4:
            return

        self._plane_program.bind()
        self._plane_program.setUniformValue("u_mvp", mvp)
        self._plane_program.setUniformValue("u_color", QtGui.QVector4D(0.5, 0.5, 0.7, 0.5)) # RGBA with alpha for translucency
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
        """Create a perspective projection matrix for the given viewport size."""
        aspect_ratio = max(1.0, width / max(1.0, float(height)))
        projection = QtGui.QMatrix4x4()
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
        self._draw_profile_plane(mvp)
        self._draw_profile_line(mvp)

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
varying float v_t;
varying float v_z;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = u_point_size;
    v_t = clamp((a_position.z - u_lo) / max(u_hi - u_lo, 1e-6), 0.0, 1.0);
    v_z = a_position.z;
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
varying float v_t;
varying vec3 v_normal;
varying float v_z;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_t = clamp((a_position.z - u_lo) / max(u_hi - u_lo, 1e-6), 0.0, 1.0);
    v_normal = normalize(a_normal);
    v_z = a_position.z;
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
