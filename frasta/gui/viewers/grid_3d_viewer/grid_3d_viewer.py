"""3D visualization widget for displaying scan surface data.

This module provides interactive 3D visualization of scan surfaces with support for:
- Multiple rendering modes (surface, wireframe, mesh)
- Level-of-detail (LOD) rendering for performance
- Customizable colormaps and value ranges
- Profile line overlay and cross-section planes
- Side-by-side comparison of two scans
- Export to image files
"""

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import logging

logger = logging.getLogger(__name__)

from ..limited_gl_view import LimitedGLView
from ...widgets.surface_control_panel import ControlsPanel

# Import specialized managers
from .lod_manager import LODManager
from .colormap_manager import ColormapManager
from .surface_renderer import SurfaceRenderer
from .profile_manager import ProfileManager
from .camera_controller import CameraController


class Grid3DViewer(QtWidgets.QWidget):
    """3D viewer widget for scan surface visualization."""
    
    def __init__(self, surface_mode='surface', parent=None):
        """Initialize the 3D grid viewer widget.
        
        Sets up the user interface, control checkboxes, and 3D view for displaying grid data.
        
        Args:
            surface_mode (str, optional): The mode for rendering surfaces ('surface', 'mesh', or 'wireframe').
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("3D Grid Viewer")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)
        
        self.two_scans_mode = True
        self.show_controls = True
        
        # Initialize 3D view
        self.view = LimitedGLView(elevation_range=None)
        
        # Initialize managers
        self.lod_manager = LODManager(self.view)
        self.colormap_manager = ColormapManager()
        self.surface_renderer = SurfaceRenderer(self.view, self.lod_manager, self.colormap_manager)
        self.profile_manager = ProfileManager(self.view)
        self.camera_controller = CameraController(self.view)
        
        # Rendering modes
        self.ref_surface_mode = surface_mode
        self.adj_surface_mode = surface_mode
        self.surface_renderer.ref_surface_mode = surface_mode
        self.surface_renderer.adj_surface_mode = surface_mode
        
        # Data cache for refresh operations
        self._ref_last = None
        self._adj_last = None
        
        # Initialize UI
        self.init_controls(layout)
        self._init_busy_ui()
        layout.addWidget(self.view)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.setStretch(0, 0)  # controls_panel
        layout.setStretch(1, 1)  # view - let it grow
        
        # Connect managers to each other
        self.lod_manager.cross_plane_item = self.profile_manager.cross_plane_item
        self.lod_manager.ref_profile_line_item = self.profile_manager.ref_profile_line_item
        self.lod_manager.adj_profile_line_item = self.profile_manager.adj_profile_line_item
        
        self.view.setCameraPosition(distance=200)
        
        self._setup_shortcuts()
    
    def init_controls(self, layout):
        """Initialize and add all UI control widgets to the layout.
        
        Creates control panels for both reference and adjusted surfaces,
        including visibility toggles, rendering mode selectors, colormap
        choosers, and range controls.
        
        Args:
            layout (QLayout): Layout to add controls to.
        """
        self.controls_panel = ControlsPanel(self)
        layout.addWidget(self.controls_panel)
        
        # Create aliases for backward compatibility
        self.checkbox_ref = self.controls_panel.ref_controls.checkbox
        self.checkbox_adj = self.controls_panel.adj_controls.checkbox
        self.checkbox_line = self.controls_panel.checkbox_line
        self.checkbox_plane = self.controls_panel.checkbox_plane
        
        self.combo_mode_r = self.controls_panel.ref_controls.combo_mode
        self.combo_mode_a = self.controls_panel.adj_controls.combo_mode
        self.combo_cmap_ref = self.controls_panel.ref_controls.combo_colormap
        self.combo_cmap_adj = self.controls_panel.adj_controls.combo_colormap
        
        # Set colormap manager widgets
        self.colormap_manager.set_widgets(
            self.controls_panel.ref_controls.spin_lo,
            self.controls_panel.ref_controls.spin_hi,
            self.controls_panel.adj_controls.spin_lo,
            self.controls_panel.adj_controls.spin_hi,
            self.controls_panel.ref_controls.chk_auto,
            self.controls_panel.adj_controls.chk_auto,
            self.controls_panel.chk_link_ranges
        )
        
        # Aliases for backward compatibility
        self.chk_auto_ref = self.controls_panel.ref_controls.chk_auto
        self.chk_auto_adj = self.controls_panel.adj_controls.chk_auto
        self.spin_lo_ref = self.controls_panel.ref_controls.spin_lo
        self.spin_hi_ref = self.controls_panel.ref_controls.spin_hi
        self.spin_lo_adj = self.controls_panel.adj_controls.spin_lo
        self.spin_hi_adj = self.controls_panel.adj_controls.spin_hi
        self.chk_link_ranges = self.controls_panel.chk_link_ranges
        
        # Ensure controls panel is visible by default
        self.controls_panel.setVisible(True)
        
        self.connect_controls()
    
    def connect_controls(self):
        """Connect the control panel signals to their respective methods."""
        # Visibility toggles
        self.controls_panel.ref_controls.visibility_changed.connect(self.surface_renderer.toggle_surface_ref)
        self.controls_panel.adj_controls.visibility_changed.connect(self.surface_renderer.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.profile_manager.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.profile_manager.toggle_cross_plane)
        
        # Mode and colormap changes
        self.controls_panel.ref_controls.mode_changed.connect(self._ui_mode_changed_r)
        self.controls_panel.ref_controls.colormap_changed.connect(self._ui_cmap_ref_changed)
        self.controls_panel.adj_controls.mode_changed.connect(self._ui_mode_changed_a)
        self.controls_panel.adj_controls.colormap_changed.connect(self._ui_cmap_adj_changed)
        
        # Range controls
        self.chk_link_ranges.toggled.connect(self._ui_link_toggled)
        self.controls_panel.ref_controls.auto_range_toggled.connect(self._ui_auto_ref_toggled)
        self.controls_panel.adj_controls.auto_range_toggled.connect(self._ui_auto_adj_toggled)
        
        self.controls_panel.ref_controls.range_lo_changed.connect(lambda _: self._ui_lohi_changed('ref'))
        self.controls_panel.ref_controls.range_hi_changed.connect(lambda _: self._ui_lohi_changed('ref'))
        self.controls_panel.adj_controls.range_lo_changed.connect(lambda _: self._ui_lohi_changed('adj'))
        self.controls_panel.adj_controls.range_hi_changed.connect(lambda _: self._ui_lohi_changed('adj'))
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for rendering modes."""
        QtWidgets.QShortcut(QtGui.QKeySequence("1"), self, 
                           activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('wireframe')))
        QtWidgets.QShortcut(QtGui.QKeySequence("2"), self, 
                           activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('surface')))
        QtWidgets.QShortcut(QtGui.QKeySequence("3"), self, 
                           activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('mesh')))
    
    def _init_busy_ui(self):
        """Initialize busy indicator overlay with progress bar."""
        # Semi-transparent overlay with progress bar
        self._busy_wrap = QtWidgets.QWidget(self)
        self._busy_wrap.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._busy_wrap.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self._busy_wrap.setVisible(False)
        
        self._busy_bar = QtWidgets.QProgressBar(self._busy_wrap)
        self._busy_bar.setRange(0, 0)  # Indeterminate
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedWidth(180)
        self._busy_bar.setStyleSheet(
            "QProgressBar{background:rgba(0,0,0,120); border-radius:6px;}"
            "QProgressBar::chunk{background:rgba(0,180,90,220);} ")
        
        # Position in bottom-right corner
        lay = QtWidgets.QHBoxLayout(self._busy_wrap)
        lay.setContentsMargins(0, 0, 8, 8)
        lay.addStretch(1)
        v = QtWidgets.QVBoxLayout()
        v.addStretch(1)
        v.addWidget(self._busy_bar)
        lay.addLayout(v)
    
    def resizeEvent(self, ev):
        """Handle resize events to update busy indicator position."""
        super().resizeEvent(ev)
        if hasattr(self, "_busy_wrap"):
            self._busy_wrap.resize(self.size())
    
    def _begin_redraw(self):
        """Begin redraw operation (show busy indicator)."""
        if not hasattr(self, "_busyDepth"):
            self._busyDepth = 0
        self._busyDepth += 1
        if self._busyDepth == 1:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self._busy_wrap.setVisible(True)
    
    def _end_redraw_now(self):
        """End redraw operation (hide busy indicator)."""
        d = getattr(self, "_busyDepth", 0)
        if d <= 1:
            self._busyDepth = 0
            self._busy_wrap.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
        else:
            self._busyDepth = d - 1
    
    def _await_next_frame_then_end(self):
        """Remove WAIT cursor only when frame is on screen."""
        # Avoid multiple connections
        if getattr(self, "_awaitingSwap", False):
            return
        self._awaitingSwap = True
        
        def _done():
            self._awaitingSwap = False
            try:
                self.view.frameSwapped.disconnect(_done)
            except Exception:
                pass
            self._end_redraw_now()
        
        # Prefer real QOpenGLWidget signal
        if hasattr(self.view, "frameSwapped"):
            try:
                self.view.frameSwapped.connect(_done)
            except Exception:
                QtCore.QTimer.singleShot(0, _done)
        else:
            # Very old Qt/QGLWidget - fallback
            QtCore.QTimer.singleShot(0, _done)
        
        # Ensure there will be a repaint
        self.view.update()
    
    def remove_existing_items(self):
        """Remove all existing 3D items from the view.
        
        Cleans up surface items, profile lines, and cross-section planes,
        and destroys any LOD surface managers.
        """
        self.surface_renderer.remove_existing_items()
        self.profile_manager.remove_existing_items()
    
    def update_data(self, reference_grid, adjusted_grid=None, line_points=None, 
                   separation=0, pixel_size_x=1.0, pixel_size_y=1.0):
        """Update the 3D view with new grid data and profile lines.
        
        Removes existing items, prepares and adds new surfaces and profile lines,
        and recenters the camera.
        
        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray, optional): The adjusted grid data.
            line_points (list or np.ndarray, optional): Points for the profile line.
            separation (float, optional): Vertical separation between surfaces.
            pixel_size_x (float, optional): Physical size of pixel in X direction (micrometers).
            pixel_size_y (float, optional): Physical size of pixel in Y direction (micrometers).
        """
        self.remove_existing_items()
        
        # Detect single vs two scan mode and profile line presence
        has_adjusted = adjusted_grid is not None
        has_profile = line_points is not None
        self.two_scans_mode = has_adjusted
        
        # Update control panel visibility based on available data
        if hasattr(self, 'controls_panel'):
            self.controls_panel.update_visibility(
                has_adjusted_surface=has_adjusted,
                has_profile_line=has_profile
            )
        
        # Prepare surfaces
        xs, ys, Z_ref, xs_idx, ys_idx = self.surface_renderer.prepare_reference_surface(
            reference_grid, dx=pixel_size_x, dy=pixel_size_y)
        
        if adjusted_grid is not None:
            Z_adj = self.surface_renderer.prepare_adjusted_surface(
                adjusted_grid, ys_idx, xs_idx, separation, Z_ref)
        else:
            Z_adj = None
        
        # Cache data for refresh operations
        self._ref_last = (xs, ys, Z_ref)
        if adjusted_grid is not None:
            self._adj_last = (xs, ys, Z_adj)
        else:
            self._adj_last = None
        
        # Update colormap manager cache
        self.colormap_manager.set_data_cache(self._ref_last, self._adj_last)
        
        # Set spinboxes according to auto calculations
        if self.colormap_manager.range_ref_auto and np.any(np.isfinite(Z_ref)):
            lo, hi = self.colormap_manager.compute_auto_lo_hi(Z_ref)
            self.colormap_manager.update_range_widgets('ref', lo, hi, auto=True)
        
        if adjusted_grid is not None and self.colormap_manager.range_adj_auto and np.any(np.isfinite(Z_adj)):
            lo, hi = self.colormap_manager.compute_auto_lo_hi(Z_adj)
            self.colormap_manager.update_range_widgets('adj', lo, hi, auto=True)
        
        # Add surfaces
        self.surface_renderer.add_reference_surface(xs, ys, Z_ref, colormap=self.colormap_manager.colormap_ref)
        
        if adjusted_grid is not None:
            self.surface_renderer.add_adjusted_surface(xs, ys, Z_adj, colormap=self.colormap_manager.colormap_adj)
        
        if not np.any(np.isfinite(Z_ref)) and not np.any(np.isfinite(Z_adj)):
            return  # Nothing to display safely
        
        # Calculate Z limits and add profile/plane
        z_min, z_max = self.surface_renderer.compute_z_limits(Z_ref, Z_adj, adjusted_grid is not None)
        margin = 0.1 * (z_max - z_min)
        z_min -= margin
        z_max += margin
        
        self.profile_manager.add_profile_and_plane(
            reference_grid, adjusted_grid, line_points, separation, 
            z_min, z_max, pixel_size_x, pixel_size_y)
        
        # Update LOD manager references
        self.lod_manager.cross_plane_item = self.profile_manager.cross_plane_item
        self.lod_manager.ref_profile_line_item = self.profile_manager.ref_profile_line_item
        self.lod_manager.adj_profile_line_item = self.profile_manager.adj_profile_line_item
        
        # Center camera
        self.camera_controller.center_camera(xs, ys, Z_ref, Z_adj, line_points, pixel_size_x, pixel_size_y)
    
    def _refresh_surfaces(self):
        """Refresh surfaces with current settings."""
        self._begin_redraw()
        try:
            if self._ref_last is not None:
                xs, ys, Z = self._ref_last
                lo, hi = self.colormap_manager.get_lo_hi_for('ref', Z)
                lod = self.lod_manager.ensure_lod('ref')
                lod.set_lod_params(target_px=1.8, hysteresis=0.3)
                lod.set_data(xs, ys, Z)
                lod.update_style(self.ref_surface_mode, self.colormap_manager.colormap_ref, (0, 1, 0, 1), lo, hi)
                lod.set_visible(True)
            if self._adj_last is not None and not np.all(np.isnan(self._adj_last[2])):
                xs, ys, Z = self._adj_last
                lo, hi = self.colormap_manager.get_lo_hi_for('adj', Z)
                lod = self.lod_manager.ensure_lod('adj')
                lod.set_lod_params(target_px=1.8, hysteresis=0.3)
                lod.set_data(xs, ys, Z)
                lod.update_style(self.adj_surface_mode, self.colormap_manager.colormap_adj, (0.2, 0.3, 1, 1), lo, hi)
                lod.set_visible(True)
        finally:
            self._await_next_frame_then_end()
    
    # UI callback methods
    def _ui_mode_changed_r(self, _idx):
        """Handle reference rendering mode change."""
        mode = self.combo_mode_r.currentData()
        self.ref_surface_mode = mode
        self.surface_renderer.ref_surface_mode = mode
        self._refresh_surfaces()
    
    def _ui_mode_changed_a(self, _idx):
        """Handle adjusted rendering mode change."""
        mode = self.combo_mode_a.currentData()
        self.adj_surface_mode = mode
        self.surface_renderer.adj_surface_mode = mode
        self._refresh_surfaces()
    
    def _ui_cmap_ref_changed(self, _idx):
        """Handle reference colormap change."""
        txt = self.combo_cmap_ref.currentText()
        self.colormap_manager.colormap_ref = None if txt == "None" else txt
        self._refresh_surfaces()
    
    def _ui_cmap_adj_changed(self, _idx):
        """Handle adjusted colormap change."""
        txt = self.combo_cmap_adj.currentText()
        self.colormap_manager.colormap_adj = None if txt == "None" else txt
        self._refresh_surfaces()
    
    def _ui_link_toggled(self, on):
        """Handle range link toggle."""
        if self.colormap_manager.ui_link_toggled(on):
            self._refresh_surfaces()
    
    def _ui_auto_ref_toggled(self, on):
        """Handle auto-ref toggle."""
        self.colormap_manager._ref_last = self._ref_last
        if self.colormap_manager.ui_auto_ref_toggled(on):
            self._refresh_surfaces()
    
    def _ui_auto_adj_toggled(self, on):
        """Handle auto-adj toggle."""
        self.colormap_manager._adj_last = self._adj_last
        if self.colormap_manager.ui_auto_adj_toggled(on):
            self._refresh_surfaces()
    
    def _ui_lohi_changed(self, which):
        """Handle manual range change."""
        if self.colormap_manager.ui_lohi_changed(which):
            self._refresh_surfaces()
    
    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent):
        """Handle right-click context menu."""
        m = QtWidgets.QMenu(self)
        group = QtWidgets.QActionGroup(m)
        a_surface = m.addAction("Surface (shaded)")
        a_surface.setCheckable(True)
        a_surface.setActionGroup(group)
        a_wireframe = m.addAction("Wireframe")
        a_wireframe.setCheckable(True)
        a_wireframe.setActionGroup(group)
        a_mesh = m.addAction("Mesh")
        a_mesh.setCheckable(True)
        a_mesh.setActionGroup(group)
        
        mode = self.ref_surface_mode
        a_surface.setChecked(mode == 'surface')
        a_wireframe.setChecked(mode == 'wireframe')
        a_mesh.setChecked(mode == 'mesh')
        
        m.addSeparator()
        sub_ref = m.addMenu("Ref colormap")
        for name in ["None", "RG", "Metrology", "viridis", "plasma", "magma"]:
            act = sub_ref.addAction(name)
            act.setCheckable(True)
            act.setChecked((self.colormap_manager.colormap_ref or "None") == name)
            act.triggered.connect(lambda _, n=name: self._set_ref_cmap_from_menu(n))
        
        if self.two_scans_mode:
            sub_adj = m.addMenu("Adj colormap")
            for name in ["None", "RG", "Metrology", "viridis", "plasma", "magma"]:
                act = sub_adj.addAction(name)
                act.setCheckable(True)
                act.setChecked((self.colormap_manager.colormap_adj or "None") == name)
                act.triggered.connect(lambda _, n=name: self._set_adj_cmap_from_menu(n))
        
        chosen = m.exec_(ev.globalPos())
        if chosen is a_surface:
            self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('surface'))
        if chosen is a_wireframe:
            self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('wireframe'))
        if chosen is a_mesh:
            self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('mesh'))
    
    def _set_ref_cmap_from_menu(self, name):
        """Set reference colormap from context menu."""
        i = self.combo_cmap_ref.findText(name)
        if i >= 0:
            self.combo_cmap_ref.setCurrentIndex(i)
    
    def _set_adj_cmap_from_menu(self, name):
        """Set adjusted colormap from context menu."""
        i = self.combo_cmap_adj.findText(name)
        if i >= 0:
            self.combo_cmap_adj.setCurrentIndex(i)
    
    def set_controls_visible(self, visible):
        """Show or hide the entire control panel.
        
        This is used to completely hide/show controls (e.g., when show_controls=False).
        For hiding only adjusted surface controls, see update_data().
        
        Args:
            visible (bool): True to show, False to hide.
        """
        self.show_controls = visible
        if hasattr(self, "controls_panel"):
            self.controls_panel.setVisible(visible)
            # When showing the panel, ref_controls are always visible
            # but adj_controls and profile controls depend on data (managed by update_visibility)
            if visible:
                self.controls_panel.ref_controls.setVisible(True)
        
        if hasattr(self, "a_tools"):
            for cb in [self.a_tools]:
                cb.setVisible(visible)
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.remove_existing_items()
        self.view.repaint()
        QtWidgets.QApplication.processEvents()
        event.accept()


# Global viewer instance
_global_3d_viewer = None


def show_3d_viewer(reference_grid, adjusted_grid=None, line_points=None, separation=0, 
                  show_controls=True, pixel_size_x=1.0, pixel_size_y=1.0):
    """Display the 3D grid viewer window with the provided data.
    
    Initializes the viewer if needed, sets control visibility, updates the 3D view
    with new data, and brings the window to the front.
    
    Args:
        reference_grid (np.ndarray): The reference grid data.
        adjusted_grid (np.ndarray, optional): The adjusted grid data.
        line_points (list or np.ndarray, optional): Points for the profile line.
        separation (float, optional): Vertical separation between surfaces.
        show_controls (bool, optional): Whether to show UI controls.
        pixel_size_x (float, optional): Physical size of pixel in X direction (micrometers).
        pixel_size_y (float, optional): Physical size of pixel in Y direction (micrometers).
    """
    global _global_3d_viewer
    if _global_3d_viewer is None:
        _global_3d_viewer = Grid3DViewer()
    
    # First update data (this will set two_scans_mode and adjust control visibility)
    _global_3d_viewer.update_data(
        reference_grid=reference_grid,
        adjusted_grid=adjusted_grid,
        line_points=line_points,
        separation=separation,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y
    )
    
    # Then set overall control panel visibility (this will restore proper adj_controls state)
    _global_3d_viewer.set_controls_visible(show_controls)
    
    logger.debug("_global_3d_viewer.show()")
    _global_3d_viewer.show()
    logger.debug("_global_3d_viewer.raise_()")
    _global_3d_viewer.raise_()
    logger.debug("_global_3d_viewer.activateWindow()")
    _global_3d_viewer.activateWindow()
    logger.debug("_global_3d_viewer ready")
