from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt  # jeśli chcesz użyć colormap matplotlib

import logging
logger = logging.getLogger(__name__)

from .limitedGLView import LimitedGLView
from .lodSurface import LODSurface

class Grid3DViewer(QtWidgets.QWidget):
    def __init__(self, surface_mode='surface', parent=None):
        """Initializes the 3D grid viewer widget.

        Sets up the user interface, control checkboxes, and 3D view for displaying grid data.
        
        Args:
            ref_surface_mode (str, optional): The mode for rendering the reference surface ('surface', 'mesh', or 'wireframe').
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("3D Grid Viewer")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        self.two_scans_mode = True

        self.ref_surface_mode = surface_mode
        self.adj_surface_mode = surface_mode
        self.init_controls(layout)
        self.connect_controls()
        # self.view = gl.GLViewWidget()
        self.view = LimitedGLView(elevation_range=None)

        self._init_busy_ui()

        layout.addWidget(self.view)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.setStretch(0, 0)  # controls_panel
        layout.setStretch(1, 1)  # view - niech rośnie

        self.view.setCameraPosition(distance=200)
        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None
        self.show_controls = True  # domyślnie
        self.colormap_ref = 'RG'
        self.colormap_adj = 'RG'

        # NEW: ustawienia zakresów
        self.range_linked = False
        self.range_ref_auto = True
        self.range_adj_auto = True
        self.range_ref = (None, None)  # (lo, hi) gdy auto=False
        self.range_adj = (None, None)

        self._ref_last = None
        self._adj_last = None

        # --- LOD settings ---
        self.use_lod = True
        self.lod_steps = (1, 2, 4, 8, 16, 32)  # można skrócić
        self.lod_ref = None
        self.lod_adj = None

        self.lod_target_px = 1.8
        self.lod_hysteresis = 0.3
        self.lod_thresholds = None      # albo słownik progów {step: (px_lo, px_hi)}
        self.lod_base_cell = None       # zwykle None -> auto z xs/ys

        # timer do auto-przełączania LOD
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.timeout.connect(self._update_lod_tick)
        self._lod_timer.start(33)  # ~30 FPS

        self._lod = {'ref': None, 'adj': None}


        self._setup_shortcuts()

    def _ensure_lod(self, which, steps=(1,2,4,8,16)):
        key = 'ref' if which == 'ref' else 'adj'
        if self._lod[key] is None:
            shader = None  # albo: shader = self._make_headlight_shader()
            s = steps or self.lod_steps
            lod = LODSurface(self.view, steps=s, shader=shader)
            # TU ustaw politykę:
            lod.set_lod_params(
                target_px=self.lod_target_px,
                hysteresis=self.lod_hysteresis,
                thresholds=self.lod_thresholds,
                base_cell=self.lod_base_cell
            )
            self._lod[key] = lod
        return self._lod[key]

    def _update_lod_tick(self):
        if self.lod_ref: self.lod_ref.update_visible()
        if self.lod_adj: self.lod_adj.update_visible()

    def create_a_tools(self):
        self.a_tools = QtWidgets.QWidget(self)

        mode_bar_a = QtWidgets.QHBoxLayout(self.a_tools)
        mode_bar_a.setContentsMargins(0,0,0,0)
        mode_bar_a.setSpacing(6)

        self.checkbox_adj = QtWidgets.QCheckBox("Adj surface:")
        self.checkbox_adj.setChecked(True)
        mode_bar_a.addWidget(self.checkbox_adj)

        # mode_bar_a.addWidget(QtWidgets.QLabel("Adj surface: "))
        mode_bar_a.addSpacing(12)

        self.combo_mode_a = QtWidgets.QComboBox()
        self.combo_mode_a.addItem("Surface (shaded)", userData='surface')
        self.combo_mode_a.addItem("Wireframe",        userData='wireframe')
        self.combo_mode_a.addItem("Mesh",             userData='mesh')
        # ustaw wartość startową wg konstruktora
        idx = self.combo_mode_a.findData(self.adj_surface_mode)
        if idx >= 0: self.combo_mode_a.setCurrentIndex(idx)

        self.combo_cmap_adj = QtWidgets.QComboBox()
        self.combo_cmap_adj.addItems(["None", "RG", "B&W", "viridis", "plasma", "magma"])
        self.combo_cmap_adj.setCurrentText('RG')

        mode_bar_a.addWidget(QtWidgets.QLabel("mode:"))
        mode_bar_a.addWidget(self.combo_mode_a)
        mode_bar_a.addSpacing(12)
        mode_bar_a.addWidget(QtWidgets.QLabel("colormap:"))
        mode_bar_a.addWidget(self.combo_cmap_adj)
        #mode_bar_a.addStretch(1)

        self.chk_auto_adj = QtWidgets.QCheckBox("Auto")
        self.chk_auto_adj.setChecked(True)
        self.spin_lo_adj = QtWidgets.QDoubleSpinBox(); self.spin_hi_adj = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_lo_adj, self.spin_hi_adj):
            sp.setDecimals(6); sp.setRange(-1e12, 1e12); sp.setSingleStep(0.1); sp.setEnabled(False)

        mode_bar_a.addWidget(QtWidgets.QLabel("Adj lo/hi:"))
        mode_bar_a.addWidget(self.spin_lo_adj)
        mode_bar_a.addWidget(self.spin_hi_adj)
        mode_bar_a.addWidget(self.chk_auto_adj)
        mode_bar_a.addStretch(1)

        self.chk_link_ranges = QtWidgets.QCheckBox("Link ranges")
        self.chk_link_ranges.setChecked(False)
        mode_bar_a.addWidget(self.chk_link_ranges)

    def create_r_tools(self):
        self.r_tools = QtWidgets.QWidget(self)

        mode_bar_r = QtWidgets.QHBoxLayout(self.r_tools)
        mode_bar_r.setContentsMargins(0,0,0,0)
        mode_bar_r.setSpacing(6)

        self.checkbox_ref = QtWidgets.QCheckBox("Ref surface:")
        self.checkbox_ref.setChecked(True)
        mode_bar_r.addWidget(self.checkbox_ref)

        # mode_bar_r.addWidget(QtWidgets.QLabel("Ref surface: "))
        mode_bar_r.addSpacing(12)

        self.combo_mode_r = QtWidgets.QComboBox()
        self.combo_mode_r.addItem("Surface (shaded)", userData='surface')
        self.combo_mode_r.addItem("Wireframe",        userData='wireframe')
        self.combo_mode_r.addItem("Mesh",             userData='mesh')
        # ustaw wartość startową wg konstruktora
        idx = self.combo_mode_r.findData(self.ref_surface_mode)
        if idx >= 0: self.combo_mode_r.setCurrentIndex(idx)

        self.combo_cmap_ref = QtWidgets.QComboBox()
        self.combo_cmap_ref.addItems(["None", "RG", "B&W", "viridis", "plasma", "magma"])
        self.combo_cmap_ref.setCurrentText('RG')

        mode_bar_r.addWidget(QtWidgets.QLabel("mode:"))
        mode_bar_r.addWidget(self.combo_mode_r)
        mode_bar_r.addSpacing(12)
        mode_bar_r.addWidget(QtWidgets.QLabel("colormap:"))
        mode_bar_r.addWidget(self.combo_cmap_ref)
        #mode_bar_r.addStretch(1)

        self.chk_auto_ref = QtWidgets.QCheckBox("Auto")
        self.chk_auto_ref.setChecked(True)
        self.spin_lo_ref = QtWidgets.QDoubleSpinBox(); self.spin_hi_ref = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_lo_ref, self.spin_hi_ref):
            sp.setDecimals(6); sp.setRange(-1e12, 1e12); sp.setSingleStep(0.1); sp.setEnabled(False)

        mode_bar_r.addWidget(QtWidgets.QLabel("Ref lo/hi:"))
        mode_bar_r.addWidget(self.spin_lo_ref)
        mode_bar_r.addWidget(self.spin_hi_ref)
        mode_bar_r.addWidget(self.chk_auto_ref)
        mode_bar_r.addStretch(1)
    
    def init_controls(self, layout):
        """Initializes the control checkboxes for toggling 3D view elements.

        Adds checkboxes for reference surface, adjusted surface, profile line, and section plane to the layout.

        Args:
            layout (QVBoxLayout): The layout to which the controls are added.
        """

        self.controls_panel = QtWidgets.QWidget(self)
        self.controls_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)

        panel = QtWidgets.QVBoxLayout(self.controls_panel)
        panel.setContentsMargins(0,0,0,0)
        panel.setSpacing(6)

        self.create_r_tools()
        for w in (self.r_tools,):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            w.setMaximumHeight(w.sizeHint().height())

        self.create_a_tools()
        for w in (self.a_tools,):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            w.setMaximumHeight(w.sizeHint().height())

        panel.addWidget(self.r_tools)
        panel.addWidget(self.a_tools)

        ctrl_layout = QtWidgets.QHBoxLayout()
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)
        for cb in [self.checkbox_line, self.checkbox_plane]:
            ctrl_layout.addWidget(cb)

        panel.addLayout(ctrl_layout)

        layout.addWidget(self.controls_panel)

    def connect_controls(self):
        """Connects the control checkboxes to their respective toggle methods.

        Sets up signal-slot connections so that toggling each checkbox shows or hides the corresponding 3D view element.
        """
        self.checkbox_ref.stateChanged.connect(self.toggle_surface_ref)
        self.checkbox_adj.stateChanged.connect(self.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.toggle_cross_plane)

        self.combo_mode_r.currentIndexChanged.connect(self._ui_mode_changed_r)
        self.combo_cmap_ref.currentIndexChanged.connect(self._ui_cmap_ref_changed)

        self.combo_mode_a.currentIndexChanged.connect(self._ui_mode_changed_a)
        self.combo_cmap_adj.currentIndexChanged.connect(self._ui_cmap_adj_changed)

        self.chk_link_ranges.toggled.connect(self._ui_link_toggled)
        self.chk_auto_ref.toggled.connect(self._ui_auto_ref_toggled)
        self.chk_auto_adj.toggled.connect(self._ui_auto_adj_toggled)

        self.spin_lo_ref.valueChanged.connect(lambda _: self._ui_lohi_changed('ref'))
        self.spin_hi_ref.valueChanged.connect(lambda _: self._ui_lohi_changed('ref'))
        self.spin_lo_adj.valueChanged.connect(lambda _: self._ui_lohi_changed('adj'))
        self.spin_hi_adj.valueChanged.connect(lambda _: self._ui_lohi_changed('adj'))


    def _compute_auto_lo_hi(self, Z, p=(2, 98)):
        # robustne percentyle, fallback na min/max
        lo, hi = np.nanpercentile(Z, p)
        if not np.isfinite(hi - lo) or hi <= lo:
            lo, hi = np.nanmin(Z), np.nanmax(Z)
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1e-6
        return float(lo), float(hi)

    def _block(self, *widgets, block=True):
        for w in widgets: w.blockSignals(block)

    def _update_range_widgets(self, which, lo, hi, auto):
        if which == 'ref':
            self._block(self.spin_lo_ref, self.spin_hi_ref, block=True)
            self.spin_lo_ref.setValue(lo); self.spin_hi_ref.setValue(hi)
            self._block(self.spin_lo_ref, self.spin_hi_ref, block=False)
            self.chk_auto_ref.setChecked(auto)
            self.spin_lo_ref.setEnabled(not auto); self.spin_hi_ref.setEnabled(not auto)
        else:
            self._block(self.spin_lo_adj, self.spin_hi_adj, block=True)
            self.spin_lo_adj.setValue(lo); self.spin_hi_adj.setValue(hi)
            self._block(self.spin_lo_adj, self.spin_hi_adj, block=False)
            self.chk_auto_adj.setChecked(auto)
            self.spin_lo_adj.setEnabled(not auto); self.spin_hi_adj.setEnabled(not auto)

        # „Link ranges” dezaktywuje pola Adj (podążają za Ref)
        linked = self.chk_link_ranges.isChecked()
        self.spin_lo_adj.setEnabled(not (linked or self.chk_auto_adj.isChecked()))
        self.spin_hi_adj.setEnabled(not (linked or self.chk_auto_adj.isChecked()))

    def _get_lo_hi_for(self, which, Z):
        # zwraca (lo, hi) biorąc pod uwagę link/auto/manual
        if which == 'adj' and self.range_linked:
            which = 'ref'   # użyj ref-owych ustawień

        auto = (self.range_ref_auto if which == 'ref' else self.range_adj_auto)
        if auto:
            return self._compute_auto_lo_hi(Z)
        rng = (self.range_ref if which == 'ref' else self.range_adj)
        lo, hi = rng
        if lo is None or hi is None or not np.isfinite(hi - lo) or hi <= lo:
            return self._compute_auto_lo_hi(Z)
        return float(lo), float(hi)

    def _init_busy_ui(self):
        # półprzezroczysty overlay z paskiem
        self._busy_wrap = QtWidgets.QWidget(self)
        self._busy_wrap.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._busy_wrap.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self._busy_wrap.setVisible(False)

        self._busy_bar = QtWidgets.QProgressBar(self._busy_wrap)
        self._busy_bar.setRange(0, 0)              # indeterminate
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedWidth(180)
        self._busy_bar.setStyleSheet(
            "QProgressBar{background:rgba(0,0,0,120); border-radius:6px;}"
            "QProgressBar::chunk{background:rgba(0,180,90,220);} ")

        # pozycjonowanie w prawym-dolnym rogu
        lay = QtWidgets.QHBoxLayout(self._busy_wrap)
        lay.setContentsMargins(0,0,8,8)
        lay.addStretch(1)
        v = QtWidgets.QVBoxLayout(); v.addStretch(1); v.addWidget(self._busy_bar)
        lay.addLayout(v)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "_busy_wrap"):
            self._busy_wrap.resize(self.size())


    def _begin_redraw(self):
        # licznik zagnieżdżeń; można wołać wielokrotnie
        if not hasattr(self, "_busyDepth"): self._busyDepth = 0
        self._busyDepth += 1
        if self._busyDepth == 1:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self._busy_wrap.setVisible(True)

    def _end_redraw_now(self):
        d = getattr(self, "_busyDepth", 0)
        if d <= 1:
            self._busyDepth = 0
            self._busy_wrap.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
        else:
            self._busyDepth = d - 1

    def _await_next_frame_then_end(self):
        """Zdejmij WAIT dopiero, gdy ramka będzie na ekranie."""
        # uniknij wielokrotnych podłączeń
        if getattr(self, "_awaitingSwap", False):
            return
        self._awaitingSwap = True

        def _done():
            self._awaitingSwap = False
            try: self.view.frameSwapped.disconnect(_done)
            except Exception: pass
            self._end_redraw_now()

        # preferuj prawdziwy sygnał QOpenGLWidget:
        if hasattr(self.view, "frameSwapped"):
            try:
                self.view.frameSwapped.connect(_done)
            except Exception:
                QtCore.QTimer.singleShot(0, _done)
        else:
            # bardzo stary Qt/QGLWidget – fallback
            QtCore.QTimer.singleShot(0, _done)

        # upewnij się, że rzeczywiście będzie repaint
        self.view.update()


    def remove_existing_items(self):
        # stare pojedyncze itemy
        for item in [self.surface_ref_item, self.surface_adj_item,
                    self.ref_profile_line_item, self.adj_profile_line_item,
                    self.cross_plane_item]:
            if item and not isinstance(item, LODSurface):
                try: self.view.removeItem(item)
                except Exception: pass

        # LOD-y
        for key in ('ref','adj'):
            if self._lod.get(key):
                self._lod[key].destroy()
                self._lod[key] = None

        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

    def update_data(self, reference_grid, adjusted_grid=None, line_points=None, separation=0):
        """Updates the 3D view with new grid data and profile lines.

        Removes existing items, prepares and adds new surfaces and profile lines, and recenters the camera.
        
        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray, optional): The adjusted grid data.
            line_points (list or np.ndarray, optional): Points for the profile line.
            separation (float, optional): Vertical separation between surfaces.
        """
        self.remove_existing_items()

        xs, ys, Z_ref, xs_idx, ys_idx = self._prepare_reference_surface(reference_grid)
        if adjusted_grid is not None:
            Z_adj = self._prepare_adjusted_surface(adjusted_grid, ys_idx, xs_idx, separation, Z_ref)
        else:
            Z_adj = None

        # tuż po wyznaczeniu xs, ys, Z_ref i Z_adj:
        self._ref_last = (xs, ys, Z_ref)
        
        if adjusted_grid is not None:
            self._adj_last = (xs, ys, Z_adj)

        # NEW: ustaw spinboksy wg auto obliczeń (nie nadpisuje manualnych wartości)
        if self.range_ref_auto and np.any(np.isfinite(Z_ref)):
            lo, hi = self._compute_auto_lo_hi(Z_ref)
            self._update_range_widgets('ref', lo, hi, auto=True)
        
        if adjusted_grid is not None and self.range_adj_auto and np.any(np.isfinite(Z_adj)):
            lo, hi = self._compute_auto_lo_hi(Z_adj)
            self._update_range_widgets('adj', lo, hi, auto=True)

        self._add_reference_surface(xs, ys, Z_ref, colormap=self.colormap_ref)
        
        if adjusted_grid is not None:
            self._add_adjusted_surface(xs, ys, Z_adj,  colormap=self.colormap_adj)

        if not np.any(np.isfinite(Z_ref)) and not np.any(np.isfinite(Z_adj)):
            return  # nothing to display safely

        z_min, z_max = self._compute_z_limits(Z_ref, Z_adj, adjusted_grid is not None)
        margin = 0.1 * (z_max - z_min)
        z_min -= margin
        z_max += margin

        self._add_profile_and_plane(reference_grid, adjusted_grid, line_points, separation, z_min, z_max)

        self._center_camera(xs, ys, Z_ref, Z_adj, line_points)

    # def _prepare_reference_surface(self, reference_grid):
    #     """Prepares the reference surface for 3D visualization.

    #     Downsamples the reference grid, replaces invalid values with NaN, and returns the axes and processed grid.

    #     Args:
    #         reference_grid (np.ndarray): The reference grid data.

    #     Returns:
    #         tuple: (xs, ys, Z_ref) where xs and ys are axis arrays and Z_ref is the processed grid.
    #     """
    #     max_points = 256
    #     step = 1 # max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // max_points)
    #     ys = np.arange(0, reference_grid.shape[0], step)
    #     xs = np.arange(0, reference_grid.shape[1], step)
    #     Z_ref = reference_grid[np.ix_(ys, xs)]
    #     Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
    #     Z_ref = np.where((Z_ref > 1e6) | (Z_ref < -1e6), np.nan, Z_ref)
    #     return xs, ys, Z_ref


    def _prepare_reference_surface(self, reference_grid,
                                max_points=512, clip_abs=1e6,
                                dx=1.0, dy=1.0, x0=0.0, y0=0.0):
        """
        Prepares a downsampled reference surface from a 2D grid for visualization or further processing.
        Parameters
        ----------
        reference_grid : np.ndarray
            2D array representing the reference surface grid.
        max_points : int, optional
            Maximum number of points along each axis after downsampling (default is 512).
        clip_abs : float, optional
            Absolute value threshold for outlier clipping; values above this are set to NaN (default is 1e6).
        dx : float, optional
            Step size along the x-axis in physical units (default is 1.0).
        dy : float, optional
            Step size along the y-axis in physical units (default is 1.0).
        x0 : float, optional
            Origin offset along the x-axis (default is 0.0).
        y0 : float, optional
            Origin offset along the y-axis (default is 0.0).
        Returns
        -------
        xs : np.ndarray
            1D array of x-axis coordinates in physical units.
        ys : np.ndarray
            1D array of y-axis coordinates in physical units.
        Z : np.ndarray
            2D array of downsampled and masked surface values.
        xs_idx : np.ndarray
            1D array of selected x indices used for downsampling.
        ys_idx : np.ndarray
            1D array of selected y indices used for downsampling.
        Notes
        -----
        - NaN and outlier values in the downsampled grid are masked and set to NaN.
        - The returned indices can be used for further processing or alignment with other grids.
        """
        logger.debug("_prepare_reference_surface() - start")
        logger.debug(f"reference_grid.shape: {reference_grid.shape}")

        h0, w0 = reference_grid.shape
        step = max(1, min(h0, w0) // max_points)

        # indeksy (INT) do downsamplingu
        ys_idx = np.arange(0, h0, step, dtype=np.int32)
        xs_idx = np.arange(0, w0, step, dtype=np.int32)

        # siatka w docelowej rozdzielczości
        Z = reference_grid[np.ix_(ys_idx, xs_idx)].astype(np.float32, copy=True)

        # maskowanie NaN / outliers (jedna maska = szybciej)
        mask = ~np.isfinite(Z) | (np.abs(Z) > clip_abs)
        if mask.any():
            Z[mask] = np.nan

        # osie w jednostkach (FLOAT) – do GLSurfacePlotItem (opcjonalnie)
        xs = x0 + dx * xs_idx.astype(np.float32)
        ys = y0 + dy * ys_idx.astype(np.float32)

        logger.debug("_prepare_reference_surface() - end")
        # ZWRACAMY TAKŻE INDEKSY, bo _prepare_adjusted_surface ich potrzebuje
        return xs, ys, Z, xs_idx, ys_idx

    def _prepare_adjusted_surface(self, adjusted_grid, ys_idx, xs_idx, separation, Z_ref, clip_abs=1e6):
        """Zwraca Z_adj o tym samym kształcie co Z_ref."""
        if adjusted_grid is not None:
            Z_adj = adjusted_grid[np.ix_(ys_idx, xs_idx)].astype(np.float32) + separation
            mask = ~np.isfinite(Z_adj) | (np.abs(Z_adj) > clip_abs)
            if mask.any():
                Z_adj[mask] = np.nan
        else:
            Z_adj = np.full_like(Z_ref, np.nan, dtype=np.float32)
        return Z_adj


    # def _upsert_gl_surface(self, item_attr, xs, ys, Z):
    #     """Ensure GLSurfacePlotItem exists and has updated geometry."""
    #     item = getattr(self, item_attr, None)
    #     if item is None or not isinstance(item, gl.GLSurfacePlotItem):
    #         # usuń poprzedni obiekt (np. z trybu 'mesh'), jeśli był
    #         if item is not None:
    #             try: self.view.removeItem(item)
    #             except Exception: pass
    #         item = gl.GLSurfacePlotItem(x=xs, y=ys, z=Z.T)
    #         setattr(self, item_attr, item)
    #         self.view.addItem(item)
    #     else:
    #         item.setData(x=xs, y=ys, z=Z.T)
    #     return item

    
    # def _style_gl_surface(self, item:gl.GLSurfacePlotItem, Z, mode, color, colormap, which):
    #     """Apply rendering mode and colors to an existing GLSurfacePlotItem."""
    #     if mode == 'wireframe':
    #         item.setShader(None)
    #         item.opts['drawFaces'] = False
    #         item.opts['drawEdges'] = True
    #         item.opts['edgeColor'] = color
    #         item.setGLOptions('opaque')
    #         item.update()
    #         return

    #     # faces mode
    #     item.opts['drawFaces'] = True
    #     item.opts['drawEdges'] = False

    #     if colormap is None:
    #         colors = np.empty((Z.shape[1], Z.shape[0], 4), dtype=np.float32)
    #         colors[...] = color  # (r,g,b,a)
    #         #item.opts['color'] = color
    #         item.setColor(color)
    #     else:    
    #         lo, hi = self._get_lo_hi_for(which, Z)
    #         zprime = np.clip((Z - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    #         zT = zprime.T

    #         if colormap == 'RG':
    #             colors = np.empty((Z.shape[1], Z.shape[0], 4), dtype=np.float32)
    #             colors[..., 0] = 1.0 - zT   # R
    #             colors[..., 1] = zT         # G
    #             colors[..., 2] = 0.0        # B
    #             colors[..., 3] = 1.0        # A
    #         elif colormap == 'B&W':
    #             colors = np.empty((Z.shape[1], Z.shape[0], 4), dtype=np.float32)
    #             colors[..., 0] = zT
    #             colors[..., 1] = zT
    #             colors[..., 2] = zT
    #             colors[..., 3] = 1.0
    #         else: # kolormapy z pyqtgraph
    #             cm = pg.colormap.get(colormap)
    #             colors = cm.map(zT, mode='float')

    #     colors = np.ascontiguousarray(colors, dtype=np.float32)
    #     item.setData(colors=colors)
    #     item.opts['computeNormals'] = True
    #     item.setShader('shaded')
    #     # item.setGLOptions('opaque')   # zmień na 'translucent' jeśli gdzieś ustawiasz A<1

    #     from pyqtgraph.opengl import shaders

    #     prog = shaders.ShaderProgram(
    #         'headlight_color',
    #         [
    #             shaders.VertexShader("""
    #                 // Wersja "legacy": używa gl_* i działa w starszych pyqtgraph
    #                 varying vec3 vN;
    #                 varying vec4 vColor;
    #                 void main() {
    #                     gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    #                     vN     = normalize(gl_NormalMatrix * gl_Normal); // normal w przestrzeni oka
    #                     vColor = gl_Color;                               // z colors=... albo setColor(...)
    #                 }
    #             """),
    #             shaders.FragmentShader("""
    #                 varying vec3 vN;
    #                 varying vec4 vColor;
    #                 void main() {
    #                     // "Headlight" - światło przyspawane do widza (w przestrzeni oka)
    #                     vec3 L = normalize(vec3(0.0, 0.0, 1.0));

    #                     // Dwustronne: odwróć normalną dla tylnej ściany
    #                     vec3 N = normalize(vN);
    #                     if (!gl_FrontFacing) N = -N;

    #                     // Diffuse + lekki ambient
    #                     float diff = max(dot(N, L), 0.0);
    #                     vec3 base = vColor.rgb;
    #                     vec3 rgb  = base * (0.15 + 0.85*diff);

    #                     // (opcjonalnie lekki połysk)
    #                     // vec3 R = reflect(-L, N);
    #                     // vec3 V = vec3(0.0, 0.0, 1.0);
    #                     // float spec = pow(max(dot(R, V), 0.0), 16.0);
    #                     // rgb += 0.12 * spec;

    #                     gl_FragColor = vec4(rgb, vColor.a);
    #                 }
    #             """)
    #         ]
    #     )

    #     item.setShader(prog)
    #     item.update()

    def _ui_link_toggled(self, on):
        self.range_linked = bool(on)
        # zaktualizuj widoczność/aktywność pól
        self._update_range_widgets('ref',
            self.spin_lo_ref.value(), self.spin_hi_ref.value(), self.chk_auto_ref.isChecked())
        self._update_range_widgets('adj',
            self.spin_lo_adj.value(), self.spin_hi_adj.value(), self.chk_auto_adj.isChecked())
        self._refresh_surfaces()

    def _ui_auto_ref_toggled(self, on):
        self.range_ref_auto = bool(on)
        # przy auto – przelicz i wstaw do spinboksów (tylko jako display)
        if self._ref_last is not None and on:
            _, _, Z = self._ref_last
            lo, hi = self._compute_auto_lo_hi(Z)
            self._update_range_widgets('ref', lo, hi, auto=True)
        else:
            self.spin_lo_ref.setEnabled(True); self.spin_hi_ref.setEnabled(True)
        self._refresh_surfaces()

    def _ui_auto_adj_toggled(self, on):
        self.range_adj_auto = bool(on)
        if self._adj_last is not None and on:
            _, _, Z = self._adj_last
            lo, hi = self._compute_auto_lo_hi(Z)
            self._update_range_widgets('adj', lo, hi, auto=True)
        else:
            # może zostać nadpisane przez „link”
            self.spin_lo_adj.setEnabled(not (self.range_linked or on))
            self.spin_hi_adj.setEnabled(not (self.range_linked or on))
        self._refresh_surfaces()

    def _ui_lohi_changed(self, which):
        # zapis manualnych zakresów + odświeżenie
        if which == 'ref':
            self.range_ref = (self.spin_lo_ref.value(), self.spin_hi_ref.value())
            if self.range_linked:
                # odśwież pola Adj wizualnie
                self._update_range_widgets('adj', *self.range_ref, auto=self.range_adj_auto)
        else:
            self.range_adj = (self.spin_lo_adj.value(), self.spin_hi_adj.value())
        self._refresh_surfaces()


    def _place_surface(self, item_attr, xs, ys, Z, mode, color, colormap, test="std"):
        logger.debug(f"_place_surface({test}) - start")
        if Z.shape != (len(ys), len(xs)) or np.all(np.isnan(Z)):
            return

        which = 'ref' if item_attr == 'surface_ref_item' else 'adj'

        if mode == 'mesh':
            # (dotychczasowy kod „mesh” zostaw bez zmian)
            old = getattr(self, item_attr, None)
            if old is not None:
                try: self.view.removeItem(old)
                except Exception: pass
            item = (self.make_voxel_mesh(Z, xs=xs, ys=ys, color=color)
                    if colormap is None
                    else self.make_voxel_mesh(Z, xs=xs, ys=ys, colormap=colormap))
            setattr(self, item_attr, item)
            if item is not None:
                self.view.addItem(item)
            # ukryj ewentualny LOD dla tego kanału
            lod = self._lod.get(which)
            if lod: lod.set_visible(False)
        else:
            # LOD dla surface/wireframe
            lod = self._ensure_lod(which)
            lod.set_visible(True)
            lod.set_data(xs, ys, Z)
            lo, hi = self._get_lo_hi_for(which, Z)
            lod.update_style(mode=mode, colormap=colormap, base_color=color, lo=lo, hi=hi)

            # dla kompatybilności: „w tym atrybucie” trzymaj wskaźnik na menedżer
            setattr(self, item_attr, lod)

        logger.debug("_place_surface() - end")


    # --- public wrappers -------------------------------------------------------

    def _add_reference_surface(self, xs, ys, Z, colormap='RG'):
        """Adds/updates the reference surface."""
        kolor = (0, 1, 0, 1)
        self._place_surface('surface_ref_item', xs, ys, Z, self.ref_surface_mode, kolor, colormap, test="add_ref")

    def _add_adjusted_surface(self, xs, ys, Z, colormap='RG'):
        """Adds/updates the adjusted surface."""
        kolor = (0.2, 0.3, 1, 1)
        self._place_surface('surface_adj_item', xs, ys, Z, self.adj_surface_mode, kolor, colormap, test="add_adj")

    # def set_view_mode(self, mode, colormap=None):
    #     # zakładam, że trzymasz ostatnią siatkę:
    #     xs, ys, Z = self._ref_last  # np. ustawiane w _place_surface

    #     if mode == 'mesh':
    #         self._place_surface('surface_ref_item', xs, ys, Z, 'mesh', (0,1,0,1), colormap)
    #         return

    #     # upewnij się, że mamy GLSurfacePlotItem (tworzy jeśli brak)
    #     item = self._upsert_gl_surface('surface_ref_item', xs, ys, Z)

    #     # przełącz w locie tryb i kolory (zero duplikacji geometrii)
    #     if mode == 'wireframe':
    #         self._style_gl_surface(item, Z, 'wireframe', (0,1,0,1), None)
    #     else:  # 'surface' / cokolwiek nie-'wireframe'
    #         self._style_gl_surface(item, Z, 'surface', (0,1,0,1), colormap)  # np. 'RG' albo 'viridis'


    def _setup_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("1"), self, activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('wireframe')))
        QtWidgets.QShortcut(QtGui.QKeySequence("2"), self, activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('surface')))
        QtWidgets.QShortcut(QtGui.QKeySequence("3"), self, activated=lambda: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('mesh')))

    # def _refresh_surfaces(self):
    #     self._begin_redraw()
    #     try:
    #         if self._ref_last is not None:
    #             xs, ys, Z = self._ref_last
    #             self._place_surface('surface_ref_item', xs, ys, Z,
    #                                 self.ref_surface_mode, (0,1,0,1), self.colormap_ref, test="refresh_ref")
    #         if self._adj_last is not None and not np.all(np.isnan(self._adj_last[2])):
    #             xs, ys, Z = self._adj_last
    #             self._place_surface('surface_adj_item', xs, ys, Z,
    #                                 self.adj_surface_mode, (0.2,0.3,1,1), self.colormap_adj, test="refresh_adj")
    #     finally:
    #         self._await_next_frame_then_end()

    def _refresh_surfaces(self):
        self._begin_redraw()
        try:
            if self._ref_last is not None:
                xs, ys, Z = self._ref_last
                lo, hi = self._get_lo_hi_for('ref', Z)
                lod = self._ensure_lod('ref')
                lod.set_lod_params(target_px=1.8, hysteresis=0.3)
                lod.set_data(xs, ys, Z)
                lod.update_style(self.ref_surface_mode, self.colormap_ref, (0,1,0,1), lo, hi)
                lod.set_visible(True)
            if self._adj_last is not None and not np.all(np.isnan(self._adj_last[2])):
                xs, ys, Z = self._adj_last
                lo, hi = self._get_lo_hi_for('adj', Z)
                lod = self._ensure_lod('adj')
                lod.set_lod_params(target_px=1.8, hysteresis=0.3)
                lod.set_data(xs, ys, Z)
                lod.update_style(self.adj_surface_mode, self.colormap_adj, (0.2,0.3,1,1), lo, hi)
                lod.set_visible(True)
        finally:
            self._await_next_frame_then_end()


    # def _ui_mode_changed(self, _idx):
    #     mode = self.combo_mode_r.currentData()
    #     self.ref_surface_mode = mode
    #     self.adj_surface_mode = mode
    #     self._refresh_surfaces()

    def _ui_mode_changed_r(self, _idx):
        mode = self.combo_mode_r.currentData()
        self.ref_surface_mode = mode
        self._refresh_surfaces()

    def _ui_mode_changed_a(self, _idx):
        mode = self.combo_mode_a.currentData()
        self.adj_surface_mode = mode
        self._refresh_surfaces()


    def _ui_cmap_ref_changed(self, _idx):
        txt = self.combo_cmap_ref.currentText()
        self.colormap_ref = None if txt == "None" else txt
        self._refresh_surfaces()

    def _ui_cmap_adj_changed(self, _idx):
        txt = self.combo_cmap_adj.currentText()
        self.colormap_adj = None if txt == "None" else txt
        self._refresh_surfaces()

    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent):
        m = QtWidgets.QMenu(self)
        group = QtWidgets.QActionGroup(m)
        a_surface   = m.addAction("Surface (shaded)"); a_surface.setCheckable(True); a_surface.setActionGroup(group)
        a_wireframe = m.addAction("Wireframe");        a_wireframe.setCheckable(True); a_wireframe.setActionGroup(group)
        a_mesh      = m.addAction("Mesh");             a_mesh.setCheckable(True); a_mesh.setActionGroup(group)

        mode = self.ref_surface_mode
        a_surface.setChecked(mode=='surface')
        a_wireframe.setChecked(mode=='wireframe')
        a_mesh.setChecked(mode=='mesh')

        m.addSeparator()
        sub_ref = m.addMenu("Ref colormap")
        for name in ["None","RG","viridis","plasma","magma"]:
            act = sub_ref.addAction(name); act.setCheckable(True)
            act.setChecked((self.colormap_ref or "None")==name)
            act.triggered.connect(lambda _, n=name: self._set_ref_cmap_from_menu(n))

        if self.two_scans_mode:
            sub_adj = m.addMenu("Adj colormap")
            for name in ["None","RG","viridis","plasma","magma"]:
                act = sub_adj.addAction(name); act.setCheckable(True)
                act.setChecked((self.colormap_adj or "None")==name)
                act.triggered.connect(lambda _, n=name: self._set_adj_cmap_from_menu(n))

        chosen = m.exec_(ev.globalPos())
        if chosen is a_surface:   self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('surface'))
        if chosen is a_wireframe: self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('wireframe'))
        if chosen is a_mesh:      self.combo_mode_r.setCurrentIndex(self.combo_mode_r.findData('mesh'))

    def _set_ref_cmap_from_menu(self, name):
        i = self.combo_cmap_ref.findText(name)
        if i >= 0: self.combo_cmap_ref.setCurrentIndex(i)

    def _set_adj_cmap_from_menu(self, name):
        i = self.combo_cmap_adj.findText(name)
        if i >= 0: self.combo_cmap_adj.setCurrentIndex(i)


    def _compute_z_limits(self, Z_ref, Z_adj, has_adjusted):
        """Computes the minimum and maximum Z values for the 3D view.

        Determines the Z-axis limits based on the reference and adjusted grids.

        Args:
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray): Processed adjusted grid.
            has_adjusted (bool): Whether the adjusted grid is present.

        Returns:
            tuple: (z_min, z_max) representing the Z-axis limits.
        """
        if has_adjusted:
            z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
            z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        else:
            z_min = np.nanmin(Z_ref)
            z_max = np.nanmax(Z_ref)
        return z_min, z_max

    def _add_profile_and_plane(self, reference_grid, adjusted_grid, line_points, separation, z_min, z_max):
        """Adds a cross-section plane and profile lines to the 3D view if line points are provided.

        Draws the cross-section plane and profile lines for both reference and adjusted grids, if available.

        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray or None): The adjusted grid data.
            line_points (list or np.ndarray): Points for the profile line.
            separation (float): Vertical separation between surfaces.
            z_min (float): Minimum Z value for the plane.
            z_max (float): Maximum Z value for the plane.
        """
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            self.cross_plane_item = self.add_cross_section_plane(pts, z_min, z_max)
            h, w = reference_grid.shape
            valid_mask = (
                (pts[:, 1] >= 0) & (pts[:, 1] < h) &
                (pts[:, 0] >= 0) & (pts[:, 0] < w)
            )
            if not np.all(valid_mask):
                logger.warning("[Grid3DViewer] Some profile points are out of bounds and will be ignored.")
                pts = pts[valid_mask]
                if len(pts) < 2:
                    return  # not enough points to plot a line
            try:
                ref_prof = reference_grid[pts[:,1], pts[:,0]]
                self.ref_profile_line_item = gl.GLLinePlotItem(
                    pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                    color=(0,0.4,0,1), width=2)
                self.view.addItem(self.ref_profile_line_item)
                if adjusted_grid is not None:
                    adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
                    self.adj_profile_line_item = gl.GLLinePlotItem(
                        pos=np.column_stack((pts[:,0], pts[:,1], adj_prof)),
                        color=(0,0,1,1), width=2)
                    self.view.addItem(self.adj_profile_line_item)
            except Exception as e:
                logger.critical("[Grid3DViewer] Failed to plot profile lines:", e)

    def _center_camera(self, xs, ys, Z_ref, Z_adj, line_points):
        """Recenters the 3D camera based on the current data bounds.

        Calculates the center of all displayed data and sets the camera position accordingly.

        Args:
            xs (np.ndarray): X-axis indices.
            ys (np.ndarray): Y-axis indices.
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray): Processed adjusted grid.
            line_points (list or np.ndarray): Points for the profile line.
        """
        all_x = list(xs)
        all_y = list(ys)
        all_z = [np.nanmin(Z_ref), np.nanmax(Z_ref)]
        
        if Z_adj is not None:
            if not np.all(np.isnan(Z_adj)):
                all_z += [np.nanmin(Z_adj), np.nanmax(Z_adj)]
        
        if line_points is not None:
            pts = np.array(line_points)
            all_x += list(pts[:,0])
            all_y += list(pts[:,1])

        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if all_z else 0
        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))

    def set_controls_visible(self, visible):
        """Shows or hides the control checkboxes in the 3D viewer.

        Sets the visibility of all control checkboxes and updates the widget.

        Args:
            visible (bool): Whether the controls should be visible.
        """
        self.two_scans_mode = visible
        self.show_controls = visible
        if hasattr(self, "checkbox_ref"):
            for cb in [self.checkbox_line, self.checkbox_plane]:
                cb.setVisible(visible)
            self.update()

        if hasattr(self, "a_tools"):
            for cb in [self.a_tools,]:
                cb.setVisible(visible)
            self.update()


    def create_verts_grid(self, Z, xs, ys, cols, rows):
        """Creates a grid of 3D vertices for a voxel mesh using provided x/y coordinates.

        For each grid point, computes the vertex position by averaging the heights of neighboring cells and assigning the appropriate x/y coordinate.

        Args:
            Z (np.ndarray): 2D array of height values.
            xs (np.ndarray): X coordinates (length = Z.shape[1]).
            ys (np.ndarray): Y coordinates (length = Z.shape[0]).
            cols (int): Number of columns in the grid.
            rows (int): Number of rows in the grid.

        Returns:
            np.ndarray: 3D array of vertex positions with shape (rows+1, cols+1, 3).
        """
        verts_grid = np.zeros((rows + 1, cols + 1, 3), dtype=np.float32)
        for i in range(rows + 1):
            for j in range(cols + 1):
                zs = []
                for di in [0, -1]:
                    for dj in [0, -1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(Z[ni, nj]):
                            zs.append(Z[ni, nj])
                z = np.mean(zs) if zs else 0.0
                x = xs[j-1] if j > 0 else xs[0]
                y = ys[i-1] if i > 0 else ys[0]
                verts_grid[i, j] = [x, y, z]
        return verts_grid


    def calculate_normals( self, verts_grid, Z, cols, rows ):
        """Calculates vertex normals for a voxel mesh based on the local surface gradient.

        Computes normals for each vertex by estimating the local surface gradient using neighboring vertices, 
        skipping locations where the underlying grid value is NaN.

        Args:
            verts_grid (np.ndarray): 3D array of vertex positions with shape (rows+1, cols+1, 3).
            Z (np.ndarray): 2D array of height values.
            cols (int): Number of columns in the grid.
            rows (int): Number of rows in the grid.

        Returns:
            np.ndarray: Flattened array of vertex normals with shape ((rows+1)*(cols+1), 3).
        """
        normals_grid = np.zeros_like(verts_grid)
        for i in range(1, rows):
            for j in range(1, cols):
                if np.isnan(Z[i-1, j-1]):
                    continue
                dzdx = (verts_grid[i, j, 2] - verts_grid[i, j-1, 2]) / (verts_grid[i, j, 0] - verts_grid[i, j-1, 0] + 1e-8)
                dzdy = (verts_grid[i, j, 2] - verts_grid[i-1, j, 2]) / (verts_grid[i, j, 1] - verts_grid[i-1, j, 1] + 1e-8)
                n = np.array([-dzdx, -dzdy, 1.0])
                n /= np.linalg.norm(n)
                normals_grid[i, j] = n
        return normals_grid.reshape(-1, 3)


    def make_voxel_mesh(self, Z, xs=None, ys=None, color=(0.0,0.7,0.0,1.0), colormap=None):
        """Creates an optimized 3D voxel mesh from a 2D grid, using the same x/y coordinates as surface mode.

        Args:
            Z (np.ndarray): 2D array of height values.
            xs (np.ndarray, optional): X coordinates (length = Z.shape[1]).
            ys (np.ndarray, optional): Y coordinates (length = Z.shape[0]).

        Returns:
            GLMeshItem: The generated 3D mesh item.
        """
    
        rows, cols = Z.shape
        if xs is None:
            xs = np.arange(cols)
        if ys is None:
            ys = np.arange(rows)

        verts_grid = self.create_verts_grid(Z, xs, ys, cols, rows)

        verts = verts_grid.reshape(-1, 3)
        idx = lambda i, j: i * (cols + 1) + j

        faces = []
        # colors = []

        #vertex_normals = self.calculate_normals( verts_grid, Z, cols, rows )

        # Górna powierzchnia
        for i in range(rows):
            for j in range(cols):
                if np.isnan(Z[i, j]):
                    continue
                v00 = idx(i, j)
                v10 = idx(i + 1, j)
                v11 = idx(i + 1, j + 1)
                v01 = idx(i, j + 1)
                faces.extend(([v00, v10, v11], [v00, v11, v01]))
                # colors.extend([[0.0, 0.7, 0.0, 1]] * 2)

        if not len(faces):
            return None

        # Mapowanie wysokości na kolory
        if colormap is None:
            vertex_colors = np.tile(color, (verts.shape[0], 1))
        elif colormap in ('RG','B&W',):
            # vertex_colors = create_colors(Z, colormap)
            logger.warning("You can't set custom colormaps in 'Mesh' mode, using solid color instead")
            vertex_colors = np.tile(color, (verts.shape[0], 1))
        else:
            z_vals = verts[:, 2]
            z_min, z_max = np.nanmin(z_vals), np.nanmax(z_vals)
            normed = (z_vals - z_min) / (z_max - z_min + 1e-8)
            #cmap = plt.get_cmap(colormap)
            #vertex_colors = cmap(normed)  # shape (N, 4)
            cmap = pg.colormap.get(colormap)
            vertex_colors = cmap.map(normed, mode='float')

        # print(f"vertex_colors.shape: {vertex_colors.shape}, vertex_colors.dtype: {vertex_colors.dtype}")

        return gl.GLMeshItem(
            vertexes=verts,
            faces=np.array(faces),
            vertexColors=vertex_colors,
            shader='shaded',
            smooth=True,
            drawEdges=False
        )


    def add_cross_section_plane(self, pts, z_min, z_max):
        """Adds a translucent cross-section plane to the 3D view.

        Creates and displays a rectangular plane between two points at the specified z-range.

        Args:
            pts (np.ndarray): Array of two (x, y) points defining the plane's endpoints.
            z_min (float): Minimum z-value for the plane.
            z_max (float): Maximum z-value for the plane.

        Returns:
            GLMeshItem: The mesh item representing the cross-section plane.
        """
        p0, p1 = pts[0], pts[-1]
        rect = np.array([
            [p0[0], p0[1], z_min],
            [p1[0], p1[1], z_min],
            [p1[0], p1[1], z_max],
            [p0[0], p0[1], z_max],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        color = np.array([0.5, 0.5, 0.7, 0.7])
        mesh = gl.GLMeshItem(vertexes=rect, faces=faces, faceColors=np.tile(color, (2,1)),
                             glOptions='translucent', smooth=False, drawEdges=False)
        self.view.addItem(mesh)
        return mesh

    # def toggle_surface_ref(self, state):
    #     if self.surface_ref_item:
    #         self.surface_ref_item.setVisible(bool(state))

    # def toggle_surface_adj(self, state):
    #     if self.surface_adj_item:
    #         self.surface_adj_item.setVisible(bool(state))

    def toggle_surface_ref(self, state):
        obj = self.surface_ref_item
        if isinstance(obj, LODSurface):
            obj.set_visible(bool(state))
        elif obj:
            obj.setVisible(bool(state))

    def toggle_surface_adj(self, state):
        obj = self.surface_adj_item
        if isinstance(obj, LODSurface):
            obj.set_visible(bool(state))
        elif obj:
            obj.setVisible(bool(state))


    def toggle_profile_line(self, state):
        if self.ref_profile_line_item:
            self.ref_profile_line_item.setVisible(bool(state))
        if self.adj_profile_line_item:
            self.adj_profile_line_item.setVisible(bool(state))

    def toggle_cross_plane(self, state):
        if self.cross_plane_item:
            self.cross_plane_item.setVisible(bool(state))

    def closeEvent(self, event):
        self.remove_existing_items()
        self.view.repaint()          # odświeżenie QWidgetu
        QtWidgets.QApplication.processEvents()  # przetwórz natychmiast
        event.accept()



_global_3d_viewer = None

def show_3d_viewer(reference_grid, adjusted_grid=None, line_points=None, separation=0, show_controls=True):
    """Displays the 3D grid viewer window with the provided data.

    Initializes the viewer if needed, sets control visibility, updates the 3D view with new data, and brings the window to the front.

    Args:
        reference_grid (np.ndarray): The reference grid data.
        adjusted_grid (np.ndarray, optional): The adjusted grid data.
        line_points (list or np.ndarray, optional): Points for the profile line.
        separation (float, optional): Vertical separation between surfaces.
        show_controls (bool, optional): Whether to show UI controls.
    """
    global _global_3d_viewer
    if _global_3d_viewer is None:
        _global_3d_viewer = Grid3DViewer()

    _global_3d_viewer.set_controls_visible(show_controls)
    _global_3d_viewer.update_data(
        reference_grid=reference_grid,
        adjusted_grid=adjusted_grid,
        line_points=line_points,
        separation=separation
    )

    logger.debug("_global_3d_viewer.show()")
    _global_3d_viewer.show()
    logger.debug("_global_3d_viewer.raise()")
    _global_3d_viewer.raise_()
    logger.debug("_global_3d_viewer.activateWindow()")
    _global_3d_viewer.activateWindow()
    logger.debug("_global_3d_viewer ready")

