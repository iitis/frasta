"""Level-of-detail (LOD) surface rendering for 3D visualization.

This module provides automatic level-of-detail management for large 3D surface
meshes, dynamically switching between different mesh densities based on camera
distance and viewport size to maintain performance.
"""

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import pyqtgraph as pg


class LODSurface:
    """Manages multiple mesh representations with automatic LOD switching.
    
    Creates and manages several GLMeshItem instances at different densities
    (steps) and automatically switches the visible level based on camera zoom
    to balance visual quality and rendering performance.
    
    Attributes:
        view: OpenGL view widget containing the meshes.
        steps (tuple): Available LOD steps (1, 2, 4, 8, etc.).
        target_px (float): Target pixels per grid cell for LOD selection.
        hysteresis (float): Hysteresis factor to prevent LOD flickering.
        data (tuple): Current mesh data (xs, ys, Z).
        visible (bool): Overall visibility flag.
    """
    def __init__(self, view, steps=(1,2,4,8,16), shader=None,
                 target_px=1.5, hysteresis=0.25, base_cell=None, thresholds=None):
        """Initialize LOD surface manager.
        
        Args:
            view: OpenGL view widget to contain the meshes.
            steps (tuple, optional): LOD step sizes. Defaults to (1,2,4,8,16).
            shader: Optional shader program for rendering.
            target_px (float, optional): Target pixels per cell. Defaults to 1.5.
            hysteresis (float, optional): LOD switching hysteresis (0-1). Defaults to 0.25.
            base_cell (float, optional): Base cell size in scene units. Auto-detected if None.
            thresholds (dict, optional): Explicit LOD thresholds {step: (px_lo, px_hi)}.
        """
        self.view = view
        self.steps = tuple(sorted(set(steps)))
        self.shader = shader
        self.items = {}
        self.data = None
        self.color = (0,1,0,1)
        self.colormap = 'RG'
        self.lohi = None
        self.mode = 'surface'
        self.visible = True

        # NOWE:
        self.target_px = float(target_px)     # docelowe px na "komĂłrkÄ™" siatki (step=1)
        self.hysteresis = float(hysteresis)   # np. 0.25 => Â±25% strefa nieczuĹ‚oĹ›ci
        self.base_cell = base_cell            # rozmiar komĂłrki w jednostkach sceny; auto z xs/ys, jeĹ›li None
        self.thresholds = thresholds          # alternatywnie: jawne progi {step: (px_lo, px_hi)}
        self._last_step = None

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update_lod)
        self._timer.start(33)


    def destroy(self):
        """Clean up all mesh items and stop update timer."""
        for it in self.items.values():
            try: self.view.removeItem(it)
            except Exception: pass
        self.items.clear()
        self._timer.stop()

    def set_visible(self, on: bool):
        """Set visibility of the LOD surface.
        
        Args:
            on (bool): True to show, False to hide.
        """
        self.visible = bool(on)
        for it in self.items.values():
            it.setVisible(self.visible and (it is self._current_item()))

    def set_data(self, xs, ys, Z):
        """Set surface data and rebuild meshes.
        
        Args:
            xs (np.ndarray): X-coordinates (1D array).
            ys (np.ndarray): Y-coordinates (1D array).
            Z (np.ndarray): Height values (2D array).
        """
        self.data = (xs, ys, Z.astype(np.float32, copy=False))

        # autodetekcja rozmiaru komĂłrki (uĹĽyj mediany, odporna na outliery)
        if self.base_cell is None and xs is not None and ys is not None:
            try:
                dx = float(np.median(np.abs(np.diff(xs)))) if len(xs) > 1 else 1.0
                dy = float(np.median(np.abs(np.diff(ys)))) if len(ys) > 1 else 1.0
                self.base_cell = max(dx, dy) if (np.isfinite(dx) and np.isfinite(dy)) else 1.0
            except Exception:
                self.base_cell = 1.0

        self._ensure_current_exists()
        self._restyle_all_existing()

    def set_lod_params(self, *, target_px=None, hysteresis=None, steps=None, thresholds=None, base_cell=None):
        """Update LOD parameters.
        
        Args:
            target_px (float, optional): New target pixels per cell.
            hysteresis (float, optional): New hysteresis factor.
            steps (tuple, optional): New LOD step sizes.
            thresholds (dict, optional): New explicit thresholds.
            base_cell (float, optional): New base cell size.
        """
        if target_px is not None:  self.target_px  = float(target_px)
        if hysteresis is not None: self.hysteresis = float(hysteresis)
        if steps is not None:      self.steps      = tuple(sorted(set(steps)))
        if thresholds is not None: self.thresholds = dict(thresholds)
        if base_cell is not None:  self.base_cell  = float(base_cell)

    def update_style(self, mode, colormap, base_color, lo, hi):
        """Update visual style of all mesh items.
        
        Args:
            mode (str): Rendering mode ('surface', 'wireframe', 'mesh').
            colormap (str): Colormap name or None for solid color.
            base_color (tuple): Base RGBA color (0-1 range).
            lo (float): Minimum value for colormap.
            hi (float): Maximum value for colormap.
        """
        self.mode = mode
        self.colormap = colormap
        self.color = base_color
        self.lohi = (float(lo), float(hi))
        self._restyle_all_existing()

    # ---------- wewnÄ™trzne ----------
    def _ensure_current_exists(self):
        s = self._pick_step()
        if s not in self.items:
            self.items[s] = self._build_item_for_step(s)

    def _current_item(self):
        if not self.items: return None
        # znajdĹş jedyny widoczny
        for s,it in self.items.items():
            # GLMeshItem ma atrybut 'visible' zamiast metody isVisible()
            if getattr(it, 'visible', False): return it
        # albo zwrĂłÄ‡ ostatni tworzony
        return next(iter(self.items.values()))

    def _restyle_all_existing(self):
        for s,it in self.items.items():
            self._apply_style(it, s)

    def _build_item_for_step(self, step):
        xs, ys, Z = self.data
        if xs is None: xs = np.arange(Z.shape[1], dtype=np.float32)
        if ys is None: ys = np.arange(Z.shape[0], dtype=np.float32)

        X, Y = np.meshgrid(xs[::step], ys[::step], indexing='xy')
        H = Z[::step, ::step]
        h, w = H.shape

        # wierzchoĹ‚ki i trĂłjkÄ…ty
        V = np.c_[X.ravel(), Y.ravel(), H.ravel()].astype(np.float32)
        idx = np.arange(h*w, dtype=np.uint32).reshape(h, w)
        f1 = np.c_[idx[:-1,:-1].ravel(), idx[1:,:-1].ravel(), idx[1:,1:].ravel()]
        f2 = np.c_[idx[:-1,:-1].ravel(), idx[1:,1:].ravel(), idx[:-1,1:].ravel()]
        faces = np.vstack([f1, f2])

        md = gl.MeshData(vertexes=V, faces=faces)
        it = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, drawFaces=True, shader='shaded')
        it._mesh = md  # <- zapisz md w itemie

        if self.shader is not None:
            it.setShader(self.shader)

        self.view.addItem(it)
        it.setVisible(False)
        # kolory/tryb bÄ™dÄ… naĹ‚oĹĽone w _apply_style
        return it

    def _apply_style(self, it, step):
        # tryb rysowania
        if self.mode == 'wireframe':
            it.opts['drawFaces'] = False
            it.opts['drawEdges'] = True
            it.opts['edgeColor'] = self.color
        else:
            it.opts['drawFaces'] = True
            it.opts['drawEdges'] = False

        # dostÄ™p do MeshData (kompatybilnie wstecz)
        md = getattr(it, '_mesh', None)
        if md is None:
            try:
                md = it.meshData()  # moĹĽe istnieÄ‡ w Twojej wersji
            except Exception:
                return  # bez MeshData nie pokolorujemy

        # wierzchoĹ‚ki i wysokoĹ›ci
        V = md.vertexes()                  # stare API: bez argumentĂłw
        z = V[:, 2].astype(np.float32, copy=False)
        finite = np.isfinite(z)

        # kolory: kolormap None => staĹ‚y kolor; w przeciwnym razie mapowanie po z
        if self.colormap is None:
            C = np.tile(np.asarray(self.color, dtype=np.float32), (V.shape[0], 1))
        else:
            # lo/hi: z GUI lub auto z danych (ignoruj NaNy)
            if self.lohi is not None and all(np.isfinite(self.lohi)):
                lo, hi = self.lohi
            elif finite.any():
                lo = float(np.nanmin(z[finite]))
                hi = float(np.nanmax(z[finite]))
                if not np.isfinite(hi - lo) or hi <= lo:
                    hi = lo + 1e-6
            else:
                lo, hi = 0.0, 1.0  # fallback gdy same NaNy

            t = np.zeros_like(z, dtype=np.float32)
            if finite.any():
                t[finite] = np.clip((z[finite] - lo) / (hi - lo + 1e-12), 0.0, 1.0)

            if self.colormap in ('RG', 'B&W'):
                if self.colormap == 'RG':
                    C = np.stack([1.0 - t, t, np.zeros_like(t), np.ones_like(t)], axis=1)
                else:  # B&W
                    C = np.stack([t, t, t, np.ones_like(t)], axis=1)
            else:
                # kolormapy z pyqtgraph
                cmap = pg.colormap.get(self.colormap)
                C = cmap.map(t, mode='float').astype(np.float32)

            # przezroczyste wierzchoĹ‚ki dla NaN
            if (~finite).any():
                C[~finite, 3] = 0.0

        # push kolorĂłw do GPU (czÄ™sto samo setVertexColors nie wystarcza)
        C = np.ascontiguousarray(C, dtype=np.float32)
        md.setVertexColors(C)
        it.setMeshData(meshdata=md)

        # Only set custom shader if one was provided during initialization
        # Otherwise, let GLMeshItem use its default 'shaded' shader
        if self.shader is not None:
            it.setShader(self.shader)
        
        it.update()

    def _pick_step(self):
        view = self.view
        fov = np.deg2rad(view.opts.get('fov', 60.0))
        dist = float(view.opts['distance'])
        px_h = max(1, view.height())
        px_per_unit = px_h / (2.0*np.tan(fov/2.0)*max(dist, 1e-6))

        cell = self.base_cell if (self.base_cell is not None and self.base_cell > 0) else 1.0
        px = px_per_unit * cell               # px przypadajÄ…ce na "komĂłrkÄ™" siatki dla step=1

        # 4a) Jawne progi: thresholds = {step: (px_lo, px_hi)} dla px_per_cell_step = px * step
        if self.thresholds:
            # Histereza: rozszerz zakres bieĹĽÄ…cego kroku
            if self._last_step in self.thresholds:
                lo, hi = self.thresholds[self._last_step]
                k = self.hysteresis
                lo *= (1.0 - k)
                hi *= (1.0 + k)
                if lo <= px * self._last_step < hi:
                    return self._last_step

            # wybierz pierwszy step, dla ktĂłrego px*step wpada w przedziaĹ‚
            for s in sorted(self.thresholds.keys()):
                lo, hi = self.thresholds[s]
                if lo <= px * s < hi:
                    self._last_step = s
                    return s
            # fallback: najbliĹĽszy step wzglÄ™dem target_px
            desired = max(1, int(np.ceil(self.target_px / max(px,1e-6))))
            s = min(self.steps, key=lambda st: abs(st - desired))
            self._last_step = s
            return s

        # 4b) Polityka "target_px" (prosta): dÄ…ĹĽ do px_per_cell_step ~ target_px
        desired = max(1, int(np.ceil(self.target_px / max(px, 1e-6))))
        # znajdĹş najbliĹĽszy dostÄ™pny step
        candidates = sorted(self.steps)
        s = None
        for st in candidates:
            if st >= desired:
                s = st; break
        if s is None:
            s = candidates[-1]

        # Histereza: trzymaj poprzedni krok dopĂłki px*s0 jest blisko target_px
        if self._last_step is not None:
            r = (px * self._last_step) / (self.target_px + 1e-6)  # 1.0 = idealnie
            band = self.hysteresis
            if (1.0 - band) <= r <= (1.0 + band):
                return self._last_step

        self._last_step = s
        return s

    def update_lod(self):
        """Update LOD level based on current camera parameters.
        
        Picks appropriate LOD step based on viewport size and distance,
        creates missing mesh items if needed, and switches visibility.
        """
        if self.data is None or not self.visible:
            return
        s = self._pick_step()
        if s not in self.items:
            self.items[s] = self._build_item_for_step(s)
            self._apply_style(self.items[s], s)
        # przeĹ‚Ä…cz widocznoĹ›Ä‡
        for k,it in self.items.items():
            it.setVisible(self.visible and (k == s))

