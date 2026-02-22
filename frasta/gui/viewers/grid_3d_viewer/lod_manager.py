"""Level-of-detail (LOD) management for 3D surface rendering.

This module handles automatic and manual LOD switching based on camera distance
and viewport resolution to maintain smooth rendering performance.
"""

from PyQt5 import QtCore
import logging

logger = logging.getLogger(__name__)

from ..lod_surface import LODSurface


class LODManager:
    """Manages LOD surfaces for reference and adjusted scans."""
    
    def __init__(self, view, lod_ref=None, lod_adj=None):
        """Initialize LOD manager.
        
        Args:
            view: The GLViewWidget for 3D rendering.
            lod_ref: Optional pre-existing LODSurface for reference.
            lod_adj: Optional pre-existing LODSurface for adjusted.
        """
        self.view = view
        self._lod = {'ref': lod_ref, 'adj': lod_adj}
        
        # LOD configuration
        self.lod_steps = (1, 2, 4, 8, 16, 32)
        self.lod_target_px = 1.8
        self.lod_hysteresis = 0.3
        self.lod_thresholds = None
        self.lod_base_cell = None
        
        # Track profile items for rendering order (initialized before timer)
        self.cross_plane_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        
        # Auto-update timer (start after all attributes are initialized)
        self._lod_timer = QtCore.QTimer()
        self._lod_timer.timeout.connect(self._update_lod_tick)
        self._lod_timer.start(33)  # ~30 FPS
    
    def ensure_lod(self, which, steps=None):
        """Ensure LODSurface exists for specified surface (ref/adj).
        
        Args:
            which (str): 'ref' or 'adj'.
            steps (tuple, optional): LOD steps to use.
            
        Returns:
            LODSurface: The LOD surface manager.
        """
        key = 'ref' if which == 'ref' else 'adj'
        if self._lod[key] is None:
            shader = None
            s = steps or self.lod_steps
            lod = LODSurface(self.view, steps=s, shader=shader)
            lod.set_lod_params(
                target_px=self.lod_target_px,
                hysteresis=self.lod_hysteresis,
                thresholds=self.lod_thresholds,
                base_cell=self.lod_base_cell
            )
            self._lod[key] = lod
        return self._lod[key]
    
    def _update_lod_tick(self):
        """Update LOD visibility every timer tick (~30 FPS)."""
        try:
            # Update LOD surfaces
            for key in ['ref', 'adj']:
                if self._lod.get(key):
                    self._lod[key].update_visible()
            
            # Ensure cross-section plane and profile lines render last
            # (after surfaces) by removing and re-adding them
            if self.cross_plane_item is not None:
                try:
                    self.view.removeItem(self.cross_plane_item)
                    self.view.addItem(self.cross_plane_item)
                except Exception:
                    pass
            
            for profile_item in [self.ref_profile_line_item, self.adj_profile_line_item]:
                if profile_item is None:
                    continue
                try:
                    if isinstance(profile_item, list):
                        for item in profile_item:
                            self.view.removeItem(item)
                            self.view.addItem(item)
                    else:
                        self.view.removeItem(profile_item)
                        self.view.addItem(profile_item)
                except Exception:
                    pass
        except Exception as e:
            # Catch any unexpected errors to prevent timer from crashing
            logger.debug(f"LOD update tick error: {e}")
    
    def set_lod_params(self, target_px=None, hysteresis=None, thresholds=None, base_cell=None):
        """Update LOD parameters.
        
        Args:
            target_px (float, optional): Target pixel size for LOD switching.
            hysteresis (float, optional): Hysteresis factor for LOD switching.
            thresholds (dict, optional): Custom thresholds per LOD step.
            base_cell (float, optional): Base cell size in world units.
        """
        if target_px is not None:
            self.lod_target_px = target_px
        if hysteresis is not None:
            self.lod_hysteresis = hysteresis
        if thresholds is not None:
            self.lod_thresholds = thresholds
        if base_cell is not None:
            self.lod_base_cell = base_cell
        
        # Update existing LOD surfaces
        for lod in self._lod.values():
            if lod:
                lod.set_lod_params(
                    target_px=self.lod_target_px,
                    hysteresis=self.lod_hysteresis,
                    thresholds=self.lod_thresholds,
                    base_cell=self.lod_base_cell
                )
    
    def get_lod(self, which):
        """Get LODSurface for specified surface.
        
        Args:
            which (str): 'ref' or 'adj'.
            
        Returns:
            LODSurface or None: The LOD surface manager.
        """
        key = 'ref' if which == 'ref' else 'adj'
        return self._lod.get(key)
    
    def destroy_lod(self, which):
        """Destroy LODSurface for specified surface.
        
        Args:
            which (str): 'ref' or 'adj'.
        """
        key = 'ref' if which == 'ref' else 'adj'
        if self._lod.get(key):
            self._lod[key].destroy()
            self._lod[key] = None
    
    def destroy_all(self):
        """Destroy all LOD surfaces."""
        for key in ['ref', 'adj']:
            if self._lod.get(key):
                self._lod[key].destroy()
                self._lod[key] = None
    
    def stop_timer(self):
        """Stop the LOD update timer."""
        if self._lod_timer:
            self._lod_timer.stop()
    
    def start_timer(self):
        """Start the LOD update timer."""
        if self._lod_timer:
            self._lod_timer.start(33)
