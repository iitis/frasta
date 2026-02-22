"""Colormap and value range management for 3D surface visualization.

This module handles:
- Automatic range calculation from data
- Manual range control with optional linking between surfaces
- Colormap selection and application
- Range widget synchronization
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ColormapManager:
    """Manages colormaps and value ranges for reference and adjusted surfaces."""
    
    def __init__(self):
        """Initialize colormap manager."""
        # Colormap selection
        self.colormap_ref = 'RG'
        self.colormap_adj = 'RG'
        
        # Range settings
        self.range_linked = False  # Link ref/adj ranges together
        self.range_ref_auto = True  # Auto-calculate ref range
        self.range_adj_auto = True  # Auto-calculate adj range
        self.range_ref = (None, None)  # (lo, hi) when auto=False
        self.range_adj = (None, None)
        
        # UI widgets (set by parent)
        self.spin_lo_ref = None
        self.spin_hi_ref = None
        self.spin_lo_adj = None
        self.spin_hi_adj = None
        self.chk_auto_ref = None
        self.chk_auto_adj = None
        self.chk_link = None
        
        # Data cache for range calculation
        self._ref_last = None
        self._adj_last = None
    
    def set_widgets(self, spin_lo_ref, spin_hi_ref, spin_lo_adj, spin_hi_adj,
                   chk_auto_ref, chk_auto_adj, chk_link):
        """Set UI widget references.
        
        Args:
            spin_lo_ref: QDoubleSpinBox for reference low value.
            spin_hi_ref: QDoubleSpinBox for reference high value.
            spin_lo_adj: QDoubleSpinBox for adjusted low value.
            spin_hi_adj: QDoubleSpinBox for adjusted high value.
            chk_auto_ref: QCheckBox for reference auto range.
            chk_auto_adj: QCheckBox for adjusted auto range.
            chk_link: QCheckBox for linking ranges.
        """
        self.spin_lo_ref = spin_lo_ref
        self.spin_hi_ref = spin_hi_ref
        self.spin_lo_adj = spin_lo_adj
        self.spin_hi_adj = spin_hi_adj
        self.chk_auto_ref = chk_auto_ref
        self.chk_auto_adj = chk_auto_adj
        self.chk_link = chk_link
    
    def set_data_cache(self, ref_last, adj_last):
        """Update cached data for range calculations.
        
        Args:
            ref_last (tuple or None): (xs, ys, Z_ref) for reference.
            adj_last (tuple or None): (xs, ys, Z_adj) for adjusted.
        """
        self._ref_last = ref_last
        self._adj_last = adj_last
    
    def compute_auto_lo_hi(self, Z):
        """Compute automatic range from data using percentile method.
        
        Args:
            Z (np.ndarray): 2D height data (may contain NaN).
            
        Returns:
            tuple: (lo, hi) range values.
        """
        valid = Z[np.isfinite(Z)]
        if len(valid) == 0:
            return (0.0, 1.0)
        
        # Use percentile to exclude outliers
        lo = np.percentile(valid, 1)
        hi = np.percentile(valid, 99)
        
        # Ensure non-zero range
        if abs(hi - lo) < 1e-9:
            span = max(abs(lo) * 0.1, 1.0)
            lo -= span
            hi += span
        
        return (float(lo), float(hi))
    
    def get_lo_hi_for(self, which, Z):
        """Get range (lo, hi) for specified surface.
        
        Respects auto/manual mode and linked setting.
        
        Args:
            which (str): 'ref' or 'adj'.
            Z (np.ndarray): 2D height data for fallback calculation.
            
        Returns:
            tuple: (lo, hi) range values.
        """
        if which == 'ref':
            if self.range_ref_auto:
                return self.compute_auto_lo_hi(Z)
            else:
                lo, hi = self.range_ref
                if lo is None or hi is None:
                    return self.compute_auto_lo_hi(Z)
                return (lo, hi)
        else:  # 'adj'
            # If linked, use ref range
            if self.range_linked:
                if self.range_ref_auto:
                    # Need to compute from ref data
                    if self._ref_last is not None:
                        _, _, Z_ref = self._ref_last
                        return self.compute_auto_lo_hi(Z_ref)
                    else:
                        return self.compute_auto_lo_hi(Z)
                else:
                    lo, hi = self.range_ref
                    if lo is None or hi is None:
                        return self.compute_auto_lo_hi(Z)
                    return (lo, hi)
            
            # Not linked - use adj-specific range
            if self.range_adj_auto:
                return self.compute_auto_lo_hi(Z)
            else:
                lo, hi = self.range_adj
                if lo is None or hi is None:
                    return self.compute_auto_lo_hi(Z)
                return (lo, hi)
    
    def update_range_widgets(self, which, lo, hi, auto=False):
        """Update spinbox widgets for specified surface.
        
        Args:
            which (str): 'ref' or 'adj'.
            lo (float): Low value.
            hi (float): High value.
            auto (bool): Whether currently in auto mode.
        """
        if which == 'ref':
            if self.spin_lo_ref and self.spin_hi_ref:
                self.spin_lo_ref.blockSignals(True)
                self.spin_hi_ref.blockSignals(True)
                self.spin_lo_ref.setValue(lo)
                self.spin_hi_ref.setValue(hi)
                self.spin_lo_ref.setEnabled(not auto)
                self.spin_hi_ref.setEnabled(not auto)
                self.spin_lo_ref.blockSignals(False)
                self.spin_hi_ref.blockSignals(False)
        else:  # 'adj'
            if self.spin_lo_adj and self.spin_hi_adj:
                # Disabled if linked OR auto
                disabled = self.range_linked or auto
                self.spin_lo_adj.blockSignals(True)
                self.spin_hi_adj.blockSignals(True)
                self.spin_lo_adj.setValue(lo)
                self.spin_hi_adj.setValue(hi)
                self.spin_lo_adj.setEnabled(not disabled)
                self.spin_hi_adj.setEnabled(not disabled)
                self.spin_lo_adj.blockSignals(False)
                self.spin_hi_adj.blockSignals(False)
    
    def ui_link_toggled(self, on):
        """Handle link checkbox toggle.
        
        Args:
            on (bool): New state of link checkbox.
            
        Returns:
            bool: True if refresh needed.
        """
        self.range_linked = bool(on)
        # Update widget states
        if self.spin_lo_ref and self.spin_hi_ref:
            self.update_range_widgets('ref',
                self.spin_lo_ref.value(), self.spin_hi_ref.value(), 
                self.chk_auto_ref.isChecked() if self.chk_auto_ref else False)
        if self.spin_lo_adj and self.spin_hi_adj:
            self.update_range_widgets('adj',
                self.spin_lo_adj.value(), self.spin_hi_adj.value(),
                self.chk_auto_adj.isChecked() if self.chk_auto_adj else False)
        return True  # Needs refresh
    
    def ui_auto_ref_toggled(self, on):
        """Handle auto-ref checkbox toggle.
        
        Args:
            on (bool): New state of auto checkbox.
            
        Returns:
            bool: True if refresh needed.
        """
        self.range_ref_auto = bool(on)
        if self._ref_last is not None and on:
            _, _, Z = self._ref_last
            lo, hi = self.compute_auto_lo_hi(Z)
            self.update_range_widgets('ref', lo, hi, auto=True)
        else:
            if self.spin_lo_ref and self.spin_hi_ref:
                self.spin_lo_ref.setEnabled(True)
                self.spin_hi_ref.setEnabled(True)
        return True  # Needs refresh
    
    def ui_auto_adj_toggled(self, on):
        """Handle auto-adj checkbox toggle.
        
        Args:
            on (bool): New state of auto checkbox.
            
        Returns:
            bool: True if refresh needed.
        """
        self.range_adj_auto = bool(on)
        if self._adj_last is not None and on:
            _, _, Z = self._adj_last
            lo, hi = self.compute_auto_lo_hi(Z)
            self.update_range_widgets('adj', lo, hi, auto=True)
        else:
            if self.spin_lo_adj and self.spin_hi_adj:
                # May be overridden by "link"
                self.spin_lo_adj.setEnabled(not (self.range_linked or on))
                self.spin_hi_adj.setEnabled(not (self.range_linked or on))
        return True  # Needs refresh
    
    def ui_lohi_changed(self, which):
        """Handle manual range spinbox change.
        
        Args:
            which (str): 'ref' or 'adj'.
            
        Returns:
            bool: True if refresh needed.
        """
        if which == 'ref':
            if self.spin_lo_ref and self.spin_hi_ref:
                self.range_ref = (self.spin_lo_ref.value(), self.spin_hi_ref.value())
                if self.range_linked:
                    # Visually update adj widgets
                    self.update_range_widgets('adj', *self.range_ref, 
                        auto=self.range_adj_auto)
        else:
            if self.spin_lo_adj and self.spin_hi_adj:
                self.range_adj = (self.spin_lo_adj.value(), self.spin_hi_adj.value())
        return True  # Needs refresh
