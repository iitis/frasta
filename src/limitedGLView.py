"""3D view widget with constrained camera rotation.

This module provides a GLViewWidget subclass that restricts camera movement
to specified azimuth and elevation ranges, useful for limiting user interaction
in 3D visualizations.
"""

import numpy as np
from pyqtgraph.opengl import GLViewWidget

class LimitedGLView(GLViewWidget):
    """OpenGL view widget with constrained camera rotation angles.
    
    Extends GLViewWidget to limit rotation ranges for azimuth (yaw) and
    elevation (pitch), preventing users from rotating the view beyond
    specified boundaries.
    
    Attributes:
        az_range (tuple or None): (min, max) azimuth range in degrees.
        el_range (tuple or None): (min, max) elevation range in degrees.
        wrap_az (bool): If True, azimuth wraps around within range.
    """
    
    def __init__(self, azimuth_range=None, elevation_range=(-90, 90),
                 wrap_azimuth=False, *args, **kwargs):
        """Initialize the limited GL view with rotation constraints.
        
        Args:
            azimuth_range (tuple, optional): (min, max) azimuth limits in degrees.
                None means no limit. Defaults to None.
            elevation_range (tuple, optional): (min, max) elevation limits in degrees.
                None means no limit. Defaults to (-90, 90).
            wrap_azimuth (bool, optional): If True, azimuth wraps within range.
                Defaults to False.
            *args: Additional positional arguments for GLViewWidget.
            **kwargs: Additional keyword arguments for GLViewWidget.
        """
        super().__init__(*args, **kwargs)
        self.az_range = azimuth_range
        self.el_range = elevation_range
        self.wrap_az = wrap_azimuth

    def orbit(self, azim, elev):
        """Rotate the camera view with constrained angles.
        
        Applies azimuth and elevation changes while respecting configured limits.
        
        Args:
            azim (float): Azimuth change in degrees.
            elev (float): Elevation change in degrees.
        """
        if self.az_range and self.az_range[0] == self.az_range[1]:
            # Zero range, no point changing azimuth
            self.update()
        else:
            # Don't call super().orbit() as it clips elev to [-90, 90]
            az = self.opts['azimuth']   + float(azim)
            el = self.opts['elevation'] + float(elev)

            # Apply constraints
            if self.az_range is not None:
                amin, amax = self.az_range
                if self.wrap_az:
                    span = (amax - amin)
                    if span != 0:
                        az = ((az - amin) % span) + amin
                else:
                    az = max(amin, min(amax, az))

            if self.el_range is not None:
                emin, emax = self.el_range
                el = max(emin, min(emax, el))

            self.opts['azimuth'] = az
            self.opts['elevation'] = el
            self.update()

    def setRotationRanges(self, azimuth_range=None, elevation_range=None, wrap_azimuth=None):
        """Set or update rotation range constraints.
        
        Args:
            azimuth_range (tuple, optional): New (min, max) azimuth range.
            elevation_range (tuple, optional): New (min, max) elevation range.
            wrap_azimuth (bool, optional): New azimuth wrapping setting.
        """
        if azimuth_range is not None:
            self.az_range = azimuth_range
        if elevation_range is not None:
            self.el_range = elevation_range
        if wrap_azimuth is not None:
            self.wrap_az = wrap_azimuth
