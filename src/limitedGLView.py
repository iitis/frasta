import numpy as np
from pyqtgraph.opengl import GLViewWidget

class LimitedGLView(GLViewWidget):
    def __init__(self, azimuth_range=None, elevation_range=(-90, 90),
                 wrap_azimuth=False, *args, **kwargs):
        """
        azimuth_range: (min,max) lub None - ogranicza/yaw (°)
        elevation_range: (min,max) lub None - ogranicza/pitch (°)
        wrap_azimuth: True -> azymut zawijany w zakresie (min,max)
        """
        super().__init__(*args, **kwargs)
        self.az_range = azimuth_range
        self.el_range = elevation_range
        self.wrap_az = wrap_azimuth

    def orbit(self, azim, elev):
        if self.az_range and self.az_range[0] == self.az_range[1]:
            # Zakres zerowy, nie ma sensu zmieniac azimuth
            self.update()
        else:
            # NIE wołamy super().orbit(), bo ono klipuje elev do [-90, 90]
            az = self.opts['azimuth']   + float(azim)
            el = self.opts['elevation'] + float(elev)

            # ograniczenia
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

    # helper do ustawiania z kodu
    def setRotationRanges(self, azimuth_range=None, elevation_range=None, wrap_azimuth=None):
        if azimuth_range is not None:
            self.az_range = azimuth_range
        if elevation_range is not None:
            self.el_range = elevation_range
        if wrap_azimuth is not None:
            self.wrap_az = wrap_azimuth
