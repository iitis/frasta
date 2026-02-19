import numpy as np

class Surface:
    """
    Unified 2D surface data model.
    
    Represents gridded surface data together with physical spacing,
    optional mask, visualization limits, and metadata.
    """

    def __init__(
        self,
        height,
        dx,
        dy,
        x0=0.0,
        y0=0.0,
        mask=None,
        unit="µm",
        metadata=None,
        vmin=None,
        vmax=None,
    ):
        self.height = np.asarray(height, dtype=float)
        self.dx = dx
        self.dy = dy
        self.x0 = x0  # Origin/offset for X coordinates
        self.y0 = y0  # Origin/offset for Y coordinates

        self.mask = mask if mask is not None else ~np.isnan(self.height)

        self.unit = unit
        self.metadata = metadata or {}

        # visualization-related (previously GridData)
        self.vmin = vmin
        self.vmax = vmax

    # ---------------------------
    # Basic properties
    # ---------------------------

    @property
    def shape(self):
        return self.height.shape

    @property
    def ny(self):
        return self.height.shape[0]

    @property
    def nx(self):
        return self.height.shape[1]

    @property
    def length(self):
        return (self.ny - 1) * self.dy

    @property
    def width(self):
        return (self.nx - 1) * self.dx

    # ---------------------------
    # Coordinate arrays (xi, yi)
    # replaces GridData.xi, yi
    # ---------------------------

    @property
    def xi(self):
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def yi(self):
        return self.y0 + np.arange(self.ny) * self.dy

    # ---------------------------
    # Utilities
    # ---------------------------

    def copy(self):
        return Surface(
            self.height.copy(),
            self.dx,
            self.dy,
            self.x0,
            self.y0,
            self.mask.copy(),
            self.unit,
            self.metadata.copy(),
            self.vmin,
            self.vmax,
        )

    def crop(self, ny, nx):
        return Surface(
            self.height[:ny, :nx],
            self.dx,
            self.dy,
            self.x0,  # Keep the same origin
            self.y0,
            self.mask[:ny, :nx],
            self.unit,
            self.metadata.copy(),
            self.vmin,
            self.vmax,
        )
