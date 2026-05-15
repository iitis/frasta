# Surface data model
from enum import Enum
from functools import cached_property

import numpy as np


class SurfaceOrientation(str, Enum):
    """Supported presentation orientations for one loaded surface.

    The enum describes how scan-space axes are interpreted before any GUI
    adapter maps them into 2D or 3D views. Only the default orientation is
    currently active, but explicit enum values make future import-time or
    per-scan orientation correction type-safe.
    """

    DEFAULT = "default"


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
        orientation: SurfaceOrientation | str = SurfaceOrientation.DEFAULT,
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
        self.orientation = (
            orientation
            if isinstance(orientation, SurfaceOrientation)
            else SurfaceOrientation(str(orientation).strip() or SurfaceOrientation.DEFAULT.value)
        )

    # ---------------------------
    # Basic properties
    # ---------------------------

    @property
    def shape(self):
        return self.height.shape

    @property
    def ny(self):
        """Number of data points in Y (rows)."""
        return self.height.shape[0]

    @property
    def nx(self):
        """Number of data points in X (columns)."""
        return self.height.shape[1]

    @property
    def length(self):
        """Physical length in Y direction."""
        return (self.ny - 1) * self.dy

    @property
    def width(self):
        """Physical width in X direction."""
        return (self.nx - 1) * self.dx

    # ---------------------------
    # Coordinate arrays (xi, yi)
    # replaces GridData.xi, yi
    # ---------------------------

    @cached_property
    def xi(self):
        """X coordinates array (cached)."""
        return self.x0 + np.arange(self.nx) * self.dx

    @cached_property
    def yi(self):
        """Y coordinates array (cached)."""
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
            self.orientation,
        )

    def crop(self, ny, nx):
        """Crop surface to ny rows and nx columns, keeping the same origin."""
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
            self.orientation,
        )
