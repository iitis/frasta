"""Shared surface data model for scan I/O and numerical tooling."""

from enum import Enum
from functools import cached_property

import numpy as np


class SurfaceOrientation(str, Enum):
    """Supported presentation orientations for one loaded surface.

    The enum describes how scan-space axes are interpreted before any viewer
    or downstream application maps them into 2D or 3D coordinates.
    """

    DEFAULT = "default"


class Surface:
    """Unified 2D surface data model.

    Represents gridded surface data together with physical spacing, optional
    mask, visualization limits, and lightweight metadata.
    """

    def __init__(
        self,
        height,
        dx,
        dy,
        x0=0.0,
        y0=0.0,
        mask=None,
        unit="um",
        metadata=None,
        vmin=None,
        vmax=None,
        orientation: SurfaceOrientation | str = SurfaceOrientation.DEFAULT,
    ):
        self.height = np.asarray(height, dtype=float)
        self.dx = dx
        self.dy = dy
        self.x0 = x0
        self.y0 = y0
        self.mask = mask if mask is not None else ~np.isnan(self.height)
        self.unit = unit
        self.metadata = metadata or {}
        self.vmin = vmin
        self.vmax = vmax
        self.orientation = (
            orientation
            if isinstance(orientation, SurfaceOrientation)
            else SurfaceOrientation(str(orientation).strip() or SurfaceOrientation.DEFAULT.value)
        )

    @property
    def shape(self):
        """Return the grid shape as ``(ny, nx)``."""

        return self.height.shape

    @property
    def ny(self):
        """Number of rows in the height map."""

        return self.height.shape[0]

    @property
    def nx(self):
        """Number of columns in the height map."""

        return self.height.shape[1]

    @property
    def length(self):
        """Physical length in the Y direction."""

        return (self.ny - 1) * self.dy

    @property
    def width(self):
        """Physical width in the X direction."""

        return (self.nx - 1) * self.dx

    @cached_property
    def xi(self):
        """Return cached X coordinates."""

        return self.x0 + np.arange(self.nx) * self.dx

    @cached_property
    def yi(self):
        """Return cached Y coordinates."""

        return self.y0 + np.arange(self.ny) * self.dy

    def copy(self):
        """Return a deep copy of the surface."""

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
        """Crop surface to ``ny`` rows and ``nx`` columns."""

        return Surface(
            self.height[:ny, :nx],
            self.dx,
            self.dy,
            self.x0,
            self.y0,
            self.mask[:ny, :nx],
            self.unit,
            self.metadata.copy(),
            self.vmin,
            self.vmax,
            self.orientation,
        )
