"""Shared orientation helpers for 2D and 3D scan presentation.

This module defines the single presentation convention used across the GUI.
The underlying height grid always remains indexed as ``grid[row, col]``.
Renderers and interaction layers should convert through these helpers instead
of applying local ad-hoc flips or transpositions.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore

from ..core import SurfaceOrientation

DEFAULT_SURFACE_ORIENTATION = SurfaceOrientation.DEFAULT
SUPPORTED_SURFACE_ORIENTATIONS = {orientation.value for orientation in SurfaceOrientation}


def normalize_surface_orientation(orientation: SurfaceOrientation | str | None) -> str:
    """Return a supported orientation identifier.

    Args:
        orientation: Optional stored orientation value.

    Returns:
        One supported orientation string.

    Raises:
        ValueError: If the orientation is unknown.
    """
    if orientation is None:
        return DEFAULT_SURFACE_ORIENTATION.value
    if isinstance(orientation, SurfaceOrientation):
        return orientation.value
    if not isinstance(orientation, str):
        return DEFAULT_SURFACE_ORIENTATION.value
    normalized = str(orientation).strip() or DEFAULT_SURFACE_ORIENTATION.value
    if normalized not in SUPPORTED_SURFACE_ORIENTATIONS:
        raise ValueError(f"Unsupported surface orientation: {normalized!r}")
    return normalized


def grid_to_image_data(grid: np.ndarray, orientation: str | None = None, copy: bool = True) -> np.ndarray:
    """Convert a grid into the shared 2D image orientation used by pyqtgraph.

    Args:
        grid: Height grid stored as ``grid[row, col]``.
        orientation: Optional stored surface orientation.
        copy: If True, return a detached array copy.

    Returns:
        Image-ready array in the shared display orientation.
    """
    normalize_surface_orientation(orientation)
    image = np.asarray(grid).T
    return image.copy() if copy else image


def build_image_rect(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str | None = None,
) -> QtCore.QRectF:
    """Build the physical image rectangle for the shared 2D view convention."""
    normalize_surface_orientation(orientation)
    rows, cols = shape
    width = cols * dx
    height = rows * dy
    return QtCore.QRectF(x0 - dx / 2.0, y0 - dy / 2.0, width, height)


def physical_to_indices(
    x_phys: float,
    y_phys: float,
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str | None = None,
) -> tuple[int, int]:
    """Convert physical coordinates to nearest ``(col, row)`` indices."""
    normalize_surface_orientation(orientation)
    col = int(round((x_phys - x0) / dx))
    row = int(round((y_phys - y0) / dy))
    return col, row


def indices_to_physical(
    col: int,
    row: int,
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str | None = None,
) -> tuple[float, float]:
    """Convert ``(col, row)`` grid indices to physical coordinates."""
    normalize_surface_orientation(orientation)
    return x0 + col * dx, y0 + row * dy


def index_to_3d_world(
    col: int | float,
    row: int | float,
    z: float,
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str | None = None,
) -> tuple[float, float, float]:
    """Convert one grid sample into the shared 3D world convention."""
    x_phys, y_phys = indices_to_physical(
        int(col) if isinstance(col, (int, np.integer)) else col,
        int(row) if isinstance(row, (int, np.integer)) else row,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        orientation=orientation,
    )
    return float(x_phys), float(-y_phys), float(z)


def points_to_3d_world(
    cols: np.ndarray,
    rows: np.ndarray,
    z_values: np.ndarray,
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    orientation: str | None = None,
) -> np.ndarray:
    """Convert arrays of grid samples into 3D world positions."""
    normalize_surface_orientation(orientation)
    cols_f = np.asarray(cols, dtype=np.float32)
    rows_f = np.asarray(rows, dtype=np.float32)
    z_f = np.asarray(z_values, dtype=np.float32)
    xs = x0 + cols_f * float(dx)
    ys = -(y0 + rows_f * float(dy))
    return np.column_stack((xs, ys, z_f)).astype(np.float32, copy=False)
