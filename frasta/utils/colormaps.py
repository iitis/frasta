"""Custom colormap helpers used across 2D and 3D viewers."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui


# Sampled from the metrology-style palette used in the EFS toolbox:
# dark navy -> blue -> cyan -> green -> yellow -> orange -> red -> pink -> white.
_METROLOGY_POS = np.array([
    0.00,
    0.08,
    0.20,
    0.32,
    0.45,
    0.58,
    0.70,
    0.80,
    0.88,
    0.94,
    1.00,
], dtype=float)

_METROLOGY_RGB = np.array([
    (0, 0, 15),
    (0, 0, 219),
    (0, 126, 248),
    (1, 248, 231),
    (80, 248, 76),
    (196, 248, 0),
    (248, 218, 0),
    (248, 154, 0),
    (248, 85, 0),
    (248, 162, 128),
    (251, 251, 251),
], dtype=np.ubyte)

_DIFFERENCE_POS = np.array([0.0, 0.5, 1.0], dtype=float)

_DIFFERENCE_RGBA = np.array([
    (49, 54, 149, 255),
    (255, 255, 255, 255),
    (165, 0, 38, 255),
], dtype=np.ubyte)


def get_colormap(name: str | None):
    """Return a pyqtgraph ColorMap, including local custom palettes."""
    if name is None:
        return None
    if name.lower() == "metrology":
        return pg.ColorMap(_METROLOGY_POS, _METROLOGY_RGB)
    if name.lower() == "difference":
        return pg.ColorMap(_DIFFERENCE_POS, _DIFFERENCE_RGBA)
    return pg.colormap.get(name)


def get_lookup_table(name: str, n: int = 256) -> np.ndarray:
    """Return a lookup table for 2D image display."""
    cmap = get_colormap(name)
    if cmap is None:
        raise ValueError("Colormap name must not be None")
    return cmap.getLookupTable(0.0, 1.0, n)


def get_gradient_brush(name: str | None):
    """Return a horizontal gradient brush matching the selected colormap."""
    if name is None or str(name).lower() in ("gray", "grey", "none"):
        return pg.mkBrush(150, 150, 150, 150)

    cmap = get_colormap(str(name))
    if cmap is None:
        return pg.mkBrush(150, 150, 150, 150)

    gradient = QtGui.QLinearGradient(0.0, 0.0, 1.0, 0.0)
    gradient.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
    positions, colors = cmap.getStops(mode="byte")
    stops = []
    for pos, color in zip(positions, colors):
        rgba = tuple(int(channel) for channel in color)
        stops.append((float(pos), QtGui.QColor(*rgba)))
    gradient.setStops(stops)
    return QtGui.QBrush(gradient)


def get_brushes_for_values(name: str | None, values, alpha: int = 220):
    """Return per-value brushes sampled from the selected colormap.

    Args:
        name (str | None): Colormap name. ``Gray`` and ``None`` return
            a neutral brush repeated for every value.
        values (array-like): Normalized values in the ``[0, 1]`` interval.
        alpha (int): Opacity channel applied to RGB colormaps.

    Returns:
        list: Sequence of ``QBrush`` instances suitable for ``BarGraphItem``.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []

    if name is None or str(name).lower() in ("gray", "grey", "none"):
        return [pg.mkBrush(150, 150, 150, 180) for _ in range(values.size)]

    cmap = get_colormap(str(name))
    if cmap is None:
        return [pg.mkBrush(150, 150, 150, 180) for _ in range(values.size)]

    colors = cmap.map(np.clip(values, 0.0, 1.0), mode="byte")
    brushes = []
    for color in np.atleast_2d(colors):
        if len(color) >= 4:
            rgba = tuple(int(channel) for channel in color[:4])
        else:
            rgba = (int(color[0]), int(color[1]), int(color[2]), int(alpha))
        brushes.append(pg.mkBrush(*rgba))
    return brushes
