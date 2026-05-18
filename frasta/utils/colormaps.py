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


def remap_normalized_colormap_values(values, curve_strength: float = 0.0) -> np.ndarray:
    """Remap normalized colormap coordinates with an endpoint-stretch curve.

    The manual curve preserves ``0 -> 0`` and ``1 -> 1`` while stretching the
    low/high color regions and compressing the middle of the palette as the
    strength increases. This mirrors the metrology-style visual tuning often
    used to make endpoint colors span a larger value interval without changing
    the numeric data range itself.

    The mapping is implemented as a symmetric power response around ``0.5``.
    Compared with the previous hyperbolic tangent formulation, the power law
    yields a noticeably stronger endpoint stretch, which better matches the
    manual metrology-style tuning expected in the viewers.

    Args:
        values: Normalized coordinates in the ``[0, 1]`` interval.
        curve_strength: Non-negative response strength. ``0`` keeps linear
            mapping, larger values progressively stretch endpoint colors.

    Returns:
        ``numpy.ndarray`` of remapped normalized coordinates.
    """
    normalized = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    strength = max(0.0, float(curve_strength))
    if strength <= 1e-9:
        return normalized

    gamma = 1.0 / (1.0 + strength)
    centered = 2.0 * normalized - 1.0
    remapped = np.sign(centered) * np.power(np.abs(centered), gamma)
    return np.clip(0.5 + 0.5 * remapped, 0.0, 1.0)


def get_lookup_table(name: str, n: int = 256, curve_strength: float = 0.0) -> np.ndarray:
    """Return a lookup table for 2D image display.

    Args:
        name: Colormap name.
        n: Number of LUT entries.
        curve_strength: Non-negative endpoint-stretch response strength.
    """
    cmap = get_colormap(name)
    if cmap is None:
        raise ValueError("Colormap name must not be None")
    sample_points = remap_normalized_colormap_values(
        np.linspace(0.0, 1.0, max(2, int(n)), dtype=float),
        curve_strength=curve_strength,
    )
    return cmap.map(sample_points, mode="byte")


def get_gradient_stops(
    name: str | None,
    samples: int = 64,
    curve_strength: float = 0.0,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Return evenly sampled gradient stops for Qt gradients and exporters.

    Args:
        name: Colormap name. ``None`` falls back to grayscale.
        samples: Number of evenly spaced stop positions to generate.
        curve_strength: Non-negative endpoint-stretch response strength.

    Returns:
        List of ``(position, rgba)`` tuples with positions in ``[0, 1]``.
    """
    resolved_samples = max(2, int(samples))
    positions = np.linspace(0.0, 1.0, resolved_samples, dtype=float)
    if name is None or str(name).lower() in ("gray", "grey", "none"):
        rgba_table = np.zeros((resolved_samples, 4), dtype=np.uint8)
        gray = np.round(positions * 255.0).astype(np.uint8, copy=False)
        rgba_table[:, 0] = gray
        rgba_table[:, 1] = gray
        rgba_table[:, 2] = gray
        rgba_table[:, 3] = 255
    else:
        rgba_table = get_lookup_table(str(name), resolved_samples, curve_strength=curve_strength)
    return [
        (float(position), tuple(int(channel) for channel in rgba))
        for position, rgba in zip(positions, np.atleast_2d(rgba_table))
    ]


def get_gradient_brush(name: str | None, curve_strength: float = 0.0):
    """Return a horizontal gradient brush matching the selected colormap."""
    if name is None or str(name).lower() in ("gray", "grey", "none"):
        return pg.mkBrush(150, 150, 150, 150)

    stops = get_gradient_stops(name, samples=64, curve_strength=curve_strength)
    if not stops:
        return pg.mkBrush(150, 150, 150, 150)

    gradient = QtGui.QLinearGradient(0.0, 0.0, 1.0, 0.0)
    gradient.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
    gradient.setStops(
        [(float(pos), QtGui.QColor(*rgba)) for pos, rgba in stops]
    )
    return QtGui.QBrush(gradient)


def get_brushes_for_values(name: str | None, values, alpha: int = 220, curve_strength: float = 0.0):
    """Return per-value brushes sampled from the selected colormap.

    Args:
        name (str | None): Colormap name. ``Gray`` and ``None`` return
            a neutral brush repeated for every value.
        values (array-like): Normalized values in the ``[0, 1]`` interval.
        alpha (int): Opacity channel applied to RGB colormaps.
        curve_strength (float): Non-negative endpoint-stretch response strength.

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

    remapped_values = remap_normalized_colormap_values(values, curve_strength=curve_strength)
    colors = cmap.map(remapped_values, mode="byte")
    brushes = []
    for color in np.atleast_2d(colors):
        if len(color) >= 4:
            rgba = tuple(int(channel) for channel in color[:4])
        else:
            rgba = (int(color[0]), int(color[1]), int(color[2]), int(alpha))
        brushes.append(pg.mkBrush(*rgba))
    return brushes
