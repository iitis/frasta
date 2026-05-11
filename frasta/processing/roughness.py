"""Minimal roughness parameters for surfaces and profiles.

The functions in this module implement a deliberately small subset of amplitude
parameters commonly used to summarize surface and profile roughness. They are
intended as lightweight processing helpers and do not replace a full ISO
surface-metrology workflow with standardized filtering and evaluation lengths.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_valid_surface_values(surface_or_grid: Any, mask=None) -> np.ndarray:
    """Return finite surface heights selected by an optional mask."""
    height = getattr(surface_or_grid, "height", surface_or_grid)
    values = np.asarray(height, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"Surface roughness requires a 2D array, got {values.ndim}D")

    if mask is None:
        mask = getattr(surface_or_grid, "mask", None)

    valid = np.isfinite(values)
    if mask is not None:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != values.shape:
            raise ValueError(
                f"Surface mask shape {mask_values.shape} does not match data shape {values.shape}"
            )
        valid &= mask_values

    selected = values[valid]
    if selected.size == 0:
        raise ValueError("Surface roughness requires at least one valid point")

    return selected


def _as_valid_profile_values(profile: Any) -> np.ndarray:
    """Return finite profile heights from a 1D array or profile-like object."""
    height = getattr(profile, "height", profile)
    values = np.asarray(height, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"Profile roughness requires a 1D array, got {values.ndim}D")

    selected = values[np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("Profile roughness requires at least one valid point")

    return selected


def _center(values: np.ndarray) -> np.ndarray:
    """Subtract the arithmetic mean from valid height values."""
    return values - float(np.mean(values))


def surface_roughness_parameters(surface_or_grid: Any, mask=None) -> dict[str, float]:
    """Compute minimal amplitude parameters for a gridded surface.

    Args:
        surface_or_grid: A ``Surface``-like object with ``height`` and optional
            ``mask`` attributes, or a 2D NumPy-compatible array.
        mask: Optional boolean mask selecting the evaluated region. When not
            provided, ``surface_or_grid.mask`` is used if available.

    Returns:
        Dictionary with ``Sa``, ``Sq``, and ``Sz``. Values are reported in the
        same unit as the input data.
    """
    z = _center(_as_valid_surface_values(surface_or_grid, mask=mask))

    sa = float(np.mean(np.abs(z)))
    sq = float(np.sqrt(np.mean(z ** 2)))
    sz = float(np.max(z) - np.min(z))

    return {
        "Sa": sa,
        "Sq": sq,
        "Sz": sz,
    }


def profile_roughness_parameters(profile: Any) -> dict[str, float]:
    """Compute minimal amplitude parameters for a one-dimensional profile.

    Args:
        profile: A profile-like object with a ``height`` attribute, or a 1D
            NumPy-compatible array. Non-finite values are ignored.

    Returns:
        Dictionary with ``Ra``, ``Rq``, and ``Rz``. Values are reported in the
        same unit as the input data. ``Rz`` is computed as the mean of the five
        highest points minus the mean of the five lowest points; for profiles
        with fewer than ten valid points it falls back to the total profile
        height.
    """
    z = _center(_as_valid_profile_values(profile))

    ra = float(np.mean(np.abs(z)))
    rq = float(np.sqrt(np.mean(z ** 2)))
    rt = float(np.max(z) - np.min(z))

    if z.size < 10:
        rz = rt
    else:
        sorted_z = np.sort(z)
        rz = float(np.mean(sorted_z[-5:]) - np.mean(sorted_z[:5]))

    return {
        "Ra": ra,
        "Rq": rq,
        "Rz": rz,
    }
