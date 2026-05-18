"""Crack-path extraction and tortuosity metrics for aligned surface pairs.

This module provides a lightweight first iteration of crack-path analysis for
FRASTA workflows. The current implementation focuses on a simple and
deterministic front definition that is easy to validate on synthetic data:

- compute the difference map between two aligned height maps,
- classify pixels as open for a user-selected separation threshold,
- extract a crack front as the first open pixel along the transverse axis,
- compute effective path length, projected length, tortuosity, and curvature.

The API is intentionally modular so later implementations can add more advanced
front definitions such as skeletonization or maximum-opening trajectories
without breaking the higher-level analysis flow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import find_contours


def _as_height_array(surface_or_grid: Any) -> np.ndarray:
    """Return a floating-point height array from a ``Surface``-like object."""
    height = getattr(surface_or_grid, "height", surface_or_grid)
    values = np.asarray(height, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"Crack-path analysis requires a 2D array, got {values.ndim}D")

    return values


def _validate_matching_shapes(reference: np.ndarray, adjusted: np.ndarray) -> None:
    """Reject surface pairs that are not sampled on the same grid."""
    if reference.shape != adjusted.shape:
        raise ValueError(
            f"Crack-path analysis requires matching grid shapes, got "
            f"{reference.shape} and {adjusted.shape}"
        )


def _normalize_axis_name(propagation_axis: str) -> str:
    """Normalize the nominal propagation axis name."""
    axis = str(propagation_axis).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("Propagation axis must be 'x' or 'y'")
    return axis


def _normalize_front_side(front_side: str) -> str:
    """Normalize the side from which the front is detected."""
    side = str(front_side).strip().lower()
    if side not in {"min", "max"}:
        raise ValueError("Front side must be 'min' or 'max'")
    return side


def _normalize_path_method(method: str) -> str:
    """Normalize the crack-path extraction method name."""
    value = str(method).strip().lower()
    if value not in {"first_open_pixel", "contour"}:
        raise ValueError("Crack-path method must be 'first_open_pixel' or 'contour'")
    return value


def _normalize_positive_step(step: float | None, fallback: float) -> float:
    """Return a positive resampling step, using *fallback* when unspecified."""
    if step is None:
        return float(fallback)
    value = float(step)
    if value <= 0.0:
        raise ValueError("Contour resampling step must be positive")
    return value


def _normalize_smoothing_window(window: int) -> int:
    """Return an odd smoothing-window length >= 1."""
    value = int(window)
    if value < 1:
        raise ValueError("Contour smoothing window must be at least 1")
    if value % 2 == 0:
        value += 1
    return value


def _deduplicate_consecutive_points(points: np.ndarray) -> np.ndarray:
    """Remove zero-length consecutive steps from a polyline."""
    if len(points) <= 1:
        return points

    deltas = np.diff(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(deltas, axis=1) > 0.0
    result = points[keep]

    if len(result) < 2:
        raise ValueError("Crack path requires at least two distinct points")

    return result


def _orient_points_along_axis(points: np.ndarray, propagation_axis: str) -> np.ndarray:
    """Orient a path so its propagation coordinate increases monotonically overall."""
    axis = _normalize_axis_name(propagation_axis)
    coord_index = 0 if axis == "x" else 1
    if points[-1, coord_index] < points[0, coord_index]:
        return np.asarray(points[::-1], dtype=float)
    return np.asarray(points, dtype=float)


def _collapse_points_by_axis(points: np.ndarray, propagation_axis: str) -> np.ndarray:
    """Reduce a contour path to one transverse value per propagation coordinate."""
    axis = _normalize_axis_name(propagation_axis)
    coord_index = 0 if axis == "x" else 1
    transverse_index = 1 - coord_index

    ordered = _orient_points_along_axis(points, axis)
    order = np.argsort(ordered[:, coord_index], kind="mergesort")
    sorted_points = ordered[order]

    unique_coords, inverse = np.unique(sorted_points[:, coord_index], return_inverse=True)
    collapsed = np.empty((len(unique_coords), 2), dtype=float)
    collapsed[:, coord_index] = unique_coords
    for idx, coord_value in enumerate(unique_coords):
        mask = inverse == idx
        collapsed[idx, transverse_index] = float(np.mean(sorted_points[mask, transverse_index]))

    return _deduplicate_consecutive_points(collapsed)


def _resample_path_by_axis(
    points: np.ndarray,
    propagation_axis: str,
    step: float,
) -> np.ndarray:
    """Resample a single-valued path to a constant propagation-axis step."""
    axis = _normalize_axis_name(propagation_axis)
    coord_index = 0 if axis == "x" else 1
    transverse_index = 1 - coord_index
    step = _normalize_positive_step(step, fallback=1.0)

    ordered = _collapse_points_by_axis(points, axis)
    coords = ordered[:, coord_index]
    transverse = ordered[:, transverse_index]
    if len(coords) < 2:
        raise ValueError("Contour resampling requires at least two ordered points")

    start = float(coords[0])
    stop = float(coords[-1])
    if stop <= start:
        raise ValueError("Contour resampling requires increasing propagation coordinates")

    count = max(2, int(np.floor((stop - start) / step + 0.5)) + 1)
    new_coords = start + np.arange(count, dtype=float) * step
    if new_coords[-1] < stop:
        new_coords = np.append(new_coords, stop)
    else:
        new_coords[-1] = stop
    new_transverse = np.interp(new_coords, coords, transverse)

    resampled = np.empty((len(new_coords), 2), dtype=float)
    resampled[:, coord_index] = new_coords
    resampled[:, transverse_index] = new_transverse
    return _deduplicate_consecutive_points(resampled)


def _smooth_path_transverse(
    points: np.ndarray,
    propagation_axis: str,
    window: int,
) -> np.ndarray:
    """Smooth only the transverse coordinate of an ordered path."""
    axis = _normalize_axis_name(propagation_axis)
    coord_index = 0 if axis == "x" else 1
    transverse_index = 1 - coord_index
    window = _normalize_smoothing_window(window)

    if window == 1 or len(points) < 3:
        return np.asarray(points, dtype=float)

    half = window // 2
    transverse = np.asarray(points[:, transverse_index], dtype=float)
    padded = np.pad(transverse, (half, half), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=float)
    smoothed_transverse = np.convolve(padded, kernel, mode="valid")

    smoothed = np.asarray(points, dtype=float).copy()
    smoothed[:, transverse_index] = smoothed_transverse
    return _deduplicate_consecutive_points(smoothed)


def crack_opening_map(
    reference_surface: Any,
    adjusted_surface: Any,
    separation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute difference, valid, and open-region maps for aligned surfaces.

    Args:
        reference_surface: Reference ``Surface``-like object or 2D array.
        adjusted_surface: Adjusted ``Surface``-like object or 2D array sampled
            on the same grid as ``reference_surface``.
        separation: Opening threshold in the same height unit as the grids.
            Pixels are classified as open when ``reference - adjusted >= separation``.

    Returns:
        Tuple ``(difference_map, open_mask, valid_mask)`` where ``difference_map``
        equals ``reference - adjusted``. Invalid pixels remain ``NaN`` in the
        difference map and are always ``False`` in ``open_mask``.
    """
    reference = _as_height_array(reference_surface)
    adjusted = _as_height_array(adjusted_surface)
    _validate_matching_shapes(reference, adjusted)

    difference = reference - adjusted
    valid_mask = np.isfinite(difference)
    open_mask = np.zeros_like(valid_mask, dtype=bool)
    open_mask[valid_mask] = difference[valid_mask] >= float(separation)

    return difference, open_mask, valid_mask


def extract_crack_path(
    open_mask: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    propagation_axis: str = "x",
    front_side: str = "min",
) -> np.ndarray:
    """Extract a simple crack-front polyline from a binary opening map.

    The current MVP front definition scans each line orthogonal to the nominal
    propagation axis and selects the first open pixel encountered from the
    chosen side. This produces one polyline point per sampled column or row.

    Args:
        open_mask: Boolean 2D array in which ``True`` marks open crack pixels.
        dx: Physical pixel spacing along the X axis.
        dy: Physical pixel spacing along the Y axis.
        propagation_axis: Nominal propagation direction, ``'x'`` or ``'y'``.
        front_side: Which side to scan from on the transverse axis, ``'min'``
            or ``'max'``.

    Returns:
        ``(N, 2)`` array of polyline points in physical coordinates ``[x, y]``.

    Raises:
        ValueError: If the mask is not 2D or does not contain a path with at
            least two points.
    """
    mask = np.asarray(open_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Open-mask crack path extraction requires a 2D array, got {mask.ndim}D")

    axis = _normalize_axis_name(propagation_axis)
    side = _normalize_front_side(front_side)

    points: list[tuple[float, float]] = []

    if axis == "x":
        for col in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, col])
            if rows.size == 0:
                continue
            row = int(rows[0] if side == "min" else rows[-1])
            points.append((float(col) * float(dx), float(row) * float(dy)))
    else:
        for row in range(mask.shape[0]):
            cols = np.flatnonzero(mask[row, :])
            if cols.size == 0:
                continue
            col = int(cols[0] if side == "min" else cols[-1])
            points.append((float(col) * float(dx), float(row) * float(dy)))

    if len(points) < 2:
        raise ValueError("Open-mask crack path extraction requires at least two sampled points")

    return _deduplicate_consecutive_points(np.asarray(points, dtype=float))


def extract_crack_path_contour(
    open_mask: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    propagation_axis: str = "x",
    resample_step: float | None = None,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Extract a crack-front polyline from the open/contact contour.

    The implementation traces the ``0.5`` level set of the binary open-region
    mask and selects the contour with the largest span along the nominal
    propagation axis. This provides a second, less axis-locked baseline method
    that follows the actual open/contact interface geometry more directly than
    the ``first_open_pixel`` scanline method.

    Args:
        open_mask: Boolean 2D array in which ``True`` marks open crack pixels.
        dx: Physical pixel spacing along the X axis.
        dy: Physical pixel spacing along the Y axis.
        propagation_axis: Nominal propagation direction, ``'x'`` or ``'y'``.
        resample_step: Optional resampling step along the propagation axis in
            physical units. When omitted, ``min(dx, dy)`` is used.
        smoothing_window: Odd moving-average window applied to the transverse
            coordinate after resampling. Even values are rounded up to the next
            odd integer.

    Returns:
        ``(N, 2)`` array of contour polyline points in physical coordinates
        ``[x, y]``.
    """
    mask = np.asarray(open_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Open-mask contour extraction requires a 2D array, got {mask.ndim}D")

    axis = _normalize_axis_name(propagation_axis)
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        raise ValueError("Open-mask contour extraction requires at least one contour")

    best_points: np.ndarray | None = None
    best_span = -np.inf
    best_length = -np.inf
    coord_index = 0 if axis == "x" else 1

    for contour in contours:
        if len(contour) < 2:
            continue
        points = np.column_stack(
            (
                contour[:, 1] * float(dx),
                contour[:, 0] * float(dy),
            )
        )
        points = _deduplicate_consecutive_points(points)
        span = float(np.max(points[:, coord_index]) - np.min(points[:, coord_index]))
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if span > best_span or (np.isclose(span, best_span) and length > best_length):
            best_points = points
            best_span = span
            best_length = length

    if best_points is None:
        raise ValueError("Open-mask contour extraction did not produce a valid polyline")

    ordered = _collapse_points_by_axis(best_points, axis)
    resampled = _resample_path_by_axis(
        ordered,
        axis,
        step=_normalize_positive_step(resample_step, fallback=min(float(dx), float(dy))),
    )
    return _smooth_path_transverse(
        resampled,
        axis,
        window=_normalize_smoothing_window(smoothing_window),
    )


def crack_path_tortuosity(
    path_points: np.ndarray,
    propagation_axis: str = "x",
) -> dict[str, float]:
    """Compute effective length, projected length, and tortuosity of a path.

    Args:
        path_points: ``(N, 2)`` array of physical polyline coordinates ``[x, y]``.
        propagation_axis: Nominal propagation direction used for projected
            length, ``'x'`` or ``'y'``.

    Returns:
        Dictionary with:
        ``effective_length``: arc length of the extracted path,
        ``projected_length``: span of the path along the propagation axis,
        ``tortuosity``: ratio ``effective_length / projected_length``.
    """
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Crack-path tortuosity requires an (N, 2) coordinate array")

    points = _deduplicate_consecutive_points(points)
    axis = _normalize_axis_name(propagation_axis)

    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    effective_length = float(np.sum(step_lengths))

    coord_index = 0 if axis == "x" else 1
    projected_length = float(np.max(points[:, coord_index]) - np.min(points[:, coord_index]))
    if projected_length <= 0.0:
        raise ValueError("Projected crack-path length must be positive to compute tortuosity")

    tortuosity = effective_length / projected_length
    return {
        "effective_length": effective_length,
        "projected_length": projected_length,
        "tortuosity": float(tortuosity),
    }


def crack_path_curvature(path_points: np.ndarray) -> dict[str, np.ndarray]:
    """Compute local tangent angle and curvature along a crack path.

    Args:
        path_points: ``(N, 2)`` array of physical polyline coordinates ``[x, y]``.

    Returns:
        Dictionary with:
        ``arc_length``: cumulative arc-length coordinate for each polyline point,
        ``tangent_angle``: unwrapped local tangent angle in radians,
        ``curvature``: local curvature estimate ``d(theta) / d(s)``.
    """
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Crack-path curvature requires an (N, 2) coordinate array")

    points = _deduplicate_consecutive_points(points)
    if len(points) < 3:
        raise ValueError("Crack-path curvature requires at least three distinct points")

    deltas = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    if np.any(step_lengths <= 0.0):
        raise ValueError("Crack-path curvature requires positive arc-length steps")

    arc_length = np.concatenate(([0.0], np.cumsum(step_lengths)))
    tangent_vectors = np.gradient(points, axis=0)
    tangent_angle = np.unwrap(np.arctan2(tangent_vectors[:, 1], tangent_vectors[:, 0]))
    curvature = np.gradient(tangent_angle, arc_length, edge_order=1)

    return {
        "arc_length": arc_length,
        "tangent_angle": tangent_angle,
        "curvature": curvature,
    }


def analyze_crack_path(
    reference_surface: Any,
    adjusted_surface: Any,
    dx: float = 1.0,
    dy: float = 1.0,
    separation: float = 0.0,
    propagation_axis: str = "x",
    front_side: str = "min",
    method: str = "first_open_pixel",
    contour_resample_step: float | None = None,
    contour_smoothing_window: int = 5,
) -> dict[str, Any]:
    """Run the complete MVP crack-path analysis on an aligned surface pair.

    Args:
        reference_surface: Reference ``Surface``-like object or 2D array.
        adjusted_surface: Adjusted ``Surface``-like object or 2D array.
        dx: Physical pixel spacing along X used to convert path coordinates.
        dy: Physical pixel spacing along Y used to convert path coordinates.
        separation: Opening threshold in the height unit of the surfaces.
        propagation_axis: Nominal propagation direction, ``'x'`` or ``'y'``.
        front_side: Side from which the front is detected on the transverse
            axis, ``'min'`` or ``'max'``.
        method: Crack-path extraction method. Supported values are
            ``'first_open_pixel'`` and ``'contour'``.
        contour_resample_step: Constant step along the propagation axis used by
            the contour method after ordering and collapsing the raw contour.
        contour_smoothing_window: Moving-average window for transverse
            smoothing used by the contour method.

    Returns:
        Dictionary containing the raw maps, extracted path, tortuosity
        summaries, and curvature arrays.
    """
    difference_map, open_mask, valid_mask = crack_opening_map(
        reference_surface,
        adjusted_surface,
        separation=separation,
    )
    path_method = _normalize_path_method(method)
    if path_method == "first_open_pixel":
        path_points = extract_crack_path(
            open_mask,
            dx=dx,
            dy=dy,
            propagation_axis=propagation_axis,
            front_side=front_side,
        )
    else:
        path_points = extract_crack_path_contour(
            open_mask,
            dx=dx,
            dy=dy,
            propagation_axis=propagation_axis,
            resample_step=contour_resample_step,
            smoothing_window=contour_smoothing_window,
        )
    tortuosity = crack_path_tortuosity(
        path_points,
        propagation_axis=propagation_axis,
    )
    curvature = crack_path_curvature(path_points)

    return {
        "difference_map": difference_map,
        "open_mask": open_mask,
        "valid_mask": valid_mask,
        "path_points": path_points,
        "path_method": path_method,
        "propagation_axis": _normalize_axis_name(propagation_axis),
        "front_side": _normalize_front_side(front_side),
        "contour_resample_step": (
            _normalize_positive_step(contour_resample_step, fallback=min(float(dx), float(dy)))
            if path_method == "contour"
            else None
        ),
        "contour_smoothing_window": (
            _normalize_smoothing_window(contour_smoothing_window)
            if path_method == "contour"
            else None
        ),
        **tortuosity,
        **curvature,
    }


def sweep_crack_path_thresholds(
    reference_surface: Any,
    adjusted_surface: Any,
    thresholds: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    propagation_axis: str = "x",
    front_side: str = "min",
    method: str = "first_open_pixel",
    contour_resample_step: float | None = None,
    contour_smoothing_window: int = 5,
) -> dict[str, np.ndarray | str | float | int | None]:
    """Evaluate crack-path metrics over a sequence of separation thresholds.

    Args:
        reference_surface: Reference ``Surface``-like object or 2D array.
        adjusted_surface: Adjusted ``Surface``-like object or 2D array.
        thresholds: One-dimensional array of thresholds to evaluate.
        dx: Physical pixel spacing along X used to convert path coordinates.
        dy: Physical pixel spacing along Y used to convert path coordinates.
        propagation_axis: Nominal propagation direction, ``'x'`` or ``'y'``.
        front_side: Side from which the first-open-pixel path is detected.
        method: Crack-path extraction method.
        contour_resample_step: Contour resampling step in physical units.
        contour_smoothing_window: Contour transverse smoothing window.

    Returns:
        Dictionary containing threshold values and per-threshold arrays for
        ``effective_length``, ``projected_length``, ``tortuosity``, and
        ``mean_abs_curvature``. Thresholds that do not produce a valid path are
        reported as ``NaN`` in the metric arrays.
    """
    values = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Threshold sweep requires a non-empty 1D threshold array")

    effective_length = np.full(values.shape, np.nan, dtype=float)
    projected_length = np.full(values.shape, np.nan, dtype=float)
    tortuosity = np.full(values.shape, np.nan, dtype=float)
    mean_abs_curvature = np.full(values.shape, np.nan, dtype=float)

    for idx, threshold in enumerate(values):
        try:
            result = analyze_crack_path(
                reference_surface,
                adjusted_surface,
                dx=dx,
                dy=dy,
                separation=float(threshold),
                propagation_axis=propagation_axis,
                front_side=front_side,
                method=method,
                contour_resample_step=contour_resample_step,
                contour_smoothing_window=contour_smoothing_window,
            )
        except ValueError:
            continue

        effective_length[idx] = float(result["effective_length"])
        projected_length[idx] = float(result["projected_length"])
        tortuosity[idx] = float(result["tortuosity"])
        mean_abs_curvature[idx] = float(np.mean(np.abs(np.asarray(result["curvature"], dtype=float))))

    return {
        "thresholds": values,
        "effective_length": effective_length,
        "projected_length": projected_length,
        "tortuosity": tortuosity,
        "mean_abs_curvature": mean_abs_curvature,
        "method": _normalize_path_method(method),
        "propagation_axis": _normalize_axis_name(propagation_axis),
        "front_side": _normalize_front_side(front_side),
        "contour_resample_step": (
            _normalize_positive_step(contour_resample_step, fallback=min(float(dx), float(dy)))
            if _normalize_path_method(method) == "contour"
            else None
        ),
        "contour_smoothing_window": (
            _normalize_smoothing_window(contour_smoothing_window)
            if _normalize_path_method(method) == "contour"
            else None
        ),
    }
