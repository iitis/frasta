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
    if axis not in {"x", "y", "angle"}:
        raise ValueError("Propagation axis must be 'x', 'y', or 'angle'")
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


def _resolve_propagation_angle(
    propagation_axis: str,
    propagation_angle_degrees: float | None = None,
) -> tuple[str, float]:
    """Return a normalized axis label and propagation angle in degrees."""
    axis = _normalize_axis_name(propagation_axis)
    if axis == "x":
        return axis, 0.0
    if axis == "y":
        return axis, 90.0
    if propagation_angle_degrees is None:
        raise ValueError("Propagation axis 'angle' requires propagation_angle_degrees")
    return axis, float(propagation_angle_degrees)


def _direction_basis(
    propagation_axis: str,
    propagation_angle_degrees: float | None = None,
) -> tuple[str, float, np.ndarray, np.ndarray]:
    """Return the propagation-angle basis vectors in world coordinates."""
    axis, angle_degrees = _resolve_propagation_angle(
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    angle_radians = np.radians(angle_degrees)
    u = np.array([np.cos(angle_radians), np.sin(angle_radians)], dtype=float)
    v = np.array([-np.sin(angle_radians), np.cos(angle_radians)], dtype=float)
    return axis, angle_degrees, u, v


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


def _project_points_to_uv(
    points: np.ndarray,
    propagation_axis: str,
    propagation_angle_degrees: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str, float, np.ndarray, np.ndarray]:
    """Project 2D points onto propagation and transverse coordinates."""
    axis, angle_degrees, u_vec, v_vec = _direction_basis(
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    u_coords = np.asarray(points, dtype=float) @ u_vec
    v_coords = np.asarray(points, dtype=float) @ v_vec
    return u_coords, v_coords, axis, angle_degrees, u_vec, v_vec


def _reconstruct_points_from_uv(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
) -> np.ndarray:
    """Map propagation/transverse coordinates back into XY points."""
    u_arr = np.asarray(u_coords, dtype=float).reshape(-1, 1)
    v_arr = np.asarray(v_coords, dtype=float).reshape(-1, 1)
    return u_arr * np.asarray(u_vec, dtype=float) + v_arr * np.asarray(v_vec, dtype=float)


def _orient_points_along_axis(
    points: np.ndarray,
    propagation_axis: str,
    propagation_angle_degrees: float | None = None,
) -> np.ndarray:
    """Orient a path so its propagation coordinate increases monotonically overall."""
    u_coords, _v_coords, _axis, _angle, _u_vec, _v_vec = _project_points_to_uv(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    if u_coords[-1] < u_coords[0]:
        return np.asarray(points[::-1], dtype=float)
    return np.asarray(points, dtype=float)


def _collapse_points_by_axis(
    points: np.ndarray,
    propagation_axis: str,
    propagation_angle_degrees: float | None = None,
    bin_step: float = 1.0,
) -> np.ndarray:
    """Reduce a path to one transverse value per propagation-coordinate bin."""
    ordered = _orient_points_along_axis(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    u_coords, v_coords, _axis, _angle, u_vec, v_vec = _project_points_to_uv(
        ordered,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    step = _normalize_positive_step(bin_step, fallback=1.0)
    order = np.argsort(u_coords, kind="mergesort")
    u_sorted = u_coords[order]
    v_sorted = v_coords[order]

    bin_indices = np.floor((u_sorted - float(u_sorted[0])) / step + 1e-9).astype(int)
    unique_bins = np.unique(bin_indices)
    collapsed_u = np.empty(len(unique_bins), dtype=float)
    collapsed_v = np.empty(len(unique_bins), dtype=float)
    for out_idx, bin_idx in enumerate(unique_bins):
        mask = bin_indices == bin_idx
        collapsed_u[out_idx] = float(np.mean(u_sorted[mask]))
        collapsed_v[out_idx] = float(np.mean(v_sorted[mask]))

    collapsed_points = _reconstruct_points_from_uv(collapsed_u, collapsed_v, u_vec, v_vec)
    return _deduplicate_consecutive_points(collapsed_points)


def _resample_path_by_axis(
    points: np.ndarray,
    propagation_axis: str,
    step: float,
    propagation_angle_degrees: float | None = None,
) -> np.ndarray:
    """Resample a single-valued path to a constant propagation-axis step."""
    step = _normalize_positive_step(step, fallback=1.0)

    ordered = _orient_points_along_axis(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    coords, transverse, _axis, _angle, u_vec, v_vec = _project_points_to_uv(
        ordered,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
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
    resampled = _reconstruct_points_from_uv(new_coords, new_transverse, u_vec, v_vec)
    return _deduplicate_consecutive_points(resampled)


def _smooth_path_transverse(
    points: np.ndarray,
    propagation_axis: str,
    window: int,
    propagation_angle_degrees: float | None = None,
) -> np.ndarray:
    """Smooth only the transverse coordinate of an ordered path."""
    window = _normalize_smoothing_window(window)

    if window == 1 or len(points) < 3:
        return np.asarray(points, dtype=float)

    ordered = _orient_points_along_axis(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    u_coords, transverse, _axis, _angle, u_vec, v_vec = _project_points_to_uv(
        ordered,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    half = window // 2
    padded = np.pad(transverse, (half, half), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=float)
    smoothed_transverse = np.convolve(padded, kernel, mode="valid")
    smoothed = _reconstruct_points_from_uv(u_coords, smoothed_transverse, u_vec, v_vec)
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
    propagation_angle_degrees: float | None = None,
) -> np.ndarray:
    """Extract a simple crack-front polyline from a binary opening map.

    The current MVP front definition scans each line orthogonal to the nominal
    propagation axis and selects the first open pixel encountered from the
    chosen side. This produces one polyline point per sampled column or row.

    Args:
        open_mask: Boolean 2D array in which ``True`` marks open crack pixels.
        dx: Physical pixel spacing along the X axis.
        dy: Physical pixel spacing along the Y axis.
        propagation_axis: Nominal propagation direction, ``'x'``, ``'y'``, or
            ``'angle'``.
        front_side: Which side to scan from on the transverse axis, ``'min'``
            or ``'max'``.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.

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

    if axis in {"x", "y"}:
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

    rows, cols = np.nonzero(mask)
    if rows.size < 2:
        raise ValueError("Open-mask crack path extraction requires at least two sampled points")

    raw_points = np.column_stack((cols.astype(float) * float(dx), rows.astype(float) * float(dy)))
    u_coords, v_coords, _axis, _angle, u_vec, v_vec = _project_points_to_uv(
        raw_points,
        axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )

    sample_step = min(float(dx), float(dy))
    u_start = float(np.min(u_coords))
    bin_indices = np.floor((u_coords - u_start) / sample_step + 1e-9).astype(int)
    unique_bins = np.unique(bin_indices)

    path_u = np.empty(len(unique_bins), dtype=float)
    path_v = np.empty(len(unique_bins), dtype=float)
    for out_idx, bin_idx in enumerate(unique_bins):
        mask_bin = bin_indices == bin_idx
        local_u = u_coords[mask_bin]
        local_v = v_coords[mask_bin]
        edge_index = int(np.argmin(local_v) if side == "min" else np.argmax(local_v))
        path_u[out_idx] = float(local_u[edge_index])
        path_v[out_idx] = float(local_v[edge_index])

    return _deduplicate_consecutive_points(
        _reconstruct_points_from_uv(path_u, path_v, u_vec, v_vec)
    )


def extract_crack_path_contour(
    open_mask: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    propagation_axis: str = "x",
    propagation_angle_degrees: float | None = None,
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
        propagation_axis: Nominal propagation direction, ``'x'``, ``'y'``, or
            ``'angle'``.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.
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
        u_coords, _v_coords, _axis, _angle, _u_vec, _v_vec = _project_points_to_uv(
            points,
            axis,
            propagation_angle_degrees=propagation_angle_degrees,
        )
        span = float(np.max(u_coords) - np.min(u_coords))
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if span > best_span or (np.isclose(span, best_span) and length > best_length):
            best_points = points
            best_span = span
            best_length = length

    if best_points is None:
        raise ValueError("Open-mask contour extraction did not produce a valid polyline")

    ordered = _collapse_points_by_axis(
        best_points,
        axis,
        propagation_angle_degrees=propagation_angle_degrees,
        bin_step=min(float(dx), float(dy)),
    )
    resampled = _resample_path_by_axis(
        ordered,
        axis,
        step=_normalize_positive_step(resample_step, fallback=min(float(dx), float(dy))),
        propagation_angle_degrees=propagation_angle_degrees,
    )
    return _smooth_path_transverse(
        resampled,
        axis,
        window=_normalize_smoothing_window(smoothing_window),
        propagation_angle_degrees=propagation_angle_degrees,
    )


def crack_path_tortuosity(
    path_points: np.ndarray,
    propagation_axis: str = "x",
    propagation_angle_degrees: float | None = None,
) -> dict[str, float]:
    """Compute effective length, projected length, and tortuosity of a path.

    Args:
        path_points: ``(N, 2)`` array of physical polyline coordinates ``[x, y]``.
        propagation_axis: Nominal propagation direction used for projected
            length, ``'x'``, ``'y'``, or ``'angle'``.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.

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
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    effective_length = float(np.sum(step_lengths))
    u_coords, _v_coords, _axis, _angle, _u_vec, _v_vec = _project_points_to_uv(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    projected_length = float(np.max(u_coords) - np.min(u_coords))
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


def crack_path_local_tortuosity(
    path_points: np.ndarray,
    propagation_axis: str = "x",
    window_length: float | None = None,
    propagation_angle_degrees: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute a sliding-window tortuosity profile along a crack path.

    Args:
        path_points: ``(N, 2)`` array of physical polyline coordinates ``[x, y]``.
        propagation_axis: Nominal propagation direction used for projected
            length, ``'x'``, ``'y'``, or ``'angle'``.
        window_length: Sliding-window length in physical units. When omitted,
            a default of one fifth of the total arc length is used, but never
            less than three median point spacings.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.

    Returns:
        Dictionary with:
        ``arc_length``: cumulative arc-length coordinate,
        ``local_tortuosity``: local tortuosity at each path point,
        ``window_length``: effective physical window length used.
    """
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Local crack-path tortuosity requires an (N, 2) coordinate array")

    points = _deduplicate_consecutive_points(points)
    deltas = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    if np.any(step_lengths <= 0.0):
        raise ValueError("Local crack-path tortuosity requires positive arc-length steps")

    arc_length = np.concatenate(([0.0], np.cumsum(step_lengths)))
    u_coords, _v_coords, _axis, _angle, _u_vec, _v_vec = _project_points_to_uv(
        points,
        propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    total_length = float(arc_length[-1])
    median_step = float(np.median(step_lengths))
    if window_length is None:
        effective_window = max(3.0 * median_step, total_length / 5.0)
    else:
        effective_window = float(window_length)
        if effective_window <= 0.0:
            raise ValueError("Local tortuosity window length must be positive")
        effective_window = max(effective_window, 3.0 * median_step)

    half_window = 0.5 * effective_window
    local_tortuosity = np.full(len(points), np.nan, dtype=float)

    for idx, center_s in enumerate(arc_length):
        left = max(0.0, center_s - half_window)
        right = min(total_length, center_s + half_window)
        mask = (arc_length >= left) & (arc_length <= right)
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            continue
        i0 = int(indices[0])
        i1 = int(indices[-1])
        effective = float(arc_length[i1] - arc_length[i0])
        projected = float(np.max(u_coords[i0 : i1 + 1]) - np.min(u_coords[i0 : i1 + 1]))
        if projected > 0.0:
            local_tortuosity[idx] = effective / projected

    return {
        "arc_length": arc_length,
        "local_tortuosity": local_tortuosity,
        "window_length": float(effective_window),
    }


def crack_path_orientation_statistics(
    path_points: np.ndarray,
    reference_angle_degrees: float | None = None,
) -> dict[str, np.ndarray | float | None]:
    """Compute tangent-orientation statistics along a crack path.

    Args:
        path_points: ``(N, 2)`` array of physical polyline coordinates ``[x, y]``.
        reference_angle_degrees: Optional external direction for alignment
            comparison, such as build or hatch direction.

    Returns:
        Dictionary with:
        ``arc_length``: cumulative arc-length coordinate,
        ``tangent_angle_degrees``: local tangent angles in degrees,
        ``orientation_degrees``: axial orientation wrapped to ``[0, 180)``,
        ``dominant_orientation_degrees``: axial circular-mean orientation,
        ``orientation_strength``: axial mean-resultant length in ``[0, 1]``,
        ``alignment_delta_degrees``: minimal absolute difference to the
        reference angle when provided.
    """
    curvature = crack_path_curvature(path_points)
    tangent_angle = np.asarray(curvature["tangent_angle"], dtype=float)
    tangent_angle_degrees = np.degrees(tangent_angle)
    orientation_degrees = np.mod(tangent_angle_degrees, 180.0)

    doubled = np.radians(2.0 * orientation_degrees)
    mean_cos = float(np.mean(np.cos(doubled)))
    mean_sin = float(np.mean(np.sin(doubled)))
    dominant_orientation = 0.5 * np.degrees(np.arctan2(mean_sin, mean_cos))
    dominant_orientation = float(np.mod(dominant_orientation, 180.0))
    orientation_strength = float(np.hypot(mean_cos, mean_sin))

    alignment_delta: float | None = None
    if reference_angle_degrees is not None:
        ref = float(reference_angle_degrees) % 180.0
        diff = abs(dominant_orientation - ref)
        alignment_delta = float(min(diff, 180.0 - diff))

    return {
        "arc_length": np.asarray(curvature["arc_length"], dtype=float),
        "tangent_angle_degrees": tangent_angle_degrees,
        "orientation_degrees": orientation_degrees,
        "dominant_orientation_degrees": dominant_orientation,
        "orientation_strength": orientation_strength,
        "alignment_delta_degrees": alignment_delta,
    }


def analyze_crack_path(
    reference_surface: Any,
    adjusted_surface: Any,
    dx: float = 1.0,
    dy: float = 1.0,
    separation: float = 0.0,
    propagation_axis: str = "x",
    propagation_angle_degrees: float | None = None,
    front_side: str = "min",
    method: str = "first_open_pixel",
    contour_resample_step: float | None = None,
    contour_smoothing_window: int = 5,
    local_window_length: float | None = None,
    reference_angle_degrees: float | None = None,
) -> dict[str, Any]:
    """Run the complete MVP crack-path analysis on an aligned surface pair.

    Args:
        reference_surface: Reference ``Surface``-like object or 2D array.
        adjusted_surface: Adjusted ``Surface``-like object or 2D array.
        dx: Physical pixel spacing along X used to convert path coordinates.
        dy: Physical pixel spacing along Y used to convert path coordinates.
        separation: Opening threshold in the height unit of the surfaces.
        propagation_axis: Nominal propagation direction, ``'x'``, ``'y'``, or
            ``'angle'``.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.
        front_side: Side from which the front is detected on the transverse
            axis, ``'min'`` or ``'max'``.
        method: Crack-path extraction method. Supported values are
            ``'first_open_pixel'`` and ``'contour'``.
        contour_resample_step: Constant step along the propagation axis used by
            the contour method after ordering and collapsing the raw contour.
        contour_smoothing_window: Moving-average window for transverse
            smoothing used by the contour method.
        local_window_length: Sliding-window length for local tortuosity.
        reference_angle_degrees: Optional external direction used for
            orientation-alignment reporting.

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
            propagation_angle_degrees=propagation_angle_degrees,
        )
    else:
        path_points = extract_crack_path_contour(
            open_mask,
            dx=dx,
            dy=dy,
            propagation_axis=propagation_axis,
            propagation_angle_degrees=propagation_angle_degrees,
            resample_step=contour_resample_step,
            smoothing_window=contour_smoothing_window,
        )
    tortuosity = crack_path_tortuosity(
        path_points,
        propagation_axis=propagation_axis,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    curvature = crack_path_curvature(path_points)
    local_tortuosity = crack_path_local_tortuosity(
        path_points,
        propagation_axis=propagation_axis,
        window_length=local_window_length,
        propagation_angle_degrees=propagation_angle_degrees,
    )
    orientation = crack_path_orientation_statistics(
        path_points,
        reference_angle_degrees=reference_angle_degrees,
    )

    return {
        "difference_map": difference_map,
        "open_mask": open_mask,
        "valid_mask": valid_mask,
        "path_points": path_points,
        "path_method": path_method,
        "propagation_axis": _resolve_propagation_angle(
            propagation_axis,
            propagation_angle_degrees=propagation_angle_degrees,
        )[0],
        "propagation_angle_degrees": _resolve_propagation_angle(
            propagation_axis,
            propagation_angle_degrees=propagation_angle_degrees,
        )[1],
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
        "local_window_length": float(local_tortuosity["window_length"]),
        "reference_angle_degrees": (
            None if reference_angle_degrees is None else float(reference_angle_degrees)
        ),
        **tortuosity,
        **curvature,
        **local_tortuosity,
        **orientation,
    }


def sweep_crack_path_thresholds(
    reference_surface: Any,
    adjusted_surface: Any,
    thresholds: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    propagation_axis: str = "x",
    propagation_angle_degrees: float | None = None,
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
        propagation_axis: Nominal propagation direction, ``'x'``, ``'y'``, or
            ``'angle'``.
        propagation_angle_degrees: Propagation direction angle in degrees when
            ``propagation_axis='angle'``.
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
                propagation_angle_degrees=propagation_angle_degrees,
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
        "propagation_axis": _resolve_propagation_angle(
            propagation_axis,
            propagation_angle_degrees=propagation_angle_degrees,
        )[0],
        "propagation_angle_degrees": _resolve_propagation_angle(
            propagation_axis,
            propagation_angle_degrees=propagation_angle_degrees,
        )[1],
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
