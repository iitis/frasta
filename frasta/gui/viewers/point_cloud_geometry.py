"""Helpers for converting regular height maps into point cloud buffers.

This module isolates the data preparation logic for the experimental
QOpenGLWidget-based 3D viewer so it can be tested without an active OpenGL
context.
"""

from __future__ import annotations

import numpy as np

from ...utils import get_colormap


def build_point_positions_from_grid(
    grid: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    z_offset: float = 0.0,
    clip_abs: float = 1e6,
    stride: int = 1,
) -> np.ndarray:
    """Build point positions from a structured height map.

    Args:
        grid: Height map represented as a 2D numpy array.
        dx: Pixel spacing in the X direction.
        dy: Pixel spacing in the Y direction.
        x0: Physical X origin.
        y0: Physical Y origin.
        z_offset: Additional Z offset applied to every valid point.
        clip_abs: Absolute clipping threshold for invalid outliers.
        stride: Sampling stride for regular decimation.

    Returns:
        Point positions with shape ``(N, 3)``.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    sampled = np.asarray(grid[::stride, ::stride], dtype=np.float32)
    valid_mask = np.isfinite(sampled) & (np.abs(sampled) <= clip_abs)
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    rows, cols = sampled.shape
    xs = x0 + dx * np.arange(0, cols * stride, stride, dtype=np.float32)
    ys = -(y0 + dy * np.arange(0, rows * stride, stride, dtype=np.float32))
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")

    z_values = sampled[valid_mask] + np.float32(z_offset)
    return np.column_stack(
        (x_grid[valid_mask], y_grid[valid_mask], z_values)
    ).astype(np.float32, copy=False)


def build_mesh_geometry_from_grid(
    grid: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    z_offset: float = 0.0,
    clip_abs: float = 1e6,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build compact mesh geometry from a structured height map.

    Args:
        grid: Height map represented as a 2D numpy array.
        dx: Pixel spacing in the X direction.
        dy: Pixel spacing in the Y direction.
        x0: Physical X origin.
        y0: Physical Y origin.
        z_offset: Additional Z offset applied to every valid vertex.
        clip_abs: Absolute clipping threshold for invalid outliers.
        stride: Sampling stride for regular decimation.

    Returns:
        Tuple ``(positions, normals, indices)`` where positions and normals
        have shape ``(N, 3)`` and indices have shape ``(M, 3)``.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    sampled = np.asarray(grid[::stride, ::stride], dtype=np.float32)
    valid_mask = np.isfinite(sampled) & (np.abs(sampled) <= clip_abs)
    if not np.any(valid_mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
        )

    rows, cols = sampled.shape
    xs = x0 + dx * np.arange(0, cols * stride, stride, dtype=np.float32)
    ys = -(y0 + dy * np.arange(0, rows * stride, stride, dtype=np.float32))
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    z_grid = sampled + np.float32(z_offset)

    vertex_ids = np.full(sampled.shape, -1, dtype=np.int32)
    valid_coords = np.argwhere(valid_mask)
    vertex_ids[valid_mask] = np.arange(len(valid_coords), dtype=np.int32)
    positions = np.column_stack(
        (
            x_grid[valid_mask],
            y_grid[valid_mask],
            z_grid[valid_mask],
        )
    ).astype(np.float32, copy=False)

    face_list: list[list[int]] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            quad = (
                valid_mask[row, col]
                and valid_mask[row + 1, col]
                and valid_mask[row, col + 1]
                and valid_mask[row + 1, col + 1]
            )
            if not quad:
                continue
            v00 = int(vertex_ids[row, col])
            v10 = int(vertex_ids[row + 1, col])
            v01 = int(vertex_ids[row, col + 1])
            v11 = int(vertex_ids[row + 1, col + 1])
            face_list.append([v00, v10, v11])
            face_list.append([v00, v11, v01])

    if not face_list:
        return (
            positions,
            np.zeros_like(positions, dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
        )

    indices = np.asarray(face_list, dtype=np.uint32)
    normals = _compute_vertex_normals(positions, indices)
    return positions, normals, indices


def build_point_cloud_from_grid(
    grid: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    z_offset: float = 0.0,
    clip_abs: float = 1e6,
    colormap: str | None = "Metrology",
    value_range: tuple[float, float] | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build point positions and colors from a structured height map.

    Args:
        grid: Height map represented as a 2D numpy array.
        dx: Pixel spacing in the X direction.
        dy: Pixel spacing in the Y direction.
        x0: Physical X origin.
        y0: Physical Y origin.
        z_offset: Additional Z offset applied to every valid point.
        clip_abs: Absolute clipping threshold for invalid outliers.
        colormap: Colormap name or ``None`` for a constant neutral color.
        value_range: Optional explicit ``(lo, hi)`` normalization range.
        stride: Sampling stride for regular decimation.

    Returns:
        Tuple ``(positions, colors)`` where ``positions`` has shape ``(N, 3)``
        and ``colors`` has shape ``(N, 4)``.
    """
    positions = build_point_positions_from_grid(
        grid,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        z_offset=z_offset,
        clip_abs=clip_abs,
        stride=stride,
    )
    if len(positions) == 0:
        return positions, np.empty((0, 4), dtype=np.float32)
    z_values = positions[:, 2]
    colors = _build_colors(z_values, colormap=colormap, value_range=value_range)
    return positions, colors


def compute_bounds(*point_sets: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned bounds for one or more point sets.

    Args:
        *point_sets: Arrays with shape ``(N, 3)`` or ``None``.

    Returns:
        Tuple ``(mins, maxs)`` as ``float32`` arrays of shape ``(3,)``.
        Empty inputs fall back to ``[-1, -1, -1]`` and ``[1, 1, 1]``.
    """
    valid_sets = [pts for pts in point_sets if pts is not None and len(pts) > 0]
    if not valid_sets:
        return (
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    stacked = np.vstack(valid_sets).astype(np.float32, copy=False)
    return np.min(stacked, axis=0), np.max(stacked, axis=0)


def compute_progressive_stride_schedule(
    grid_shape: tuple[int, int],
    cloud_count: int = 1,
    target_initial_points: int = 120_000,
    min_stride: int = 1,
) -> list[int]:
    """Compute a progressive stride schedule for responsive initial rendering.

    Args:
        grid_shape: Input grid shape as ``(rows, cols)``.
        cloud_count: Number of simultaneously rendered point clouds.
        target_initial_points: Target upper bound for the first displayed stage.
        min_stride: Smallest automatically scheduled stride.

    Returns:
        Descending list of strides ending with ``1``.
    """
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        return [max(1, min_stride)]

    min_stride = max(1, int(min_stride))

    total_points = max(1, rows * cols * max(1, cloud_count))
    initial_ratio = total_points / max(1, target_initial_points)
    initial_stride = _next_power_of_two(int(np.ceil(np.sqrt(initial_ratio))))
    initial_stride = max(initial_stride, min_stride)

    schedule = [max(1, initial_stride)]
    while schedule[-1] > min_stride:
        next_stride = max(min_stride, schedule[-1] // 2)
        if next_stride == schedule[-1]:
            break
        schedule.append(next_stride)
    if schedule[-1] != min_stride:
        schedule.append(min_stride)
    return schedule


def build_colormap_lut(
    colormap: str | None,
    size: int = 256,
) -> np.ndarray:
    """Build a compact RGBA lookup table for GPU color mapping.

    Args:
        colormap: Colormap name or ``None`` for a constant neutral color.
        size: Number of LUT samples.

    Returns:
        RGBA LUT with shape ``(size, 4)`` and dtype ``uint8``.
    """
    size = max(2, int(size))
    sample_points = np.linspace(0.0, 1.0, size, dtype=np.float32)

    if colormap is None:
        rgba = np.tile(
            np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32),
            (size, 1),
        )
    elif colormap == "RG":
        rgba = np.stack(
            [
                1.0 - sample_points,
                sample_points,
                np.zeros_like(sample_points),
                np.ones_like(sample_points),
            ],
            axis=1,
        )
    elif colormap == "B&W":
        rgba = np.stack(
            [
                sample_points,
                sample_points,
                sample_points,
                np.ones_like(sample_points),
            ],
            axis=1,
        )
    else:
        cmap = get_colormap(colormap)
        rgba = cmap.map(sample_points, mode="float").astype(np.float32, copy=False)

    rgba = np.clip(rgba, 0.0, 1.0)
    return np.round(rgba * 255.0).astype(np.uint8, copy=False)


def _build_colors(
    z_values: np.ndarray,
    colormap: str | None,
    value_range: tuple[float, float] | None,
) -> np.ndarray:
    """Map point heights to RGBA colors."""
    if len(z_values) == 0:
        return np.empty((0, 4), dtype=np.float32)

    if colormap is None:
        return np.tile(
            np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32),
            (len(z_values), 1),
        )

    if value_range is None:
        lo = float(np.min(z_values))
        hi = float(np.max(z_values))
    else:
        lo, hi = value_range

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((z_values - lo) / (hi - lo), 0.0, 1.0)
    if colormap == "RG":
        return np.stack(
            [1.0 - normalized, normalized, np.zeros_like(normalized), np.ones_like(normalized)],
            axis=1,
        ).astype(np.float32, copy=False)
    if colormap == "B&W":
        return np.stack(
            [normalized, normalized, normalized, np.ones_like(normalized)],
            axis=1,
        ).astype(np.float32, copy=False)

    cmap = get_colormap(colormap)
    return cmap.map(normalized, mode="float").astype(np.float32, copy=False)


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _compute_vertex_normals(
    positions: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute smooth vertex normals from triangle indices."""
    normals = np.zeros_like(positions, dtype=np.float32)
    p0 = positions[indices[:, 0]]
    p1 = positions[indices[:, 1]]
    p2 = positions[indices[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)

    for corner in range(3):
        np.add.at(normals, indices[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    normals[valid] /= lengths[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32, copy=False)
