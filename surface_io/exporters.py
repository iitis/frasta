"""Shared export functions for common scan formats."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def save_npz(fname, scans):
    """Save scans to FRASTA-style NPZ."""

    save_dict = {"frasta_info": 1, "frasta_cnt": len(scans)}
    for i, (name, surface) in enumerate(scans):
        save_dict[f"name_{i:02}"] = name
        save_dict[f"height_{i:02}"] = surface.height
        save_dict[f"dx_{i:02}"] = surface.dx
        save_dict[f"dy_{i:02}"] = surface.dy
        save_dict[f"x0_{i:02}"] = surface.x0
        save_dict[f"y0_{i:02}"] = surface.y0
    np.savez_compressed(fname, **save_dict)
    logger.info("Saved %s scans to %s (Surface format)", len(scans), fname)


def save_h5(fname, scans):
    """Save scans to FRASTA-style HDF5."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 scan export.") from exc

    with h5py.File(fname, "w") as file:
        file.attrs["frasta_info"] = 1
        file.attrs["frasta_cnt"] = len(scans)
        for i, (name, surface) in enumerate(scans):
            group = file.create_group(f"tab_{i:02}")
            group.create_dataset("name", data=name.encode("utf-8"))
            group.create_dataset("height", data=surface.height, compression="gzip")
            group.create_dataset("dx", data=surface.dx)
            group.create_dataset("dy", data=surface.dy)
            group.create_dataset("x0", data=surface.x0)
            group.create_dataset("y0", data=surface.y0)
    logger.info("Saved %s scans to %s (Surface format)", len(scans), fname)


def save_stl(fname, surface, binary=True, max_points=50000000):
    """Save one surface as STL mesh."""

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for STL scan export.") from exc

    xi = surface.xi
    yi = surface.yi
    grid = surface.height
    height_count, width_count = grid.shape
    valid_points = np.count_nonzero(~np.isnan(grid))
    total_points = height_count * width_count

    if valid_points > max_points:
        stride = int(np.ceil(np.sqrt(valid_points / max_points)))
        nan_percent = 100 * (1 - valid_points / total_points)
        logger.info(
            "Grid size %sx%s (%s points, %s valid, %.1f%% NaN) exceeds max_points=%s. "
            "Downsampling with stride=%s to ~%s points",
            height_count,
            width_count,
            f"{total_points:,}",
            f"{valid_points:,}",
            nan_percent,
            f"{max_points:,}",
            stride,
            f"{valid_points // (stride * stride):,}",
        )
        grid = grid[::stride, ::stride]
        xi = xi[::stride]
        yi = yi[::stride]

    xi_mm = xi / 1000.0
    yi_mm = yi / 1000.0
    grid_mm = grid / 1000.0
    xx, yy = np.meshgrid(xi_mm, yi_mm)
    valid_mask = ~np.isnan(grid_mm)
    x_flat = xx[valid_mask]
    y_flat = yy[valid_mask]
    z_flat = grid_mm[valid_mask]

    if len(x_flat) == 0:
        raise ValueError("Grid contains no valid data points")

    vertices = np.column_stack([x_flat, y_flat, z_flat])
    height_count, width_count = grid.shape
    index_map = np.full((height_count, width_count), -1, dtype=np.int32)
    valid_indices = np.where(valid_mask)
    for idx, (i, j) in enumerate(zip(valid_indices[0], valid_indices[1])):
        index_map[i, j] = idx

    faces = []
    for i in range(height_count - 1):
        for j in range(width_count - 1):
            idx_00 = index_map[i, j]
            idx_10 = index_map[i + 1, j]
            idx_01 = index_map[i, j + 1]
            idx_11 = index_map[i + 1, j + 1]
            if idx_00 >= 0 and idx_10 >= 0 and idx_11 >= 0:
                faces.append([idx_00, idx_10, idx_11])
            if idx_00 >= 0 and idx_11 >= 0 and idx_01 >= 0:
                faces.append([idx_00, idx_11, idx_01])

    if len(faces) == 0:
        raise ValueError("Could not generate any triangular faces from the grid")

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.export(fname, file_type="stl" if binary else "stl_ascii")
    logger.info("Saved mesh with %s vertices and %s faces to %s", len(vertices), len(faces), fname)
