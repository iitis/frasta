"""Shared loading functions for common scan formats."""

import logging

import h5py
import numpy as np
import pandas as pd
import trimesh

from .surface import Surface

logger = logging.getLogger(__name__)


def suggest_units(fname):
    """Suggest coordinate units based on data sample heuristics."""

    try:
        sample = pd.read_csv(
            fname,
            sep=r"[;,\t ]+",
            engine="python",
            header=None,
            names=["x", "y", "z"],
            nrows=5000,
        )
        x, y, z = sample["x"].values, sample["y"].values, sample["z"].values
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        px_x_raw = np.median(dx[dx > 0])
        px_y_raw = np.median(dy[dy > 0])
        typical_step_xy = np.median([px_x_raw, px_y_raw])
        z_range = np.nanmax(z) - np.nanmin(z)
        suggested_xy = "mm" if typical_step_xy < 0.1 else "um"
        suggested_z = "mm" if z_range < 1 else "um"
        logger.debug("Detected typical step XY: %s, suggesting: %s", typical_step_xy, suggested_xy)
        logger.debug("Detected Z range: %s, suggesting: %s", z_range, suggested_z)
        return suggested_xy, suggested_z
    except Exception as exc:
        logger.warning("Could not analyze data sample: %s", exc)
        return "um", "um"


def load_csv_data(fname, units_xy="um", units_z="um", progress_callback=None):
    """Load and grid CSV scan data into one ``Surface``."""

    chunk_size = 100_000
    with open(fname, encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    chunks = []

    if progress_callback:
        progress_callback(0)

    for i, chunk in enumerate(
        pd.read_csv(
            fname,
            sep=r"[;,\t ]+",
            engine="python",
            header=None,
            names=["x", "y", "z"],
            chunksize=chunk_size,
        )
    ):
        chunks.append(chunk)
        if progress_callback:
            progress_callback(int(20 + 30 * (i * chunk_size / total)))

    df = pd.concat(chunks, ignore_index=True)
    x, y, z = df["x"].values.copy(), df["y"].values.copy(), df["z"].values.copy()

    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    if units_xy == "mm":
        logger.info("Converting XY from millimetres to micrometres.")
        x = x * 1000
        y = y * 1000
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
    else:
        logger.info("XY already in micrometres - no conversion.")

    if units_z == "mm":
        logger.info("Converting Z from millimetres to micrometres.")
        z = z * 1000
    else:
        logger.info("Z already in micrometres - no conversion.")

    dx = np.median(dx[dx > 0]).round(2)
    dy = np.median(dy[dy > 0]).round(2)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_size_x = int((x_max - x_min) / dx) + 1
    grid_size_y = int((y_max - y_min) / dy) + 1

    grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int32)
    point_count = len(x)

    for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ix = int(round((xi - x_min) / dx))
        iy = int(round((yi - y_min) / dy))
        if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
            if np.isnan(grid[iy, ix]):
                grid[iy, ix] = zi
            else:
                grid[iy, ix] += zi
            counts[iy, ix] += 1
        if progress_callback and idx % max(1, point_count // 50) == 0:
            progress_callback(50 + int(49 * idx / point_count))

    mask_dup = counts > 1
    grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
    xi_grid = np.linspace(x_min, x_max, grid_size_x)
    yi_grid = np.linspace(y_min, y_max, grid_size_y)

    if progress_callback:
        progress_callback(100)

    return Surface(
        height=grid,
        dx=dx,
        dy=dy,
        x0=xi_grid[0],
        y0=yi_grid[0],
        unit="um",
    )


def load_npz_data(fname):
    """Load one or more surfaces from FRASTA-style NPZ."""

    data = np.load(fname)
    if "frasta_info" not in data:
        raise ValueError("NPZ does not contain grid data")

    results = []
    cnt = data["frasta_cnt"]
    for i in range(cnt):
        name = str(data[f"name_{i:02}"])
        if f"height_{i:02}" in data:
            grid = data[f"height_{i:02}"]
        elif f"grid_{i:02}" in data:
            grid = data[f"grid_{i:02}"]
        else:
            raise ValueError(f"Missing grid/height data for scan {i}")

        if f"dx_{i:02}" in data:
            dx = float(data[f"dx_{i:02}"])
            dy = float(data[f"dy_{i:02}"])
        elif f"px_{i:02}" in data:
            dx = float(data[f"px_{i:02}"])
            dy = float(data[f"py_{i:02}"])
        else:
            raise ValueError(f"Missing pixel size data for scan {i}")

        if f"x0_{i:02}" in data and f"y0_{i:02}" in data:
            x0 = float(data[f"x0_{i:02}"])
            y0 = float(data[f"y0_{i:02}"])
            xi = x0 + np.arange(grid.shape[1]) * dx
            yi = y0 + np.arange(grid.shape[0]) * dy
        elif f"xi_{i:02}" in data and f"yi_{i:02}" in data:
            xi = data[f"xi_{i:02}"]
            yi = data[f"yi_{i:02}"]
        else:
            xi = np.arange(grid.shape[1]) * dx
            yi = np.arange(grid.shape[0]) * dy
            logger.warning("Scan %s missing coordinate data, assuming origin at (0, 0)", i)

        results.append(
            Surface(
                height=grid,
                dx=dx,
                dy=dy,
                x0=xi[0] if len(xi) > 0 else 0.0,
                y0=yi[0] if len(yi) > 0 else 0.0,
                unit="um",
                metadata={"name": name},
            )
        )

    return results


def load_h5_data(fname):
    """Load one or more surfaces from FRASTA-style HDF5."""

    results = []
    with h5py.File(fname, "r") as file:
        if "frasta_info" not in file.attrs:
            raise ValueError("HDF5 does not contain grid data")

        cnt = file.attrs.get("frasta_cnt", 0)
        for i in range(cnt):
            group_name = f"tab_{i:02}"
            if group_name not in file:
                continue

            group = file[group_name]
            name = group["name"][()].decode("utf-8")
            if "height" in group:
                grid = group["height"][:]
            elif "grid" in group:
                grid = group["grid"][:]
            else:
                raise ValueError(f"Missing grid/height data for scan {i}")

            if "dx" in group:
                dx = float(group["dx"][()])
                dy = float(group["dy"][()])
            elif "px_x" in group:
                dx = float(group["px_x"][()])
                dy = float(group["px_y"][()])
            else:
                raise ValueError(f"Missing pixel size data for scan {i}")

            if "x0" in group and "y0" in group:
                x0 = float(group["x0"][()])
                y0 = float(group["y0"][()])
                xi = x0 + np.arange(grid.shape[1]) * dx
                yi = y0 + np.arange(grid.shape[0]) * dy
            elif "xi" in group and "yi" in group:
                xi = group["xi"][:]
                yi = group["yi"][:]
            else:
                xi = np.arange(grid.shape[1]) * dx
                yi = np.arange(grid.shape[0]) * dy
                logger.warning("Scan %s missing coordinate data, assuming origin at (0, 0)", i)

            results.append(
                Surface(
                    height=grid,
                    dx=dx,
                    dy=dy,
                    x0=xi[0],
                    y0=yi[0],
                    unit="um",
                    metadata={"name": name},
                )
            )

    return results


def load_stl_data(fname, resolution=None, progress_callback=None):
    """Load one STL mesh and sample it into a regular height map."""

    if progress_callback:
        progress_callback(10)

    try:
        mesh = trimesh.load_mesh(fname)
        if progress_callback:
            progress_callback(30)

        bounds = mesh.bounds
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        x_min *= 1000
        x_max *= 1000
        y_min *= 1000
        y_max *= 1000
        z_min *= 1000
        z_max *= 1000

        if resolution is None:
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            resolution = max_range / 500.0 if max_range > 0 else 1.0

        dx = dy = resolution
        grid_size_x = int((x_max - x_min) / dx) + 1
        grid_size_y = int((y_max - y_min) / dy) + 1
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        xx, yy = np.meshgrid(xi_grid, yi_grid)

        if progress_callback:
            progress_callback(50)

        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
        ray_origins = np.column_stack(
            [
                xx.ravel() / 1000.0,
                yy.ravel() / 1000.0,
                np.full(xx.size, (z_max / 1000.0) + 1.0),
            ]
        )
        ray_directions = np.tile(np.array([0, 0, -1]), (len(ray_origins), 1))
        locations, index_ray, _index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
        )

        if progress_callback:
            progress_callback(80)

        if len(locations) > 0:
            for idx, location in zip(index_ray, locations):
                row = idx // grid_size_x
                col = idx % grid_size_x
                grid[row, col] = location[2] * 1000.0
        else:
            raise ValueError("No intersections found - mesh may be empty or improperly oriented")

        if progress_callback:
            progress_callback(100)

        logger.info("STL loaded: %sx%s grid, resolution: %.2f um", grid_size_x, grid_size_y, dx)
        return Surface(
            height=grid,
            dx=dx,
            dy=dy,
            x0=xi_grid[0],
            y0=yi_grid[0],
            unit="um",
        )
    except Exception as exc:
        logger.error("Error loading STL file: %s", exc)
        raise ValueError(f"Failed to load STL file: {exc}") from exc
