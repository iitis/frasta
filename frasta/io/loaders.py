"""Data loading functions for various scan formats.

This module provides functions for loading scan data from CSV, NPZ, HDF5, and STL formats.
"""

import numpy as np
import pandas as pd
import h5py
import trimesh
import logging

logger = logging.getLogger(__name__)


def suggest_units(fname):
    """Suggests coordinate units based on data sample heuristics.
    
    Analyzes a sample of CSV data to suggest whether coordinates are in
    millimeters or micrometers.
    
    Args:
        fname (str): Path to CSV file.
        
    Returns:
        tuple: (suggested_xy, suggested_z) where each is 'mm' or 'um'.
    """
    try:
        sample = pd.read_csv(
            fname,
            sep=r'[;,\t ]+',
            engine='python',
            header=None,
            names=['x', 'y', 'z'],
            nrows=5000
        )
        x, y, z = sample['x'].values, sample['y'].values, sample['z'].values
        
        # Heurystyka dla XY
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        px_x_raw = np.median(dx[dx > 0])
        px_y_raw = np.median(dy[dy > 0])
        typical_step_xy = np.median([px_x_raw, px_y_raw])
        
        # Heurystyka dla Z - sprawdź zakres wartości
        z_range = np.nanmax(z) - np.nanmin(z)
        
        # Jeśli typical_step < 0.1, prawdopodobnie mm dla XY
        # Jeśli zakres Z < 1, prawdopodobnie mm dla Z
        suggested_xy = 'mm' if typical_step_xy < 0.1 else 'um'
        suggested_z = 'mm' if z_range < 1 else 'um'
        
        logger.debug(f"Detected typical step XY: {typical_step_xy}, suggesting: {suggested_xy}")
        logger.debug(f"Detected Z range: {z_range}, suggesting: {suggested_z}")
        
        return suggested_xy, suggested_z
    except Exception as e:
        logger.warning(f"Could not analyze data sample: {e}")
        return 'um', 'um'


def load_csv_data(fname, units_xy='um', units_z='um', progress_callback=None):
    """Loads and grids CSV scan data.
    
    Loads large CSV files in chunks, converts coordinate units, grids the
    point cloud data onto a regular 2D array, and averages duplicate points.
    
    Args:
        fname (str): Path to the CSV file to load.
        units_xy (str): Unit for X and Y coordinates: 'mm' or 'um'.
        units_z (str): Unit for Z coordinate: 'mm' or 'um'.
        progress_callback (callable, optional): Function to call with progress updates (0-100).
        
    Returns:
        tuple: (grid, xi, yi, px_x, px_y) containing:
            - grid: 2D numpy array of Z values
            - xi: 1D array of X coordinates
            - yi: 1D array of Y coordinates
            - dx: Pixel size in X
            - dy: Pixel size in Y
    """
    chunk_size = 100_000
    total = sum(1 for _ in open(fname, encoding="utf-8"))
    chunks = []
    
    if progress_callback:
        progress_callback(0)

    for i, chunk in enumerate(pd.read_csv(
            fname,
            sep=r'[;,\t ]+',
            engine='python',
            header=None,
            names=['x', 'y', 'z'],
            chunksize=chunk_size)):

        chunks.append(chunk)
        if progress_callback:
            progress_callback(int(20 + 30 * (i * chunk_size / total)))

    df = pd.concat(chunks, ignore_index=True)
    # Create copies to avoid read-only array issues
    x, y, z = df['x'].values.copy(), df['y'].values.copy(), df['z'].values.copy()

    # Oblicz typowe kroki w x i y
    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    px_x_raw = np.median(dx[dx > 0])
    px_y_raw = np.median(dy[dy > 0])
    typical_step = np.median([px_x_raw, px_y_raw])
    logger.debug(f"typical_step XY: {typical_step}")
    
    # Konwersja XY na podstawie wybranej przez użytkownika jednostki
    if units_xy == 'mm':
        logger.info("Przeliczam XY z milimetrów na mikrometry.")
        x = x * 1000
        y = y * 1000
        # trzeba przeliczyć dx/dy jeszcze raz po skalowaniu
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
    else:
        logger.info("Używam XY w mikrometrach - brak konwersji.")
    
    # Konwersja Z niezależnie od XY
    if units_z == 'mm':
        logger.info("Przeliczam Z z milimetrów na mikrometry.")
        z = z * 1000
    else:
        logger.info("Używam Z w mikrometrach - brak konwersji.")

    dx = np.median(dx[dx > 0]).round(2)
    dy = np.median(dy[dy > 0]).round(2)

    logger.debug(f"px_x: {dx}, px_y: {dy}")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_size_x = int((x_max - x_min) / dx) + 1
    grid_size_y = int((y_max - y_min) / dy) + 1

    grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int32)
    N = len(x)
    
    for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ix = int(round((xi - x_min) / dx))
        iy = int(round((yi - y_min) / dy))
        if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
            if np.isnan(grid[iy, ix]):
                grid[iy, ix] = zi
            else:
                grid[iy, ix] += zi
            counts[iy, ix] += 1
        if progress_callback and idx % max(1, N//50) == 0:
            progress_callback(50 + int(49 * idx / N))

    mask_dup = (counts > 1)
    grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
    xi_grid = np.linspace(x_min, x_max, grid_size_x)
    yi_grid = np.linspace(y_min, y_max, grid_size_y)
    
    if progress_callback:
        progress_callback(100)
    
    return grid, xi_grid, yi_grid, dx, dy


def load_npz_data(fname):
    """Loads scan data from NPZ format.
    
    Supports both new format (height, dx, dy, x0, y0) and legacy format 
    (grid, px_x, px_y, xi, yi arrays).
    
    Args:
        fname (str): Path to NPZ file.
        
    Returns:
        list: List of tuples (name, grid, xi, yi, dx, dy) for each scan.
        
    Raises:
        ValueError: If NPZ file doesn't contain grid data.
    """
    data = np.load(fname)
    if 'frasta_info' not in data:
        raise ValueError("NPZ does not contain grid data")
    
    results = []
    cnt = data['frasta_cnt']
    for i in range(cnt):
        name = str(data[f"name_{i:02}"])

        # Support both naming conventions: new (height) and legacy (grid)
        if f"height_{i:02}" in data:
            grid = data[f"height_{i:02}"]
        elif f"grid_{i:02}" in data:
            grid = data[f"grid_{i:02}"]
        else:
            raise ValueError(f"Missing grid/height data for scan {i}")
        
        # Support both naming conventions: new (dx, dy) and legacy (px, py)
        if f"dx_{i:02}" in data:
            dx = float(data[f"dx_{i:02}"])
            dy = float(data[f"dy_{i:02}"])
        elif f"px_{i:02}" in data:
            dx = float(data[f"px_{i:02}"])
            dy = float(data[f"py_{i:02}"])
        else:
            raise ValueError(f"Missing pixel size data for scan {i}")
        
        # Check for new format with x0, y0
        if f"x0_{i:02}" in data and f"y0_{i:02}" in data:
            # New format: reconstruct xi, yi from x0, y0, dx, dy
            x0 = float(data[f"x0_{i:02}"])
            y0 = float(data[f"y0_{i:02}"])
            xi = x0 + np.arange(grid.shape[1]) * dx
            yi = y0 + np.arange(grid.shape[0]) * dy
            logger.debug(f"Loaded scan {i} with new format: x0={x0}, y0={y0}")
        elif f"xi_{i:02}" in data and f"yi_{i:02}" in data:
            # Legacy format: use stored xi, yi arrays
            xi = data[f"xi_{i:02}"]
            yi = data[f"yi_{i:02}"]
            logger.debug(f"Loaded scan {i} with legacy format: xi[0]={xi[0]}, yi[0]={yi[0]}")
        else:
            # Fallback: generate from origin (0, 0)
            xi = np.arange(grid.shape[1]) * dx
            yi = np.arange(grid.shape[0]) * dy
            logger.warning(f"Scan {i} missing coordinate data, assuming origin at (0, 0)")

        results.append((name, grid, xi, yi, dx, dy))
    
    return results


def load_h5_data(fname):
    """Loads scan data from HDF5 format.
    
    Supports both new format (height, dx, dy, x0, y0) and legacy format 
    (grid, px_x, px_y, xi, yi arrays).
    
    Args:
        fname (str): Path to HDF5 file.
        
    Returns:
        list: List of tuples (name, grid, xi, yi, dx, dy) for each scan.
        
    Raises:
        ValueError: If HDF5 file doesn't contain grid data.
    """
    results = []
    with h5py.File(fname, 'r') as f:
        if 'frasta_info' not in f.attrs:
            raise ValueError("HDF5 does not contain grid data")

        cnt = f.attrs.get('frasta_cnt', 0)
        for i in range(cnt):
            group_name = f"tab_{i:02}"
            if group_name not in f:
                continue
            
            group = f[group_name]
            name = group["name"][()].decode("utf-8")
            
            # Support both naming conventions
            if "height" in group:
                grid = group["height"][:]
            elif "grid" in group:
                grid = group["grid"][:]
            else:
                raise ValueError(f"Missing grid/height data for scan {i}")
            
            # Support both naming conventions for pixel size
            if "dx" in group:
                dx = float(group["dx"][()])
                dy = float(group["dy"][()])
            elif "px_x" in group:
                dx = float(group["px_x"][()])
                dy = float(group["px_y"][()])
            else:
                raise ValueError(f"Missing pixel size data for scan {i}")
            
            # Check for new format with x0, y0
            if "x0" in group and "y0" in group:
                x0 = float(group["x0"][()])
                y0 = float(group["y0"][()])
                xi = x0 + np.arange(grid.shape[1]) * dx
                yi = y0 + np.arange(grid.shape[0]) * dy
                logger.debug(f"Loaded scan {i} with new format: x0={x0}, y0={y0}")
            elif "xi" in group and "yi" in group:
                xi = group["xi"][:]
                yi = group["yi"][:]
                logger.debug(f"Loaded scan {i} with legacy format")
            else:
                xi = np.arange(grid.shape[1]) * dx
                yi = np.arange(grid.shape[0]) * dy
                logger.warning(f"Scan {i} missing coordinate data, assuming origin at (0, 0)")
            
            results.append((name, grid, xi, yi, dx, dy))
    
    return results


def load_stl_data(fname, resolution=None, progress_callback=None):
    """Loads scan data from STL format.
    
    Loads a 3D STL mesh and projects it onto a 2D grid by sampling the Z-coordinate
    at regular XY intervals. The mesh is converted to a height map suitable for
    surface analysis.
    
    Args:
        fname (str): Path to STL file (ASCII or binary).
        resolution (float, optional): Target pixel size in micrometers. If None, 
            automatically determines resolution based on mesh bounds (aims for ~500 points per axis).
        progress_callback (callable, optional): Function to call with progress updates (0-100).
        
    Returns:
        tuple: (grid, xi, yi, px_x, px_y) containing:
            - grid: 2D numpy array of Z values (height map)
            - xi: 1D array of X coordinates
            - yi: 1D array of Y coordinates
            - dx: Pixel size in X
            - dy: Pixel size in Y
            
    Raises:
        ValueError: If STL file cannot be loaded or is empty.
    """
    if progress_callback:
        progress_callback(10)
    
    try:
        # Load STL mesh using trimesh
        mesh = trimesh.load_mesh(fname)
        
        if progress_callback:
            progress_callback(30)
        
        # Get mesh bounds
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        
        # Convert to micrometers (assuming STL is in mm)
        x_min *= 1000
        x_max *= 1000
        y_min *= 1000
        y_max *= 1000
        z_min *= 1000
        z_max *= 1000
        
        # Determine resolution
        if resolution is None:
            # Auto-resolution: aim for ~500 points in the larger dimension
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            resolution = max_range / 500.0 if max_range > 0 else 1.0
        
        dx = dy = resolution
        
        # Create grid
        grid_size_x = int((x_max - x_min) / dx) + 1
        grid_size_y = int((y_max - y_min) / dy) + 1
        
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        
        # Create meshgrid for sampling
        xx, yy = np.meshgrid(xi_grid, yi_grid)
        
        if progress_callback:
            progress_callback(50)
        
        # Sample Z values from mesh using ray casting
        # For each XY point, cast a ray downward and find intersection
        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
        
        # Prepare ray origins (above the mesh) and directions (downward)
        # Convert back to mm for trimesh ray casting
        ray_origins = np.column_stack([
            xx.ravel() / 1000.0,
            yy.ravel() / 1000.0,
            np.full(xx.size, (z_max / 1000.0) + 1.0)  # Start above mesh
        ])
        ray_directions = np.array([0, 0, -1])  # Cast downward
        ray_directions = np.tile(ray_directions, (len(ray_origins), 1))
        
        # Perform ray-mesh intersection
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
        
        if progress_callback:
            progress_callback(80)
        
        # Fill grid with Z values from intersections
        if len(locations) > 0:
            for idx, location in zip(index_ray, locations):
                row = idx // grid_size_x
                col = idx % grid_size_x
                # Convert Z back to micrometers
                grid[row, col] = location[2] * 1000.0
        else:
            raise ValueError("No intersections found - mesh may be empty or improperly oriented")
        
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"STL loaded: {grid_size_x}x{grid_size_y} grid, resolution: {dx:.2f} μm")
        
        return grid, xi_grid, yi_grid, dx, dy
        
    except Exception as e:
        logger.error(f"Error loading STL file: {e}")
        raise ValueError(f"Failed to load STL file: {e}")
