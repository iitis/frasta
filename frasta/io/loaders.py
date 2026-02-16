"""Data loading functions for various scan formats.

This module provides functions for loading scan data from CSV, NPZ, and HDF5 formats.
"""

import numpy as np
import pandas as pd
import h5py
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
            - px_x: Pixel size in X
            - px_y: Pixel size in Y
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

    px_x = np.median(dx[dx > 0]).round(2)
    px_y = np.median(dy[dy > 0]).round(2)

    logger.debug(f"px_x: {px_x}, px_y: {px_y}")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_size_x = int((x_max - x_min) / px_x) + 1
    grid_size_y = int((y_max - y_min) / px_y) + 1

    grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int32)
    N = len(x)
    
    for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ix = int(round((xi - x_min) / px_x))
        iy = int(round((yi - y_min) / px_y))
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
    
    return grid, xi_grid, yi_grid, px_x, px_y


def load_npz_data(fname):
    """Loads scan data from NPZ format.
    
    Args:
        fname (str): Path to NPZ file.
        
    Returns:
        list: List of tuples (name, grid, xi, yi, px_x, px_y) for each scan.
        
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
        grid = data[f"grid_{i:02}"]
        xi = data[f"xi_{i:02}"]
        yi = data[f"yi_{i:02}"]
        px_x = data[f"px_{i:02}"]
        px_y = data[f"py_{i:02}"]
        results.append((name, grid, xi, yi, px_x, px_y))
    
    return results


def load_h5_data(fname):
    """Loads scan data from HDF5 format.
    
    Args:
        fname (str): Path to HDF5 file.
        
    Returns:
        list: List of tuples (name, grid, xi, yi, px_x, px_y) for each scan.
        
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
            grid = group["grid"][:]
            xi = group["xi"][:]
            yi = group["yi"][:]
            px_x = group["px_x"][()]
            px_y = group["px_y"][()]
            results.append((name, grid, xi, yi, px_x, px_y))
    
    return results
