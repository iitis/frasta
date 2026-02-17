"""Data export functions for scan data.

This module provides functions for saving scan data to NPZ, HDF5, and STL formats.
"""

import numpy as np
import h5py
import trimesh
import logging

logger = logging.getLogger(__name__)


def save_npz(fname, scans):
    """Saves scan data to NPZ format.
    
    Args:
        fname (str): Path to output NPZ file.
        scans (list): List of tuples (name, grid, xi, yi, px_x, px_y).
    """
    save_dict = {'frasta_info': 1, 'frasta_cnt': len(scans)}
    
    for i, (name, grid, xi, yi, px_x, px_y) in enumerate(scans):
        save_dict[f"name_{i:02}"] = name
        save_dict[f"grid_{i:02}"] = grid
        save_dict[f"xi_{i:02}"] = xi
        save_dict[f"yi_{i:02}"] = yi
        save_dict[f"px_{i:02}"] = px_x
        save_dict[f"py_{i:02}"] = px_y
    
    np.savez_compressed(fname, **save_dict)
    logger.info(f"Saved {len(scans)} scans to {fname}")


def save_h5(fname, scans):
    """Saves scan data to HDF5 format.
    
    Args:
        fname (str): Path to output HDF5 file.
        scans (list): List of tuples (name, grid, xi, yi, px_x, px_y).
    """
    with h5py.File(fname, 'w') as f:
        f.attrs['frasta_info'] = 1
        f.attrs['frasta_cnt'] = len(scans)
        
        for i, (name, grid, xi, yi, px_x, px_y) in enumerate(scans):
            group_name = f"tab_{i:02}"
            group = f.create_group(group_name)
            group.create_dataset("name", data=name.encode("utf-8"))
            group.create_dataset("grid", data=grid, compression="gzip")
            group.create_dataset("xi", data=xi, compression="gzip")
            group.create_dataset("yi", data=yi, compression="gzip")
            group.create_dataset("px_x", data=px_x)
            group.create_dataset("px_y", data=px_y)
    
    logger.info(f"Saved {len(scans)} scans to {fname}")


def save_stl(fname, grid, xi, yi, binary=True):
    """Saves a single scan as an STL mesh file.
    
    Converts a 2D height map (grid) to a 3D triangular mesh and saves it
    in STL format. NaN values in the grid are excluded from the mesh.
    
    Args:
        fname (str): Path to output STL file.
        grid (np.ndarray): 2D array of Z values (height map).
        xi (np.ndarray): 1D array of X coordinates in micrometers.
        yi (np.ndarray): 1D array of Y coordinates in micrometers.
        binary (bool): If True, save as binary STL; if False, save as ASCII STL.
    
    Raises:
        ValueError: If grid contains no valid data points.
    """
    # Convert coordinates from micrometers to millimeters for STL
    xi_mm = xi / 1000.0
    yi_mm = yi / 1000.0
    grid_mm = grid / 1000.0
    
    # Create meshgrid
    xx, yy = np.meshgrid(xi_mm, yi_mm)
    
    # Flatten arrays and filter out NaN values
    valid_mask = ~np.isnan(grid_mm)
    x_flat = xx[valid_mask]
    y_flat = yy[valid_mask]
    z_flat = grid_mm[valid_mask]
    
    if len(x_flat) == 0:
        raise ValueError("Grid contains no valid data points")
    
    # Create vertex array
    vertices = np.column_stack([x_flat, y_flat, z_flat])
    
    # Create index mapping for valid points
    h, w = grid.shape
    index_map = np.full((h, w), -1, dtype=np.int32)
    valid_indices = np.where(valid_mask)
    for idx, (i, j) in enumerate(zip(valid_indices[0], valid_indices[1])):
        index_map[i, j] = idx
    
    # Generate triangular faces by connecting neighboring valid vertices
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            # Get indices of the four corners of current quad
            idx_00 = index_map[i, j]       # top-left
            idx_10 = index_map[i + 1, j]   # bottom-left
            idx_01 = index_map[i, j + 1]   # top-right
            idx_11 = index_map[i + 1, j + 1]  # bottom-right
            
            # Create triangles only if all corners are valid
            # Triangle 1: (00, 10, 11)
            if idx_00 >= 0 and idx_10 >= 0 and idx_11 >= 0:
                faces.append([idx_00, idx_10, idx_11])
            
            # Triangle 2: (00, 11, 01)
            if idx_00 >= 0 and idx_11 >= 0 and idx_01 >= 0:
                faces.append([idx_00, idx_11, idx_01])
    
    if len(faces) == 0:
        raise ValueError("Could not generate any triangular faces from the grid")
    
    faces = np.array(faces)
    
    # Create trimesh mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export to STL
    file_type = 'stl' if binary else 'stl_ascii'
    mesh.export(fname, file_type=file_type)
    
    logger.info(f"Saved mesh with {len(vertices)} vertices and {len(faces)} faces to {fname}")
