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
    
    Saves in new Surface format (height, dx, dy, x0, y0).
    
    Args:
        fname (str): Path to output NPZ file.
        scans (list): List of tuples (name, Surface).
    """
    save_dict = {'frasta_info': 1, 'frasta_cnt': len(scans)}
    
    for i, (name, surface) in enumerate(scans):
        save_dict[f"name_{i:02}"] = name
        save_dict[f"height_{i:02}"] = surface.height
        save_dict[f"dx_{i:02}"] = surface.dx
        save_dict[f"dy_{i:02}"] = surface.dy
        save_dict[f"x0_{i:02}"] = surface.x0
        save_dict[f"y0_{i:02}"] = surface.y0
    
    np.savez_compressed(fname, **save_dict)
    logger.info(f"Saved {len(scans)} scans to {fname} (Surface format)")


def save_h5(fname, scans):
    """Saves scan data to HDF5 format.
    
    Saves in new Surface format (height, dx, dy, x0, y0).
    
    Args:
        fname (str): Path to output HDF5 file.
        scans (list): List of tuples (name, Surface).
    """
    with h5py.File(fname, 'w') as f:
        f.attrs['frasta_info'] = 1
        f.attrs['frasta_cnt'] = len(scans)
        
        for i, (name, surface) in enumerate(scans):
            group_name = f"tab_{i:02}"
            group = f.create_group(group_name)
            group.create_dataset("name", data=name.encode("utf-8"))
            group.create_dataset("height", data=surface.height, compression="gzip")
            group.create_dataset("dx", data=surface.dx)
            group.create_dataset("dy", data=surface.dy)
            group.create_dataset("x0", data=surface.x0)
            group.create_dataset("y0", data=surface.y0)
    
    logger.info(f"Saved {len(scans)} scans to {fname} (Surface format)")


def save_stl(fname, surface, binary=True):
    """Saves a single scan as an STL mesh file.
    
    Converts a 2D height map to a 3D triangular mesh and saves it
    in STL format. NaN values are excluded from the mesh.
    
    Args:
        fname (str): Path to output STL file.
        surface (Surface): Surface object containing height data and coordinates.
        binary (bool): If True, save as binary STL; if False, save as ASCII STL.
    
    Raises:
        ValueError: If surface contains no valid data points.
    """
    # Extract coordinate arrays from Surface (includes x0, y0)
    xi = surface.xi
    yi = surface.yi
    grid = surface.height
    
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
