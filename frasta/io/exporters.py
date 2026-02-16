"""Data export functions for scan data.

This module provides functions for saving scan data to NPZ and HDF5 formats.
"""

import numpy as np
import h5py
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
