"""Compatibility wrapper around shared ``surface_io`` loaders."""

from surface_io.loaders import load_csv_data, load_h5_data, load_npz_data, load_stl_data, suggest_units

__all__ = [
    "load_csv_data",
    "load_h5_data",
    "load_npz_data",
    "load_stl_data",
    "suggest_units",
]
