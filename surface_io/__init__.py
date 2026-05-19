"""Shared scan I/O package extracted from FRASTA for cross-project reuse."""

from .surface import Surface, SurfaceOrientation
from .loaders import (
    load_csv_data,
    load_h5_data,
    load_npz_data,
    load_stl_data,
    suggest_units,
)
from .exporters import save_h5, save_npz, save_stl
from .parsers import (
    get_scan_reader,
    get_surface_parser,
    load_alicona_al3d,
    load_dicom_series,
    load_sensofar_plux,
    load_scan_file,
    load_surface_file,
    register_scan_reader,
    register_surface_parser,
)


@register_scan_reader(".csv", ".dat", ".txt")
def _load_xyz_text_as_scan_list(fname, **kwargs):
    """Adapt text XYZ loading to the unified multi-scan reader contract."""

    return [load_csv_data(fname, **kwargs)]


@register_scan_reader(".npz")
def _load_npz_as_scan_list(fname, **kwargs):
    """Adapt NPZ loading to the unified multi-scan reader contract."""

    return load_npz_data(fname, **kwargs)


@register_scan_reader(".h5")
def _load_h5_as_scan_list(fname, **kwargs):
    """Adapt HDF5 loading to the unified multi-scan reader contract."""

    return load_h5_data(fname, **kwargs)


@register_scan_reader(".stl")
def _load_stl_as_scan_list(fname, **kwargs):
    """Adapt STL loading to the unified multi-scan reader contract."""

    return [load_stl_data(fname, **kwargs)]


__all__ = [
    "Surface",
    "SurfaceOrientation",
    "get_scan_reader",
    "get_surface_parser",
    "load_alicona_al3d",
    "load_csv_data",
    "load_dicom_series",
    "load_h5_data",
    "load_npz_data",
    "load_sensofar_plux",
    "load_scan_file",
    "load_stl_data",
    "load_surface_file",
    "register_scan_reader",
    "register_surface_parser",
    "save_h5",
    "save_npz",
    "save_stl",
    "suggest_units",
]
