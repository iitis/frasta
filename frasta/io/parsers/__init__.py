"""Compatibility wrapper around shared ``surface_io`` parsers."""

from surface_io.parsers import (
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

__all__ = [
    "get_scan_reader",
    "get_surface_parser",
    "load_alicona_al3d",
    "load_dicom_series",
    "load_sensofar_plux",
    "load_scan_file",
    "load_surface_file",
    "register_scan_reader",
    "register_surface_parser",
]
