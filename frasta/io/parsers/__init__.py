"""Reusable surface parsers for instrument-specific file formats.

Each parser in this package returns a single :class:`frasta.core.Surface`
instance and stays independent from GUI concerns. This makes the parsing
layer reusable in other programs that need only the normalized surface model.
"""

from .alicona import load_alicona_al3d
from .registry import get_surface_parser, load_surface_file, register_surface_parser

__all__ = [
    "get_surface_parser",
    "load_alicona_al3d",
    "load_surface_file",
    "register_surface_parser",
]
