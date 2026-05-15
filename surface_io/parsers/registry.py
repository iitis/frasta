"""Registry for parsers that normalize files into shared ``Surface`` objects."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..surface import Surface

SurfaceParser = Callable[..., Surface]
ScanReader = Callable[..., list[Surface]]

_SURFACE_PARSERS: dict[str, SurfaceParser] = {}
_SCAN_READERS: dict[str, ScanReader] = {}


def _normalize_suffixes(*suffixes: str) -> tuple[str, ...]:
    """Normalize file suffixes to lowercase dot-prefixed strings."""

    return tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    )


def register_surface_parser(*suffixes: str) -> Callable[[SurfaceParser], SurfaceParser]:
    """Register a single-surface parser for one or more suffixes."""

    normalized_suffixes = _normalize_suffixes(*suffixes)

    def decorator(parser: SurfaceParser) -> SurfaceParser:
        def as_scan_reader(fname: str, **kwargs) -> list[Surface]:
            return [parser(fname, **kwargs)]

        for suffix in normalized_suffixes:
            _SURFACE_PARSERS[suffix] = parser
            _SCAN_READERS[suffix] = as_scan_reader
        return parser

    return decorator


def register_scan_reader(*suffixes: str) -> Callable[[ScanReader], ScanReader]:
    """Register a multi-scan reader for one or more suffixes."""

    normalized_suffixes = _normalize_suffixes(*suffixes)

    def decorator(reader: ScanReader) -> ScanReader:
        for suffix in normalized_suffixes:
            _SCAN_READERS[suffix] = reader
        return reader

    return decorator


def get_surface_parser(path_or_suffix: str) -> SurfaceParser:
    """Return the parser registered for a path or suffix."""

    suffix = Path(path_or_suffix).suffix.lower()
    if not suffix and path_or_suffix.startswith("."):
        suffix = path_or_suffix.lower()
    parser = _SURFACE_PARSERS.get(suffix)
    if parser is None:
        supported = ", ".join(sorted(_SURFACE_PARSERS))
        raise ValueError(
            f"Unsupported surface parser format '{suffix or path_or_suffix}'. "
            f"Supported parser formats: {supported or '(none)'}."
        )
    return parser


def get_scan_reader(path_or_suffix: str) -> ScanReader:
    """Return the reader registered for a path or suffix."""

    suffix = Path(path_or_suffix).suffix.lower()
    if not suffix and path_or_suffix.startswith("."):
        suffix = path_or_suffix.lower()
    reader = _SCAN_READERS.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(_SCAN_READERS))
        raise ValueError(
            f"Unsupported scan reader format '{suffix or path_or_suffix}'. "
            f"Supported reader formats: {supported or '(none)'}."
        )
    return reader


def load_surface_file(fname: str, **kwargs) -> Surface:
    """Load one file with a single-surface parser."""

    parser = get_surface_parser(fname)
    return parser(fname, **kwargs)


def load_scan_file(fname: str, **kwargs) -> list[Surface]:
    """Load one file into a normalized list of surfaces."""

    reader = get_scan_reader(fname)
    return reader(fname, **kwargs)
