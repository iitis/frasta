"""Registry for parsers that normalize external files into ``Surface`` objects."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ...core import Surface

SurfaceParser = Callable[..., Surface]

_SURFACE_PARSERS: dict[str, SurfaceParser] = {}


def register_surface_parser(*suffixes: str) -> Callable[[SurfaceParser], SurfaceParser]:
    """Register a parser function for one or more file suffixes.

    Args:
        *suffixes: File suffixes handled by the parser, with or without the
            leading dot.

    Returns:
        Callable[[SurfaceParser], SurfaceParser]: Decorator registering the
        parser in the module-level registry.
    """

    normalized_suffixes = tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    )

    def decorator(parser: SurfaceParser) -> SurfaceParser:
        for suffix in normalized_suffixes:
            _SURFACE_PARSERS[suffix] = parser
        return parser

    return decorator


def get_surface_parser(path_or_suffix: str) -> SurfaceParser:
    """Return the parser registered for a file path or suffix.

    Args:
        path_or_suffix: Full file path or bare suffix.

    Returns:
        SurfaceParser: Registered parser function.

    Raises:
        ValueError: If no parser is registered for the suffix.
    """

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


def load_surface_file(fname: str, **kwargs) -> Surface:
    """Load a single-surface instrument file using the parser registry.

    Args:
        fname: Path to the input file.
        **kwargs: Parser-specific keyword arguments.

    Returns:
        Surface: Parsed surface in FRASTA's canonical data model.
    """

    parser = get_surface_parser(fname)
    return parser(fname, **kwargs)
