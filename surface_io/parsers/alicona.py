"""Parser for Alicona Imaging ``.al3d`` surface files."""

from __future__ import annotations

import logging
import math
import os
import struct

import numpy as np

from ..surface import Surface
from .registry import register_surface_parser

logger = logging.getLogger(__name__)

_MAGIC = b"AliconaImaging\x00\r\n"
_TAG_LAYOUT = "<20s30s2s"
_TAG_SIZE = struct.calcsize(_TAG_LAYOUT)
_COMMENT_SIZE = 256


def _decode_tag_bytes(raw: bytes) -> str:
    """Decode one AL3D tag field."""

    return raw.split(b"\x00", 1)[0].decode("latin-1").strip()


def _read_tag(handle) -> tuple[str, str]:
    """Read one fixed-size AL3D header tag."""

    raw = handle.read(_TAG_SIZE)
    if len(raw) != _TAG_SIZE:
        raise ValueError("Unexpected end of file while reading AL3D header tag.")
    key_raw, value_raw, crlf = struct.unpack(_TAG_LAYOUT, raw)
    if crlf != b"\r\n":
        raise ValueError("Invalid AL3D header tag terminator.")
    key = _decode_tag_bytes(key_raw)
    value = _decode_tag_bytes(value_raw)
    if not key:
        raise ValueError("Encountered an empty AL3D header tag key.")
    return key, value


def _read_comment_block(handle) -> str:
    """Read and validate the fixed-size AL3D comment block."""

    raw = handle.read(_COMMENT_SIZE)
    if len(raw) != _COMMENT_SIZE:
        raise ValueError("Unexpected end of file while reading AL3D comment block.")
    if raw[-2:] != b"\r\n":
        raise ValueError("Invalid AL3D comment block terminator.")
    payload = raw[:-2]
    payload = payload.split(b"\x00", 1)[0]
    return payload.decode("latin-1").strip()


def _read_required_int(tags: dict[str, str], key: str) -> int:
    """Read one required integer-valued AL3D tag."""

    try:
        return int(tags[key])
    except KeyError as exc:
        raise ValueError(f"Missing required AL3D tag '{key}'.") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid integer value in AL3D tag '{key}'.") from exc


def _read_required_float(tags: dict[str, str], key: str) -> float:
    """Read one required float-valued AL3D tag."""

    try:
        return float(tags[key])
    except KeyError as exc:
        raise ValueError(f"Missing required AL3D tag '{key}'.") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid float value in AL3D tag '{key}'.") from exc


def _replace_invalid_values(data: np.ndarray, invalid_value: float) -> np.ndarray:
    """Replace AL3D invalid-pixel sentinels with ``NaN``."""

    if math.isnan(invalid_value):
        return np.where(np.isnan(data), np.nan, data)
    tolerance = max(abs(invalid_value) * 1.5e-7, 1.5e-7)
    invalid_mask = np.isclose(data, invalid_value, atol=tolerance, rtol=0.0)
    data = data.copy()
    data[invalid_mask] = np.nan
    return data


@register_surface_parser(".al3d")
def load_alicona_al3d(fname: str, progress_callback=None) -> Surface:
    """Load one Alicona AL3D file into a shared ``Surface``."""

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    if progress_callback:
        progress_callback(10)

    try:
        with open(fname, "rb") as handle:
            magic = handle.read(len(_MAGIC))
            if magic != _MAGIC:
                raise ValueError("Not a valid Alicona AL3D file: magic header mismatch.")

            version_key, version_value = _read_tag(handle)
            if version_key != "Version":
                raise ValueError("Invalid AL3D header: first tag must be 'Version'.")

            count_key, count_value = _read_tag(handle)
            if count_key != "TagCount":
                raise ValueError("Invalid AL3D header: second tag must be 'TagCount'.")

            try:
                tag_count = int(count_value)
            except ValueError as exc:
                raise ValueError("Invalid AL3D TagCount value.") from exc

            tags: dict[str, str] = {"Version": version_value, "TagCount": count_value}
            for _ in range(tag_count):
                key, value = _read_tag(handle)
                tags[key] = value

            comment = _read_comment_block(handle)

            if progress_callback:
                progress_callback(40)

            cols = _read_required_int(tags, "Cols")
            rows = _read_required_int(tags, "Rows")
            depth_offset = _read_required_int(tags, "DepthImageOffset")
            step_x_um = _read_required_float(tags, "PixelSizeXMeter") * 1e6
            step_y_um = _read_required_float(tags, "PixelSizeYMeter") * 1e6
            invalid_value = float(tags.get("InvalidPixelValue", "nan"))

            if cols <= 0 or rows <= 0:
                raise ValueError("AL3D surface dimensions must be positive.")
            if step_x_um <= 0.0 or step_y_um <= 0.0:
                raise ValueError("AL3D pixel sizes must be positive.")

            rowstride = ((cols * np.dtype("<f4").itemsize) + 7) // 8 * 8
            handle.seek(depth_offset)
            grid = np.empty((rows, cols), dtype=np.float32)
            for row_index in range(rows):
                row_bytes = handle.read(rowstride)
                if len(row_bytes) != rowstride:
                    raise ValueError("Unexpected end of file while reading AL3D depth image.")
                grid[row_index, :] = np.frombuffer(row_bytes[: cols * 4], dtype="<f4")

            if progress_callback:
                progress_callback(75)

            height_um = _replace_invalid_values(grid, invalid_value).astype(np.float64, copy=False) * 1e6

        metadata = {
            "name": os.path.splitext(os.path.basename(fname))[0],
            "format": "alicona_al3d",
            "source_path": fname,
            "comment": comment,
            "al3d_tags": tags.copy(),
        }
        surface = Surface(
            height=height_um,
            dx=step_x_um,
            dy=step_y_um,
            x0=0.0,
            y0=0.0,
            unit="um",
            metadata=metadata,
        )

        if progress_callback:
            progress_callback(100)

        logger.info(
            "Loaded AL3D surface '%s': %sx%s grid, dx=%.6f um, dy=%.6f um",
            metadata["name"],
            cols,
            rows,
            step_x_um,
            step_y_um,
        )
        return surface
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Failed to load AL3D file '%s': %s", fname, exc)
        raise ValueError(f"Failed to load AL3D file '{fname}': {exc}") from exc
