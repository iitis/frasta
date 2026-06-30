"""Parser for Digital Surf ``.sur`` surface files."""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path

import numpy as np

from ..surface import Surface
from .registry import register_surface_parser

logger = logging.getLogger(__name__)

_MAGIC_CLASSIC = "DIGITAL SURF"
_MAGIC_COMPRESSED = "DSCOMPRESSED"
_HEADER_SIZE = 512

_TYPE_PROFILE = 1
_TYPE_SURFACE = 2
_TYPE_RGB_INTENSITY_SURFACE = 16

_UNIT_TO_UM = {
    "m": 1_000_000.0,
    "cm": 10_000.0,
    "mm": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 0.001,
}


def _read_int16(buffer: bytes, offset: int) -> int:
    """Read one little-endian signed 16-bit integer."""

    return struct.unpack_from("<h", buffer, offset)[0]


def _read_uint32(buffer: bytes, offset: int) -> int:
    """Read one little-endian unsigned 32-bit integer."""

    return struct.unpack_from("<I", buffer, offset)[0]


def _read_int32(buffer: bytes, offset: int) -> int:
    """Read one little-endian signed 32-bit integer."""

    return struct.unpack_from("<i", buffer, offset)[0]


def _read_float32(buffer: bytes, offset: int) -> float:
    """Read one little-endian 32-bit float."""

    return struct.unpack_from("<f", buffer, offset)[0]


def _decode_text(buffer: bytes, offset: int, length: int) -> str:
    """Decode one fixed-width header text field."""

    text = buffer[offset:offset + length].decode("latin-1", errors="ignore")
    text = text.replace("\xe6", "\xb5")
    return text.replace("\x00", "").strip()


def _normalize_unit(unit: str) -> str:
    """Normalize a SUR unit string to one of the known spellings."""

    unit = unit.strip().replace("Â", "").replace("μ", "µ")
    if not unit:
        return "um"
    lowered = unit.lower()
    if lowered in {"µm", "um"}:
        return "um"
    return lowered


def _unit_to_um(unit: str) -> float:
    """Convert one SUR unit string to a micrometre multiplier."""

    normalized = _normalize_unit(unit)
    scale = _UNIT_TO_UM.get(normalized)
    if scale is None:
        logger.warning("Unknown SUR unit '%s'; assuming micrometres.", unit)
        return 1.0
    return scale


def _read_header(buffer: bytes) -> dict[str, object]:
    """Parse the fixed-width SUR header into a plain dictionary."""

    code = _decode_text(buffer, 0, 12)
    header = {
        "code": code,
        "format": _read_int16(buffer, 12),
        "n_objects": _read_int16(buffer, 14),
        "version_number": _read_int16(buffer, 16),
        "studiable_type": _read_int16(buffer, 18),
        "name_object": _decode_text(buffer, 20, 30),
        "name_operator": _decode_text(buffer, 50, 30),
        "non_measured_points": _read_int16(buffer, 86),
        "bits_per_point": _read_int16(buffer, 98),
        "min_point": _read_int32(buffer, 100),
        "max_point": _read_int32(buffer, 104),
        "n_points_per_line": _read_int32(buffer, 108),
        "n_lines": _read_int32(buffer, 112),
        "n_total_points": _read_int32(buffer, 116),
        "spacing_x": _read_float32(buffer, 120),
        "spacing_y": _read_float32(buffer, 124),
        "spacing_z": _read_float32(buffer, 128),
        "unit_step_x": _decode_text(buffer, 180, 16),
        "unit_step_y": _decode_text(buffer, 196, 16),
        "unit_step_z": _decode_text(buffer, 212, 16),
        "unit_x": _decode_text(buffer, 228, 16),
        "unit_y": _decode_text(buffer, 244, 16),
        "unit_z": _decode_text(buffer, 260, 16),
        "xunit_ratio": _read_float32(buffer, 276),
        "yunit_ratio": _read_float32(buffer, 280),
        "zunit_ratio": _read_float32(buffer, 284),
        "seconds": _read_int16(buffer, 306),
        "minutes": _read_int16(buffer, 308),
        "hours": _read_int16(buffer, 310),
        "day": _read_int16(buffer, 312),
        "month": _read_int16(buffer, 314),
        "year": _read_int16(buffer, 316),
        "measurement_duration": _read_float32(buffer, 320),
        "compressed_data_size": _read_uint32(buffer, 324),
        "length_comment": _read_int16(buffer, 334),
        "length_private": _read_int16(buffer, 336),
        "client_zone": _decode_text(buffer, 338, 128),
        "offset_x": _read_float32(buffer, 466),
        "offset_y": _read_float32(buffer, 470),
        "offset_z": _read_float32(buffer, 474),
    }
    return header


def _read_compressed_data(raw: bytes, expected_point_bytes: int) -> bytes:
    """Read and decompress the chunked SUR data block."""

    if len(raw) < 4:
        raise ValueError("Compressed SUR payload is too short to contain directory metadata.")

    offset = 0
    directory_count = _read_uint32(raw, offset)
    offset += 4
    if directory_count <= 0:
        raise ValueError("Compressed SUR payload contains no data directories.")

    chunks: list[tuple[int, int]] = []
    for _ in range(directory_count):
        if offset + 8 > len(raw):
            raise ValueError("Compressed SUR directory is truncated.")
        raw_size = _read_uint32(raw, offset)
        zipped_size = _read_uint32(raw, offset + 4)
        offset += 8
        chunks.append((raw_size, zipped_size))

    decompressed = bytearray()
    for raw_size, zipped_size in chunks:
        if offset + zipped_size > len(raw):
            raise ValueError("Compressed SUR chunk extends past end of file.")
        chunk = raw[offset:offset + zipped_size]
        offset += zipped_size
        decoded = zlib.decompress(chunk)
        if len(decoded) != raw_size:
            raise ValueError(
                "Compressed SUR chunk size mismatch: "
                f"expected {raw_size} bytes, got {len(decoded)}."
            )
        decompressed.extend(decoded)

    if len(decompressed) != expected_point_bytes:
        raise ValueError(
            "Compressed SUR payload size mismatch: "
            f"expected {expected_point_bytes} bytes, got {len(decompressed)}."
        )
    return bytes(decompressed)


def _is_supported_studiable_type(studiable_type: int, name_object: str, name_operator: str) -> bool:
    """Check if the SUR object type can be treated as one height surface."""

    if studiable_type in {_TYPE_SURFACE, _TYPE_RGB_INTENSITY_SURFACE}:
        return True
    return studiable_type == _TYPE_PROFILE and name_object == "SCRATCH" and name_operator == "csm"


@register_surface_parser(".sur", ".spro", ".ssur")
def load_digital_surf_sur(fname: str, progress_callback=None) -> Surface:
    """Load one Digital Surf ``.sur`` file into a shared ``Surface``."""

    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {fname}")

    raw = path.read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"File '{fname}' is too short to be a valid SUR file.")

    if progress_callback:
        progress_callback(10)

    header = _read_header(raw[:_HEADER_SIZE])
    code = str(header["code"])
    if code not in {_MAGIC_CLASSIC, _MAGIC_COMPRESSED}:
        raise ValueError(f"Unsupported SUR signature '{code}' in file '{fname}'.")
    if int(header["version_number"]) != 1:
        raise ValueError(
            f"Unsupported SUR version {header['version_number']} in file '{fname}'."
        )

    if int(header["n_objects"]) != 1:
        raise ValueError(
            f"SUR file '{fname}' contains {header['n_objects']} objects; "
            "multi-object SUR files are not supported yet."
        )

    if not _is_supported_studiable_type(
        int(header["studiable_type"]),
        str(header["name_object"]),
        str(header["name_operator"]),
    ):
        raise ValueError(
            f"SUR studiable type {header['studiable_type']} is not supported for file '{fname}'."
        )

    nx = int(header["n_points_per_line"])
    ny = int(header["n_lines"])
    total_points = int(header["n_total_points"])
    if nx <= 0 or ny <= 0 or total_points != nx * ny:
        raise ValueError(f"Invalid SUR surface dimensions in file '{fname}'.")

    bits_per_point = int(header["bits_per_point"])
    if bits_per_point not in {16, 32}:
        raise ValueError(f"Unsupported SUR point size {bits_per_point} in file '{fname}'.")
    dtype = np.int16 if bits_per_point == 16 else np.int32
    point_bytes = total_points * (bits_per_point // 8)

    data_offset = _HEADER_SIZE + int(header["length_comment"]) + int(header["length_private"])
    if data_offset > len(raw):
        raise ValueError(f"SUR file '{fname}' header points past end of file.")

    payload = raw[data_offset:]
    if code == _MAGIC_COMPRESSED:
        payload = _read_compressed_data(payload, point_bytes)
    elif len(payload) < point_bytes:
        raise ValueError(f"SUR file '{fname}' data section is truncated.")
    else:
        payload = payload[:point_bytes]

    if progress_callback:
        progress_callback(60)

    grid = np.frombuffer(payload, dtype=dtype, count=total_points).reshape(ny, nx).astype(np.float64, copy=False)

    z_scale_um = _unit_to_um(str(header["unit_step_z"]))
    x_scale_um = _unit_to_um(str(header["unit_step_x"]))
    y_scale_um = _unit_to_um(str(header["unit_step_y"]))
    x0_scale_um = _unit_to_um(str(header["unit_x"]) or str(header["unit_step_x"]))
    y0_scale_um = _unit_to_um(str(header["unit_y"]) or str(header["unit_step_y"]))

    invalid_mask = None
    if int(header["non_measured_points"]) == 1:
        invalid_value = int(header["min_point"]) - 2
        invalid_mask = grid == invalid_value

    height_um = grid * float(header["spacing_z"]) * z_scale_um
    height_um += float(header["offset_z"]) * z_scale_um
    if invalid_mask is not None:
        height_um = height_um.copy()
        height_um[invalid_mask] = np.nan

    dx_um = float(header["spacing_x"]) * x_scale_um
    dy_um = float(header["spacing_y"]) * y_scale_um
    x0_um = float(header["offset_x"]) * x0_scale_um
    y0_um = float(header["offset_y"]) * y0_scale_um

    metadata = {
        "name": str(header["name_object"]) or path.stem,
        "format": "digital_surf_sur",
        "source_path": fname,
        "operator": str(header["name_operator"]),
        "studiable_type": int(header["studiable_type"]),
        "acquisition_duration_s": float(header["measurement_duration"]),
        "timestamp": (
            f"{int(header['year']):04d}-{int(header['month']):02d}-{int(header['day']):02d} "
            f"{int(header['hours']):02d}:{int(header['minutes']):02d}:{int(header['seconds']):02d}"
        ),
        "sur_header": header,
    }

    if progress_callback:
        progress_callback(100)

    logger.info(
        "Loaded SUR surface '%s': %sx%s grid, dx=%.6f um, dy=%.6f um",
        metadata["name"],
        nx,
        ny,
        dx_um,
        dy_um,
    )
    return Surface(
        height=height_um,
        dx=dx_um,
        dy=dy_um,
        x0=x0_um,
        y0=y0_um,
        unit="um",
        metadata=metadata,
    )
