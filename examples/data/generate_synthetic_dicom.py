"""Generate a small synthetic DICOM series for parser and GUI smoke tests.

The writer intentionally avoids a ``pydicom`` dependency so the demo data can
be regenerated in a plain FRASTA environment. The produced files are minimal
Explicit VR Little Endian grayscale CT slices with deterministic metadata.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).with_name("synthetic_dicom_series")
TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.1"
CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"
IMPLEMENTATION_UID = "1.2.826.0.1.3680043.10.5432.1"
STUDY_UID = "1.2.826.0.1.3680043.10.5432.100"
SERIES_UID = "1.2.826.0.1.3680043.10.5432.200"


def _even_length_bytes(payload: bytes, pad_byte: bytes = b" ") -> bytes:
    """Pad one payload to the even length required by DICOM."""

    if len(payload) % 2 == 1:
        payload += pad_byte
    return payload


def _encode_text(value: str, *, pad_byte: bytes = b" ") -> bytes:
    """Encode one text-like DICOM value with even-length padding."""

    return _even_length_bytes(value.encode("ascii"), pad_byte=pad_byte)


def _encode_ui(value: str) -> bytes:
    """Encode one UID value using the DICOM-required NUL padding."""

    return _even_length_bytes(value.encode("ascii"), pad_byte=b"\x00")


def _pack_short_vr(tag: tuple[int, int], vr: str, value: bytes) -> bytes:
    """Pack one Explicit VR element that stores a 16-bit length."""

    return struct.pack("<HH2sH", tag[0], tag[1], vr.encode("ascii"), len(value)) + value


def _pack_long_vr(tag: tuple[int, int], vr: str, value: bytes) -> bytes:
    """Pack one Explicit VR element that stores a 32-bit length."""

    return (
        struct.pack("<HH2s2sI", tag[0], tag[1], vr.encode("ascii"), b"\x00\x00", len(value))
        + value
    )


def _build_file_meta(sop_instance_uid: str) -> bytes:
    """Build the DICOM File Meta Information group."""

    elements = [
        _pack_long_vr((0x0002, 0x0001), "OB", b"\x00\x01"),
        _pack_short_vr((0x0002, 0x0002), "UI", _encode_ui(CT_IMAGE_STORAGE_UID)),
        _pack_short_vr((0x0002, 0x0003), "UI", _encode_ui(sop_instance_uid)),
        _pack_short_vr((0x0002, 0x0010), "UI", _encode_ui(TRANSFER_SYNTAX_UID)),
        _pack_short_vr((0x0002, 0x0012), "UI", _encode_ui(IMPLEMENTATION_UID)),
    ]
    body = b"".join(elements)
    group_length = _pack_short_vr((0x0002, 0x0000), "UL", struct.pack("<I", len(body)))
    return group_length + body


def _build_dataset(
    pixel_array: np.ndarray,
    *,
    sop_instance_uid: str,
    instance_number: int,
    z_position_mm: float,
) -> bytes:
    """Build one minimal CT slice dataset with signed 16-bit pixels."""

    rows, cols = pixel_array.shape
    pixel_bytes = np.asarray(pixel_array, dtype="<i2").tobytes()
    image_position = f"-12.5\\8.0\\{z_position_mm:.3f}"
    image_orientation = "1\\0\\0\\0\\1\\0"
    pixel_spacing = "0.180\\0.220"
    window_center = "20"
    window_width = "280"

    parts = [
        _pack_short_vr((0x0008, 0x0016), "UI", _encode_ui(CT_IMAGE_STORAGE_UID)),
        _pack_short_vr((0x0008, 0x0018), "UI", _encode_ui(sop_instance_uid)),
        _pack_short_vr((0x0008, 0x0060), "CS", _encode_text("CT")),
        _pack_short_vr((0x0008, 0x103E), "LO", _encode_text("Synthetic FRASTA CT demo")),
        _pack_short_vr((0x0010, 0x0010), "PN", _encode_text("Synthetic^FRASTA")),
        _pack_short_vr((0x0010, 0x0020), "LO", _encode_text("FRASTA_DEMO")),
        _pack_short_vr((0x0020, 0x000D), "UI", _encode_ui(STUDY_UID)),
        _pack_short_vr((0x0020, 0x000E), "UI", _encode_ui(SERIES_UID)),
        _pack_short_vr((0x0020, 0x0013), "IS", _encode_text(str(instance_number))),
        _pack_short_vr((0x0020, 0x0032), "DS", _encode_text(image_position)),
        _pack_short_vr((0x0020, 0x0037), "DS", _encode_text(image_orientation)),
        _pack_short_vr((0x0020, 0x1041), "DS", _encode_text(f"{z_position_mm:.3f}")),
        _pack_short_vr((0x0028, 0x0002), "US", struct.pack("<H", 1)),
        _pack_short_vr((0x0028, 0x0004), "CS", _encode_text("MONOCHROME2")),
        _pack_short_vr((0x0028, 0x0010), "US", struct.pack("<H", rows)),
        _pack_short_vr((0x0028, 0x0011), "US", struct.pack("<H", cols)),
        _pack_short_vr((0x0028, 0x0030), "DS", _encode_text(pixel_spacing)),
        _pack_short_vr((0x0028, 0x0100), "US", struct.pack("<H", 16)),
        _pack_short_vr((0x0028, 0x0101), "US", struct.pack("<H", 16)),
        _pack_short_vr((0x0028, 0x0102), "US", struct.pack("<H", 15)),
        _pack_short_vr((0x0028, 0x0103), "US", struct.pack("<H", 1)),
        _pack_short_vr((0x0028, 0x1052), "DS", _encode_text("-1000")),
        _pack_short_vr((0x0028, 0x1053), "DS", _encode_text("2")),
        _pack_short_vr((0x0028, 0x1050), "DS", _encode_text(window_center)),
        _pack_short_vr((0x0028, 0x1051), "DS", _encode_text(window_width)),
        _pack_long_vr((0x7FE0, 0x0010), "OW", _even_length_bytes(pixel_bytes, pad_byte=b"\x00")),
    ]
    return b"".join(parts)


def build_demo_slices() -> list[np.ndarray]:
    """Create two deterministic signed-intensity slices before HU rescaling."""

    rows, cols = 32, 40
    x = np.linspace(-1.0, 1.0, cols)
    y = np.linspace(-1.0, 1.0, rows)
    xx, yy = np.meshgrid(x, y)

    slice_a = (
        -120
        + 90 * np.exp(-3.5 * (xx * xx + yy * yy))
        + 18 * np.sin(2.0 * np.pi * xx)
        - 12 * yy
    )
    slice_b = (
        -80
        + 65 * np.exp(-4.2 * ((xx - 0.18) ** 2 + (yy + 0.12) ** 2))
        - 20 * np.cos(1.5 * np.pi * yy)
        + 10 * xx
    )
    return [np.rint(slice_a).astype(np.int16), np.rint(slice_b).astype(np.int16)]


def write_demo_series(path: Path) -> Path:
    """Write the synthetic DICOM series to ``path`` and return that directory."""

    path.mkdir(parents=True, exist_ok=True)
    for index, pixel_array in enumerate(build_demo_slices(), start=1):
        sop_instance_uid = f"{SERIES_UID}.{index}"
        payload = bytearray()
        payload.extend(b"\x00" * 128)
        payload.extend(b"DICM")
        payload.extend(_build_file_meta(sop_instance_uid))
        payload.extend(
            _build_dataset(
                pixel_array,
                sop_instance_uid=sop_instance_uid,
                instance_number=index,
                z_position_mm=1.5 * index,
            )
        )
        output_path = path / f"slice_{index:02d}.dcm"
        output_path.write_bytes(payload)
    return path


def main() -> None:
    """Generate the demo DICOM series and print its directory path."""

    path = write_demo_series(OUTPUT_DIR)
    print(path)


if __name__ == "__main__":
    main()
