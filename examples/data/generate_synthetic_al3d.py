"""Generate a small synthetic Alicona AL3D demo file.

The generated file is useful for manual parser checks and GUI smoke tests.
It contains a smooth synthetic surface with one invalid pixel encoded using
the AL3D invalid-value mechanism.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"AliconaImaging\x00\r\n"
TAG_LAYOUT = "<20s30s2s"
COMMENT_SIZE = 256
OUTPUT_PATH = Path(__file__).with_name("synthetic_alicona_demo.al3d")


def pack_tag(key: str, value: str | int | float) -> bytes:
    """Pack one fixed-size AL3D header tag."""

    return struct.pack(
        TAG_LAYOUT,
        key.encode("latin-1"),
        str(value).encode("latin-1"),
        b"\r\n",
    )


def build_demo_surface() -> np.ndarray:
    """Create a smooth synthetic surface in meters."""

    rows, cols = 18, 24
    x = np.linspace(-1.0, 1.0, cols)
    y = np.linspace(-1.0, 1.0, rows)
    xx, yy = np.meshgrid(x, y)

    # Smooth dome with a shallow trough and one invalid point.
    height_um = (
        18.0 * np.exp(-2.8 * (xx * xx + yy * yy))
        - 4.0 * np.exp(-18.0 * ((xx - 0.35) ** 2 + (yy + 0.20) ** 2))
        + 1.5 * xx
    )
    height_um[6, 11] = np.nan
    return height_um * 1e-6


def write_demo_file(path: Path) -> Path:
    """Write the synthetic AL3D file to ``path``."""

    invalid_value = -9999.0
    grid_m = build_demo_surface().astype(np.float32)
    rows, cols = grid_m.shape
    dx_m = 3.5e-6
    dy_m = 5.0e-6

    tags = [
        ("Cols", cols),
        ("Rows", rows),
        ("IconOffset", 0),
        ("DepthImageOffset", 0),
        ("InvalidPixelValue", invalid_value),
        ("PixelSizeYMeter", dy_m),
        ("PixelSizeXMeter", dx_m),
        ("NumberOfPlanes", 0),
        ("TextureImageOffset", 0),
    ]
    tag_size = struct.calcsize(TAG_LAYOUT)
    depth_offset = len(MAGIC) + (2 + len(tags)) * tag_size + COMMENT_SIZE
    tags[3] = ("DepthImageOffset", depth_offset)

    comment = (
        "Synthetic Alicona AL3D demo generated for FRASTA parser checks."
    ).encode("latin-1")
    comment_block = comment + b"\x00" * (COMMENT_SIZE - len(comment) - 2) + b"\r\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(pack_tag("Version", 1))
        handle.write(pack_tag("TagCount", len(tags)))
        for key, value in tags:
            handle.write(pack_tag(key, value))
        handle.write(comment_block)

        rowstride = ((cols * 4) + 7) // 8 * 8
        for row in grid_m:
            encoded_row = np.where(np.isnan(row), invalid_value, row).astype("<f4").tobytes()
            handle.write(encoded_row)
            handle.write(b"\x00" * (rowstride - len(encoded_row)))

    return path


def main() -> None:
    """Generate the demo AL3D file and print its path."""

    path = write_demo_file(OUTPUT_PATH)
    print(path)


if __name__ == "__main__":
    main()
