"""Tests for reusable single-surface instrument parsers."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from frasta.core import Surface
from frasta.io import (
    get_scan_reader,
    get_surface_parser,
    load_alicona_al3d,
    load_scan_file,
    load_surface_file,
)

_AL3D_MAGIC = b"AliconaImaging\x00\r\n"
_AL3D_TAG_LAYOUT = "<20s30s2s"
_AL3D_TAG_SIZE = struct.calcsize(_AL3D_TAG_LAYOUT)
_AL3D_COMMENT_SIZE = 256


def _pack_al3d_tag(key: str, value: str | int | float) -> bytes:
    """Create one fixed-size AL3D header tag."""

    return struct.pack(
        _AL3D_TAG_LAYOUT,
        key.encode("latin-1"),
        str(value).encode("latin-1"),
        b"\r\n",
    )


def _build_al3d_file(
    grid_m: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    invalid_value: float,
    comment: str = "",
) -> bytes:
    """Build a minimal AL3D byte stream for parser tests."""

    rows, cols = grid_m.shape
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
    depth_offset = len(_AL3D_MAGIC) + (2 + len(tags)) * _AL3D_TAG_SIZE + _AL3D_COMMENT_SIZE
    tags[3] = ("DepthImageOffset", depth_offset)

    header = bytearray()
    header.extend(_AL3D_MAGIC)
    header.extend(_pack_al3d_tag("Version", 1))
    header.extend(_pack_al3d_tag("TagCount", len(tags)))
    for key, value in tags:
        header.extend(_pack_al3d_tag(key, value))

    comment_bytes = comment.encode("latin-1")
    if len(comment_bytes) > _AL3D_COMMENT_SIZE - 3:
        raise ValueError("AL3D comment too long for test fixture.")
    comment_block = comment_bytes + b"\x00" * (_AL3D_COMMENT_SIZE - len(comment_bytes) - 2) + b"\r\n"
    header.extend(comment_block)

    rowstride = ((cols * 4) + 7) // 8 * 8
    payload = bytearray()
    for row in grid_m.astype("<f4"):
        raw_row = row.tobytes()
        payload.extend(raw_row)
        payload.extend(b"\x00" * (rowstride - len(raw_row)))

    return bytes(header + payload)


class TestSurfaceParsers:
    """Test suite for reusable instrument parsers."""

    def test_load_alicona_al3d_returns_surface(self, tmp_path):
        """AL3D parser should normalize depth data into a ``Surface``."""

        invalid = -9999.0
        grid_m = np.array(
            [
                [1.0e-6, invalid, 3.0e-6],
                [4.0e-6, 5.0e-6, 6.0e-6],
            ],
            dtype=np.float32,
        )
        payload = _build_al3d_file(
            grid_m,
            dx_m=2.5e-6,
            dy_m=4.0e-6,
            invalid_value=invalid,
            comment="Synthetic Alicona sample",
        )
        path = tmp_path / "synthetic.al3d"
        path.write_bytes(payload)

        progress = []
        surface = load_alicona_al3d(str(path), progress_callback=progress.append)

        assert isinstance(surface, Surface)
        assert surface.height.shape == (2, 3)
        assert surface.dx == pytest.approx(2.5)
        assert surface.dy == pytest.approx(4.0)
        assert surface.height[0, 0] == pytest.approx(1.0)
        assert np.isnan(surface.height[0, 1])
        assert surface.height[1, 2] == pytest.approx(6.0)
        assert surface.metadata["name"] == "synthetic"
        assert surface.metadata["format"] == "alicona_al3d"
        assert surface.metadata["comment"] == "Synthetic Alicona sample"
        assert surface.metadata["al3d_tags"]["Cols"] == "3"
        assert progress[0] == 10
        assert progress[-1] == 100

    def test_load_surface_file_dispatches_registered_parser(self, tmp_path):
        """Registry-based loading should dispatch ``.al3d`` files automatically."""

        payload = _build_al3d_file(
            np.array([[1.0e-6, 2.0e-6]], dtype=np.float32),
            dx_m=1.0e-6,
            dy_m=3.0e-6,
            invalid_value=float("nan"),
        )
        path = tmp_path / "dispatch.al3d"
        path.write_bytes(payload)

        parser = get_surface_parser(str(path))
        surface = load_surface_file(str(path))

        assert parser is load_alicona_al3d
        assert surface.height.shape == (1, 2)
        assert surface.height[0, 1] == pytest.approx(2.0)

    def test_load_scan_file_normalizes_single_surface_readers(self, tmp_path):
        """Unified scan loading should wrap single-surface readers in a list."""

        payload = _build_al3d_file(
            np.array([[1.0e-6, 2.0e-6, 3.0e-6]], dtype=np.float32),
            dx_m=2.0e-6,
            dy_m=4.0e-6,
            invalid_value=float("nan"),
        )
        path = tmp_path / "single.al3d"
        path.write_bytes(payload)

        reader = get_scan_reader(str(path))
        surfaces = load_scan_file(str(path))

        assert callable(reader)
        assert len(surfaces) == 1
        assert surfaces[0].height.shape == (1, 3)
        assert surfaces[0].dx == pytest.approx(2.0)

    def test_load_scan_file_handles_multi_surface_archives(self, temp_npz_file):
        """Unified scan loading should preserve multi-surface archive readers."""

        reader = get_scan_reader(temp_npz_file)
        surfaces = load_scan_file(temp_npz_file)

        assert callable(reader)
        assert len(surfaces) == 1
        assert surfaces[0].metadata["name"] == "test_scan"

    def test_get_surface_parser_rejects_unknown_suffix(self):
        """Unknown suffixes should fail with a clear error."""

        with pytest.raises(ValueError, match="Unsupported surface parser format"):
            get_surface_parser(".unknown")

    def test_get_scan_reader_rejects_unknown_suffix(self):
        """Unknown suffixes should fail with a clear reader-registry error."""

        with pytest.raises(ValueError, match="Unsupported scan reader format"):
            get_scan_reader(".unknown")
