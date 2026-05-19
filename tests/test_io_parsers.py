"""Tests for reusable instrument parsers and scan readers."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from frasta.core import Surface
from frasta.io import (
    get_scan_reader,
    get_surface_parser,
    load_alicona_al3d,
    load_dicom_series,
    load_sensofar_plux,
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


def _build_plux_file(
    layers_um: list[np.ndarray],
    *,
    dx_um: float,
    dy_um: float,
    author: str = "Unit Test",
    timestamp: str = "2026-05-17T12:34:56",
    recipe_comment: str = "Synthetic recipe",
) -> bytes:
    """Build a minimal PLUX ZIP archive for parser tests."""

    if not layers_um:
        raise ValueError("PLUX fixture requires at least one layer.")

    yres, xres = layers_um[0].shape
    info_items = [
        ("Device", "Sensofar S Test"),
        ("Objective", "20x"),
        ("Technique", "Confocal"),
        ("Comment", "Synthetic PLUX sample"),
    ]
    info_xml = "\n".join(
        f"""
    <ITEM_{index}>
      <NAME>{name}</NAME>
      <VALUE>{value}</VALUE>
    </ITEM_{index}>""".rstrip()
        for index, (name, value) in enumerate(info_items)
    )
    layer_xml = "\n".join(
        f"""
  <LAYER_{layer_id}>
    <FILENAME_Z>LAYER_{layer_id}.raw</FILENAME_Z>
    <POSITION_X>{1.5 * layer_id:.2f}</POSITION_X>
    <POSITION_Y>{2.5 * layer_id:.2f}</POSITION_Y>
    <POSITION_Z>{3.5 * layer_id:.2f}</POSITION_Z>
  </LAYER_{layer_id}>""".rstrip()
        for layer_id, _layer in enumerate(layers_um)
    )
    index_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<xml>
  <GENERAL>
    <AUTHOR>{author}</AUTHOR>
    <DATE>{timestamp}</DATE>
    <IMAGE_SIZE_X>{xres}</IMAGE_SIZE_X>
    <IMAGE_SIZE_Y>{yres}</IMAGE_SIZE_Y>
    <FOV_X>{dx_um}</FOV_X>
    <FOV_Y>{dy_um}</FOV_Y>
  </GENERAL>
  <INFO>
    <SIZE>{len(info_items)}</SIZE>
{info_xml}
  </INFO>
{layer_xml}
</xml>
"""
    recipe_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<xml>
  <CAPTURE>
    <STEP>Coarse</STEP>
    <COMMENT>{recipe_comment}</COMMENT>
  </CAPTURE>
</xml>
"""

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.xml", index_xml.encode("utf-8"))
        archive.writestr("./recipe.txt", recipe_xml.encode("utf-8"))
        for layer_id, layer in enumerate(layers_um):
            archive.writestr(f"LAYER_{layer_id}.raw", np.asarray(layer, dtype="<f4").tobytes())
    return payload.getvalue()


def _build_dicom_file(
    path: Path,
    pixel_array: np.ndarray,
    *,
    series_instance_uid: str,
    sop_instance_uid: str,
    instance_number: int,
    z_position_mm: float,
    spacing_row_mm: float = 0.2,
    spacing_col_mm: float = 0.3,
    intercept: float = -1024.0,
    slope: float = 2.0,
) -> None:
    """Build one minimal grayscale DICOM file for parser tests."""

    pydicom = pytest.importorskip("pydicom")

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    dataset = pydicom.dataset.FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.StudyInstanceUID = pydicom.uid.generate_uid()
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Synthetic CT"
    dataset.InstanceNumber = instance_number
    dataset.ImagePositionPatient = [10.0, 20.0, z_position_mm]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [spacing_row_mm, spacing_col_mm]
    dataset.SliceLocation = z_position_mm
    dataset.Rows = int(pixel_array.shape[0])
    dataset.Columns = int(pixel_array.shape[1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 1
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.RescaleIntercept = intercept
    dataset.RescaleSlope = slope
    dataset.PixelData = np.asarray(pixel_array, dtype=np.int16).tobytes()
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.save_as(str(path), write_like_original=False)


class TestSurfaceParsers:
    """Test suite for reusable instrument parsers and scan readers."""

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

    def test_load_sensofar_plux_returns_all_height_layers(self, tmp_path):
        """PLUX reader should normalize each ``LAYER_*`` height file into a ``Surface``."""

        payload = _build_plux_file(
            [
                np.array([[1.0, np.nan], [3.5, 4.5]], dtype=np.float32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ],
            dx_um=0.8,
            dy_um=1.2,
        )
        path = tmp_path / "synthetic.plux"
        path.write_bytes(payload)

        progress = []
        surfaces = load_sensofar_plux(str(path), progress_callback=progress.append)

        assert len(surfaces) == 2
        assert all(isinstance(surface, Surface) for surface in surfaces)
        assert surfaces[0].height.shape == (2, 2)
        assert np.isnan(surfaces[0].height[0, 1])
        assert surfaces[0].dx == pytest.approx(0.8)
        assert surfaces[0].dy == pytest.approx(1.2)
        assert surfaces[0].metadata["format"] == "sensofar_plux"
        assert surfaces[0].metadata["info"]["Device"] == "Sensofar S Test"
        assert surfaces[0].metadata["recipe"]["xml/CAPTURE/STEP"] == "Coarse"
        assert surfaces[1].metadata["name"] == "synthetic_L1"
        assert progress[0] == 10
        assert progress[-1] == 100

    def test_load_scan_file_dispatches_registered_plux_reader(self, tmp_path):
        """Registry-based scan loading should dispatch ``.plux`` files automatically."""

        payload = _build_plux_file(
            [np.array([[10.0, 11.0, 12.0]], dtype=np.float32)],
            dx_um=1.5,
            dy_um=2.5,
        )
        path = tmp_path / "dispatch.plux"
        path.write_bytes(payload)

        reader = get_scan_reader(str(path))
        surfaces = load_scan_file(str(path))

        assert callable(reader)
        assert len(surfaces) == 1
        assert surfaces[0].height[0, 2] == pytest.approx(12.0)
        assert surfaces[0].metadata["name"] == "dispatch"

    def test_load_dicom_series_returns_sorted_surfaces(self, tmp_path):
        """DICOM reader should normalize one same-series directory subset into surfaces."""

        pytest.importorskip("pydicom")
        series_uid = "1.2.826.0.1.3680043.2.1125.10"
        second_slice = tmp_path / "slice_02.dcm"
        first_slice = tmp_path / "slice_01.dcm"
        other_series = tmp_path / "other.dcm"

        _build_dicom_file(
            second_slice,
            np.array([[10, 20], [30, 40]], dtype=np.int16),
            series_instance_uid=series_uid,
            sop_instance_uid="1.2.826.0.1.3680043.2.1125.10.2",
            instance_number=2,
            z_position_mm=2.0,
        )
        _build_dicom_file(
            first_slice,
            np.array([[1, 2], [3, 4]], dtype=np.int16),
            series_instance_uid=series_uid,
            sop_instance_uid="1.2.826.0.1.3680043.2.1125.10.1",
            instance_number=1,
            z_position_mm=1.0,
        )
        _build_dicom_file(
            other_series,
            np.array([[99, 98], [97, 96]], dtype=np.int16),
            series_instance_uid="1.2.826.0.1.3680043.2.1125.11",
            sop_instance_uid="1.2.826.0.1.3680043.2.1125.11.1",
            instance_number=1,
            z_position_mm=5.0,
        )

        progress = []
        surfaces = load_dicom_series(str(second_slice), progress_callback=progress.append)

        assert len(surfaces) == 2
        assert all(isinstance(surface, Surface) for surface in surfaces)
        assert surfaces[0].height.shape == (2, 2)
        assert surfaces[0].height[0, 0] == pytest.approx(-1022.0)
        assert surfaces[1].height[1, 1] == pytest.approx(-944.0)
        assert surfaces[0].dx == pytest.approx(300.0)
        assert surfaces[0].dy == pytest.approx(200.0)
        assert surfaces[0].x0 == pytest.approx(10000.0)
        assert surfaces[0].y0 == pytest.approx(20000.0)
        assert surfaces[0].metadata["format"] == "dicom"
        assert surfaces[0].metadata["series_instance_uid"] == series_uid
        assert progress[0] == 5
        assert progress[-1] == 100

    def test_load_scan_file_dispatches_registered_dicom_reader(self, tmp_path):
        """Registry-based scan loading should dispatch ``.dcm`` files automatically."""

        pytest.importorskip("pydicom")
        path = tmp_path / "single_slice.dcm"
        _build_dicom_file(
            path,
            np.array([[7, 8, 9]], dtype=np.int16),
            series_instance_uid="1.2.826.0.1.3680043.2.1125.12",
            sop_instance_uid="1.2.826.0.1.3680043.2.1125.12.1",
            instance_number=1,
            z_position_mm=0.0,
        )

        reader = get_scan_reader(str(path))
        surfaces = load_scan_file(str(path))

        assert callable(reader)
        assert reader is load_dicom_series
        assert len(surfaces) == 1
        assert surfaces[0].height[0, 2] == pytest.approx(-1006.0)

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
