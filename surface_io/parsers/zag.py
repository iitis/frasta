"""Keyence VR ``.zag`` surface parser."""

from __future__ import annotations

import io
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..surface import Surface
from .registry import register_scan_reader, register_surface_parser

_VALUE_SUFFIX = "e57e75b1-707b-4a6f-a095-1485b8b95efb"
_HEIGHT_KEY = "HeightImageKeys.ImageKey"
_TEXTURE_KEY = "TextureImageKeys.ImageKey"
_ERROR_KEY = "ErrorImageKeys.ImageKey"
_WIDTH_KEY = "ImageSizeKeys.WidthKey"
_HEIGHT_SIZE_KEY = "ImageSizeKeys.HeightKey"
_UNIT_KEY = "DisplayUnitKeys.UnitKey"
_CALIBRATION_KEY = "CalibrationKeys.ScannedDateTime"
_FILE_NAME_KEY = "FileItemAccessor"
_DEVICE_MODEL_KEY = "VrStorageKeys.DeviceModelNameKey"
_SCANNED_AT_KEY = "ScannedInformationKeys.ScannedDateTimeWithOffset"


@dataclass(frozen=True)
class _ZAGMeasurement:
    """Minimal metadata needed to load one measurement payload."""

    path: str
    original_file_name: str | None
    storage_keys: dict[str, str]


def _read_entry_bytes(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one entry and transparently decode embedded Zstandard frames."""

    raw = archive.read(name)
    if raw.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading this ZAG file requires the optional 'zstandard' package."
            ) from exc

        decompressor = zstandard.ZstdDecompressor()
        try:
            return decompressor.decompress(raw)
        except zstandard.ZstdError:
            with decompressor.stream_reader(io.BytesIO(raw)) as reader:
                return reader.read()
    return raw


def _decode_small_value(raw: bytes):
    """Decode compact metadata values stored as text or scalar numbers."""

    if not raw:
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None and text and all(31 < ord(ch) < 127 for ch in text):
        return text

    if len(raw) == 4:
        return struct.unpack("<I", raw)[0]
    if len(raw) == 8:
        return struct.unpack("<d", raw)[0]
    return raw


def _load_measurements(archive: zipfile.ZipFile) -> list[_ZAGMeasurement]:
    """Parse the measurement table from the ZAG metadata XML."""

    metadata_name = next(
        info.filename
        for info in archive.infolist()
        if info.filename.endswith("/ded120b2-27e3-49d3-89d3-4f1d9046d96d")
    )
    root = ET.fromstring(_read_entry_bytes(archive, metadata_name).decode("utf-8-sig"))

    measurements: list[_ZAGMeasurement] = []
    for measurement_node in root.findall("MeasurementData"):
        storage_keys = {
            item.attrib["Name"]: (item.text or "")
            for item in measurement_node.find("StorageKeys").findall("StorageKey")
        }
        measurements.append(
            _ZAGMeasurement(
                path=(measurement_node.findtext("Path") or "").strip(),
                original_file_name=measurement_node.findtext("OriginalFileName"),
                storage_keys=storage_keys,
            )
        )
    if not measurements:
        raise ValueError("ZAG archive does not contain any measurement entries.")
    return measurements


def _measurement_prefix(measurement: _ZAGMeasurement) -> str:
    """Return the prefix used by one measurement branch in the archive."""

    return f"3ce05d80-8452-4446-9274-76595a40bd4f/{measurement.path}/9eedddb6-0cf2-4021-9ffc-4e46618e365b"


def _get_storage_value(
    archive: zipfile.ZipFile,
    measurement: _ZAGMeasurement,
    key_name: str,
) -> bytes:
    """Resolve one storage key payload for a measurement."""

    key_id = measurement.storage_keys[key_name]
    entry_name = f"{_measurement_prefix(measurement)}/{key_id}/{_VALUE_SUFFIX}"
    return _read_entry_bytes(archive, entry_name)


def _parse_calibration(xml_bytes: bytes) -> tuple[float, float]:
    """Return XY pixel pitch and Z scale in micrometres."""

    root = ET.fromstring(xml_bytes.decode("utf-8-sig"))
    meter_per_pixel = float(root.findtext("./XYCalibration/MeterPerPixel"))
    meter_per_unit = float(root.findtext("./ZCalibration/MeterPerUnit"))
    return meter_per_pixel * 1e6, meter_per_unit * 1e6


def _load_measurement_surface(
    archive: zipfile.ZipFile,
    measurement: _ZAGMeasurement,
) -> Surface:
    """Load one ZAG measurement branch into a shared ``Surface``."""

    width = int(_decode_small_value(_get_storage_value(archive, measurement, _WIDTH_KEY)))
    height = int(_decode_small_value(_get_storage_value(archive, measurement, _HEIGHT_SIZE_KEY)))
    unit = str(_decode_small_value(_get_storage_value(archive, measurement, _UNIT_KEY)) or "Millimeter")
    dx_um, z_scale_um = _parse_calibration(_get_storage_value(archive, measurement, _CALIBRATION_KEY))

    height_payload = _get_storage_value(archive, measurement, _HEIGHT_KEY)
    raw_height = np.frombuffer(height_payload, dtype="<i4")
    expected_size = width * height
    if raw_height.size != expected_size:
        raise ValueError(
            f"Unexpected ZAG height payload size for measurement {measurement.path}: "
            f"got {raw_height.size}, expected {expected_size}."
        )

    height_map = raw_height.reshape(height, width).astype(np.float64, copy=False) * z_scale_um
    invalid_mask = raw_height.reshape(height, width) == -42830
    if invalid_mask.any():
        height_map = height_map.copy()
        height_map[invalid_mask] = np.nan

    metadata = {
        "name": _decode_small_value(_get_storage_value(archive, measurement, _FILE_NAME_KEY))
        or Path(measurement.original_file_name or f"measurement_{measurement.path}").stem,
        "format": "keyence_zag",
        "source_path": measurement.original_file_name,
        "device_model": _decode_small_value(_get_storage_value(archive, measurement, _DEVICE_MODEL_KEY)),
        "display_unit": unit,
        "measurement_path": measurement.path,
        "scanned_at": _decode_small_value(_get_storage_value(archive, measurement, _SCANNED_AT_KEY)),
        "xy_step_um": dx_um,
        "z_step_um": z_scale_um,
    }

    if _TEXTURE_KEY in measurement.storage_keys:
        metadata["has_texture"] = True
    if _ERROR_KEY in measurement.storage_keys:
        metadata["has_error_image"] = True

    return Surface(
        height=height_map,
        dx=dx_um,
        dy=dx_um,
        unit="um",
        metadata=metadata,
    )


@register_scan_reader(".zag")
def load_keyence_zag(fname: str, **kwargs) -> list[Surface]:
    """Load all measurement branches from one Keyence VR ``.zag`` archive."""

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported ZAG loader options: {unsupported}")

    with zipfile.ZipFile(fname) as archive:
        measurements = _load_measurements(archive)
        return [_load_measurement_surface(archive, measurement) for measurement in measurements]


@register_surface_parser(".zag")
def load_keyence_zag_surface(fname: str, *, index: int = 0, **kwargs) -> Surface:
    """Load one selected measurement from a Keyence VR ``.zag`` archive."""

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported ZAG loader options: {unsupported}")

    with zipfile.ZipFile(fname) as archive:
        measurements = _load_measurements(archive)
        try:
            measurement = measurements[index]
        except IndexError as exc:
            raise IndexError(
                f"Measurement index {index} is out of range for ZAG file '{fname}' "
                f"with {len(measurements)} measurements."
            ) from exc
        return _load_measurement_surface(archive, measurement)
