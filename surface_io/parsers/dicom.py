"""Parser for DICOM image files normalized into shared ``Surface`` objects."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..surface import Surface
from .registry import register_scan_reader

logger = logging.getLogger(__name__)

_DICOM_SUFFIXES = (".dcm", ".dicom")


def _import_pydicom():
    """Import ``pydicom`` lazily so the package stays optional until used."""

    try:
        import pydicom
    except ImportError as exc:
        raise ValueError(
            "DICOM support requires the optional 'pydicom' dependency. "
            "Install it with 'pip install pydicom'."
        ) from exc
    return pydicom


@contextmanager
def _relaxed_pydicom_validation():
    """Temporarily relax pydicom validation for imperfect vendor files."""

    pydicom = _import_pydicom()
    previous_mode = pydicom.config.settings.reading_validation_mode
    try:
        pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
        with pydicom.config.disable_value_validation():
            yield pydicom
    finally:
        pydicom.config.settings.reading_validation_mode = previous_mode


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert one DICOM scalar-like value to ``float`` with fallback."""

    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_list(values: Any) -> list[float]:
    """Convert one DICOM multi-value field into a ``list[float]``."""

    if values is None:
        return []
    try:
        return [float(value) for value in values]
    except TypeError:
        return [_safe_float(values)]
    except ValueError:
        return [_safe_float(value) for value in values]


def _slice_sort_key(dataset, fallback_index: int = 0) -> tuple[float, float]:
    """Return a stable sort key for one DICOM slice dataset."""

    slice_location = _safe_float(getattr(dataset, "SliceLocation", None), default=np.nan)
    if not np.isnan(slice_location):
        return (slice_location, float(fallback_index))

    image_position = _safe_list(getattr(dataset, "ImagePositionPatient", None))
    if len(image_position) >= 3:
        return (image_position[2], float(fallback_index))

    instance_number = _safe_float(getattr(dataset, "InstanceNumber", None), default=np.nan)
    if not np.isnan(instance_number):
        return (instance_number, float(fallback_index))

    return (float(fallback_index), float(fallback_index))


def _read_dataset(path: str, *, stop_before_pixels: bool):
    """Read one DICOM dataset with relaxed validation settings."""

    with _relaxed_pydicom_validation() as pydicom:
        return pydicom.dcmread(path, force=True, stop_before_pixels=stop_before_pixels)


def _iter_series_paths(seed_path: str) -> list[str]:
    """Collect DICOM files that belong to the same series as ``seed_path``."""

    seed_dataset = _read_dataset(seed_path, stop_before_pixels=True)
    seed_series_uid = getattr(seed_dataset, "SeriesInstanceUID", None)
    seed_path_obj = Path(seed_path)
    directory = seed_path_obj.parent

    if seed_series_uid in (None, ""):
        return [seed_path]

    candidate_paths: list[str] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in _DICOM_SUFFIXES:
            continue
        try:
            dataset = _read_dataset(str(entry), stop_before_pixels=True)
        except Exception:
            logger.debug("Skipping unreadable DICOM candidate '%s'.", entry)
            continue
        if getattr(dataset, "SeriesInstanceUID", None) == seed_series_uid:
            candidate_paths.append(str(entry))

    return candidate_paths or [seed_path]


def _read_spacing_um(dataset) -> tuple[float, float]:
    """Read physical pixel spacing and convert it from mm to um."""

    spacing = getattr(dataset, "PixelSpacing", None)
    if spacing is None:
        spacing = getattr(dataset, "ImagerPixelSpacing", None)
    spacing_values = _safe_list(spacing)
    if len(spacing_values) >= 2:
        dy_um = spacing_values[0] * 1000.0
        dx_um = spacing_values[1] * 1000.0
        if dx_um > 0.0 and dy_um > 0.0:
            return dx_um, dy_um
    return 1.0, 1.0


def _read_origin_um(dataset) -> tuple[float, float]:
    """Read the in-plane origin from ``ImagePositionPatient`` when present."""

    position = _safe_list(getattr(dataset, "ImagePositionPatient", None))
    if len(position) >= 2:
        return position[0] * 1000.0, position[1] * 1000.0
    return 0.0, 0.0


def _rescale_pixels(dataset) -> np.ndarray:
    """Decode and rescale one DICOM pixel array into floating-point values."""

    with _relaxed_pydicom_validation():
        pixel_array = np.asarray(dataset.pixel_array, dtype=np.float64)
    slope = _safe_float(getattr(dataset, "RescaleSlope", 1.0), default=1.0)
    intercept = _safe_float(getattr(dataset, "RescaleIntercept", 0.0), default=0.0)
    return pixel_array * slope + intercept


def _surface_metadata(dataset, fname: str, base_name: str, frame_index: int | None) -> dict[str, Any]:
    """Build one metadata dictionary for a normalized DICOM surface."""

    metadata = {
        "name": base_name if frame_index is None else f"{base_name}_F{frame_index:03d}",
        "format": "dicom",
        "source_path": fname,
        "modality": str(getattr(dataset, "Modality", "")),
        "series_instance_uid": str(getattr(dataset, "SeriesInstanceUID", "")),
        "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(dataset, "StudyInstanceUID", "")),
        "series_description": str(getattr(dataset, "SeriesDescription", "")),
        "instance_number": getattr(dataset, "InstanceNumber", None),
        "slice_location": getattr(dataset, "SliceLocation", None),
        "patient_position": str(getattr(dataset, "PatientPosition", "")),
        "window_center": getattr(dataset, "WindowCenter", None),
        "window_width": getattr(dataset, "WindowWidth", None),
        "rescale_slope": _safe_float(getattr(dataset, "RescaleSlope", 1.0), default=1.0),
        "rescale_intercept": _safe_float(getattr(dataset, "RescaleIntercept", 0.0), default=0.0),
    }
    if frame_index is not None:
        metadata["frame_index"] = frame_index
    return metadata


def _dataset_to_surfaces(dataset, fname: str, surface_index: int) -> list[Surface]:
    """Convert one DICOM dataset into one or more shared ``Surface`` objects."""

    pixels = _rescale_pixels(dataset)
    if pixels.ndim == 2:
        frames = [pixels]
        frame_indices = [None]
    elif pixels.ndim == 3:
        frames = [pixels[index, :, :] for index in range(pixels.shape[0])]
        frame_indices = list(range(pixels.shape[0]))
    else:
        raise ValueError(
            f"Unsupported DICOM pixel array rank {pixels.ndim}. Only 2D slices and 3D frame stacks are supported."
        )

    dx_um, dy_um = _read_spacing_um(dataset)
    x0_um, y0_um = _read_origin_um(dataset)
    stem = Path(fname).stem
    base_name = stem if len(frames) == 1 else f"{stem}_S{surface_index:03d}"

    surfaces: list[Surface] = []
    for frame_index, frame in zip(frame_indices, frames):
        metadata = _surface_metadata(dataset, fname, base_name, frame_index)
        surfaces.append(
            Surface(
                height=np.asarray(frame, dtype=np.float64),
                dx=dx_um,
                dy=dy_um,
                x0=x0_um,
                y0=y0_um,
                unit="um",
                metadata=metadata,
            )
        )
    return surfaces


@register_scan_reader(*_DICOM_SUFFIXES)
def load_dicom_series(fname: str, progress_callback=None) -> list[Surface]:
    """Load one DICOM file or a same-series directory subset into surfaces."""

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    if progress_callback:
        progress_callback(5)

    try:
        series_paths = _iter_series_paths(fname)
        datasets = []
        for index, path in enumerate(series_paths):
            datasets.append((path, _read_dataset(path, stop_before_pixels=False)))
            if progress_callback and series_paths:
                progress_callback(10 + int(35 * (index + 1) / len(series_paths)))

        datasets.sort(key=lambda item: _slice_sort_key(item[1], fallback_index=series_paths.index(item[0])))

        surfaces: list[Surface] = []
        total = max(len(datasets), 1)
        for dataset_index, (path, dataset) in enumerate(datasets):
            surfaces.extend(_dataset_to_surfaces(dataset, path, dataset_index))
            if progress_callback:
                progress_callback(45 + int(55 * (dataset_index + 1) / total))

        if not surfaces:
            raise ValueError("No readable image slices were found in the selected DICOM series.")

        logger.info(
            "Loaded DICOM input '%s': %s dataset(s), %s surface(s).",
            fname,
            len(datasets),
            len(surfaces),
        )
        return surfaces
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.error("Failed to load DICOM input '%s': %s", fname, exc)
        raise ValueError(f"Failed to load DICOM file '{fname}': {exc}") from exc
