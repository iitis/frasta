"""Parser for Sensofar ``.plux`` ZIP surface archives."""

from __future__ import annotations

import logging
import os
import re
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

import numpy as np

from ..surface import Surface
from .registry import register_scan_reader

logger = logging.getLogger(__name__)

_LAYER_TAG_RE = re.compile(r"^LAYER_(\d+)$")
_INFO_ITEM_RE = re.compile(r"^ITEM_(\d+)$")


def _read_archive_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one archive member while tolerating optional ``./`` prefixes."""

    candidates = [name, name.lstrip("./"), f"./{name.lstrip('./')}"]
    for candidate in candidates:
        try:
            return archive.read(candidate)
        except KeyError:
            continue
    raise KeyError(name)


def _get_required_text(root: ET.Element, path: str) -> str:
    """Return non-empty XML text for one required path."""

    text = root.findtext(path)
    if text is None:
        raise ValueError(f"Missing required PLUX XML field '{path}'.")
    text = text.strip()
    if not text:
        raise ValueError(f"Empty required PLUX XML field '{path}'.")
    return text


def _parse_iso_datetime(value: str) -> str:
    """Normalize the PLUX timestamp for metadata storage when possible."""

    value = value.strip()
    if not value:
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


def _parse_info_section(root: ET.Element) -> dict[str, str]:
    """Extract flattened metadata entries from the ``INFO`` XML section."""

    info: dict[str, str] = {}
    info_section = root.find("INFO")
    if info_section is None:
        return info

    for item in info_section:
        if not _INFO_ITEM_RE.match(item.tag):
            continue
        name = item.findtext("NAME")
        value = item.findtext("VALUE")
        if name is None or value is None:
            continue
        name = name.strip()
        value = value.strip()
        if name and value:
            info[name] = value
    return info


def _flatten_xml_leaves(node: ET.Element, prefix: str = "") -> dict[str, str]:
    """Flatten XML leaf text nodes into a ``path -> value`` dictionary."""

    tag = node.tag.strip()
    current_prefix = f"{prefix}/{tag}" if prefix else tag
    children = [child for child in node if isinstance(child.tag, str)]
    if not children:
        text = (node.text or "").strip()
        return {current_prefix: text} if text else {}

    flattened: dict[str, str] = {}
    for child in children:
        flattened.update(_flatten_xml_leaves(child, current_prefix))
    return flattened


def _parse_recipe_metadata(archive: zipfile.ZipFile) -> dict[str, str]:
    """Parse optional recipe metadata stored as XML-like text."""

    for candidate in ("recipe.txt", "./recipe.txt"):
        try:
            raw = _read_archive_member(archive, candidate)
            break
        except KeyError:
            continue
    else:
        return {}

    try:
        root = ET.fromstring(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ET.ParseError):
        return {}
    return _flatten_xml_leaves(root)


def _iter_layer_elements(root: ET.Element) -> list[tuple[int, ET.Element]]:
    """Return sorted ``(layer_id, element)`` pairs defined in the PLUX XML."""

    layers: list[tuple[int, ET.Element]] = []
    for child in root:
        if not isinstance(child.tag, str):
            continue
        match = _LAYER_TAG_RE.match(child.tag.strip())
        if not match:
            continue
        layers.append((int(match.group(1)), child))
    layers.sort(key=lambda item: item[0])
    return layers


@register_scan_reader(".plux")
def load_sensofar_plux(fname: str, progress_callback=None) -> list[Surface]:
    """Load one Sensofar PLUX archive into one or more shared ``Surface`` objects."""

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    if progress_callback:
        progress_callback(10)

    try:
        with zipfile.ZipFile(fname, "r") as archive:
            try:
                xml_payload = _read_archive_member(archive, "index.xml")
            except KeyError as exc:
                raise ValueError("Not a valid Sensofar PLUX file: missing index.xml.") from exc

            try:
                root = ET.fromstring(xml_payload.decode("utf-8-sig"))
            except (UnicodeDecodeError, ET.ParseError) as exc:
                raise ValueError(f"Invalid PLUX XML metadata: {exc}") from exc

            if progress_callback:
                progress_callback(35)

            xres = int(_get_required_text(root, "./GENERAL/IMAGE_SIZE_X"))
            yres = int(_get_required_text(root, "./GENERAL/IMAGE_SIZE_Y"))
            dx_um = float(_get_required_text(root, "./GENERAL/FOV_X"))
            dy_um = float(_get_required_text(root, "./GENERAL/FOV_Y"))
            timestamp = _parse_iso_datetime(root.findtext("./GENERAL/DATE", default=""))
            author = root.findtext("./GENERAL/AUTHOR", default="").strip()

            if xres <= 0 or yres <= 0:
                raise ValueError("PLUX surface dimensions must be positive.")
            if dx_um <= 0.0 or dy_um <= 0.0:
                raise ValueError("PLUX pixel sizes must be positive.")

            info_metadata = _parse_info_section(root)
            recipe_metadata = _parse_recipe_metadata(archive)
            layer_elements = _iter_layer_elements(root)
            if not layer_elements:
                raise ValueError("PLUX archive does not define any LAYER_* elements.")

            if progress_callback:
                progress_callback(55)

            surfaces: list[Surface] = []
            filename_stem = os.path.splitext(os.path.basename(fname))[0]
            expected_size = xres * yres

            for index, (layer_id, layer_element) in enumerate(layer_elements):
                data_name = layer_element.findtext("FILENAME_Z")
                if data_name is None or not data_name.strip():
                    logger.debug("Skipping PLUX layer %s without FILENAME_Z.", layer_id)
                    continue

                try:
                    raw_grid = _read_archive_member(archive, data_name.strip())
                except KeyError as exc:
                    raise ValueError(
                        f"PLUX layer {layer_id} references missing data file '{data_name.strip()}'."
                    ) from exc

                grid = np.frombuffer(raw_grid, dtype="<f4")
                if grid.size != expected_size:
                    raise ValueError(
                        f"PLUX layer {layer_id} has {grid.size} samples, expected {expected_size}."
                    )
                grid = grid.reshape((yres, xres)).astype(np.float64, copy=False)

                layer_metadata = {
                    "name": filename_stem if len(layer_elements) == 1 else f"{filename_stem}_L{layer_id}",
                    "format": "sensofar_plux",
                    "source_path": fname,
                    "author": author,
                    "timestamp": timestamp,
                    "layer_id": layer_id,
                    "layer_position_x_um": layer_element.findtext("POSITION_X", default="").strip(),
                    "layer_position_y_um": layer_element.findtext("POSITION_Y", default="").strip(),
                    "layer_position_z_um": layer_element.findtext("POSITION_Z", default="").strip(),
                    "info": info_metadata.copy(),
                    "recipe": recipe_metadata.copy(),
                }

                surfaces.append(
                    Surface(
                        height=grid,
                        dx=dx_um,
                        dy=dy_um,
                        x0=0.0,
                        y0=0.0,
                        unit="um",
                        metadata=layer_metadata,
                    )
                )

                if progress_callback:
                    progress_callback(55 + int(40 * (index + 1) / len(layer_elements)))

        if not surfaces:
            raise ValueError("PLUX archive does not contain any readable height layers.")

        if progress_callback:
            progress_callback(100)

        logger.info(
            "Loaded PLUX archive '%s': %s layer(s), grid=%sx%s, dx=%.6f um, dy=%.6f um",
            filename_stem,
            len(surfaces),
            xres,
            yres,
            dx_um,
            dy_um,
        )
        return surfaces
    except (FileNotFoundError, ValueError):
        raise
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid Sensofar PLUX ZIP archive: {exc}") from exc
    except Exception as exc:
        logger.error("Failed to load PLUX file '%s': %s", fname, exc)
        raise ValueError(f"Failed to load PLUX file '{fname}': {exc}") from exc
