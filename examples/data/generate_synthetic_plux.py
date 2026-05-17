"""Generate a small synthetic Sensofar PLUX demo file.

The generated archive is useful for manual parser checks and GUI smoke tests.
It contains two synthetic height layers encoded as little-endian float32
grids together with the minimal XML metadata expected by the PLUX reader.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path(__file__).with_name("synthetic_sensofar_demo.plux")


def build_demo_layers() -> list[np.ndarray]:
    """Create two smooth synthetic surface layers in micrometers."""

    rows, cols = 20, 28
    x = np.linspace(-1.0, 1.0, cols)
    y = np.linspace(-1.0, 1.0, rows)
    xx, yy = np.meshgrid(x, y)

    # Layer 0: smooth dome with one missing sample.
    layer_0 = (
        22.0 * np.exp(-3.2 * (xx * xx + yy * yy))
        - 6.0 * np.exp(-20.0 * ((xx - 0.30) ** 2 + (yy + 0.10) ** 2))
        + 1.8 * xx
    )
    layer_0[7, 13] = np.nan

    # Layer 1: slightly shifted variant to exercise multi-layer loading.
    layer_1 = (
        18.0 * np.exp(-2.4 * ((xx + 0.12) ** 2 + (yy - 0.08) ** 2))
        + 2.5 * np.sin(np.pi * xx) * np.cos(0.6 * np.pi * yy)
        - 1.2 * yy
    )
    layer_1[11, 8] = np.nan

    return [layer_0.astype(np.float32), layer_1.astype(np.float32)]


def build_index_xml(
    *,
    xres: int,
    yres: int,
    dx_um: float,
    dy_um: float,
    layer_count: int,
) -> str:
    """Build the minimal `index.xml` payload for the synthetic archive."""

    info_items = [
        ("Device", "Sensofar S synthetic"),
        ("Objective", "20x"),
        ("Technique", "Confocal"),
        ("Measurement Type", "Topography"),
        ("Algorithm", "Synthetic generator"),
        ("Comment", "Synthetic Sensofar PLUX demo for FRASTA parser checks."),
    ]
    info_xml = "\n".join(
        f"""
    <ITEM_{index}>
      <NAME>{name}</NAME>
      <VALUE>{value}</VALUE>
    </ITEM_{index}>""".rstrip()
        for index, (name, value) in enumerate(info_items)
    )
    layers_xml = "\n".join(
        f"""
  <LAYER_{layer_id}>
    <FILENAME_Z>LAYER_{layer_id}.raw</FILENAME_Z>
    <POSITION_X>{1.25 * layer_id:.2f}</POSITION_X>
    <POSITION_Y>{2.50 * layer_id:.2f}</POSITION_Y>
    <POSITION_Z>{3.75 * layer_id:.2f}</POSITION_Z>
  </LAYER_{layer_id}>""".rstrip()
        for layer_id in range(layer_count)
    )

    return f"""<?xml version="1.0" encoding="utf-8"?>
<xml>
  <GENERAL>
    <AUTHOR>FRASTA-toolbox</AUTHOR>
    <DATE>2026-05-17T12:00:00</DATE>
    <IMAGE_SIZE_X>{xres}</IMAGE_SIZE_X>
    <IMAGE_SIZE_Y>{yres}</IMAGE_SIZE_Y>
    <FOV_X>{dx_um}</FOV_X>
    <FOV_Y>{dy_um}</FOV_Y>
  </GENERAL>
  <INFO>
    <SIZE>{len(info_items)}</SIZE>
{info_xml}
  </INFO>
{layers_xml}
</xml>
"""


def build_recipe_xml() -> str:
    """Build the optional `recipe.txt` payload stored in the PLUX archive."""

    return """<?xml version="1.0" encoding="utf-8"?>
<xml>
  <CAPTURE>
    <STEP>Coarse</STEP>
    <COMMENT>Synthetic recipe metadata for parser verification.</COMMENT>
  </CAPTURE>
</xml>
"""


def write_demo_file(path: Path) -> Path:
    """Write the synthetic PLUX archive to ``path``."""

    layers = build_demo_layers()
    yres, xres = layers[0].shape
    dx_um = 0.85
    dy_um = 1.10
    index_xml = build_index_xml(
        xres=xres,
        yres=yres,
        dx_um=dx_um,
        dy_um=dy_um,
        layer_count=len(layers),
    )
    recipe_xml = build_recipe_xml()

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.xml", index_xml.encode("utf-8"))
        archive.writestr("./recipe.txt", recipe_xml.encode("utf-8"))
        for layer_id, layer in enumerate(layers):
            archive.writestr(f"LAYER_{layer_id}.raw", np.asarray(layer, dtype="<f4").tobytes())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.getvalue())
    return path


def main() -> None:
    """Generate the demo PLUX file and print its path."""

    path = write_demo_file(OUTPUT_PATH)
    print(path)


if __name__ == "__main__":
    main()
