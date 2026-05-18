"""Generate deterministic synthetic datasets for crack-path GUI checks.

The generator creates small aligned surface pairs tailored to the current
contact-map and crack-path tortuosity workflow. Each output dataset contains
two surfaces saved as an NPZ file plus a JSON manifest with the recommended
analysis settings and the reference metrics computed from the current
``first_open_pixel`` crack-path backend.

Generated files:

- ``crack_path_straight_demo.npz`` / ``.json``
- ``crack_path_wavy_demo.npz`` / ``.json``
- ``crack_path_y_axis_demo.npz`` / ``.json``
- ``crack_path_realistic_demo.npz`` / ``.json``

Usage:
    python examples/data/generate_crack_path_demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure the local package is importable when the script is run directly.
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from frasta.core import Surface
from frasta.io.exporters import save_npz
from frasta.processing.crack_path import analyze_crack_path


OUTPUT_DIR = Path(__file__).resolve().parent
GRID_SHAPE = (96, 128)
DX = 4.0
DY = 4.0
OPENING_UM = 18.0
THRESHOLD_UM = 9.0
RNG_REALISTIC = np.random.default_rng(20260518)


def _base_surface(shape: tuple[int, int]) -> np.ndarray:
    """Create a smooth deterministic reference topography in micrometers."""
    rows, cols = shape
    x = np.arange(cols, dtype=float) * DX
    y = np.arange(rows, dtype=float) * DY
    xx, yy = np.meshgrid(x, y)
    return (
        0.9 * np.sin(2.0 * np.pi * xx / 180.0)
        + 0.6 * np.cos(2.0 * np.pi * yy / 220.0)
        + 0.25 * np.sin(2.0 * np.pi * (xx + yy) / 150.0)
    ).astype(float)


def _base_surface_realistic(shape: tuple[int, int]) -> np.ndarray:
    """Create a richer deterministic topography with multi-scale roughness."""
    rows, cols = shape
    x = np.arange(cols, dtype=float) * DX
    y = np.arange(rows, dtype=float) * DY
    xx, yy = np.meshgrid(x, y)

    low_freq = (
        2.8 * np.sin(2.0 * np.pi * xx / 310.0 + 0.25)
        + 2.1 * np.cos(2.0 * np.pi * yy / 270.0 - 0.6)
        + 1.6 * np.sin(2.0 * np.pi * (xx + 0.7 * yy) / 420.0)
    )
    mid_freq = (
        1.1 * np.sin(2.0 * np.pi * xx / 74.0 + 0.8)
        * np.cos(2.0 * np.pi * yy / 88.0 - 0.2)
        + 0.9 * np.cos(2.0 * np.pi * (xx - yy) / 63.0)
    )
    high_freq = 0.55 * RNG_REALISTIC.standard_normal(shape)

    # A few broad deterministic asperity clusters make the map less idealized.
    asperities = np.zeros(shape, dtype=float)
    for cx_um, cy_um, amp, sx_um, sy_um in (
        (180.0, 120.0, 5.0, 34.0, 28.0),
        (420.0, 210.0, -4.2, 40.0, 24.0),
        (620.0, 300.0, 3.6, 30.0, 36.0),
        (760.0, 110.0, -3.1, 22.0, 30.0),
    ):
        asperities += amp * np.exp(
            -0.5 * (((xx - cx_um) / sx_um) ** 2 + ((yy - cy_um) / sy_um) ** 2)
        )

    return (low_freq + mid_freq + high_freq + asperities).astype(float)


def _add_holes(
    values: np.ndarray,
    rng: np.random.Generator,
    hole_count: int = 14,
    radius_range: tuple[int, int] = (2, 5),
) -> np.ndarray:
    """Punch circular NaN holes into a surface to emulate measurement gaps."""
    result = np.asarray(values, dtype=float).copy()
    rows, cols = result.shape
    min_r, max_r = radius_range

    for _ in range(hole_count):
        cy = int(rng.integers(10, rows - 10))
        cx = int(rng.integers(10, cols - 10))
        radius = int(rng.integers(min_r, max_r + 1))
        yy, xx = np.ogrid[:rows, :cols]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        result[mask] = np.nan

    return result


def _opening_from_front_rows(front_rows: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Build an opening field for fronts that advance along the X axis."""
    rows, cols = shape
    opening = np.zeros(shape, dtype=float)
    for col in range(cols):
        row0 = int(front_rows[col])
        opening[row0:, col] = OPENING_UM
    return opening


def _opening_from_front_cols(front_cols: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Build an opening field for fronts that advance along the Y axis."""
    rows, cols = shape
    opening = np.zeros(shape, dtype=float)
    for row in range(rows):
        col0 = int(front_cols[row])
        opening[row, col0:] = OPENING_UM
    return opening


def _save_dataset(
    stem: str,
    surface_a: Surface,
    surface_b: Surface,
    analysis_settings: dict[str, object],
) -> None:
    """Save one NPZ dataset and a JSON manifest with expected crack-path metrics."""
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    json_path = OUTPUT_DIR / f"{stem}.json"

    save_npz(
        str(npz_path),
        [
            ("Surface_A", surface_a),
            ("Surface_B", surface_b),
        ],
    )

    result = analyze_crack_path(
        surface_a,
        surface_b,
        dx=surface_a.dx,
        dy=surface_a.dy,
        separation=float(analysis_settings["threshold_um"]),
        propagation_axis=str(analysis_settings["propagation_axis"]),
        front_side=str(analysis_settings["front_side"]),
    )
    abs_curvature = np.abs(np.asarray(result["curvature"], dtype=float))

    manifest = {
        "dataset": stem,
        "grid_shape": list(surface_a.height.shape),
        "dx_um": float(surface_a.dx),
        "dy_um": float(surface_a.dy),
        **analysis_settings,
        "expected_metrics": {
            "path_method": str(result["path_method"]),
            "path_point_count": int(len(result["path_points"])),
            "effective_length_um": round(float(result["effective_length"]), 6),
            "projected_length_um": round(float(result["projected_length"]), 6),
            "tortuosity": round(float(result["tortuosity"]), 8),
            "mean_abs_curvature_inv_um": round(float(np.mean(abs_curvature)), 8),
            "max_abs_curvature_inv_um": round(float(np.max(abs_curvature)), 8),
        },
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved: {npz_path.name}")
    print(f"Saved: {json_path.name}")
    print(
        "  expected tortuosity = "
        f"{manifest['expected_metrics']['tortuosity']:.8f}"
    )


def generate_straight_front_dataset() -> None:
    """Create a pair with a straight crack front and unit tortuosity."""
    base = _base_surface(GRID_SHAPE)
    front_rows = np.full(GRID_SHAPE[1], 28, dtype=int)
    opening = _opening_from_front_rows(front_rows, GRID_SHAPE)
    surface_a = Surface(height=base, dx=DX, dy=DY)
    surface_b = Surface(height=base - opening, dx=DX, dy=DY)

    _save_dataset(
        "crack_path_straight_demo",
        surface_a,
        surface_b,
        {
            "description": (
                "Straight front. Use propagation axis X and front side Min. "
                "Expected tortuosity is exactly 1."
            ),
            "threshold_um": THRESHOLD_UM,
            "propagation_axis": "x",
            "front_side": "min",
        },
    )


def generate_wavy_front_dataset() -> None:
    """Create a pair with a sinusoid-like front and tortuosity > 1."""
    base = _base_surface(GRID_SHAPE)
    cols = np.arange(GRID_SHAPE[1], dtype=float)
    front_rows = np.round(
        32.0
        + 7.0 * np.sin(2.0 * np.pi * cols / 36.0)
        + 3.0 * np.sin(2.0 * np.pi * cols / 15.0 + 0.5)
    ).astype(int)
    front_rows = np.clip(front_rows, 12, GRID_SHAPE[0] - 12)

    opening = _opening_from_front_rows(front_rows, GRID_SHAPE)
    surface_a = Surface(height=base, dx=DX, dy=DY)
    surface_b = Surface(height=base - opening, dx=DX, dy=DY)

    _save_dataset(
        "crack_path_wavy_demo",
        surface_a,
        surface_b,
        {
            "description": (
                "Wavy front. Use propagation axis X and front side Min. "
                "Expected tortuosity is reproducibly greater than 1."
            ),
            "threshold_um": THRESHOLD_UM,
            "propagation_axis": "x",
            "front_side": "min",
        },
    )


def generate_y_axis_dataset() -> None:
    """Create a pair that exercises the Y-axis propagation mode."""
    base = _base_surface(GRID_SHAPE)
    rows = np.arange(GRID_SHAPE[0], dtype=float)
    front_cols = np.round(
        34.0
        + 10.0 * np.sin(2.0 * np.pi * rows / 42.0 + 0.2)
        + 0.22 * rows
    ).astype(int)
    front_cols = np.clip(front_cols, 10, GRID_SHAPE[1] - 14)

    opening = _opening_from_front_cols(front_cols, GRID_SHAPE)
    surface_a = Surface(height=base, dx=DX, dy=DY)
    surface_b = Surface(height=base - opening, dx=DX, dy=DY)

    _save_dataset(
        "crack_path_y_axis_demo",
        surface_a,
        surface_b,
        {
            "description": (
                "Front extracted one row at a time. Use propagation axis Y and "
                "front side Min to verify the alternate orientation path."
            ),
            "threshold_um": THRESHOLD_UM,
            "propagation_axis": "y",
            "front_side": "min",
        },
    )


def generate_realistic_dataset() -> None:
    """Create a more realistic synthetic pair with irregular opening behavior.

    The front is no longer a clean sinusoid. It combines broad waviness,
    shorter-scale perturbations, local bulges, and partial re-contact bridges.
    The opening field also grows gradually behind the front instead of jumping
    directly to one constant value, which makes threshold sweeps more similar
    to real measured data.
    """
    shape = (176, 256)
    dx = 4.0
    dy = 4.0
    rows, cols = shape
    x = np.arange(cols, dtype=float) * dx
    y = np.arange(rows, dtype=float) * dy
    xx, yy = np.meshgrid(x, y)

    base = _base_surface_realistic(shape)
    cols_idx = np.arange(cols, dtype=float)

    front_rows = (
        58.0
        + 13.0 * np.sin(2.0 * np.pi * cols_idx / 92.0 + 0.25)
        + 7.0 * np.sin(2.0 * np.pi * cols_idx / 33.0 - 0.9)
        + 0.11 * cols_idx
    )
    front_rows += 10.0 * np.exp(-0.5 * ((cols_idx - 70.0) / 14.0) ** 2)
    front_rows -= 8.0 * np.exp(-0.5 * ((cols_idx - 165.0) / 18.0) ** 2)
    front_rows += 6.0 * np.exp(-0.5 * ((cols_idx - 220.0) / 9.0) ** 2)
    front_rows = np.clip(np.round(front_rows), 18, rows - 24).astype(int)

    opening = np.zeros(shape, dtype=float)
    for col in range(cols):
        row0 = front_rows[col]
        depth_px = np.arange(rows - row0, dtype=float)
        if depth_px.size == 0:
            continue

        # Opening ramps up with depth from the front and saturates gradually.
        ramp = 26.0 * (1.0 - np.exp(-depth_px / 14.0))

        # Column-wise modulation creates local differences in bridge opening.
        modulation = (
            1.0
            + 0.18 * np.sin(2.0 * np.pi * col / 47.0 + 0.7)
            + 0.09 * np.cos(2.0 * np.pi * col / 19.0 - 0.3)
        )
        profile = ramp * modulation

        # Local partial re-contact bridge near one irregular region.
        bridge = 5.5 * np.exp(-0.5 * ((col - 142.0) / 16.0) ** 2)
        profile -= bridge * np.exp(-0.5 * ((depth_px - 7.0) / 3.8) ** 2)

        # A second weaker bridge nearer the right edge.
        bridge2 = 3.5 * np.exp(-0.5 * ((col - 208.0) / 11.0) ** 2)
        profile -= bridge2 * np.exp(-0.5 * ((depth_px - 10.0) / 4.5) ** 2)

        opening[row0:, col] = np.clip(profile, 0.0, None)

    # Add a weak global COD gradient and a few local patches so that the
    # threshold sweep does not behave like an idealized binary step.
    opening += 1.8 * (xx / float(np.max(x)))
    opening += 1.6 * np.exp(-0.5 * (((xx - 360.0) / 42.0) ** 2 + ((yy - 240.0) / 36.0) ** 2))
    opening -= 1.2 * np.exp(-0.5 * (((xx - 660.0) / 38.0) ** 2 + ((yy - 170.0) / 28.0) ** 2))
    opening = np.clip(opening, 0.0, None)

    # Slightly break perfect conjugacy with low-amplitude deterministic mismatch.
    mismatch = (
        0.75 * np.sin(2.0 * np.pi * xx / 120.0 + 0.5)
        * np.cos(2.0 * np.pi * yy / 108.0 - 0.1)
        + 0.35 * RNG_REALISTIC.standard_normal(shape)
    )

    rng_holes = np.random.default_rng(20260519)
    surface_a = Surface(height=_add_holes(base, rng_holes), dx=dx, dy=dy)
    surface_b = Surface(height=_add_holes(base - opening + mismatch, rng_holes), dx=dx, dy=dy)

    _save_dataset(
        "crack_path_realistic_demo",
        surface_a,
        surface_b,
        {
            "description": (
                "More realistic irregular front with multi-scale roughness, "
                "gradual opening growth, local contact bridges, and sparse "
                "measurement holes. Start with first_open_pixel at 12 µm, then "
                "compare against contour with moderate smoothing."
            ),
            "threshold_um": 12.0,
            "propagation_axis": "x",
            "front_side": "min",
        },
    )


def main() -> None:
    """Generate all crack-path demo datasets in ``examples/data``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_straight_front_dataset()
    print()
    generate_wavy_front_dataset()
    print()
    generate_y_axis_dataset()
    print()
    generate_realistic_dataset()


if __name__ == "__main__":
    main()
