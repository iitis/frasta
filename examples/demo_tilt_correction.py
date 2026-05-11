"""Demo: tilt correction on a pair of conjugate fracture surfaces.

Dataset: examples/data/fracture_tilt_demo.npz
  Surface_A      – reference fracture half (rough, with measurement holes)
  Surface_B_tilted – conjugate half with a systematic mounting tilt and holes

Scenario
--------
After an optical profilometer scan the specimen was not perfectly level,
introducing a linear ramp across Surface B.  Before any FRASTA analysis the
tilt must be removed so that the difference map reflects only the true fracture
geometry rather than the mounting error.

Processing pipeline (no GUI required)
--------------------------------------
1. Load both surfaces from NPZ.
2. Fill NaN measurement holes with nearest-neighbour interpolation.
3. Remove the relative tilt (plane fit to the difference map).
4. Remove the residual constant height offset.
5. Compute the difference map  D = A − B.
6. Report RMSE before and after correction.
7. Save a four-panel comparison figure to examples/output/.

Run from repository root:
    python examples/demo_tilt_correction.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from frasta.io.loaders    import load_npz_data
from frasta.processing.interpolation import fill_holes
from frasta.processing.alignment     import remove_relative_tilt, remove_relative_offset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "data" / "fracture_tilt_demo.npz"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-generate data if missing
if not DATA_FILE.exists():
    print("Demo data not found – generating …")
    import subprocess
    subprocess.run(
        [sys.executable,
         str(Path(__file__).parent / "data" / "generate_demo_data.py")],
        check=True,
    )

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
surfaces = load_npz_data(str(DATA_FILE))
surf_A, surf_B = surfaces[0], surfaces[1]

Z_A_raw = surf_A.height.copy()
Z_B_raw = surf_B.height.copy()

print(f"Loaded: {DATA_FILE.name}")
print(f"  Grid: {Z_A_raw.shape[0]} × {Z_A_raw.shape[1]}, "
      f"dx={surf_A.dx} µm, dy={surf_A.dy} µm")
print(f"  NaN holes – Surface A: {np.isnan(Z_A_raw).sum()}, "
      f"Surface B: {np.isnan(Z_B_raw).sum()}")

# ---------------------------------------------------------------------------
# Step 1 – fill measurement holes
# ---------------------------------------------------------------------------
Z_A = fill_holes(Z_A_raw)
Z_B = fill_holes(Z_B_raw)

# ---------------------------------------------------------------------------
# Step 2 – difference map BEFORE correction
# ---------------------------------------------------------------------------
valid_before = ~(np.isnan(Z_A) | np.isnan(Z_B))
D_before = np.where(valid_before, Z_A - Z_B, np.nan)
rmse_before = float(np.sqrt(np.nanmean(D_before ** 2)))

# ---------------------------------------------------------------------------
# Step 3 – remove relative tilt  (fits a plane to D, subtracts it from B)
# ---------------------------------------------------------------------------
mask_full = np.ones(Z_A.shape, dtype=bool)
Z_B_lev = remove_relative_tilt(Z_A, Z_B, mask_full)
Z_A_lev = Z_A.copy()

# ---------------------------------------------------------------------------
# Step 4 – remove residual constant offset
# ---------------------------------------------------------------------------
Z_B_lev = remove_relative_offset(Z_A_lev, Z_B_lev, mask_full)

# ---------------------------------------------------------------------------
# Step 5 – difference map AFTER correction
# ---------------------------------------------------------------------------
valid_after = ~(np.isnan(Z_A_lev) | np.isnan(Z_B_lev))
D_after = np.where(valid_after, Z_A_lev - Z_B_lev, np.nan)
rmse_after = float(np.sqrt(np.nanmean(D_after ** 2)))

print(f"\nRMSE of difference map")
print(f"  Before tilt removal : {rmse_before:.2f} µm")
print(f"  After  tilt removal : {rmse_after:.2f} µm")
print(f"  Improvement factor  : {rmse_before / rmse_after:.1f}×")

# ---------------------------------------------------------------------------
# Step 6 – figure
# ---------------------------------------------------------------------------
extent = [0, surf_A.dx * Z_A_raw.shape[1], 0, surf_A.dy * Z_A_raw.shape[0]]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Tilt correction demo  –  fracture_tilt_demo", fontsize=13)

def _imshow(ax, data, title, cmap="RdYlBu_r", label="height [µm]"):
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

_imshow(axes[0, 0], Z_A,      "Surface A (holes filled)")
_imshow(axes[0, 1], Z_B,      "Surface B – before correction\n(conjugate + tilt + separation)")
_imshow(axes[1, 0], D_before,
        f"Difference map BEFORE\n(RMSE = {rmse_before:.1f} µm)",
        cmap="bwr", label="A − B [µm]")
_imshow(axes[1, 1], D_after,
        f"Difference map AFTER tilt + offset removal\n(RMSE = {rmse_after:.1f} µm)",
        cmap="bwr", label="A − B [µm]")

plt.tight_layout()
out_path = OUT_DIR / "demo_tilt_correction.png"
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved: {out_path}")
