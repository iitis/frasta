"""Demo: crack-opening displacement and contact map from conjugate surfaces.

Dataset: examples/data/fracture_contact_demo.npz
  Surface_A – reference fracture half (small-amplitude roughness, with holes)
  Surface_B – conjugate half with a spatially varying crack-opening
              displacement (COD) that increases from 0 µm at the left edge to
              35 µm at the right edge.  The crack-arrest line lies near
              x ≈ 150 µm; the region to its right is the fully propagated
              fatigue zone.

Sign convention
---------------
  Z_B ≈ −Z_A − COD(x)
  D = A − B ≈ 2·Z_A + COD(x) ≈ COD(x)   (since roughness std ≈ 2 µm)

  D ≈ 0   →  in contact (left of arrest line)
  D >> 0  →  crack open (right, fully separated)

Contact threshold  s  selects pixels where D < s:
  small s  → conservative; only the deepest contact zone
  large s  → includes the transitional fringe near the arrest line

Processing pipeline (no GUI required)
--------------------------------------
1. Load both surfaces from NPZ.
2. Fill NaN measurement holes.
3. Compute the difference map  D = A − B.
4. Build binary contact maps at two threshold values.
5. Report contact fraction at each threshold.
6. Save a three-panel figure to examples/output/.

Note: no tilt or offset removal is applied — D is physically meaningful
as the local crack-opening displacement (absolute scale in µm).

Run from repository root:
    python examples/demo_contact_map.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from frasta.io.loaders               import load_npz_data
from frasta.processing.interpolation import fill_holes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "data" / "fracture_contact_demo.npz"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"  Physical size: {Z_A_raw.shape[1] * surf_A.dx:.0f} µm × "
      f"{Z_A_raw.shape[0] * surf_A.dy:.0f} µm")

# ---------------------------------------------------------------------------
# Step 1 – fill holes
# ---------------------------------------------------------------------------
Z_A = fill_holes(Z_A_raw)
Z_B = fill_holes(Z_B_raw)

# No tilt removal needed; no offset removal — we want D ≈ COD (absolute).

# ---------------------------------------------------------------------------
# Step 2 – difference map (COD field)
# ---------------------------------------------------------------------------
valid = ~(np.isnan(Z_A) | np.isnan(Z_B))
D = np.where(valid, Z_A - Z_B, np.nan)

cod_mean = float(np.nanmean(D))
cod_p5   = float(np.nanpercentile(D,  5))
cod_p95  = float(np.nanpercentile(D, 95))

print(f"\nDifference map D = A - B (approx. COD)")
print(f"  Mean   : {cod_mean:.1f} µm")
print(f"  5th pct: {cod_p5:.1f} µm  (near-contact zone)")
print(f"  95th pct: {cod_p95:.1f} µm  (open zone)")

# ---------------------------------------------------------------------------
# Step 3 – contact maps  (D < threshold  →  contact)
# ---------------------------------------------------------------------------
dx_phys = surf_A.dx * surf_A.dy   # µm² per pixel

THRESHOLD_STRICT =  8.0   # µm – conservative: core contact zone (D ≈ 0–8)
THRESHOLD_LOOSE  = 16.0   # µm – includes transitional fringe

# contact = gap is smaller than threshold
contact_strict = valid & (D < THRESHOLD_STRICT)
contact_loose  = valid & (D < THRESHOLD_LOOSE)

area_strict = contact_strict.sum() * dx_phys * 1e-6  # mm²
area_loose  = contact_loose.sum()  * dx_phys * 1e-6

frac_strict = contact_strict.sum() / valid.sum() * 100
frac_loose  = contact_loose.sum()  / valid.sum() * 100

print(f"\nContact fraction (threshold = {THRESHOLD_STRICT} µm): "
      f"{area_strict:.4f} mm²  ({frac_strict:.1f} % of valid area)")
print(f"Contact fraction (threshold = {THRESHOLD_LOOSE} µm): "
      f"{area_loose:.4f} mm²  ({frac_loose:.1f} % of valid area)")

# ---------------------------------------------------------------------------
# Step 5 – figure
# ---------------------------------------------------------------------------
extent = [0, surf_A.dx * Z_A.shape[1], 0, surf_A.dy * Z_A.shape[0]]

contact_cmap = ListedColormap(["#d62728", "#aec7e8"])   # red=open, blue=contact

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("COD and contact-map demo  –  fracture_contact_demo", fontsize=12)

def _im(ax, data, title, **kw):
    im = ax.imshow(data, origin="lower", extent=extent, aspect="equal", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    return im

# Panel 1: Surface A
im1 = _im(axes[0], Z_A, "Surface A (hole-filled)", cmap="RdYlBu_r")
plt.colorbar(im1, ax=axes[0], label="height [µm]", fraction=0.046, pad=0.04)

# Panel 2: Difference map (COD)
lim = np.nanpercentile(np.abs(D), 98)
im2 = _im(axes[1], D,
          "Difference map D = A − B\n(crack-opening displacement)",
          cmap="plasma", vmin=0, vmax=lim)
plt.colorbar(im2, ax=axes[1], label="COD [µm]", fraction=0.046, pad=0.04)

# Panel 3: Contact map (strict threshold)
contact_img = np.where(valid, contact_strict.astype(float), np.nan)
im3 = _im(axes[2], contact_img,
          f"Contact map  (threshold = {THRESHOLD_STRICT} µm)\n"
          f"blue = contact ({frac_strict:.1f} %), red = open",
          cmap=contact_cmap, vmin=0, vmax=1)

plt.tight_layout()
out_path = OUT_DIR / "demo_contact_map.png"
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved: {out_path}")
