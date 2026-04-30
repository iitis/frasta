"""Generate synthetic fracture-surface demo datasets.

Creates two NPZ files in this directory:

  fracture_tilt_demo.npz
      Two conjugate fracture-surface halves where one half carries a
      systematic tilt from measurement mounting.  Scenario: raw output
      from an optical profilometer where the specimen was not level.
      Workflow: fill holes → remove relative tilt → remove offset → inspect
      the residual difference map.

  fracture_contact_demo.npz
      Two conjugate fracture-surface halves already free of tilt, but with
      a spatially varying crack-opening displacement (COD).  The COD is
      ~0 µm near the left-centre region (last contact front / crack arrest
      zone) and rises to ~30 µm near the edges (fully separated region).
      Scenario: a mode-I fatigue fracture whose last-cycle contact front is
      preserved.  Workflow: fill holes → remove offset → compute difference
      map → apply contact threshold to visualise the contact region.

Both surfaces use a 150 × 150 grid with dx = dy = 5 µm (physical extent
750 µm × 750 µm) and heights in micrometres.  Measurement artefact holes
(NaN) are scattered randomly in both surfaces.

Usage (from repository root or examples/data/):
    python examples/data/generate_demo_data.py
"""

import sys
from pathlib import Path

import numpy as np

# Ensure frasta package is importable when run from any directory
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from frasta.core import Surface
from frasta.io.exporters import save_npz

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

N = 150          # grid size (points)
DX = DY = 5.0   # grid spacing, µm
RNG_A = np.random.default_rng(42)
RNG_B = np.random.default_rng(123)

x = np.arange(N) * DX   # µm
y = np.arange(N) * DY
X, Y = np.meshgrid(x, y)


def _add_holes(Z: np.ndarray, rng, n_holes: int = 10, max_radius: int = 4) -> np.ndarray:
    """Punch circular NaN holes into a surface to simulate artefacts."""
    Z = Z.copy().astype(float)
    for _ in range(n_holes):
        cx, cy = rng.integers(8, N - 8, size=2)
        r = rng.integers(2, max_radius + 1)
        yi, xi = np.ogrid[-cy : N - cy, -cx : N - cx]
        Z[xi * xi + yi * yi <= r * r] = np.nan
    return Z


# ---------------------------------------------------------------------------
# Dataset 1 – fracture_tilt_demo
# ---------------------------------------------------------------------------
# Surface A: multi-scale rough fracture topography
Z_base1 = (
    30.0 * np.sin(2 * np.pi * X / 400) * np.cos(2 * np.pi * Y / 500)
    + 15.0 * np.sin(2 * np.pi * X / 150 + 0.8)
    + 10.0 * np.cos(2 * np.pi * Y / 200 + 1.2)
    +  5.0 * RNG_A.standard_normal((N, N))
)

# Surface B: conjugate face (approximately −A) + mean separation + tilt + noise
#   tilt_x = 0.06 µm/µm → ~45 µm total across 750 µm  (clearly visible in GUI)
#   tilt_y = 0.03 µm/µm → ~22 µm total across 750 µm
SEPARATION1 = 40.0   # µm mean separation
TILT_X1     =  0.06  # µm per µm in X
TILT_Y1     =  0.03  # µm per µm in Y

Z_B1 = (
    -Z_base1
    + SEPARATION1
    + TILT_X1 * (X - X.mean())
    + TILT_Y1 * (Y - Y.mean())
    + 3.0 * RNG_A.standard_normal((N, N))
)

surf_A1 = Surface(height=_add_holes(Z_base1, RNG_A), dx=DX, dy=DY)
surf_B1 = Surface(height=_add_holes(Z_B1,    RNG_A), dx=DX, dy=DY)

# ---------------------------------------------------------------------------
# Dataset 2 – fracture_contact_demo
# ---------------------------------------------------------------------------
# Surface A: small-amplitude roughness (std ~2 µm).  Large-scale form is
# absent so that the difference map D = A − B is dominated by the COD.
Z_base2 = 2.0 * RNG_B.standard_normal((N, N))

# COD profile: crack propagated from right to left; the last contact front
# (crack-arrest line) lies near x ≈ 150 µm (left 20% of the scan).
# COD increases smoothly from 0 (left edge) to 35 µm (right edge).
# This represents a mode-I fatigue crack whose arrest line is preserved.
cod = np.clip((X - 150.0) / 600.0, 0.0, 1.0) * 35.0   # µm, range 0–35

# Surface B: conjugate face (flipped) MINUS the opening.
# Convention: D = A − B = 2*Z_base2 + cod ≈ cod  (roughness small).
#   D ≈ 0   → contact  (left of crack-arrest line)
#   D >> 0  → open     (fully separated fatigue region on the right)
Z_B2 = (
    -Z_base2
    - cod
    + 1.5 * RNG_B.standard_normal((N, N))
)

surf_A2 = Surface(height=_add_holes(Z_base2, RNG_B), dx=DX, dy=DY)
surf_B2 = Surface(height=_add_holes(Z_B2,    RNG_B), dx=DX, dy=DY)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

tilt_path    = out_dir / "fracture_tilt_demo.npz"
contact_path = out_dir / "fracture_contact_demo.npz"

save_npz(str(tilt_path),    [("Surface_A", surf_A1), ("Surface_B_tilted", surf_B1)])
save_npz(str(contact_path), [("Surface_A", surf_A2), ("Surface_B",        surf_B2)])

print(f"Saved: {tilt_path}")
print(f"Saved: {contact_path}")
print()
print("Dataset 1 (fracture_tilt_demo):")
print(f"  Grid: {N}x{N}, dx=dy={DX} µm  →  {N*DX:.0f} µm × {N*DY:.0f} µm")
print(f"  Surface_A      : fracture topography, height range "
      f"{np.nanmin(surf_A1.height):.1f} – {np.nanmax(surf_A1.height):.1f} µm")
print(f"  Surface_B_tilted: conjugate + tilt ({TILT_X1}/{TILT_Y1} µm/µm) + "
      f"separation {SEPARATION1} µm")
print()
print("Dataset 2 (fracture_contact_demo):")
print(f"  Grid: {N}x{N}, dx=dy={DX} µm  →  {N*DX:.0f} µm × {N*DY:.0f} µm")
print(f"  Surface_A: small-roughness fracture (std ~2 µm)")
print(f"  Surface_B: conjugate, COD 0–35 µm left→right (D≈0 left of x=150 µm)")
