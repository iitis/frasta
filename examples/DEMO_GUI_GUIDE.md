# Demo datasets – GUI walkthrough

This guide explains how to explore the two synthetic fracture-surface datasets
in the FRASTA-toolbox graphical interface.  The same processing steps are also
available as standalone Python scripts in this directory.

---

## Prerequisites

1. Generate the demo files (run once from the repository root):
   ```
   python examples/data/generate_demo_data.py
   ```
   This creates two files in `examples/data/`:
   - `fracture_tilt_demo.npz`
   - `fracture_contact_demo.npz`

2. Start the application:
   ```
   python main.py
   ```

---

## Dataset 1 – Tilt correction (`fracture_tilt_demo.npz`)

**Scenario.**  Two optical-profilometer scans of conjugate fracture faces.
`Surface_B_tilted` carries a systematic linear ramp introduced by an unlevel
mounting of the specimen.  The objective is to remove this tilt so that the
difference map reflects only the true fracture topography.

### Steps

1. **Load Surface A**
   - *File → Open scan* → select `fracture_tilt_demo.npz`
   - Both surfaces load into separate tabs (`Surface_A`, `Surface_B_tilted`).

2. **Inspect holes**
   - In each tab, circular white (NaN) patches are visible — simulated
     measurement artefacts.
   - *Processing dialog → Hole filling* (nearest-neighbour) on both tabs.

3. **Switch to Alignment view**
   - Click the *Alignment* icon or select it from the toolbar.
   - The initial difference map will show a strong diagonal gradient —
     this is the mounting tilt.

4. **Remove tilt and offset**
   - *Processing → Remove relative tilt* — fits and subtracts a plane from
     the difference map.
   - *Processing → Remove relative offset* — zeros the mean height difference.
   - The difference map should now appear flat, centred near 0 µm.

5. **Quantitative check**
   - Hover over the difference map; the status bar shows local height values.
   - The RMSE should drop from ≈ 50 µm to ≈ 10 µm after correction.

6. **Export** (optional)
   - *File → Save session* to save the aligned pair as a new NPZ.
   - *File → Export difference map* to save a CSV.

**Script equivalent:** `python examples/demo_tilt_correction.py`

---

## Dataset 2 – Contact map and COD (`fracture_contact_demo.npz`)

**Scenario.**  Two conjugate faces of a mode-I fatigue fracture.  The surfaces
are already free of systematic tilt.  The difference map encodes the
**crack-opening displacement (COD)**: near-zero values in the left-centre
region correspond to the last contact front (crack arrest zone); larger values
near the edges indicate the fully separated fatigue region.

### Steps

1. **Load both surfaces**
   - *File → Open scan* → select `fracture_contact_demo.npz`
   - Tabs `Surface_A` and `Surface_B` appear.

2. **Fill holes**
   - *Processing → Hole filling* on both tabs.

3. **Remove offset** (no tilt in this dataset)
   - *Processing → Remove relative offset* in Alignment view.

4. **Inspect the difference map**
   - The Alignment view shows a smooth gradient from deep blue (low COD, left
     centre) through yellow to red (high COD, edges).
   - The colour scale represents local COD in µm.

5. **Build a contact map**
   - Switch to *Profile / Cross-section* view.
   - Place a horizontal section line across the surface centre.
   - In the profile panel set the *Separation* slider to **8 µm** (strict
     threshold): the contact region appears highlighted in the profile.
   - Increase the threshold to **15 µm** to include the transitional fringe.
   - The *contact area* and *fraction of valid area* are reported in the
     status bar.

6. **Visualise in 3D**
   - The 3D view shows Surface A with the contact region overlaid as a
     colour mask.

7. **Export**
   - *File → Export contact map* saves the binary contact mask as a CSV.
   - *File → Export profile* saves the extracted cross-section.

**Script equivalent:** `python examples/demo_contact_map.py`

---

## Key parameters

| Parameter | Dataset 1 | Dataset 2 | Effect |
|-----------|-----------|-----------|--------|
| Grid size | 150 × 150 | 150 × 150 | |
| Pixel size | 5 µm × 5 µm | 5 µm × 5 µm | |
| Physical extent | 750 µm × 750 µm | 750 µm × 750 µm | |
| Mean separation | 40 µm | 3–33 µm (spatially varying) | |
| Tilt | 0.06 / 0.03 µm/µm | none | |
| NaN holes | ~10 per surface | ~10 per surface | |
| Suggested contact threshold | — | 8–15 µm | |

---

## Further reading

- `docs/METHODS.md` — mathematical definitions (difference map, contact map,
  interpolation, alignment)
- `docs/QUICK_REFERENCE.md` — concise list of all processing operations
- `examples/advanced_processing.py` — scripted examples of all preprocessing
  functions
