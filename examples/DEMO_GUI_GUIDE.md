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

## Dataset 3 – Crack-path tortuosity demos

Generate the dedicated crack-path datasets:
```
python examples/data/generate_crack_path_demo_data.py
```

This creates:
- `crack_path_straight_demo.npz`
- `crack_path_wavy_demo.npz`
- `crack_path_y_axis_demo.npz`
- `crack_path_realistic_demo.npz`

Each file also gets a companion `.json` manifest containing the recommended
threshold, propagation axis, front side, and expected crack-path metrics.

### Suggested checks

1. Open one of the NPZ files from `examples/data/`.
2. Launch `Tools -> Contact map analysis...`.
3. Select `Surface_A` and `Surface_B`.
4. Set the threshold from the JSON manifest, typically `9 µm`.
5. Match the propagation axis and front side from the same JSON file.
6. Compare the GUI values for:
   - `Effective length`
   - `Projected length`
   - `Tortuosity`
   - `Mean |curvature|`

### Expected behavior by dataset

- `crack_path_straight_demo`: straight front, `tortuosity = 1`.
- `crack_path_wavy_demo`: undulating front, `tortuosity > 1`.
- `crack_path_y_axis_demo`: same workflow, but with propagation along `Y`.
- `crack_path_realistic_demo`: larger, less idealized front with gradual
  opening, local bridges, sparse holes, and stronger method sensitivity than
  the simpler control cases.

### Recommended first pass for `crack_path_realistic_demo`

1. Start with `first_open_pixel`, threshold `12 µm`, axis `X`, side `Min`.
2. Inspect whether the current threshold falls on a stable `tau(s)` plateau.
3. Switch to `contour`, then try:
   - `Resample = 4 µm`
   - `Smooth win = 5` or `7`
4. Compare whether the contour path remains close to the first-open-pixel
   estimate or becomes strongly threshold-sensitive.

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
     measurement artifacts.
   - *Processing dialog → Hole filling* (nearest-neighbor) on both tabs.

3. **Switch to Alignment view**
   - Click the *Alignment* icon or select it from the toolbar.
   - The initial difference map will show a strong diagonal gradient —
     this is the mounting tilt.

4. **Remove tilt and offset**
   - *Processing → Remove relative tilt* — fits and subtracts a plane from
     the difference map.
   - *Processing → Remove relative offset* — zeros the mean height difference.
   - The difference map should now appear flat, centered near 0 µm.

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
**crack-opening displacement (COD)**: near-zero values in the left-center
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
     center) through yellow to red (high COD, edges).
   - The color scale represents local COD in µm.

5. **Build a contact map**
   - Switch to *Profile / Cross-section* view.
   - Place a horizontal section line across the surface center.
   - In the profile panel set the *Separation* slider to **8 µm** (strict
     threshold): the contact region appears highlighted in the profile.
   - Increase the threshold to **15 µm** to include the transitional fringe.
   - The *contact area* and *fraction of valid area* are reported in the
     status bar.

6. **Visualise in 3D**
   - The 3D view shows Surface A with the contact region overlaid as a
     color mask.

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
