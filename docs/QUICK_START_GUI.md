# Quick Start — FRASTA-toolbox GUI

This guide walks you through two representative workflows using the included
synthetic demo datasets. It is intended for first-time users who want to
verify that the software is installed correctly and explore the main
interactive features before working with their own data.

Use this guide when you want a practical first session in the GUI. If you only
need a short list of operations, skip to [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md).
If you want implementation details and computational assumptions, continue with
[`METHODS.md`](METHODS.md) after finishing the workflows below.

---

## Step 0 — Installation and data preparation

1. Install dependencies (from the repository root):
   ```
   pip install -r requirements.txt
   ```
2. Generate the demo datasets (run once):
   ```
   python examples/data/generate_demo_data.py
   ```
   This creates two NPZ files in `examples/data/`:
   - `fracture_tilt_demo.npz`
   - `fracture_contact_demo.npz`

3. Start the application:
   ```
   python main.py
   ```
   Run from the repository root so that icons and resources resolve correctly.

---

## Workflow 1 — Tilt correction (`fracture_tilt_demo.npz`)

**What it shows.** One surface carries a systematic linear ramp from a
simulated unlevel mounting. You will remove this tilt and observe the
flattened difference map.

### 1.1  Load the dataset

- **File → Open scan** → navigate to `examples/data/fracture_tilt_demo.npz`
- Two tabs appear: **Surface\_A** and **Surface\_B\_tilted**

### 1.2  Inspect and fill measurement holes

- In the **Surface\_A** tab you will see circular white patches — these are
  NaN artifacts (simulated reflectance dropouts).
- Open **Scan Actions → Fill holes** (nearest-neighbor); repeat for
  **Surface\_B\_tilted**.
- If ROI is active, only holes inside the selected ROI are filled.
- The patches disappear and the height map becomes continuous.

### 1.3  Switch to Alignment view

- Click the **Alignment** button in the toolbar (or select it from the
  **View** menu).
- The difference map panel shows a strong diagonal gradient — this is the
  mounting tilt of Surface B.

### 1.4  Remove tilt and offset

- **Processing → Remove relative tilt**  
  A plane is fitted to the difference map and subtracted from Surface B.
  The gradient should largely disappear.
- **Processing → Remove relative offset**  
  Removes the remaining constant height difference.
- The difference map should now appear flat, centered near 0 µm.

### 1.5  Quantitative check

- Hover over the difference map; the status bar shows the local height value.
- The displayed color scale range should shrink substantially compared to the
  initial state (from ~±50 µm to ~±10 µm or less).

### 1.6  Save the session

- **File → Save session** saves both aligned surfaces into a new NPZ file,
  preserving grid spacing and coordinate origins.

**Script version:** `python examples/demo_tilt_correction.py`
(produces a four-panel figure in `examples/output/`)

---

## Workflow 2 — Contact map and crack-opening displacement (`fracture_contact_demo.npz`)

**What it shows.** Two conjugate fracture faces with a spatially varying
crack-opening displacement (COD): near zero on the left (contact /
crack-arrest line) and rising to ~35 µm on the right (fully separated
fatigue region). You will visualize the COD field and identify the contact
zone.

### 2.1  Load the dataset

- **File → Open scan** → select `examples/data/fracture_contact_demo.npz`
- Two tabs appear: **Surface\_A** and **Surface\_B**

### 2.2  Fill holes

- **Scan Actions → Fill holes** on both tabs.
- If ROI is active, only holes inside the selected ROI are filled.

### 2.3  Inspect the difference map

- Switch to **Alignment view**.
- The difference map shows a smooth gradient from dark (low COD, left edge)
  to bright (high COD, right edge).
- The color scale represents local COD in µm; values near zero indicate
  the crack-arrest line.

### 2.4  Build a contact map

- Switch to **Cross-section / Profile view**.
- Place a horizontal section line near the center of the surface (drag
  across the 2D map).
- In the profile panel, locate the **Separation** slider and set it to
  **8 µm** (conservative threshold): the left portion of the profile is
  highlighted as contact.
- Increase the threshold to **16 µm** to include the transitional fringe.
- The status bar reports the contact area and fraction of valid area.

### 2.5  Three-dimensional context

- The **3D view** shows Surface A with the contact overlay; rotate the view
  to inspect the spatial distribution of the contact zone relative to the
  fracture topography.
- The 3D view opens in a scan-oriented perspective: the horizontal scan axis
  follows the same left-to-right direction as the main 2D view.

### 2.6  Export results

- **File → Export contact map** saves the binary mask as a CSV.
- **File → Export profile** saves the extracted cross-section.

**Script version:** `python examples/demo_contact_map.py`
(produces a three-panel figure in `examples/output/`)

---

## Key parameters at a glance

| Parameter           | Dataset 1 (tilt)       | Dataset 2 (contact)     |
|---------------------|------------------------|-------------------------|
| Grid size           | 150 × 150              | 150 × 150               |
| Pixel size          | 5 µm × 5 µm            | 5 µm × 5 µm             |
| Physical extent     | 750 µm × 750 µm        | 750 µm × 750 µm         |
| Mean separation     | 40 µm                  | 0–35 µm (spatially varying) |
| Tilt                | 0.06 / 0.03 µm/µm      | none                    |
| NaN holes           | ~10 per surface        | ~10 per surface         |
| Suggested threshold | —                      | 8–16 µm                 |

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| Application window does not open | Run from repository root: `python main.py` |
| 3D view is black or missing | Desktop session must support OpenGL; check GPU drivers |
| Icons missing | Ensure you are in the repository root directory |
| `ModuleNotFoundError` | Activate the virtual environment or run `pip install -r requirements.txt` |

---

## Further reading

- [`examples/DEMO_GUI_GUIDE.md`](../examples/DEMO_GUI_GUIDE.md) — extended
  step-by-step GUI walkthrough with screenshots references
- [`docs/METHODS.md`](METHODS.md) — mathematical definitions (difference map,
  contact map, interpolation, alignment)
- [`docs/QUICK_REFERENCE.md`](QUICK_REFERENCE.md) — concise list of all
  processing operations and API functions
- [`examples/demo_tilt_correction.py`](../examples/demo_tilt_correction.py) —
  script version of Workflow 1
- [`examples/demo_contact_map.py`](../examples/demo_contact_map.py) — script
  version of Workflow 2
