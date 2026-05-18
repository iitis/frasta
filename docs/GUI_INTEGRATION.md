# GUI Integration - Advanced Processing

This document describes the GUI integration of advanced processing features in FRASTA-toolbox.

## Overview

Advanced processing functions have been integrated into the main FRASTA-toolbox GUI with:
- **New menu**: "Processing" menubar with 4 operations
- **Toolbar buttons**: Quick access to filtering, morphology, transforms
- **Parameter dialogs**: User-friendly interfaces for all operations
- **Progress feedback**: Cursor changes and message boxes

---

## New GUI Features

### Menu Structure

A new **"Processing"** menu has been added between "Scan Actions" and "Tools":

```
File | Edit | Scan Actions | Processing | Tools | Help
                                   |
                                   +-- Advanced Filtering...
                                   +-- Morphology & Leveling...
                                   +-- Geometric Transforms...
                                   +-- ----------------------
                                    +-- Auto-Register Surfaces...
```

The **Edit** menu also contains **ROI settings...**, a dialog for:
- enabling or disabling the ROI,
- choosing the ROI behavior mode,
- selecting the ROI shape,
- entering the ROI position and size numerically in physical units.

The same menu now also exposes **Undo ROI delete**, which restores the most
recent delete-inside or delete-outside operation performed through the active
ROI. After a successful ROI delete, the status bar reports how many valid grid
points were removed and reminds the user that the operation can be undone.

The **Tools** menu also contains **Scan info...**, a read-only summary dialog for:
- grid shape and physical spacing,
- coordinate extents and origin,
- valid-data coverage and basic height statistics,
- stored scan metadata such as the imported scan name.

### Experimental 3D Viewer Layout

The experimental point/mesh-based 3D viewer now separates its controls into:
- a **top action bar** for global view settings and export actions,
- a **left sidebar** for grouped visibility and surface-style controls,
- the **main OpenGL viewport** on the right.

For very large scans, the GUI also applies adaptive display limits:
- the main 2D scan tab shows a regularly decimated preview when the native grid
  would otherwise make repeated redraws too expensive,
- the experimental 3D viewer refines only down to a practical point or mesh
  budget instead of always forcing stride `1`,
- the experimental 3D viewer exposes a `Quality` preset selector
  (`Fast` / `Balanced` / `High` / `Ultra` / `Manual`) so the user can consciously trade
  off fidelity against responsiveness. `Ultra` plays the role of the old
  full-resolution mode, but very large mesh scenes may still be capped to avoid
  an effectively unbounded build time,
- when `Manual` is selected, a `Target stride` slider becomes active so the
  user can explicitly choose the settled final stride, including `1` for
  maximum-fidelity inspection or screenshot export at the cost of performance
  when the resulting mesh still fits within the OpenGL buffer-allocation
  limits used by the Qt backend. Extremely large mesh scenes may still be
  clamped to a coarser stride to avoid impossible uploads.

These display-side limits affect responsiveness only. Processing, measurement,
and export still use the original full-resolution surface data.

The top action bar contains:
- render mode,
- projection mode,
- auto-range mode (`Full` or `Percentile`) for 3D color normalization,
- a live render-status line reporting the current mode, stride, and whether the
  mesh or point representation is still refining,
- point size,
- background-color actions,
- screenshot and colorbar export actions.

The `Points` render mode now uses lightweight per-point normal estimation and
shaded circular splats instead of flat square pixels, so it remains visually
useful as a faster alternative when the full mesh path is too expensive.

When `Full resolution` is enabled, active camera interaction such as orbiting,
panning, or zooming now temporarily switches the viewport to a decimated point
preview. After a short idle delay the viewer jumps directly to the settled
target stride. If the selected mode is `mesh`, the non-interactive view first
shows points at that same target density and only later swaps to triangles once
the mesh worker finishes building indices and mesh normals in the background.

The sidebar groups controls into:
- **View** - visibility toggles for reference surface, adjusted surface, profile line, section plane, and the `Z=0` reference rectangle,
  plus the quality preset (`Fast`, `Balanced`, `High`, `Ultra`, `Manual`) and
  the always-visible target-stride slider that stays disabled outside `Manual`,
  adopts the current settled target stride when `Manual` is enabled, and then
  applies later edits after a short idle delay so dragging the slider does not
  rebuild the scene on every intermediate value, plus section-plane opacity and
  color controls,
- **Reference** - colormap, range, and clipping controls for the reference surface,
- **Adjusted** - matching controls for the adjusted surface, shown only when a second surface is available.

To keep the sidebar narrow enough for the 3D viewport, the reference and
adjusted range controls use a compact vertical `Min` / `Max` arrangement
instead of two wide spin boxes in one row.

When the experimental 3D viewer is opened from the profile-analysis window,
dragging the profile ROI updates the 3D profile line and section plane live in
the already-open viewer. This live sync keeps the current camera and surface
geometry intact and refreshes only the profile overlay.

The section plane in the experimental 3D viewer can now be restyled directly
from the sidebar. Its visibility remains independent, while opacity and color
can be adjusted without rebuilding the surface geometry.

Its height can now be controlled in three modes:
- **Dynamic** - follows the current profile while keeping a minimum visible vertical extent derived from the scene footprint,
- **Maximum** - spans the full loaded scene height range,
- **Manual** - uses a slider to interpolate from the dynamic local height up to the maximum scene-wide height.

This keeps the plane readable on nearly flat surfaces while also making it
easier to prepare clearer screenshots for publications and presentations.

The profile-analysis window now uses a horizontal splitter only for the two
main views. Compact profile controls sit in one shared row above them, so the
profile plot keeps most of the width while the FRASTA map stays in a narrower
right-hand column. The FRASTA map preserves its aspect ratio through the
underlying ViewBox rather than by forcing a constant widget size. When free
space appears around the fitted FRASTA map, a thin outline marks the true
image boundary.

Both the 2D and experimental 3D colorbar exporters now use a shared layout
and formatting engine with larger labels, regular rounded major ticks, an
explicit zero tick when the display range crosses zero, and an optional
histogram beside the color ramp. The export dialogs also expose label
precision and font-size controls so screenshots can be tuned for different
units and publication layouts.

The dialog supports two behavior modes:
- **Shared across scans** - one ROI geometry reused for every scan
- **Independent per scan** - each tab keeps its own ROI position and size

The FRASTA binary and profile docks still remember their saved positions, but
they no longer auto-open empty after application startup. They stay hidden
until a scan pair or a saved FRASTA session is loaded into them.

The FRASTA binary map now follows the same 2D Y-axis orientation as the main
scan views and the manual overlay view. This keeps profile lines, point picks,
and the linked 3D section plane in one consistent coordinate convention.

Internally, this orientation contract is now centralized in a shared GUI
adapter rather than being reimplemented separately in each viewer. The scan
data model also carries a dedicated `orientation` enum property so future
import-time or per-scan orientation correction can be introduced from one
source of truth without changing each individual view path.

### Toolbar

Three new buttons added to the toolbar (after "Set tilt", before "View 3D"):
- **Advanced Filtering** - Opens filter selection dialog
- **Morphology & Leveling** - Opens morphology operations dialog
- **Geometric Transforms** - Opens transform dialog

---

## Feature Dialogs

### 1. ROI Settings Dialog

**Access**: `Edit -> ROI settings...`

**Available controls**:
- **ROI enabled** - show or hide the ROI
- **Mode** - shared across scans or independent per scan
- **Units** - the native scan unit plus a nearby smaller or larger unit when meaningful
- **ROI type** - currently circle or rectangle
- **Geometry** - center coordinates plus radius or width/height

**Behavior**:
- In **shared** mode, the dialog updates the common ROI used by all scans.
- In **independent** mode, the dialog updates only the ROI stored for the current tab.
- The ROI remains interactive in the image view after being created from the dialog.
- The ROI geometry is stored in physical coordinates rather than pixel indices, so the displayed shape remains correct even when `dx` and `dy` differ.
- ROI delete operations report how many valid points were removed. If no active
  ROI exists on the current tab, the delete is skipped and the status bar
  explains why.

---

### 2. Advanced Filtering Dialog

**Access**: `Processing -> Advanced Filtering...` or toolbar button

**Available Filters**:
- **Bilateral Filter** - Edge-preserving smoothing
  - Parameters: Spatial Sigma (1-20 px), Range Sigma (1-50)
  - Use for: Smoothing while preserving fracture edges
  
- **Median Filter** - Robust outlier removal
  - Parameters: Window Size (3-15, odd values)
  - Use for: Removing measurement spikes and outliers
  
- **Morphological Opening** - Peak removal
  - Parameters: Structure Size (3-15)
  - Use for: Removing noise peaks smaller than structure
  
- **Morphological Closing** - Valley filling
  - Parameters: Structure Size (3-15)
  - Use for: Filling small valleys and holes
  
- **Robust Gaussian Filter** - Outlier-resistant smoothing
  - Parameters: Sigma (0.5-10), Max Iterations (1-10), Threshold (1-5sigma)
  - Use for: Gaussian smoothing with automatic outlier rejection

**Workflow**:
1. Load a scan
2. Click "Advanced Filtering" toolbar button or menu item
3. Select filter type from dropdown
4. Adjust parameters (descriptions appear below each parameter)
5. Click OK and choose whether to replace the current scan or create a derived tab

**Result**: The filtered data can either replace the current scan or be stored in a new tab.
If an ROI is visible, filtering is automatically restricted to that ROI. In
per-scan mode, the ROI of the active tab is used.

---

### 3. Morphology & Leveling Dialog

**Access**: `Processing -> Morphology & Leveling...` or toolbar button

**Available Operations**:
- **Level by Plane (Least Squares)** - Fast tilt removal
  - No parameters
  - Use for: Quick tilt removal when data is clean
  
- **Level by Plane (Robust RANSAC)** - Outlier-resistant leveling
  - Parameters: Max Iterations (100-10000), Inlier Threshold (0.1-100 nm)
  - Use for: Tilt removal with outliers or contamination
  
- **Remove Polynomial Form** - Curvature correction
  - Parameters: Polynomial Order (1-5)
  - Use for: Removing bending, warping, or systematic curvature
  - Order 1 = plane, 2 = parabolic, 3+ = higher order
  
- **Threshold Grid** - Value-based masking
  - Parameters: Lower Bound, Upper Bound
  - Use for: Masking extreme values (become NaN)

**Workflow**:
1. Load a scan
2. Click "Morphology & Leveling" toolbar button
3. Select operation from dropdown
4. Configure parameters
5. Click OK and choose whether to replace the current scan or create a derived tab

**Result**: The processed data can either replace the current scan or be stored in a new tab.
If an ROI is visible, leveling and polynomial form removal use that ROI automatically.

---

### 4. Geometric Transforms Dialog

**Access**: `Processing -> Geometric Transforms...` or toolbar button

**Available Transforms**:
- **Rotate Grid** - Rotate surface by angle
  - Parameters: Angle (-180 deg to +180 deg), Interpolation (Nearest/Linear/Cubic)
  - Use for: Aligning surface orientation
  - Cubic interpolation recommended for best quality
  
- **Rescale Grid** - Change resolution
  - Parameters: Scale Factor (0.1-10), Interpolation
  - Use for: Upsampling (>1) or downsampling (<1)
  - Scale 2.0 = double resolution, 0.5 = half resolution
  
- **Crop to Valid Region** - Automatic cropping
  - Parameters: Margin (0-50 pixels)
  - Use for: Removing NaN borders automatically
  - Margin adds padding around valid region

**Workflow**:
1. Load a scan
2. Click "Geometric Transforms" toolbar button
3. Select transform type
4. Set parameters
5. Click OK and choose whether to replace the current scan or create a derived tab

**Result**: 
- The transformed data can either replace the current scan or be stored in a new tab
- Pixel sizes are updated automatically
- Coordinate arrays are regenerated
- Success message shows new dimensions

---

### 5. Auto-Register Surfaces Dialog

**Access**: `Processing -> Auto-Register Surfaces...`

**Requirements**: At least 2 loaded scans

**Parameters**:
- **Reference Surface** - Surface to align to (stays fixed)
- **Moving Surface** - Surface to transform (will be aligned)
- **Registration Method**:
  - **Cross-Correlation** - Fast, 2D translation only
  - **ICP (Iterative Closest Point)** - 3D rigid transform (translation + rotation)
 - **Refine ICP alignment (slower)** - Optional final height-RMSE refinement for ICP; useful mainly for small, distinctive ROIs
 - **Auto reject mismatched areas (ICP)** - Optional second ICP pass on an automatically selected low-mismatch overlap region; useful when scans contain burrs or broken fragments

If cross-correlation is selected for scans of different sizes, the GUI offers
cropping both scans to their shared rectangular area before registration.
When a visible ROI is active, automatic registration is restricted to that ROI
by masking everything outside the selected region. In per-scan mode, the
reference and moving scans each use their own stored ROI geometry.

**Workflow**:
1. Load 2+ scans in different tabs
2. Click `Processing -> Auto-Register Surfaces...`
3. Select reference surface (stays fixed)
4. Select moving surface (will be transformed)
5. Choose registration method
6. Optionally enable ICP refinement for a small ROI
7. Optionally enable automatic mismatched-area rejection for ICP
8. Click OK

**Result**:
- The registered moving surface can either replace the original moving tab or be stored in a new tab
- Success dialog shows:
  - Registration method used
  - Translation values (pixels)
  - Rotation angle (degrees, if ICP)
  - RMSE (registration quality metric)

**Use Cases**:
- Aligning opposing fracture surfaces
- Comparing before/after scans

### Overlay viewer assistance

The manual scan-comparison window includes one helper action:
- **Auto align (ICP, Experimental)** - estimates translation and in-plane rotation with a fast ICP pass and writes both into the manual sliders

This action does not save anything on its own; it only provides a starting
point for further manual refinement before accepting the alignment.

The same window now refreshes the difference map automatically while the
translation or rotation sliders are moving:
- a throttled lower-resolution preview is rendered during dragging,
- a reduced-resolution working difference map is rendered automatically after a
  short pause or immediately when the slider is released.

To keep routine use simpler, the difference panel now defaults to compact
**Center** and **+/-** controls with an **Auto** button. The auto mode estimates
the center from the current mean difference and expands the range around that
center. Each value is available both as a spin box and as a synchronized
slider for faster interactive tuning. The visibility and blinking toggles now
sit in one compact row, and the alignment / accept actions sit beside the
range controls instead of below them. The full interactive histogram is hidden
by default and can be opened on demand through **Advanced histogram**.

This split update path keeps the overlay responsive on larger grids without
changing the final accepted transform. The visual comparison window now favors
interactive responsiveness over pixel-exact difference rendering; the final
accepted transform is still applied to the original full-resolution moving
surface when the user confirms the alignment. The splitter between the overlay
view and the difference map can be dragged freely, including collapsing either
side almost completely when the user wants to focus on only one view.

---

## Usage Tips

### General Workflow
1. **Load scan** - Use File -> Open or toolbar
2. **Pre-process** (optional) - Remove outliers, fill holes
3. **Apply processing** - Use new Processing menu
4. **Visualize** - View 3D, profiles, or comparisons
5. **Save** - Export processed data

### Best Practices
- **Backup original data** - Processing operations modify the current scan directly
- **Start simple** - Try least-squares leveling before robust methods
- **Check parameters** - Hover over parameter fields for tooltips
- **Use appropriate filters**:
  - Bilateral for edge preservation
  - Median for spike removal
  - Gaussian for general smoothing
- **Registration order matters** - Choose reference surface carefully

### Performance
- **Bilateral filter** - Slowest, best quality (5-10 seconds for 512x512)
- **Median filter** - Fast (1-2 seconds)
- **Morphology** - Fast (1-2 seconds)
- **Leveling** - Very fast (<1 second)
- **Robust leveling** - Medium (2-5 seconds)
- **ICP registration** - Medium (3-10 seconds)

### Error Handling
All operations include:
- OK Wait cursor during processing
- OK Error dialogs with details
- OK Success confirmations
- OK Parameter validation
- OK Safe failure (data preserved on error)

---

## Keyboard Shortcuts

Currently, advanced processing features have no keyboard shortcuts assigned. You can add them by modifying the action definitions in `main_window.py`:

```python
self.actions["filter"].setShortcut("Ctrl+Shift+F")
self.actions["morphology"].setShortcut("Ctrl+Shift+L")
self.actions["transform"].setShortcut("Ctrl+Shift+T")
```

---

## Technical Details

### Code Structure

**Dialog Classes** (`frasta/gui/dialogs/processing_dialog.py`):
- `FilterDialog` - Filter selection and parameters
- `MorphologyDialog` - Morphology operation selection
- `TransformDialog` - Geometric transform selection
- `RegistrationDialog` - Surface pair selection

**Main Window Methods** (`frasta/gui/main_window.py`):
- `apply_advanced_filter()` - Handles filter application
- `apply_morphology()` - Handles morphology operations
- `apply_transform()` - Handles geometric transforms
- `auto_register_surfaces()` - Handles surface registration

### Data Flow
```
User clicks button/menu
    v
Dialog opens (parameter input)
    v
User clicks OK
    v
get_*_config() extracts parameters
    v
Processing function called
    v
tab.grid updated
    v
tab.update_histogram() refreshes display
    v
Success message shown
```

### Error Recovery
- Try-except blocks catch all processing errors
- Original data preserved if operation fails
- Detailed error messages shown to user
- Wait cursor always restored

---

## Future Enhancements

Potential improvements:
- [ ] Preview mode (show before/after comparison)
- [ ] Undo/redo functionality
- [ ] Batch processing (apply to all open scans)
- [ ] Custom keyboard shortcuts
- [ ] Icons for toolbar buttons
- [ ] Parameter presets/favorites
- [ ] Processing history log
- [ ] Real-time parameter preview
- [ ] Multi-threaded processing with progress bar
- [ ] Export processing pipeline as script

---

## Known Issues

1. **No undo** - Processing operations directly modify the scan. Save backups before processing.
2. **No live preview** - Morphology and leveling operations still apply after confirmation rather than through an interactive preview.
3. **No progress bar** - Long operations (bilateral filter) show wait cursor but no percentage.
4. **No icons** - Toolbar buttons use text labels only (icons can be added later).

---

## Related Documentation

- [Advanced Processing API](ADVANCED_PROCESSING.md) - Full function reference
- [Quick Reference](QUICK_REFERENCE.md) - Function cheat sheet
- [Examples](../examples/) - Python examples and visualizations

---

## Support

For issues or questions:
1. Check [ADVANCED_PROCESSING.md](ADVANCED_PROCESSING.md) for function details
2. Run [examples/advanced_processing.py](../examples/advanced_processing.py) to verify installation
3. Check console output for detailed error messages
4. Verify that the Python environment has SciPy and scikit-learn installed

---

**Last Updated**: February 2026  
**FRASTA-toolbox Version**: 1.x (with advanced processing integration)
