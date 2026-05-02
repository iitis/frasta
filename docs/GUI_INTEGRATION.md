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

### Toolbar

Three new buttons added to the toolbar (after "Set tilt", before "View 3D"):
- **Advanced Filtering** - Opens filter selection dialog
- **Morphology & Leveling** - Opens morphology operations dialog
- **Geometric Transforms** - Opens transform dialog

---

## Feature Dialogs

### 1. Advanced Filtering Dialog

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

**Result**: The filtered data can either replace the current scan or be stored in a new tab
If an ROI is visible, filtering is automatically restricted to that ROI.

---

### 2. Morphology & Leveling Dialog

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

**Result**: The processed data can either replace the current scan or be stored in a new tab
If an ROI is visible, leveling and polynomial form removal use that ROI automatically.

---

### 3. Geometric Transforms Dialog

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

### 4. Auto-Register Surfaces Dialog

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
by masking everything outside the selected region.

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
- **Auto align (ICP)** - estimates translation and in-plane rotation with a fast ICP pass and writes both into the manual sliders

This action does not save anything on its own; it only provides a starting
point for further manual refinement before accepting the alignment.

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
