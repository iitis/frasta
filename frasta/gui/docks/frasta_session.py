"""FRASTA session save / restore.

A ``.frasta`` file is a ZIP archive containing:

* ``grids.npz``  — ``ref_grid``, ``adj_grid``, ``dx``, ``dy``
* ``session.json`` — UI settings, analysis state, profile line endpoints

Usage::

    from frasta.gui.docks.frasta_session import save_session, load_session
    save_session("analysis.frasta", controller)
    load_session("analysis.frasta", controller)
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .frasta_controller import FrastaController

import logging
logger = logging.getLogger(__name__)

FRASTA_SESSION_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_session(path: str, controller: "FrastaController") -> None:
    """Save the current FRASTA analysis state to a ``.frasta`` file.

    Parameters
    ----------
    path:
        Destination path (should end in ``.frasta``).
    controller:
        :class:`FrastaController` whose state is serialised.

    Raises
    ------
    ValueError
        If no data has been loaded into the controller yet.
    """
    bd = controller.binary_dock
    pd = controller.profile_dock

    if bd._diff_map is None:
        raise ValueError("No data loaded – nothing to save.")

    # ---- Collect binary-dock state ----
    map_mode    = bd._map_mode
    binary_cmap = bd._combo_binary_cmap.currentData() or "gray"
    diff_cmap   = bd._combo_cmap.currentText()
    diff_center = bd._spinbox_diff_center.value()
    diff_range  = bd._spinbox_diff_range.value()
    separation  = bd._spinbox.value()
    dx, dy      = bd._dx, bd._dy

    # Profile-line endpoints (c0, r0, c1, r1) stored in controller
    endpoints = controller._current_endpoints

    # ---- Collect profile-dock state ----
    window_size_um = pd._spinbox_window.value()
    snap_to_plot   = pd._checkbox_snap.isChecked()

    curve_visibility: dict[str, bool] = {}
    for name, (item, _color) in pd._curve_items.items():
        try:
            curve_visibility[name] = bool(item.isVisible())
        except Exception:
            curve_visibility[name] = True

    session_data = {
        "frasta_version": FRASTA_SESSION_VERSION,
        "saved": datetime.now().isoformat(),
        "separation": separation,
        "dx": dx,
        "dy": dy,
        "map_mode": map_mode,
        "binary_cmap": binary_cmap,
        "diff_cmap": diff_cmap,
        "diff_center": diff_center,
        "diff_range": diff_range,
        "profile_endpoints": list(endpoints) if endpoints is not None else None,
        "window_size_um": window_size_um,
        "snap_to_plot": snap_to_plot,
        "curve_visibility": curve_visibility,
    }

    # ---- Write ZIP ----
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Grids
        buf = io.BytesIO()
        arrays: dict = {"dx": np.float64(dx), "dy": np.float64(dy)}
        if bd._grid1 is not None:
            arrays["ref_grid"] = bd._grid1
        if bd._grid2 is not None:
            arrays["adj_grid"] = bd._grid2
        np.savez_compressed(buf, **arrays)
        zf.writestr("grids.npz", buf.getvalue())

        # Session JSON
        zf.writestr("session.json", json.dumps(session_data, indent=2))

    logger.info("FRASTA session saved to %s", path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_session(path: str, controller: "FrastaController") -> None:
    """Restore FRASTA analysis state from a ``.frasta`` file.

    Parameters
    ----------
    path:
        Path to an existing ``.frasta`` session file.
    controller:
        :class:`FrastaController` whose state will be overwritten.

    Raises
    ------
    ValueError
        If the file is missing required data arrays.
    """
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("session.json") as jf:
            session = json.load(jf)

        with zf.open("grids.npz") as nf:
            buf    = io.BytesIO(nf.read())
            grids  = np.load(buf)
            ref_grid = grids["ref_grid"] if "ref_grid" in grids else None
            adj_grid = grids["adj_grid"] if "adj_grid" in grids else None
            dx = float(grids["dx"]) if "dx" in grids else float(session.get("dx", 1.0))
            dy = float(grids["dy"]) if "dy" in grids else float(session.get("dy", 1.0))

    if ref_grid is None or adj_grid is None:
        raise ValueError("Session file is missing ref_grid or adj_grid.")

    bd  = controller.binary_dock
    pd  = controller.profile_dock
    sep = float(session.get("separation", 0.0))

    # Load grids — this resets ROI to full extent and calls _update_plot()
    diff = ref_grid - adj_grid
    bd.set_data(diff, ref_grid, adj_grid, dx=dx, dy=dy)

    # Restore separation without triggering a second full update
    bd._spinbox.blockSignals(True)
    bd._spinbox.setValue(sep)
    bd._spinbox.blockSignals(False)
    bd._sync_slider_from_spinbox()

    # Restore ROI endpoints
    endpoints = session.get("profile_endpoints")
    if endpoints is not None and len(endpoints) == 4:
        c0, r0, c1, r1 = endpoints
        bd.restore_roi(float(c0), float(r0), float(c1), float(r1))

    # Restore display settings
    bd.restore_display_settings(
        map_mode    = session.get("map_mode", "binary"),
        binary_cmap = session.get("binary_cmap", "gray"),
        diff_cmap   = session.get("diff_cmap", "CET-R4"),
        diff_center = float(session.get("diff_center", 0.0)),
        diff_range  = float(session.get("diff_range", 1.0)),
    )

    # Restore profile-dock settings
    pd.restore_settings(
        window_size_um  = float(session.get("window_size_um", 500.0)),
        snap_to_plot    = bool(session.get("snap_to_plot", True)),
        curve_visibility = session.get("curve_visibility", {}),
    )

    # Final redraw with all restored settings
    bd._update_plot()

    logger.info("FRASTA session loaded from %s", path)
