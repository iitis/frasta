"""Tests for the contact-map dialog and linked crack-path window."""

from __future__ import annotations

import numpy as np
from unittest.mock import patch

from frasta.core import Surface
from frasta.gui.dialogs.contact_map_dialog import ContactMapDialog


def _build_surface_pair_with_wavy_front() -> tuple[Surface, Surface, np.ndarray]:
    """Create a synthetic aligned pair with a deterministic open-front shape."""
    front_rows = np.array([1, 2, 1, 2, 1], dtype=int)
    reference = np.ones((5, 5), dtype=float)
    adjusted = np.ones((5, 5), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    return Surface(reference, dx=2.0, dy=3.0), Surface(adjusted, dx=2.0, dy=3.0), front_rows


def test_contact_map_dialog_opens_linked_crack_path_dialog(qapp):
    """Contact-map dialog should open one linked crack-path analysis window."""
    surface_a, surface_b, _front_rows = _build_surface_pair_with_wavy_front()

    dialog = ContactMapDialog(surface_a, surface_b)
    try:
        assert dialog._crack_path_dialog is None
        dialog._open_crack_path_dialog()
        assert dialog._crack_path_dialog is not None
        assert dialog._crack_path_dialog._analysis_result is not None
        first_dialog = dialog._crack_path_dialog

        dialog._open_crack_path_dialog()
        assert dialog._crack_path_dialog is first_dialog
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_contact_map_dialog_syncs_threshold_to_linked_crack_path_dialog(qapp):
    """Threshold edits in the contact-map dialog should propagate to the child window."""
    surface_a, surface_b, _front_rows = _build_surface_pair_with_wavy_front()

    dialog = ContactMapDialog(surface_a, surface_b)
    try:
        dialog._open_crack_path_dialog()
        child = dialog._crack_path_dialog
        assert child is not None

        dialog._threshold_spin.setValue(3.0)
        assert child._threshold_spin.value() == 3.0
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_contact_map_dialog_exports_contact_statistics_to_json(qapp, tmp_path):
    """JSON export should remain focused on contact-map statistics."""
    surface_a, surface_b, _front_rows = _build_surface_pair_with_wavy_front()
    output_path = tmp_path / "contact_map_metrics.json"

    dialog = ContactMapDialog(surface_a, surface_b, title_a="A", title_b="B")
    try:
        with patch(
            "frasta.gui.dialogs.contact_map_dialog.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(output_path), "JSON file (*.json)"),
        ):
            dialog._export_json()

        import json
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["surface_a"] == "A"
        assert data["surface_b"] == "B"
        assert "crack_path_method" not in data
        assert data["contact_fraction_pct"] >= 0.0
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
