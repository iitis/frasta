"""Tests for the dedicated crack-path analysis dialog."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from frasta.core import Surface
from frasta.gui.dialogs.crack_path_dialog import CrackPathDialog


def _build_surface_pair_with_wavy_front() -> tuple[Surface, Surface]:
    """Create a synthetic aligned pair with a deterministic open-front shape."""
    front_rows = np.array([1, 2, 1, 2, 1], dtype=int)
    reference = np.ones((5, 5), dtype=float)
    adjusted = np.ones((5, 5), dtype=float)

    for col, row in enumerate(front_rows):
        adjusted[row:, col] = -1.0

    return Surface(reference, dx=2.0, dy=3.0), Surface(adjusted, dx=2.0, dy=3.0)


def test_crack_path_dialog_initializes_with_analysis_result(qapp):
    """Dialog initialization should compute crack-path metrics and overlay data."""
    surface_a, surface_b = _build_surface_pair_with_wavy_front()

    dialog = CrackPathDialog(surface_a, surface_b)
    try:
        assert dialog._analysis_result is not None
        assert dialog._threshold_sweep_result is not None
        assert dialog._lbl_status.text() == "OK"
        assert dialog._lbl_tortuosity.text() != "—"

        x_data, y_data = dialog._path_curve.getData()
        assert x_data is not None and y_data is not None
        assert len(x_data) > 0
        assert len(y_data) > 0
        sx, sy = dialog._sweep_curve.getData()
        assert sx is not None and sy is not None
        assert len(sx) == 41
        assert len(sy) == 41
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_crack_path_dialog_contour_method_changes_result(qapp):
    """The contour method should be selectable and produce a valid result."""
    surface_a, surface_b = _build_surface_pair_with_wavy_front()

    dialog = CrackPathDialog(surface_a, surface_b)
    try:
        base_points = np.asarray(dialog._analysis_result["path_points"], dtype=float)
        dialog._method_combo.setCurrentIndex(1)

        assert dialog._analysis_result is not None
        assert dialog._analysis_result["path_method"] == "contour"
        assert dialog._lbl_status.text() == "OK"
        contour_points = np.asarray(dialog._analysis_result["path_points"], dtype=float)
        assert contour_points.shape[0] >= 2
        assert not np.array_equal(contour_points, base_points)
        assert dialog._resample_spin.isEnabled() is True
        assert dialog._smoothing_spin.isEnabled() is True
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_crack_path_dialog_smoothing_control_reduces_reported_curvature(qapp):
    """Increasing contour smoothing should not increase the reported mean absolute curvature."""
    surface_a, surface_b = _build_surface_pair_with_wavy_front()

    dialog = CrackPathDialog(surface_a, surface_b)
    try:
        assert dialog._resample_spin.isEnabled() is False
        assert dialog._smoothing_spin.isEnabled() is False

        dialog._method_combo.setCurrentIndex(1)
        dialog._smoothing_spin.setValue(1)
        raw_value = float(dialog._lbl_curvature.text().split()[0])

        dialog._smoothing_spin.setValue(9)
        smooth_value = float(dialog._lbl_curvature.text().split()[0])
        assert smooth_value <= raw_value
        assert dialog._current_threshold_line.value() == dialog._threshold_spin.value()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_crack_path_dialog_exports_metrics_to_json(qapp, tmp_path):
    """JSON export should include the current crack-path metrics."""
    surface_a, surface_b = _build_surface_pair_with_wavy_front()
    output_path = tmp_path / "crack_path_metrics.json"

    dialog = CrackPathDialog(surface_a, surface_b, title_a="A", title_b="B")
    try:
        with patch(
            "frasta.gui.dialogs.crack_path_dialog.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(output_path), "JSON file (*.json)"),
        ):
            dialog._export_json()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["surface_a"] == "A"
        assert data["surface_b"] == "B"
        assert data["method"] == "first_open_pixel"
        assert data["path_point_count"] > 0
        assert data["tortuosity"] is not None
        assert data["mean_abs_curvature_inv_um"] is not None
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
