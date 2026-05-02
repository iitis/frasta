"""Scan information dialog for FRASTA-toolbox."""

from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtWidgets


def unit_to_mm_factor(unit_label: str | None) -> float | None:
    """Return a conversion factor from native scan units to millimeters."""
    normalized = (unit_label or "").strip().lower()
    mapping = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "µm": 0.001,
        "um": 0.001,
        "micrometer": 0.001,
        "micrometers": 0.001,
        "nm": 0.000001,
        "nanometer": 0.000001,
        "nanometers": 0.000001,
    }
    return mapping.get(normalized)


class ScanInfoDialog(QtWidgets.QDialog):
    """Read-only dialog showing geometry, statistics, and metadata of a scan."""

    def __init__(self, tab, tab_title: str, parent=None):
        """Initialize the scan information dialog.

        Args:
            tab: Active scan tab providing grid and coordinate data.
            tab_title (str): Visible title of the tab in the main window.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.tab = tab
        self.tab_title = tab_title
        self.setWindowTitle("Scan information")
        self.resize(560, 520)
        self._init_ui()

    def _init_ui(self):
        """Build the dialog layout."""
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Summary of the active scan, including geometry, valid-data coverage, "
            "value statistics, and stored metadata.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary_group = QtWidgets.QGroupBox("Geometry and statistics", self)
        summary_layout = QtWidgets.QFormLayout(summary_group)

        summary = self._build_summary_map()
        for label, value in summary:
            summary_layout.addRow(f"{label}:", self._create_value_label(value))

        layout.addWidget(summary_group)

        metadata_group = QtWidgets.QGroupBox("Metadata", self)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_group)
        metadata_view = QtWidgets.QPlainTextEdit(self)
        metadata_view.setReadOnly(True)
        metadata_view.setPlainText(self._build_metadata_text())
        metadata_layout.addWidget(metadata_view)
        layout.addWidget(metadata_group, 1)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=self)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        button_box.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(button_box)

    @staticmethod
    def _create_value_label(value: str) -> QtWidgets.QLabel:
        """Create a selectable value label."""
        label = QtWidgets.QLabel(value)
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _format_number(value: float, decimals: int = 4) -> str:
        """Format a floating-point value for human-readable display."""
        if not np.isfinite(value):
            return "n/a"
        if value == 0:
            return "0"
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.{decimals}e}"
        return f"{value:.{decimals}f}"

    def _format_with_unit(self, value: float, unit_label: str) -> str:
        """Format a value in native units and in millimeters when available."""
        native_text = f"{self._format_number(value)} {unit_label}"
        factor = unit_to_mm_factor(unit_label)
        if factor is None:
            return native_text
        value_mm = value * factor
        return f"{native_text} ({self._format_number(value_mm)} mm)"

    def _build_summary_map(self) -> list[tuple[str, str]]:
        """Collect scan geometry and scalar statistics for display."""
        grid = np.asarray(self.tab.grid, dtype=float)
        unit_label = getattr(self.tab, "unit", "a.u.")
        xi = np.asarray(self.tab.xi) if self.tab.xi is not None else np.array([])
        yi = np.asarray(self.tab.yi) if self.tab.yi is not None else np.array([])
        dx = float(self.tab.dx) if self.tab.dx is not None else math.nan
        dy = float(self.tab.dy) if self.tab.dy is not None else math.nan

        valid_mask = np.isfinite(grid)
        valid_count = int(valid_mask.sum())
        total_count = int(grid.size)
        invalid_count = total_count - valid_count
        coverage = (100.0 * valid_count / total_count) if total_count else 0.0

        if valid_count:
            values = grid[valid_mask]
            z_min = self._format_with_unit(float(np.min(values)), unit_label)
            z_max = self._format_with_unit(float(np.max(values)), unit_label)
            z_mean = self._format_with_unit(float(np.mean(values)), unit_label)
            z_std = self._format_with_unit(float(np.std(values)), unit_label)
        else:
            z_min = z_max = z_mean = z_std = "n/a"

        x_min = float(xi[0]) if xi.size else 0.0
        x_max = float(xi[-1]) if xi.size else 0.0
        y_min = float(yi[0]) if yi.size else 0.0
        y_max = float(yi[-1]) if yi.size else 0.0
        width = float((xi[-1] - xi[0])) if xi.size > 1 else 0.0
        height = float((yi[-1] - yi[0])) if yi.size > 1 else 0.0

        return [
            ("Tab", self.tab_title),
            ("Source name", str(getattr(self.tab.get_surface(), "metadata", {}).get("name", self.tab_title))),
            ("Grid shape", f"{grid.shape[0]} rows x {grid.shape[1]} columns"),
            ("Native unit", unit_label),
            ("Spacing dx", self._format_with_unit(dx, unit_label)),
            ("Spacing dy", self._format_with_unit(dy, unit_label)),
            ("Origin x0", self._format_with_unit(x_min, unit_label)),
            ("Origin y0", self._format_with_unit(y_min, unit_label)),
            ("X range", f"{self._format_with_unit(x_min, unit_label)} to {self._format_with_unit(x_max, unit_label)}"),
            ("Y range", f"{self._format_with_unit(y_min, unit_label)} to {self._format_with_unit(y_max, unit_label)}"),
            ("Width", self._format_with_unit(width, unit_label)),
            ("Height", self._format_with_unit(height, unit_label)),
            ("Valid points", f"{valid_count} / {total_count} ({coverage:.2f}%)"),
            ("Missing or invalid points", str(invalid_count)),
            ("Z minimum", z_min),
            ("Z maximum", z_max),
            ("Z mean", z_mean),
            ("Z standard deviation", z_std),
        ]

    def _build_metadata_text(self) -> str:
        """Serialize the surface metadata dictionary to a readable text block."""
        metadata = getattr(self.tab.get_surface(), "metadata", {}) or {}
        if not metadata:
            return "No additional metadata stored for this scan."

        lines = []
        for key in sorted(metadata):
            value = metadata[key]
            if isinstance(value, (list, tuple)):
                value_text = ", ".join(str(item) for item in value)
            else:
                value_text = str(value)
            lines.append(f"{key}: {value_text}")
        return "\n".join(lines)
