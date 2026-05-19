"""ROI settings dialog for FRASTA-toolbox."""

from __future__ import annotations

from PyQt5 import QtWidgets


class ROIDialog(QtWidgets.QDialog):
    """Dialog for editing ROI mode, shape, geometry, and display units."""

    UNIT_LADDER: list[tuple[str, str, float]] = [
        ("nm", "Nanometers (nm)", 1e-6),
        ("µm", "Micrometers (µm)", 1e-3),
        ("mm", "Millimeters (mm)", 1.0),
    ]

    def __init__(self, roi_config: dict, unit_label: str, parent=None):
        """Initialize the ROI settings dialog.

        Args:
            roi_config (dict): Initial ROI configuration in native scan units.
            unit_label (str): Native physical unit label of the active scan.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("ROI settings")
        self.unit_label = self._normalize_unit_label(unit_label)
        self.display_unit_options = self._build_display_unit_options(self.unit_label)
        self._last_units_mode = "native"
        self._polygon_points_native: tuple[tuple[float, float], ...] = ()
        self._init_ui()
        self._load_config(roi_config)
        self._update_enabled_state()
        self._update_shape_fields()

    @classmethod
    def _normalize_unit_label(cls, unit_label: str | None) -> str:
        """Map equivalent unit strings to the dialog's canonical labels."""
        normalized = (unit_label or "").strip().lower()
        alias_map = {
            "nm": "nm",
            "nanometer": "nm",
            "nanometers": "nm",
            "µm": "µm",
            "um": "µm",
            "micrometer": "µm",
            "micrometers": "µm",
            "mm": "mm",
            "millimeter": "mm",
            "millimeters": "mm",
        }
        return alias_map.get(normalized, unit_label or "native units")

    @classmethod
    def _build_display_unit_options(cls, native_unit_label: str) -> dict[str, dict[str, float | str]]:
        """Return available display units around the native unit when recognized."""
        options: dict[str, dict[str, float | str]] = {
            "native": {
                "label": f"Native ({native_unit_label})",
                "suffix": native_unit_label,
                "factor": 1.0,
            }
        }

        canonical_units = [entry[0] for entry in cls.UNIT_LADDER]
        if native_unit_label not in canonical_units:
            return options

        native_index = canonical_units.index(native_unit_label)
        candidate_indices = [native_index - 1, native_index + 1]
        for index in candidate_indices:
            if 0 <= index < len(cls.UNIT_LADDER):
                token, label, mm_factor = cls.UNIT_LADDER[index]
                native_mm_factor = cls.UNIT_LADDER[native_index][2]
                options[token] = {
                    "label": label,
                    "suffix": token,
                    "factor": native_mm_factor / mm_factor,
                }
        return options

    def _init_ui(self):
        """Build the dialog widgets."""
        layout = QtWidgets.QVBoxLayout(self)

        mode_group = QtWidgets.QGroupBox("Behavior")
        mode_layout = QtWidgets.QFormLayout(mode_group)
        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItem("Shared across scans", "global")
        self.mode_combo.addItem("Independent per scan", "per_scan")
        self.units_combo = QtWidgets.QComboBox(self)
        for token, option in self.display_unit_options.items():
            self.units_combo.addItem(str(option["label"]), token)
        self.units_combo.currentIndexChanged.connect(self._reload_geometry_for_units)
        self.enabled_checkbox = QtWidgets.QCheckBox("ROI enabled", self)
        self.enabled_checkbox.toggled.connect(self._update_enabled_state)
        mode_layout.addRow("Mode:", self.mode_combo)
        mode_layout.addRow("Units:", self.units_combo)
        mode_layout.addRow(self.enabled_checkbox)
        layout.addWidget(mode_group)

        shape_group = QtWidgets.QGroupBox("Shape")
        shape_layout = QtWidgets.QFormLayout(shape_group)
        self.shape_combo = QtWidgets.QComboBox(self)
        self.shape_combo.addItem("Circle", "circle")
        self.shape_combo.addItem("Ring", "ring")
        self.shape_combo.addItem("Rectangle", "rectangle")
        self.shape_combo.addItem("Polygon", "polygon")
        self.shape_combo.currentIndexChanged.connect(self._update_shape_fields)
        shape_layout.addRow("ROI type:", self.shape_combo)
        layout.addWidget(shape_group)

        geometry_group = QtWidgets.QGroupBox("Geometry")
        self.geometry_layout = QtWidgets.QFormLayout(geometry_group)
        self.center_x_spin = self._create_double_spin()
        self.center_y_spin = self._create_double_spin()
        self.radius_spin = self._create_double_spin(minimum=0.0001)
        self.inner_radius_spin = self._create_double_spin(minimum=0.0)
        self.width_spin = self._create_double_spin(minimum=0.0001)
        self.height_spin = self._create_double_spin(minimum=0.0001)

        self.center_x_label = QtWidgets.QLabel(self)
        self.center_y_label = QtWidgets.QLabel(self)
        self.radius_label = QtWidgets.QLabel(self)
        self.inner_radius_label = QtWidgets.QLabel(self)
        self.width_label = QtWidgets.QLabel(self)
        self.height_label = QtWidgets.QLabel(self)
        self.geometry_layout.addRow(self.center_x_label, self.center_x_spin)
        self.geometry_layout.addRow(self.center_y_label, self.center_y_spin)
        self.geometry_layout.addRow(self.radius_label, self.radius_spin)
        self.geometry_layout.addRow(self.inner_radius_label, self.inner_radius_spin)
        self.geometry_layout.addRow(self.width_label, self.width_spin)
        self.geometry_layout.addRow(self.height_label, self.height_spin)
        layout.addWidget(geometry_group)

        self.scope_label = QtWidgets.QLabel(self)
        self.scope_label.setWordWrap(True)
        self.scope_label.setStyleSheet("color: gray; font-style: italic;")
        self.mode_combo.currentIndexChanged.connect(self._update_scope_text)
        layout.addWidget(self.scope_label)

        self.shape_help_label = QtWidgets.QLabel(self)
        self.shape_help_label.setWordWrap(True)
        self.shape_help_label.setStyleSheet("color: gray;")
        layout.addWidget(self.shape_help_label)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _create_double_spin(minimum: float = -1e6, maximum: float = 1e6) -> QtWidgets.QDoubleSpinBox:
        """Create a double spin box suitable for geometry input."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(1.0)
        return spin

    def _native_to_display(self, value: float, units_mode: str | None = None) -> float:
        """Convert a native-unit value to the currently selected display units."""
        mode = self.units_combo.currentData() if units_mode is None else units_mode
        factor = float(self.display_unit_options.get(mode, {}).get("factor", 1.0))
        return value * factor

    def _display_to_native(self, value: float, units_mode: str | None = None) -> float:
        """Convert a displayed value back to native scan units."""
        mode = self.units_combo.currentData() if units_mode is None else units_mode
        factor = float(self.display_unit_options.get(mode, {}).get("factor", 1.0))
        return value / factor

    def _load_config(self, roi_config: dict):
        """Populate widgets from the supplied ROI configuration."""
        mode_index = self.mode_combo.findData(roi_config.get("mode", "global"))
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)

        preferred_units = next((token for token in ("mm", "µm", "nm") if token in self.display_unit_options), "native")
        units_index = self.units_combo.findData(preferred_units)
        if units_index >= 0:
            self.units_combo.setCurrentIndex(units_index)
        self._last_units_mode = self.units_combo.currentData()

        self.enabled_checkbox.setChecked(bool(roi_config.get("enabled", False)))

        shape = roi_config.get("shape", "circle")
        shape_index = self.shape_combo.findData(shape)
        if shape_index >= 0:
            self.shape_combo.setCurrentIndex(shape_index)

        self._polygon_points_native = tuple(
            (float(point[0]), float(point[1])) for point in roi_config.get("points", ())
        )
        self._set_geometry_fields_from_native(roi_config)
        self._update_scope_text()

    def _set_geometry_fields_from_native(self, roi_config: dict):
        """Populate geometry widgets from native-unit configuration values."""
        center_x, center_y = roi_config.get("center", (0.0, 0.0))
        size_x, size_y = roi_config.get("size", (100.0, 100.0))
        inner_size = float(roi_config.get("inner_size", size_x * 0.5))
        self.center_x_spin.setValue(self._native_to_display(center_x))
        self.center_y_spin.setValue(self._native_to_display(center_y))
        self.radius_spin.setValue(max(self._native_to_display(size_x / 2.0), self.radius_spin.minimum()))
        self.inner_radius_spin.setValue(max(self._native_to_display(inner_size / 2.0), self.inner_radius_spin.minimum()))
        self.width_spin.setValue(max(self._native_to_display(size_x), self.width_spin.minimum()))
        self.height_spin.setValue(max(self._native_to_display(size_y), self.height_spin.minimum()))
        self._update_geometry_labels()

    def _reload_geometry_for_units(self):
        """Repaint geometry values after a unit selection change."""
        old_units_mode = self._last_units_mode
        current_native = self.get_roi_config(units_mode=old_units_mode)
        self._set_geometry_fields_from_native(current_native)
        self._last_units_mode = self.units_combo.currentData()

    def _update_enabled_state(self):
        """Enable or disable geometry widgets according to ROI visibility."""
        enabled = self.enabled_checkbox.isChecked()
        self.shape_combo.setEnabled(enabled)
        self._update_shape_fields()
        self._update_geometry_labels()

    def _update_shape_fields(self):
        """Show geometry fields relevant to the selected shape."""
        enabled = self.enabled_checkbox.isChecked()
        circle_selected = self.shape_combo.currentData() == "circle"
        ring_selected = self.shape_combo.currentData() == "ring"
        polygon_selected = self.shape_combo.currentData() == "polygon"

        if polygon_selected and not self._polygon_points_native:
            center_x = self._display_to_native(self.center_x_spin.value())
            center_y = self._display_to_native(self.center_y_spin.value())
            width = self._display_to_native(self.width_spin.value())
            height = self._display_to_native(self.height_spin.value())
            radius_x = max(width / 2.0, 1e-6)
            radius_y = max(height / 2.0, 1e-6)
            self._polygon_points_native = (
                (center_x, center_y - radius_y),
                (center_x + radius_x * 0.85, center_y - radius_y * 0.5),
                (center_x + radius_x * 0.85, center_y + radius_y * 0.5),
                (center_x, center_y + radius_y),
                (center_x - radius_x * 0.85, center_y + radius_y * 0.5),
                (center_x - radius_x * 0.85, center_y - radius_y * 0.5),
            )

        self.radius_label.setVisible(circle_selected or ring_selected)
        self.radius_spin.setVisible(circle_selected or ring_selected)
        self.inner_radius_label.setVisible(ring_selected)
        self.inner_radius_spin.setVisible(ring_selected)
        self.width_label.setVisible(not circle_selected and not ring_selected and not polygon_selected)
        self.width_spin.setVisible(not circle_selected and not ring_selected and not polygon_selected)
        self.height_label.setVisible(not circle_selected and not ring_selected and not polygon_selected)
        self.height_spin.setVisible(not circle_selected and not ring_selected and not polygon_selected)

        self.center_x_spin.setEnabled(enabled and not polygon_selected)
        self.center_y_spin.setEnabled(enabled and not polygon_selected)
        self.radius_spin.setEnabled(enabled and (circle_selected or ring_selected))
        self.inner_radius_spin.setEnabled(enabled and ring_selected)
        self.width_spin.setEnabled(enabled and not circle_selected and not ring_selected and not polygon_selected)
        self.height_spin.setEnabled(enabled and not circle_selected and not ring_selected and not polygon_selected)
        self.shape_help_label.setVisible(enabled and polygon_selected)
        if polygon_selected:
            self.shape_help_label.setText(
                "Polygon vertices are edited directly in the image view after the ROI is created."
            )
        elif ring_selected:
            self.shape_help_label.setText(
                "Ring ROI uses the outer circle for interactive move/scale; set the inner radius numerically here."
            )
        else:
            self.shape_help_label.clear()
        self._update_geometry_labels()

    def _current_unit_suffix(self) -> str:
        """Return the unit suffix shown beside geometry labels."""
        mode = self.units_combo.currentData()
        return str(self.display_unit_options.get(mode, {}).get("suffix", self.unit_label))

    def _update_geometry_labels(self):
        """Refresh geometry labels to match the selected display units."""
        suffix = self._current_unit_suffix()
        self.center_x_label.setText(f"Center X ({suffix}):")
        self.center_y_label.setText(f"Center Y ({suffix}):")
        self.radius_label.setText(f"Radius ({suffix}):")
        self.inner_radius_label.setText(f"Inner Radius ({suffix}):")
        self.width_label.setText(f"Width ({suffix}):")
        self.height_label.setText(f"Height ({suffix}):")

    def _update_scope_text(self):
        """Explain how the current dialog settings will be applied."""
        if self.mode_combo.currentData() == "global":
            self.scope_label.setText(
                "Shared mode uses one ROI geometry for all scans. Applying these "
                "settings updates the common ROI."
            )
        else:
            self.scope_label.setText(
                "Independent mode stores one ROI per tab. Applying these settings "
                "updates the current scan only."
            )

    def get_roi_config(self, units_mode: str | None = None) -> dict:
        """Return the ROI configuration selected in the dialog."""
        shape = self.shape_combo.currentData()
        enabled = self.enabled_checkbox.isChecked()
        center_x = self._display_to_native(self.center_x_spin.value(), units_mode=units_mode)
        center_y = self._display_to_native(self.center_y_spin.value(), units_mode=units_mode)

        if shape == "polygon":
            if not self._polygon_points_native:
                width = self._display_to_native(self.width_spin.value(), units_mode=units_mode)
                height = self._display_to_native(self.height_spin.value(), units_mode=units_mode)
                self._polygon_points_native = (
                    (center_x, center_y - height / 2.0),
                    (center_x + width / 2.0, center_y),
                    (center_x, center_y + height / 2.0),
                    (center_x - width / 2.0, center_y),
                )
            polygon_points = self._polygon_points_native
            if polygon_points:
                x_values = [point[0] for point in polygon_points]
                y_values = [point[1] for point in polygon_points]
                center = ((min(x_values) + max(x_values)) / 2.0, (min(y_values) + max(y_values)) / 2.0)
                size = (max(max(x_values) - min(x_values), 1e-6), max(max(y_values) - min(y_values), 1e-6))
            else:
                center = (center_x, center_y)
                size = (1.0, 1.0)
            return {
                "mode": self.mode_combo.currentData(),
                "enabled": enabled,
                "shape": shape,
                "center": center,
                "size": size,
                "points": polygon_points,
            }

        if shape == "circle":
            radius = self._display_to_native(self.radius_spin.value(), units_mode=units_mode)
            size = (radius * 2.0, radius * 2.0)
            inner_size = 0.0
        elif shape == "ring":
            outer_radius = self._display_to_native(self.radius_spin.value(), units_mode=units_mode)
            inner_radius = self._display_to_native(self.inner_radius_spin.value(), units_mode=units_mode)
            inner_radius = min(max(inner_radius, 0.0), outer_radius * 0.999)
            self.inner_radius_spin.blockSignals(True)
            self.inner_radius_spin.setValue(self._native_to_display(inner_radius, units_mode=units_mode))
            self.inner_radius_spin.blockSignals(False)
            size = (outer_radius * 2.0, outer_radius * 2.0)
            inner_size = inner_radius * 2.0
        else:
            size = (
                self._display_to_native(self.width_spin.value(), units_mode=units_mode),
                self._display_to_native(self.height_spin.value(), units_mode=units_mode),
            )
            inner_size = 0.0

        return {
            "mode": self.mode_combo.currentData(),
            "enabled": enabled,
            "shape": shape,
            "center": (center_x, center_y),
            "size": size,
            "points": tuple(),
            "inner_size": inner_size,
        }
