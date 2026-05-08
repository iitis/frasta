"""Main window for FRASTA-toolbox application.

This module provides the main window structure and routing to specialized controllers.
The main window delegates functionality to:
- ROIController: ROI operations and masking
- FileController: File loading and saving
- ProcessingController: Data processing operations
- RegistrationController: Scan comparison and registration
- MenuBuilder: Menu and action management
- ToolbarBuilder: Toolbar creation
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon

from ..dialogs import AboutDialog, ScanInfoDialog
from ..scan_tab import ScanTab
from ..viewers import (
    show_point_3d_viewer,
)
from ...core import Surface

from .roi_controller import ROIController
from .file_controller import FileController
from .processing_controller import ProcessingController
from .registration_controller import RegistrationController
from .menu_builder import MenuBuilder
from .toolbar_builder import ToolbarBuilder

import logging
logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window for the scan loader and hole filler tool.

    Provides a multi-tab interface for loading, viewing, processing, and saving 2D scan data.
    Delegates specific functionality to specialized controllers for better code organization.
    """

    def __init__(self):
        """Initializes the main window and sets up the user interface.

        Sets up the tab widget, controllers, menus, and toolbar.
        """
        super().__init__()
        self.setWindowTitle("FRASTA-toolbox")
        self.setGeometry(100, 100, 1000, 600)

        # Setup tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)

        # Initialize controllers
        self.roi_controller = ROIController(self)
        self.file_controller = FileController(self)
        self.processing_controller = ProcessingController(self)
        self.registration_controller = RegistrationController(self)
        
        # Initialize menu and toolbar builders
        self.menu_builder = MenuBuilder(self)
        self.toolbar_builder = ToolbarBuilder(self, self.menu_builder)
        
        # Build UI
        self.menu_builder.create_actions()
        self.menu_builder.connect_actions()
        self.menu_builder.create_menubar()
        self.toolbar = self.toolbar_builder.create_toolbar()

        # Connect tab changes to ROI controller
        self.tabs.currentChanged.connect(self.roi_controller.move_roi_to_current_tab)
        self.tabs.currentChanged.connect(self.sync_colormap_selector)

    def close_tab(self, index: int):
        """Close a tab at given index.
        
        Args:
            index (int): Tab index to close
        """
        widget = self.tabs.widget(index)
        if widget is not None:
            self.roi_controller.remove_tab_state(widget)
            self.tabs.removeTab(index)
            widget.deleteLater()

    def current_tab(self):
        """Get currently active tab.
        
        Returns:
            ScanTab or None: Current tab widget
        """
        return self.tabs.currentWidget()

    def make_unique_tab_title(self, base_title: str) -> str:
        """Return a tab title that does not collide with existing tabs.

        Args:
            base_title (str): Preferred tab title.

        Returns:
            str: Unique tab title derived from the requested base title.
        """
        existing_titles = {
            self.tabs.tabText(index)
            for index in range(self.tabs.count())
        }
        if base_title not in existing_titles:
            return base_title

        suffix = 2
        while f"{base_title} ({suffix})" in existing_titles:
            suffix += 1
        return f"{base_title} ({suffix})"

    def create_surface_tab(self, surface: Surface, title: str) -> ScanTab:
        """Create a new scan tab and populate it with surface data.

        Args:
            surface (Surface): Surface data to display in the new tab.
            title (str): Preferred tab title.

        Returns:
            ScanTab: Newly created and selected scan tab.
        """
        source_tab = self.current_tab()
        tab = ScanTab()
        unique_title = self.make_unique_tab_title(title)
        self.tabs.addTab(tab, unique_title)
        self.tabs.setCurrentWidget(tab)
        tab.set_surface(surface)
        self.roi_controller.initialize_tab_roi_state(tab, source_tab=source_tab)
        self.roi_controller.move_roi_to_current_tab(self.tabs.currentIndex())
        return tab

    def prompt_result_target(
        self,
        title: str,
        message: str,
        overwrite_label: str,
        new_tab_label: str = "Create new tab",
    ) -> str | None:
        """Ask whether a processing result should overwrite or create a tab.

        Args:
            title (str): Dialog title.
            message (str): Short explanatory message.
            overwrite_label (str): Label for the overwrite action.
            new_tab_label (str): Label for creating new output tab(s).

        Returns:
            str | None: ``"new_tab"`` or ``"overwrite"`` when chosen,
            otherwise ``None`` if the dialog is cancelled.
        """
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        new_button = dialog.addButton(new_tab_label, QtWidgets.QMessageBox.AcceptRole)
        overwrite_button = dialog.addButton(overwrite_label, QtWidgets.QMessageBox.ActionRole)
        dialog.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        dialog.exec_()

        if dialog.clickedButton() == new_button:
            return "new_tab"
        if dialog.clickedButton() == overwrite_button:
            return "overwrite"
        return None

    def view3d(self):
        """Show the default 3D viewer for the current tab."""
        if tab := self.current_tab():
            # Przekaż rozmiary pikseli dla prawidłowych proporcji
            dx = getattr(tab, 'dx', 1.0)
            dy = getattr(tab, 'dy', 1.0)
            show_point_3d_viewer(tab.grid, pixel_size_x=dx, pixel_size_y=dy)

    def view3d_points(self):
        """Show the experimental point-based 3D viewer for current tab."""
        if tab := self.current_tab():
            dx = getattr(tab, 'dx', 1.0)
            dy = getattr(tab, 'dy', 1.0)
            show_point_3d_viewer(tab.grid, pixel_size_x=dx, pixel_size_y=dy)

    def export_2d_image(self):
        """Export the active 2D scan view as a PNG image."""
        if tab := self.current_tab():
            tab.export_2d_image()

    def export_2d_colorbar(self):
        """Export the active 2D scan colorbar as a PNG image."""
        if tab := self.current_tab():
            tab.export_2d_colorbar()

    def toggle_colormap_current_tab(self):
        """Toggle colormap for current tab."""
        if tab := self.current_tab():
            tab.toggle_colormap()
            self.sync_colormap_selector()

    def set_current_tab_colormap(self, name: str):
        """Apply selected 2D colormap to the current tab."""
        if tab := self.current_tab():
            tab.set_colormap(name)
            self.sync_colormap_selector()

    def sync_colormap_selector(self, _index: int | None = None):
        """Synchronize toolbar colormap selector with the active tab."""
        combo = getattr(self.toolbar_builder, "colormap_combo", None)
        action = self.menu_builder.actions.get("colormap")
        if combo is None:
            return

        combo.blockSignals(True)
        try:
            if tab := self.current_tab():
                name = tab.get_colormap_name()
                idx = combo.findText(name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                if action is not None:
                    action.setChecked(name != "Gray")
            else:
                idx = combo.findText("Gray")
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                if action is not None:
                    action.setChecked(False)
        finally:
            combo.blockSignals(False)

    def set_zero_point_mode(self):
        """Enable zero point selection mode for current tab."""
        if tab := self.current_tab():
            tab.set_zero_point_mode()

    def set_tilt_mode(self):
        """Enable tilt correction mode for current tab."""
        if tab := self.current_tab():
            tab.set_tilt_mode()

    def show_about_dialog(self):
        """Show about dialog."""
        dlg = AboutDialog(self)
        dlg.exec_()

    def show_scan_info_dialog(self):
        """Show a read-only dialog with information about the active scan."""
        tab = self.current_tab()
        if tab is None:
            QtWidgets.QMessageBox.information(
                self,
                "Scan information",
                "Load or select a scan tab first.",
            )
            return

        dialog = ScanInfoDialog(
            tab=tab,
            tab_title=self.tabs.tabText(self.tabs.currentIndex()),
            parent=self,
        )
        dialog.exec_()

    def closeEvent(self, event):
        """Handle window close event.
        
        Args:
            event: Close event
        """
        self.file_controller.settings.setValue("recentFiles", self.file_controller.recent_files)
        event.accept()
