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

from ..dialogs import AboutDialog
from ..viewers import show_3d_viewer

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

    def close_tab(self, index: int):
        """Close a tab at given index.
        
        Args:
            index (int): Tab index to close
        """
        widget = self.tabs.widget(index)
        if widget is not None:
            self.tabs.removeTab(index)
            widget.deleteLater()

    def current_tab(self):
        """Get currently active tab.
        
        Returns:
            ScanTab or None: Current tab widget
        """
        return self.tabs.currentWidget()

    def view3d(self):
        """Show 3D viewer for current tab."""
        if tab := self.current_tab():
            # Przekaż rozmiary pikseli dla prawidłowych proporcji
            dx = getattr(tab, 'dx', 1.0)
            dy = getattr(tab, 'dy', 1.0)
            show_3d_viewer(tab.grid, show_controls=True, pixel_size_x=dx, pixel_size_y=dy)

    def toggle_colormap_current_tab(self):
        """Toggle colormap for current tab."""
        if tab := self.current_tab():
            tab.toggle_colormap()

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

    def closeEvent(self, event):
        """Handle window close event.
        
        Args:
            event: Close event
        """
        self.file_controller.settings.setValue("recentFiles", self.file_controller.recent_files)
        event.accept()
