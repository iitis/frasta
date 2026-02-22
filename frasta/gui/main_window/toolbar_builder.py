"""Toolbar builder for main window.

Handles toolbar creation and organization.
"""

from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)


class ToolbarBuilder:
    """Builder for toolbar."""
    
    def __init__(self, main_window, menu_builder):
        """Initialize toolbar builder.
        
        Args:
            main_window: Reference to MainWindow instance
            menu_builder: Reference to MenuBuilder instance for accessing actions
        """
        self.main_window = main_window
        self.menu_builder = menu_builder
    
    def create_toolbar(self):
        """Create toolbar with commonly used actions."""
        toolbar = self.main_window.addToolBar("Tools")
        actions = self.menu_builder.actions
        
        # File operations
        toolbar.addAction(actions["open"])
        toolbar.addAction(actions["save_scan"])
        toolbar.addAction(actions["save_multi"])
        toolbar.addSeparator()
        
        # Basic processing
        toolbar.addAction(actions["repair"])
        toolbar.addAction(actions["flipUD"])
        toolbar.addAction(actions["flipLR"])
        toolbar.addAction(actions["rot90"])
        toolbar.addAction(actions["inverse"])
        toolbar.addAction(actions["zero"])
        toolbar.addAction(actions["tilt"])
        toolbar.addAction(actions["colormap"])
        toolbar.addSeparator()
        
        # Advanced processing
        toolbar.addAction(actions["filter"])
        toolbar.addAction(actions["morphology"])
        toolbar.addAction(actions["transform"])
        toolbar.addSeparator()
        
        # Tools
        toolbar.addAction(actions["view3d"])
        toolbar.addAction(actions["compare"])
        toolbar.addAction(actions["profile"])
        toolbar.addSeparator()
        
        # Help and exit
        toolbar.addAction(actions["about"])
        toolbar.addAction(actions["exit"])

        toolbar.setStyleSheet("QToolButton { color: #222; }")
        
        return toolbar
