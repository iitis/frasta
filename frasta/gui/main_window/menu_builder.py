"""Menu builder for main window.

Handles menu and action creation:
- Creating QActions with icons and shortcuts
- Connecting actions to controller methods
- Building menu structure
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon

from ...utils import resource_path

import logging
logger = logging.getLogger(__name__)


class MenuBuilder:
    """Builder for menus and actions."""
    
    def __init__(self, main_window):
        """Initialize menu builder.
        
        Args:
            main_window: Reference to MainWindow instance
        """
        self.main_window = main_window
        self.actions = {}
        self.recent_menu = None
    
    def create_actions(self):
        """Create all QActions with icons and tooltips."""
        self.actions = {
            "open": QtWidgets.QAction("Open...", self.main_window),
            "save_scan": QtWidgets.QAction("Save current scan...", self.main_window),
            "save_multi": QtWidgets.QAction("Save multiple scans...", self.main_window),
            "fill": QtWidgets.QAction("Fill holes", self.main_window),
            "repair": QtWidgets.QAction("Remove holes and outliers", self.main_window),
            "flipUD": QtWidgets.QAction("Flip Up/Down", self.main_window),
            "flipLR": QtWidgets.QAction("Flip Left/Right", self.main_window),
            "rot90": QtWidgets.QAction("Rotate 90-Left", self.main_window),
            "inverse": QtWidgets.QAction("Inverse Z", self.main_window),
            "zero": QtWidgets.QAction("Set zero point", self.main_window),
            "tilt": QtWidgets.QAction("Set tilt", self.main_window),
            "colormap": QtWidgets.QAction("Toggle colormap", self.main_window),
            "view3d": QtWidgets.QAction("View 3d...", self.main_window),
            "view3d_legacy": QtWidgets.QAction("Legacy 3d view...", self.main_window),
            "compare": QtWidgets.QAction("Scan positioning...", self.main_window),
            "profile": QtWidgets.QAction("Profile analysis...", self.main_window),
            "scan_info": QtWidgets.QAction("Scan info...", self.main_window),
            "about": QtWidgets.QAction("About...", self.main_window),
            "exit": QtWidgets.QAction("Exit", self.main_window),
            # Advanced processing actions
            "filter": QtWidgets.QAction("Advanced Filtering...", self.main_window),
            "morphology": QtWidgets.QAction("Morphology && Leveling...", self.main_window),
            "transform": QtWidgets.QAction("Geometric Transforms...", self.main_window),
            "register": QtWidgets.QAction("Auto-Register Surfaces...", self.main_window),
            "roughness": QtWidgets.QAction("Surface roughness summary...", self.main_window),
            # ROI actions
            "del_outside": QtWidgets.QAction("outside of the mask", self.main_window),
            "del_inside": QtWidgets.QAction("inside of the mask", self.main_window),
            "roi_settings": QtWidgets.QAction("ROI settings...", self.main_window),
        }

        # Set icons
        self.actions["open"].setIcon(QIcon(resource_path("icons/icons8-open-file1-50.png")))
        self.actions["save_scan"].setIcon(QIcon(resource_path("icons/icons8-save1-50.png")))
        self.actions["save_multi"].setIcon(QIcon(resource_path("icons/icons8-save2-50.png")))
        self.actions["repair"].setIcon(QIcon(resource_path("icons/icons8-job-50.png")))
        self.actions["flipUD"].setIcon(QIcon(resource_path("icons/flipUD.png")))
        self.actions["flipLR"].setIcon(QIcon(resource_path("icons/flipLR.png")))
        self.actions["rot90"].setIcon(QIcon(resource_path("icons/icons8-rotate-left-50.png")))
        self.actions["inverse"].setIcon(QIcon(resource_path("icons/icons8-invert-50.png")))
        self.actions["zero"].setIcon(QIcon(resource_path("icons/icons8-eyedropper-50.png")))
        self.actions["tilt"].setIcon(QIcon(resource_path("icons/icons8-tilt-64.png")))
        self.actions["colormap"].setIcon(QIcon(resource_path("icons/icons8-color-palette-50.png")))
        self.actions["view3d"].setIcon(QIcon(resource_path("icons/icons8-3d-80.png")))
        self.actions["view3d_legacy"].setIcon(QIcon(resource_path("icons/icons8-3d-80.png")))
        self.actions["compare"].setIcon(QIcon(resource_path("icons/icons8-compare-50.png")))
        self.actions["profile"].setIcon(QIcon(resource_path("icons/icons8-graph-50.png")))
        self.actions["about"].setIcon(QIcon(resource_path("icons/icons8-about-50.png")))
        self.actions["exit"].setIcon(QIcon(resource_path("icons/icons8-exit-50.png")))
        self.actions["filter"].setIcon(QIcon(resource_path("icons/icons8-filter-50.png")))
        self.actions["morphology"].setIcon(QIcon(resource_path("icons/icons8-filter2-50.png")))
        self.actions["transform"].setIcon(QIcon(resource_path("icons/icons8-transform-64.png")))

        # Set checkable
        self.actions["colormap"].setCheckable(True)
        self.actions["colormap"].setChecked(False)
        # Set tooltips for advanced processing actions
        self.actions["filter"].setToolTip("Apply advanced filtering (bilateral, median, morphological)")
        self.actions["morphology"].setToolTip("Level surface and remove polynomial forms")
        self.actions["transform"].setToolTip("Rotate, rescale, or crop grid")
        self.actions["register"].setToolTip("Automatically align two surfaces")
        self.actions["roughness"].setToolTip("Show minimal Sa, Sq, and Sz summary for the current scan")
        self.actions["roi_settings"].setToolTip("Choose ROI mode, shape, position, and size")
        self.actions["scan_info"].setToolTip("Show geometry, statistics, and metadata for the active scan")
    
    def connect_actions(self):
        """Connect all actions to their respective handlers."""
        # Get controller references
        file_ctrl = self.main_window.file_controller
        proc_ctrl = self.main_window.processing_controller
        reg_ctrl = self.main_window.registration_controller
        roi_ctrl = self.main_window.roi_controller
        
        # File actions
        self.actions["open"].triggered.connect(file_ctrl.open_file)
        self.actions["save_scan"].triggered.connect(file_ctrl.save_single_scan)
        self.actions["save_multi"].triggered.connect(file_ctrl.save_multiple_scans)
        self.actions["exit"].triggered.connect(self.main_window.close)
        
        # Processing actions
        self.actions["fill"].triggered.connect(proc_ctrl.fill_holes)
        self.actions["repair"].triggered.connect(proc_ctrl.repair_grid)
        self.actions["flipUD"].triggered.connect(proc_ctrl.flipUD_scan)
        self.actions["flipLR"].triggered.connect(proc_ctrl.flipLR_scan)
        self.actions["rot90"].triggered.connect(proc_ctrl.scan_rot90)
        self.actions["inverse"].triggered.connect(proc_ctrl.invert_scan)
        self.actions["filter"].triggered.connect(proc_ctrl.apply_advanced_filter)
        self.actions["morphology"].triggered.connect(proc_ctrl.apply_morphology)
        self.actions["transform"].triggered.connect(proc_ctrl.apply_transform)
        self.actions["roughness"].triggered.connect(proc_ctrl.show_surface_roughness_summary)
        
        # Registration actions
        self.actions["compare"].triggered.connect(reg_ctrl.compare_scans)
        self.actions["profile"].triggered.connect(reg_ctrl.start_profile_analysis)
        self.actions["register"].triggered.connect(reg_ctrl.auto_register_surfaces)
        self.actions["scan_info"].triggered.connect(self.main_window.show_scan_info_dialog)
        
        # ROI actions
        self.actions["del_outside"].triggered.connect(roi_ctrl.del_outside_mask)
        self.actions["del_inside"].triggered.connect(roi_ctrl.del_inside_mask)
        self.actions["roi_settings"].triggered.connect(roi_ctrl.open_roi_settings_dialog)
        
        # View actions
        self.actions["zero"].triggered.connect(self.main_window.set_zero_point_mode)
        self.actions["tilt"].triggered.connect(self.main_window.set_tilt_mode)
        self.actions["colormap"].triggered.connect(self.main_window.toggle_colormap_current_tab)
        self.actions["view3d"].triggered.connect(self.main_window.view3d)
        self.actions["view3d_legacy"].triggered.connect(self.main_window.view3d_legacy)
        self.actions["about"].triggered.connect(self.main_window.show_about_dialog)
    
    def create_menubar(self):
        """Create menubar with organized menu structure."""
        menubar = self.main_window.menuBar()

        menu_structure = [
            ("&File", [
                "open",
                "save_scan",
                "save_multi",
                ("recent_menu", []),
                "separator",
                "exit"
            ]),
            ("&Edit", [
                "roi_settings",
                ("delete", [
                    "del_outside",
                    "del_inside"
                ])
            ]),
            ("Scan &Actions", [
                "fill", "repair", "flipUD", "flipLR", "rot90", "inverse", "zero"#, "colormap"
            ]),
            ("&Processing", [
                "filter", "morphology", "transform", "roughness", "separator", "register"
            ]),
            ("&Tools", [
                "view3d",
                "compare", "profile", "scan_info"
            ]),
            ("&Help", [
                "about"
            ])
        ]

        # Tworzymy recent_menu przed budowaniem menu
        self.recent_menu = QtWidgets.QMenu("Recent files", self.main_window)
        self.update_recent_files_menu()

        def add_menu_items(menu, items):
            for item in items:
                if item == "separator":
                    menu.addSeparator()
                elif isinstance(item, tuple):
                    submenu_name, subitems = item
                    if submenu_name == "recent_menu":
                        menu.addMenu(self.recent_menu)
                    else:
                        submenu = QtWidgets.QMenu(submenu_name, self.main_window)
                        add_menu_items(submenu, subitems)
                        menu.addMenu(submenu)
                else:
                    menu.addAction(self.actions[item])

        for menu_name, items in menu_structure:
            menu = menubar.addMenu(menu_name)
            add_menu_items(menu, items)
            if menu_name == "&Tools" and self.main_window.is_legacy_3d_viewer_enabled():
                menu.insertAction(
                    self.actions["compare"],
                    self.actions["view3d_legacy"],
                )
    
    def update_recent_files_menu(self):
        """Update recent files menu with current list."""
        self.recent_menu.clear()
        recent_files = self.main_window.file_controller.recent_files
        if not recent_files:
            action = QtWidgets.QAction("No recent files", self.main_window)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        for path in recent_files:
            action = QtWidgets.QAction(path, self.main_window)
            action.triggered.connect(lambda checked, p=path: self.main_window.file_controller.open_file_from_recent(p))
            self.recent_menu.addAction(action)
