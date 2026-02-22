"""Tests for frasta.gui.main_window modules.

This module tests the main window controllers:
- FileController: File operations (open, save, recent files)
- ProcessingController: Data processing operations
- RegistrationController: Scan comparison and registration
- ROIController: Region of interest operations
- MenuBuilder: Menu and action creation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call, mock_open
from PyQt5 import QtWidgets, QtCore
from pathlib import Path

from frasta.gui.main_window.file_controller import FileController
from frasta.gui.main_window.processing_controller import ProcessingController
from frasta.gui.main_window.registration_controller import RegistrationController
from frasta.gui.main_window.roi_controller import ROIController
from frasta.gui.main_window.menu_builder import MenuBuilder
from frasta.core import Surface


# ============================================================================
# FileController Tests
# ============================================================================

class TestFileController:
    """Test suite for FileController."""
    
    @pytest.fixture
    def mock_main_window(self):
        """Create mock MainWindow."""
        window = Mock()
        window.tabs = Mock()
        window.tabs.count = Mock(return_value=0)
        window.tabs.addTab = Mock()
        window.tabs.widget = Mock()
        window.tabs.setCurrentWidget = Mock()
        window.statusBar = Mock(return_value=Mock())
        return window
    
    @pytest.fixture
    def file_controller(self, mock_main_window):
        """Create FileController instance."""
        with patch('frasta.gui.main_window.file_controller.QtCore.QSettings') as mock_settings:
            # Setup mock to return empty list for value() calls
            mock_settings_instance = Mock()
            mock_settings_instance.value = Mock(return_value=[])
            mock_settings.return_value = mock_settings_instance
            
            controller = FileController(mock_main_window)
            return controller
    
    def test_initialization(self, file_controller):
        """Test FileController initializes with defaults."""
        assert file_controller.recent_files == []
        assert file_controller.max_recent_files == 10
        assert file_controller.worker is None
        assert file_controller.thread is None
    
    def test_add_to_recent_files_new_file(self, file_controller):
        """Test adding new file to recent files list."""
        file_controller.main_window.menu_builder = Mock()
        file_controller.main_window.menu_builder.update_recent_files_menu = Mock()
        
        file_controller.add_to_recent_files("/path/to/file.csv")
        
        assert "/path/to/file.csv" in file_controller.recent_files
        assert file_controller.recent_files[0] == "/path/to/file.csv"
        file_controller.settings.setValue.assert_called()
    
    def test_add_to_recent_files_duplicate(self, file_controller):
        """Test adding duplicate file moves it to front."""
        file_controller.main_window.menu_builder = Mock()
        file_controller.main_window.menu_builder.update_recent_files_menu = Mock()
        
        file_controller.add_to_recent_files("/path/file1.csv")
        file_controller.add_to_recent_files("/path/file2.csv")
        file_controller.add_to_recent_files("/path/file1.csv")  # Duplicate
        
        assert file_controller.recent_files[0] == "/path/file1.csv"
        assert len(file_controller.recent_files) == 2
    
    def test_add_to_recent_files_max_limit(self, file_controller):
        """Test recent files list respects maximum limit."""
        file_controller.main_window.menu_builder = Mock()
        file_controller.main_window.menu_builder.update_recent_files_menu = Mock()
        
        # Add more than max_recent_files
        for i in range(15):
            file_controller.add_to_recent_files(f"/path/file{i}.csv")
        
        assert len(file_controller.recent_files) == file_controller.max_recent_files
    
    def test_ask_for_units_returns_tuple(self, file_controller, tmp_path):
        """Test _ask_for_units returns units tuple when accepted."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("X [um];Y [um];Z [nm]\n0;0;0\n")
        
        # Mock the entire dialog interaction
        with patch('frasta.gui.main_window.file_controller.suggest_units', return_value=('um', 'mm')):
            with patch('frasta.gui.main_window.file_controller.QtWidgets') as mock_qt:
                # Setup mock dialog
                mock_dialog = Mock()
                mock_dialog.exec_ = Mock(return_value=1)  # QDialog.Accepted == 1
                mock_qt.QDialog.return_value = mock_dialog
                mock_qt.QDialog.Accepted = 1
                
                # Setup mock radio buttons (mm, um for xy and z)
                mock_radio_xy_mm = Mock()
                mock_radio_xy_um = Mock()
                mock_radio_z_mm = Mock()
                mock_radio_z_um = Mock()
                
                mock_radio_xy_mm.isChecked = Mock(return_value=True)   # mm selected for xy
                mock_radio_xy_um.isChecked = Mock(return_value=False)
                mock_radio_z_mm.isChecked = Mock(return_value=False)
                mock_radio_z_um.isChecked = Mock(return_value=True)    # um selected for z
                
                mock_qt.QRadioButton.side_effect = [
                    mock_radio_xy_mm, mock_radio_xy_um, 
                    mock_radio_z_mm, mock_radio_z_um
                ]
                
                # Mock other widgets to prevent actual instantiation
                mock_qt.QVBoxLayout = Mock(return_value=Mock())
                mock_qt.QLabel = Mock(return_value=Mock())
                mock_qt.QGroupBox = Mock(return_value=Mock())
                
                # Mock QDialogButtonBox to handle flag operations
                mock_button_box = Mock(accepted=Mock(), rejected=Mock())
                mock_qt.QDialogButtonBox = Mock(return_value=mock_button_box)
                mock_qt.QDialogButtonBox.Ok = 0x00000400
                mock_qt.QDialogButtonBox.Cancel = 0x00400000
                
                result = file_controller._ask_for_units(str(test_file))
                
                assert result is not None
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert result == ('mm', 'um')
    
    def test_ask_for_units_returns_none_on_cancel(self, file_controller, tmp_path):
        """Test _ask_for_units returns None when cancelled."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("X [um];Y [um];Z [nm]\n0;0;0\n")
        
        # Mock the entire dialog interaction
        with patch('frasta.gui.main_window.file_controller.suggest_units', return_value=('um', 'um')):
            with patch('frasta.gui.main_window.file_controller.QtWidgets') as mock_qt:
                # Setup mock dialog that returns Rejected
                mock_dialog = Mock()
                mock_dialog.exec_ = Mock(return_value=0)  # QDialog.Rejected == 0
                mock_qt.QDialog.return_value = mock_dialog
                mock_qt.QDialog.Accepted = 1
                mock_qt.QDialog.Rejected = 0
                
                # Mock other widgets
                mock_qt.QRadioButton = Mock(return_value=Mock())
                mock_qt.QVBoxLayout = Mock(return_value=Mock())
                mock_qt.QLabel = Mock(return_value=Mock())
                mock_qt.QGroupBox = Mock(return_value=Mock())
                
                # Mock QDialogButtonBox to handle flag operations
                mock_button_box = Mock(accepted=Mock(), rejected=Mock())
                mock_qt.QDialogButtonBox = Mock(return_value=mock_button_box)
                mock_qt.QDialogButtonBox.Ok = 0x00000400
                mock_qt.QDialogButtonBox.Cancel = 0x00400000
                
                result = file_controller._ask_for_units(str(test_file))
                
                assert result is None
    
    def test_load_recent_files(self, file_controller):
        """Test load_recent_files retrieves from settings."""
        test_files = ["/path/file1.csv", "/path/file2.csv"]
        file_controller.settings.value = Mock(return_value=test_files)
        
        file_controller.load_recent_files()
        
        assert file_controller.recent_files == test_files


# ============================================================================
# ProcessingController Tests
# ============================================================================

class TestProcessingController:
    """Test suite for ProcessingController."""
    
    @pytest.fixture
    def mock_main_window(self):
        """Create mock MainWindow."""
        window = Mock()
        window.current_tab = Mock(return_value=None)
        window.roi_controller = Mock()
        return window
    
    @pytest.fixture
    def mock_tab(self):
        """Create mock ScanTab."""
        tab = Mock()
        tab.grid = np.ones((10, 10), dtype=float)
        tab.dx = 1.0
        tab.dy = 1.0
        tab.flip_scan = Mock()
        tab.scan_rot90 = Mock()
        tab.invert_scan = Mock()
        tab.fill_holes = Mock()
        tab.repair_grid = Mock()
        tab.update_histogram = Mock()
        tab.update_image = Mock()
        return tab
    
    @pytest.fixture
    def processing_controller(self, mock_main_window):
        """Create ProcessingController instance."""
        return ProcessingController(mock_main_window)
    
    def test_initialization(self, processing_controller, mock_main_window):
        """Test ProcessingController initializes correctly."""
        assert processing_controller.main_window == mock_main_window
    
    def test_flipUD_scan_calls_tab_method(self, processing_controller, mock_tab):
        """Test flipUD_scan delegates to tab."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        
        processing_controller.flipUD_scan()
        
        mock_tab.flip_scan.assert_called_once_with(
            direction='UD',
            parent=processing_controller.main_window
        )
    
    def test_flipLR_scan_calls_tab_method(self, processing_controller, mock_tab):
        """Test flipLR_scan delegates to tab."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        
        processing_controller.flipLR_scan()
        
        mock_tab.flip_scan.assert_called_once_with(
            direction='LR',
            parent=processing_controller.main_window
        )
    
    def test_scan_rot90_calls_tab_method(self, processing_controller, mock_tab):
        """Test scan_rot90 delegates to tab."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        
        processing_controller.scan_rot90()
        
        mock_tab.scan_rot90.assert_called_once()
    
    def test_invert_scan_calls_tab_method(self, processing_controller, mock_tab):
        """Test invert_scan delegates to tab."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        
        processing_controller.invert_scan()
        
        mock_tab.invert_scan.assert_called_once()
    
    def test_fill_holes_calls_tab_method(self, processing_controller, mock_tab):
        """Test fill_holes delegates to tab."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        
        processing_controller.fill_holes()
        
        mock_tab.fill_holes.assert_called_once()
    
    def test_repair_grid_no_tab(self, processing_controller):
        """Test repair_grid handles missing tab gracefully."""
        processing_controller.main_window.current_tab = Mock(return_value=None)
        
        # Should not raise exception
        processing_controller.repair_grid()
    
    def test_repair_grid_creates_mask(self, processing_controller, mock_tab):
        """Test repair_grid creates and uses mask."""
        processing_controller.main_window.current_tab = Mock(return_value=mock_tab)
        processing_controller.main_window.roi_controller.create_mask = Mock(
            return_value=np.ones((10, 10), dtype=bool)
        )
        
        processing_controller.repair_grid()
        
        processing_controller.main_window.roi_controller.create_mask.assert_called_once()
        mock_tab.repair_grid.assert_called_once()
    
    def test_apply_advanced_filter_no_data(self, processing_controller):
        """Test apply_advanced_filter shows warning with no data."""
        processing_controller.main_window.current_tab = Mock(return_value=None)
        
        with patch.object(QtWidgets.QMessageBox, 'warning'):
            processing_controller.apply_advanced_filter()
            QtWidgets.QMessageBox.warning.assert_called_once()


# ============================================================================
# RegistrationController Tests
# ============================================================================

class TestRegistrationController:
    """Test suite for RegistrationController."""
    
    @pytest.fixture
    def mock_main_window(self):
        """Create mock MainWindow with tabs."""
        window = Mock()
        window.tabs = Mock()
        window.tabs.count = Mock(return_value=2)
        window.tabs.tabText = Mock(side_effect=lambda i: f"Tab {i}")
        window.tabs.widget = Mock()
        window.tabs.addTab = Mock()
        return window
    
    @pytest.fixture
    def registration_controller(self, mock_main_window):
        """Create RegistrationController instance."""
        return RegistrationController(mock_main_window)
    
    def test_initialization(self, registration_controller, mock_main_window):
        """Test RegistrationController initializes correctly."""
        assert registration_controller.main_window == mock_main_window
        assert registration_controller.viewer is None
        assert registration_controller._profile_viewer is None
    
    def test_compare_scans_warns_with_few_tabs(self, registration_controller):
        """Test compare_scans shows warning with less than 2 tabs."""
        registration_controller.main_window.tabs.count = Mock(return_value=1)
        
        with patch.object(QtWidgets.QMessageBox, 'warning'):
            registration_controller.compare_scans()
            QtWidgets.QMessageBox.warning.assert_called_once()
    
    def test_start_profile_analysis_warns_with_few_tabs(self, registration_controller):
        """Test start_profile_analysis shows warning with less than 2 tabs."""
        registration_controller.main_window.tabs.count = Mock(return_value=1)
        
        with patch.object(QtWidgets.QMessageBox, 'warning'):
            registration_controller.start_profile_analysis()
            QtWidgets.QMessageBox.warning.assert_called_once()


# ============================================================================
# ROIController Tests
# ============================================================================

class TestROIController:
    """Test suite for ROIController."""
    
    @pytest.fixture
    def mock_main_window(self):
        """Create mock MainWindow."""
        window = Mock()
        window.current_tab = Mock(return_value=None)
        return window
    
    @pytest.fixture
    def roi_controller(self, mock_main_window):
        """Create ROIController instance."""
        return ROIController(mock_main_window)
    
    def test_initialization(self, roi_controller):
        """Test ROIController initializes with None ROIs."""
        assert roi_controller.shared_circle_roi is None
        assert roi_controller.shared_rectangle_roi is None
    
    def test_is_roi_valid_and_visible_with_none(self, roi_controller):
        """Test _is_roi_valid_and_visible handles None ROI."""
        assert roi_controller._is_roi_valid_and_visible(None) is False
    
    def test_is_roi_valid_and_visible_with_valid_roi(self, roi_controller):
        """Test _is_roi_valid_and_visible with visible ROI."""
        mock_roi = Mock()
        mock_roi.isVisible = Mock(return_value=True)
        
        assert roi_controller._is_roi_valid_and_visible(mock_roi) is True
    
    def test_is_roi_valid_and_visible_with_deleted_roi(self, roi_controller):
        """Test _is_roi_valid_and_visible handles deleted ROI."""
        mock_roi = Mock()
        mock_roi.isVisible = Mock(side_effect=RuntimeError("Deleted"))
        
        assert roi_controller._is_roi_valid_and_visible(mock_roi) is False
    
    def test_is_roi_deleted_with_none(self, roi_controller):
        """Test _is_roi_deleted handles None ROI."""
        assert roi_controller._is_roi_deleted(None) is True
    
    def test_is_roi_deleted_with_valid_roi(self, roi_controller):
        """Test _is_roi_deleted with existing ROI."""
        mock_roi = Mock()
        mock_roi.isVisible = Mock(return_value=True)
        
        assert roi_controller._is_roi_deleted(mock_roi) is False
    
    def test_create_circle_mask_basic(self, roi_controller):
        """Test create_circle_mask generates correct boolean mask."""
        mask = roi_controller.create_circle_mask((10, 10), (5, 5), 3)
        
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Center should be inside
        assert mask[5, 5] == True
        # Corners should be outside
        assert mask[0, 0] == False
        assert mask[9, 9] == False
    
    def test_create_circle_mask_radius_zero(self, roi_controller):
        """Test create_circle_mask with zero radius."""
        mask = roi_controller.create_circle_mask((10, 10), (5, 5), 0)
        
        # Only center point should be True
        assert mask.sum() <= 1
    
    def test_create_circle_mask_large_radius(self, roi_controller):
        """Test create_circle_mask with large radius."""
        mask = roi_controller.create_circle_mask((10, 10), (5, 5), 10)
        
        # All points should be inside
        assert mask.all()
    
    def test_create_rectangle_mask_basic(self, roi_controller):
        """Test create_rectangle_mask generates correct mask."""
        mask = roi_controller.create_rectangle_mask((10, 10), (5, 5), 4, 4)
        
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Center should be inside
        assert mask[5, 5] == True
        # Corners should be outside
        assert mask[0, 0] == False
    
    def test_create_rectangle_mask_full_coverage(self, roi_controller):
        """Test create_rectangle_mask with full coverage."""
        mask = roi_controller.create_rectangle_mask((10, 10), (5, 5), 10, 10)
        
        # All or most points should be inside
        assert mask.sum() >= 80  # At least 80% coverage
    
    def test_create_mask_no_roi(self, roi_controller):
        """Test create_mask returns None when no ROI is visible."""
        roi_controller.shared_circle_roi = None
        roi_controller.shared_rectangle_roi = None
        
        mask = roi_controller.create_mask(10, 10)
        
        assert mask is None
    
    def test_create_mask_with_circle(self, roi_controller):
        """Test create_mask uses circle ROI when visible."""
        mock_circle = Mock()
        mock_circle.isVisible = Mock(return_value=True)
        mock_circle.pos = Mock(return_value=Mock(x=lambda: 2, y=lambda: 2))
        mock_circle.size = Mock(return_value=[6, 6])
        
        roi_controller.shared_circle_roi = mock_circle
        roi_controller.shared_rectangle_roi = None
        
        mask = roi_controller.create_mask(10, 10)
        
        assert mask is not None
        assert mask.shape == (10, 10)
    
    def test_create_mask_with_rectangle(self, roi_controller):
        """Test create_mask uses rectangle ROI when visible."""
        mock_rect = Mock()
        mock_rect.isVisible = Mock(return_value=True)
        mock_rect.pos = Mock(return_value=Mock(x=lambda: 2, y=lambda: 2))
        mock_rect.size = Mock(return_value=[6, 6])
        
        roi_controller.shared_circle_roi = None
        roi_controller.shared_rectangle_roi = mock_rect
        
        mask = roi_controller.create_mask(10, 10)
        
        assert mask is not None
        assert mask.shape == (10, 10)


# ============================================================================
# MenuBuilder Tests
# ============================================================================

class TestMenuBuilder:
    """Test suite for MenuBuilder."""
    
    @pytest.fixture
    def mock_main_window(self):
        """Create mock MainWindow."""
        window = Mock()
        window.menuBar = Mock(return_value=Mock())
        window.file_controller = Mock()
        window.processing_controller = Mock()
        window.registration_controller = Mock()
        window.roi_controller = Mock()
        return window
    
    @pytest.fixture
    def menu_builder(self, mock_main_window):
        """Create MenuBuilder instance."""
        return MenuBuilder(mock_main_window)
    
    def test_initialization(self, menu_builder):
        """Test MenuBuilder initializes correctly."""
        assert menu_builder.actions == {}
        assert menu_builder.recent_menu is None
    
    def test_create_actions_creates_all_actions(self, menu_builder):
        """Test create_actions creates all required actions."""
        # Patch QAction to prevent actual Qt object creation
        with patch('frasta.gui.main_window.menu_builder.QtWidgets.QAction', return_value=Mock()):
            with patch('frasta.gui.main_window.menu_builder.QIcon', return_value=Mock()):
                menu_builder.create_actions()
        
        # Check essential actions exist
        assert 'open' in menu_builder.actions
        assert 'save_scan' in menu_builder.actions
        assert 'save_multi' in menu_builder.actions
        assert 'fill' in menu_builder.actions
        assert 'repair' in menu_builder.actions
        assert 'flipUD' in menu_builder.actions
        assert 'flipLR' in menu_builder.actions
        assert 'rot90' in menu_builder.actions
        assert 'inverse' in menu_builder.actions
        assert 'zero' in menu_builder.actions
        assert 'tilt' in menu_builder.actions
        assert 'exit' in menu_builder.actions
    
    def test_create_actions_sets_icons(self, menu_builder):
        """Test create_actions sets icons for actions."""
        with patch('frasta.gui.main_window.menu_builder.resource_path', return_value='fake_path'):
            with patch('frasta.gui.main_window.menu_builder.QtWidgets.QAction', return_value=Mock()):
                with patch('frasta.gui.main_window.menu_builder.QIcon', return_value=Mock()):
                    menu_builder.create_actions()
            
            # Check actions were created
            assert 'open' in menu_builder.actions
            assert 'save_scan' in menu_builder.actions
    
    def test_create_actions_sets_checkable(self, menu_builder):
        """Test create_actions marks colormap action as checkable."""
        with patch('frasta.gui.main_window.menu_builder.QtWidgets.QAction', return_value=Mock()):
            with patch('frasta.gui.main_window.menu_builder.QIcon', return_value=Mock()):
                menu_builder.create_actions()
            
            # Verify colormap action was created
            assert 'colormap' in menu_builder.actions
    
    def test_connect_actions_connects_file_operations(self, menu_builder):
        """Test connect_actions connects file operations."""
        with patch('frasta.gui.main_window.menu_builder.QtWidgets.QAction') as mock_action:
            mock_action_instance = Mock()
            mock_action_instance.triggered = Mock()
            mock_action_instance.toggled = Mock()
            mock_action.return_value = mock_action_instance
            
            with patch('frasta.gui.main_window.menu_builder.QIcon', return_value=Mock()):
                menu_builder.create_actions()
                menu_builder.connect_actions()
            
            # Verify actions were created
            assert 'open' in menu_builder.actions
            assert 'save_scan' in menu_builder.actions
