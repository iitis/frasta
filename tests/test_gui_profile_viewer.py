"""Tests for frasta.gui.dialogs.profile_viewer modules.

This module tests the profile viewer components:
- DataManager: Data loading and saving
- VisualizationManager: 3D visualization and statistics
- ROIHandler: Profile line placement and interaction
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPointF
import json

from frasta.gui.dialogs.profile_viewer.data_manager import DataManager
from frasta.gui.dialogs.profile_viewer.visualization_manager import VisualizationManager
from frasta.gui.dialogs.profile_viewer.roi_handler import ROIHandler
from frasta.core import Surface


# ============================================================================
# DataManager Tests
# ============================================================================

class TestDataManager:
    """Test suite for DataManager."""
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock ProfileViewer parent."""
        parent = Mock()
        parent.sigma = 1.0
        parent.reference_grid = None
        parent.adjusted_grid = None
        parent.ref_pixel_um = QPointF(1.0, 1.0)
        parent.adj_pixel_um = QPointF(1.0, 1.0)
        parent.reference_grid_smooth = None
        parent.adjusted_grid_smooth = None
        parent.valid_mask = None
        parent.adjusted_grid_corrected = None
        parent.centralWidget = Mock(return_value=Mock())
        parent.statusBar = Mock(return_value=Mock())
        parent.progress_bar = Mock()
        parent.progress_bar.setVisible = Mock()
        parent.progress_bar.setRange = Mock()
        parent.worker = None
        parent.update_plot = Mock(return_value=(10, 10))  # Return valid shape tuple
        parent.roi_handler = Mock()
        parent.roi_handler.redraw_roi = Mock()
        parent.visualization_manager = Mock()
        parent.visualization_manager.resize_image_view = Mock()
        parent.image_view = Mock()
        mock_view = Mock()
        mock_view.setAspectLocked = Mock()
        mock_view.setLimits = Mock()
        mock_view.setRange = Mock()
        parent.image_view.getView = Mock(return_value=mock_view)
        parent.cc_full = []  # For profile line coordinates
        parent.rr_full = []  # For profile line coordinates
        parent.separation = 0.0
        return parent
    
    @pytest.fixture
    def data_manager(self, mock_parent):
        """Create DataManager instance."""
        return DataManager(mock_parent)
    
    def test_initialization(self, data_manager, mock_parent):
        """Test DataManager initializes correctly."""
        assert data_manager.parent == mock_parent
    
    def test_load_new_data_opens_dialog(self, data_manager):
        """Test load_new_data opens file dialog."""
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=("", "")):
            data_manager.load_new_data()
            
            QtWidgets.QFileDialog.getOpenFileName.assert_called_once()
    
    def test_load_new_data_loads_file(self, data_manager):
        """Test load_new_data triggers loading when file selected."""
        test_file = "/path/to/test.h5"
        
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(test_file, "")):
            with patch.object(data_manager, 'load_data_from_file'):
                data_manager.load_new_data()
                
                data_manager.load_data_from_file.assert_called_once_with(test_file)
    
    def test_load_data_from_file_creates_worker(self, data_manager, mock_parent):
        """Test load_data_from_file creates ProfileWorker."""
        test_file = "/path/to/test.h5"
        
        # Mock all Qt application interactions
        with patch('frasta.gui.dialogs.profile_viewer.data_manager.QtWidgets.QApplication') as mock_app:
            with patch('frasta.gui.dialogs.profile_viewer.data_manager.ProfileWorker') as mock_worker:
                mock_app.setOverrideCursor = Mock()
                mock_worker_instance = Mock()
                mock_worker.return_value = mock_worker_instance
                mock_worker_instance.finished = Mock()
                mock_worker_instance.error = Mock()
                mock_worker_instance.start = Mock()
                
                # Mock parent methods
                mock_parent.centralWidget = Mock(return_value=Mock())
                mock_parent.statusBar = Mock(return_value=Mock())
                mock_parent.progress_bar = Mock()
                mock_parent.sigma = 1.0
                
                data_manager.load_data_from_file(test_file)
                
                # Verify worker was created
                mock_worker.assert_called_once_with(test_file, mock_parent.sigma)
    
    def test_on_worker_error_shows_message(self, data_manager, mock_parent):
        """Test on_worker_error displays error message."""
        error_msg = "Test error message"
        
        with patch.object(QtWidgets.QMessageBox, 'critical'):
            data_manager.on_worker_error(error_msg)
            
            QtWidgets.QMessageBox.critical.assert_called_once()
            mock_parent.progress_bar.setVisible.assert_called_with(False)
    
    def test_on_worker_finished_sets_data(self, data_manager, mock_parent):
        """Test on_worker_finished processes result data."""
        result = {
            'reference_grid': np.ones((10, 10), dtype=float),
            'adjusted_grid': np.ones((10, 10), dtype=float) * 2
        }
        
        with patch.object(data_manager, 'set_data'):
            data_manager.on_worker_finished(result)
            
            data_manager.set_data.assert_called_once()
            mock_parent.progress_bar.setVisible.assert_called_with(False)
    
    def test_set_surfaces_calls_set_data(self, data_manager):
        """Test set_surfaces converts Surface objects to grids."""
        surface1 = Surface(
            height=np.ones((5, 5), dtype=float),
            dx=1.5, dy=2.0, x0=0, y0=0
        )
        surface2 = Surface(
            height=np.ones((5, 5), dtype=float) * 2,
            dx=1.5, dy=2.0, x0=0, y0=0
        )
        
        with patch.object(data_manager, 'set_data'):
            data_manager.set_surfaces(surface1, surface2)
            
            data_manager.set_data.assert_called_once()
            call_args = data_manager.set_data.call_args[0]
            assert np.array_equal(call_args[0], surface1.height)
            assert np.array_equal(call_args[1], surface2.height)
            assert call_args[2] == 1.5
            assert call_args[3] == 2.0
    
    def test_set_data_basic(self, data_manager, mock_parent):
        """Test set_data stores grid data."""
        grid1 = np.ones((10, 10), dtype=float) * 5
        grid2 = np.ones((10, 10), dtype=float) * 10
        
        data_manager.set_data(grid1, grid2, 1.0, 1.0, 1.0, 1.0)
        
        assert np.array_equal(mock_parent.reference_grid, grid1)
        assert np.array_equal(mock_parent.adjusted_grid, grid2)
        assert mock_parent.ref_pixel_um.x() == 1.0
        assert mock_parent.adj_pixel_um.x() == 1.0
    
    def test_set_data_creates_valid_mask(self, data_manager, mock_parent):
        """Test set_data creates valid data mask."""
        grid1 = np.ones((10, 10), dtype=float)
        grid1[3:5, 4:6] = np.nan
        grid2 = np.ones((10, 10), dtype=float)
        grid2[2:4, 5:7] = np.nan
        
        data_manager.set_data(grid1, grid2, 1.0, 1.0, 1.0, 1.0)
        
        # Valid mask should exclude NaN regions
        assert isinstance(mock_parent.valid_mask, np.ndarray)
        assert mock_parent.valid_mask.dtype == bool
        assert not mock_parent.valid_mask[3, 4]  # NaN in grid1
        assert not mock_parent.valid_mask[2, 5]  # NaN in grid2
    
    def test_set_data_handles_no_overlap(self, data_manager, mock_parent):
        """Test set_data handles case with no valid overlapping data."""
        grid1 = np.full((5, 5), np.nan, dtype=float)
        grid2 = np.full((5, 5), np.nan, dtype=float)
        
        with patch.object(QtWidgets.QMessageBox, 'critical'):
            data_manager.set_data(grid1, grid2, 1.0, 1.0, 1.0, 1.0)
            
            # Should show error message
            QtWidgets.QMessageBox.critical.assert_called_once()


# ============================================================================
# VisualizationManager Tests
# ============================================================================

class TestVisualizationManager:
    """Test suite for VisualizationManager."""
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock ProfileViewer parent."""
        parent = Mock()
        parent.reference_grid_smooth = np.random.randn(50, 50) + 100
        parent.adjusted_grid_corrected = np.random.randn(50, 50) + 105
        parent.separation = 5.0
        parent.ref_pixel_um = QPointF(1.0, 1.0)
        parent.adj_pixel_um = QPointF(1.0, 1.0)
        parent.image_view = Mock()
        parent.image_view.getView = Mock()
        parent.rr_full = []  # Empty list instead of None
        parent.cc_full = []  # Empty list instead of None
        parent._preview_win = None  # For preview window handling
        return parent
    
    @pytest.fixture
    def viz_manager(self, mock_parent):
        """Create VisualizationManager instance."""
        return VisualizationManager(mock_parent)
    
    def test_initialization(self, viz_manager, mock_parent):
        """Test VisualizationManager initializes correctly."""
        assert viz_manager.parent == mock_parent
    
    def test_show_3d_view_basic(self, viz_manager, mock_parent):
        """Test show_3d_view opens 3D viewer."""
        mock_viewbox = Mock()
        mock_viewbox.viewRange = Mock(return_value=([0, 50], [0, 50]))
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch('frasta.gui.dialogs.profile_viewer.visualization_manager.show_3d_viewer'):
            viz_manager.show_3d_view()
            
            # Should call show_3d_viewer with grids
            from frasta.gui.dialogs.profile_viewer.visualization_manager import show_3d_viewer
            show_3d_viewer.assert_called_once()
    
    def test_show_3d_view_with_profile_line(self, viz_manager, mock_parent):
        """Test show_3d_view includes profile line points."""
        mock_viewbox = Mock()
        mock_viewbox.viewRange = Mock(return_value=([0, 50], [0, 50]))
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        # Set profile line data
        mock_parent.rr_full = np.array([10, 20, 30])
        mock_parent.cc_full = np.array([10, 20, 30])
        
        with patch('frasta.gui.dialogs.profile_viewer.visualization_manager.show_3d_viewer'):
            viz_manager.show_3d_view()
            
            from frasta.gui.dialogs.profile_viewer.visualization_manager import show_3d_viewer
            call_kwargs = show_3d_viewer.call_args[1]
            assert call_kwargs['line_points'] is not None
    
    def test_show_preview_creates_window(self, viz_manager, mock_parent):
        """Test show_preview creates preview window."""
        fragment = np.random.randn(20, 20)
        
        # Ensure _preview_win is None initially
        mock_parent._preview_win = None
        
        with patch('frasta.gui.dialogs.profile_viewer.visualization_manager.pg') as mock_pg:
            mock_window = Mock()
            mock_pg.ImageView.return_value = mock_window
            
            viz_manager.show_preview(fragment, title="Test Preview")
            
            mock_pg.ImageView.assert_called_once()
            mock_window.setWindowTitle.assert_called_with("Test Preview")
            mock_window.show.assert_called_once()
    
    def test_resize_image_view_landscape(self, viz_manager, mock_parent):
        """Test resize_image_view calculates size for landscape image."""
        shape = (300, 500)  # Width > Height
        
        viz_manager.resize_image_view(shape)
        
        mock_parent.image_view.setFixedSize.assert_called_once()
        call_args = mock_parent.image_view.setFixedSize.call_args[0]
        width, height = call_args
        # Width should be base (500), height proportionally smaller
        assert width == 500
        assert height < width
    
    def test_resize_image_view_portrait(self, viz_manager, mock_parent):
        """Test resize_image_view calculates size for portrait image."""
        shape = (500, 300)  # Height > Width
        
        viz_manager.resize_image_view(shape)
        
        mock_parent.image_view.setFixedSize.assert_called_once()
        call_args = mock_parent.image_view.setFixedSize.call_args[0]
        width, height = call_args
        # Height should be base (500), width proportionally smaller
        assert height == 500
        assert width < height
    
    def test_resize_image_view_square(self, viz_manager, mock_parent):
        """Test resize_image_view handles square images."""
        shape = (400, 400)
        
        viz_manager.resize_image_view(shape)
        
        mock_parent.image_view.setFixedSize.assert_called_once()
        call_args = mock_parent.image_view.setFixedSize.call_args[0]
        width, height = call_args
        # Should be equal for square
        assert width == height
    
    def test_get_viewbox_ranges_int_basic(self, viz_manager, mock_parent):
        """Test get_viewbox_ranges_int returns integer ranges."""
        mock_viewbox = Mock()
        mock_viewbox.viewRange = Mock(return_value=([5.7, 45.2], [10.3, 40.8]))
        mock_viewbox.mapToParent = Mock(side_effect=lambda p: p)
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        x_min, x_max, y_min, y_max = viz_manager.get_viewbox_ranges_int()
        
        assert isinstance(x_min, int)
        assert isinstance(x_max, int)
        assert isinstance(y_min, int)
        assert isinstance(y_max, int)
    
    def test_get_viewbox_ranges_int_with_overflow(self, viz_manager, mock_parent):
        """Test get_viewbox_ranges_int with overflow flag."""
        mock_viewbox = Mock()
        mock_viewbox.viewRange = Mock(return_value=([5.7, 45.2], [10.3, 40.8]))
        mock_viewbox.mapToParent = Mock(side_effect=lambda p: p)
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        x_min, x_max, y_min, y_max = viz_manager.get_viewbox_ranges_int(overflow=True)
        
        # With overflow, should use floor for min, ceil-1 for max
        assert x_min == int(np.floor(5.7))  # = 5
        assert x_max == int(np.ceil(45.2)) - 1  # = 45
        assert y_min == int(np.floor(10.3))  # = 10
        assert y_max == int(np.ceil(40.8)) - 1  # = 40


# ============================================================================
# ROIHandler Tests
# ============================================================================

class TestROIHandler:
    """Test suite for ROIHandler."""
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock ProfileViewer parent."""
        parent = Mock()
        parent.reference_grid_smooth = np.random.randn(100, 100) + 100
        parent.adjusted_grid_corrected = np.random.randn(100, 100) + 105
        parent.image_view = Mock()
        parent.image_view.getView = Mock()
        parent.line_roi = None
        parent.line_drag_active = False
        parent.x1 = 0
        parent.y1 = 0
        parent.x2 = 0
        parent.y2 = 0
        parent.roi_endpoint_markers = []
        parent.roi_endpoint_labels = []
        return parent
    
    @pytest.fixture
    def roi_handler(self, mock_parent):
        """Create ROIHandler instance."""
        return ROIHandler(mock_parent)
    
    def test_initialization(self, roi_handler, mock_parent):
        """Test ROIHandler initializes correctly."""
        assert roi_handler.parent == mock_parent
    
    def test_on_image_click_with_shift(self, roi_handler, mock_parent):
        """Test on_image_click starts ROI placement with Shift key."""
        mock_event = Mock()
        mock_event.modifiers = Mock(return_value=QtCore.Qt.ShiftModifier)
        mock_event.scenePos = Mock(return_value=Mock())
        mock_event.accept = Mock()
        
        mock_viewbox = Mock()
        mock_scene_point = Mock()
        mock_scene_point.x = Mock(return_value=25)
        mock_scene_point.y = Mock(return_value=35)
        mock_viewbox.mapSceneToView = Mock(return_value=mock_scene_point)
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch.object(roi_handler, 'redraw_roi'):
            with patch.object(roi_handler, 'update_profile_from_roi'):
                roi_handler.on_image_click(mock_event)
                
                # Should set coordinates and enter drag mode
                assert mock_parent.x1 == 25
                assert mock_parent.y1 == 35
                assert mock_parent.line_drag_active is True
                mock_event.accept.assert_called_once()
    
    def test_on_image_click_without_shift(self, roi_handler, mock_parent):
        """Test on_image_click passes through without Shift key."""
        mock_event = Mock()
        mock_event.modifiers = Mock(return_value=QtCore.Qt.NoModifier)
        
        mock_viewbox = Mock()
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch('frasta.gui.dialogs.profile_viewer.roi_handler.pg.ViewBox.mousePressEvent') as mock_press:
            roi_handler.on_image_click(mock_event)
            
            # Should pass event to parent ViewBox
            mock_press.assert_called_once()
    
    def test_on_image_mouse_release_ends_drag(self, roi_handler, mock_parent):
        """Test on_image_mouse_release ends drag mode."""
        mock_parent.line_drag_active = True
        mock_event = Mock()
        mock_event.accept = Mock()
        
        roi_handler.on_image_mouse_release(mock_event)
        
        assert mock_parent.line_drag_active is False
        mock_event.accept.assert_called_once()
    
    def test_on_image_mouse_move_during_drag(self, roi_handler, mock_parent):
        """Test on_image_mouse_move updates endpoint during drag."""
        mock_parent.line_drag_active = True
        mock_event = Mock()
        mock_event.scenePos = Mock(return_value=Mock())
        mock_event.accept = Mock()
        
        mock_viewbox = Mock()
        mock_scene_point = Mock()
        mock_scene_point.x = Mock(return_value=50)
        mock_scene_point.y = Mock(return_value=60)
        mock_viewbox.mapSceneToView = Mock(return_value=mock_scene_point)
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch.object(roi_handler, 'redraw_roi'):
            with patch.object(roi_handler, 'update_profile_from_roi'):
                roi_handler.on_image_mouse_move(mock_event)
                
                # Should update second endpoint
                assert mock_parent.x2 == 50
                assert mock_parent.y2 == 60
                mock_event.accept.assert_called_once()
    
    def test_on_image_mouse_move_clips_bounds(self, roi_handler, mock_parent):
        """Test on_image_mouse_move clips coordinates to image bounds."""
        mock_parent.line_drag_active = True
        mock_event = Mock()
        mock_event.scenePos = Mock(return_value=Mock())
        mock_event.accept = Mock()
        
        mock_viewbox = Mock()
        mock_scene_point = Mock()
        mock_scene_point.x = Mock(return_value=200)  # Out of bounds
        mock_scene_point.y = Mock(return_value=200)
        mock_viewbox.mapSceneToView = Mock(return_value=mock_scene_point)
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch.object(roi_handler, 'redraw_roi'):
            with patch.object(roi_handler, 'update_profile_from_roi'):
                roi_handler.on_image_mouse_move(mock_event)
                
                # Should clip to image bounds
                assert mock_parent.x2 <= 99
                assert mock_parent.y2 <= 99
    
    def test_redraw_roi_creates_line(self, roi_handler, mock_parent):
        """Test redraw_roi creates LineROI."""
        mock_parent.x1 = 10
        mock_parent.y1 = 15
        mock_parent.x2 = 50
        mock_parent.y2 = 60
        
        mock_viewbox = Mock()
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch('frasta.gui.dialogs.profile_viewer.roi_handler.pg.LineROI') as mock_line_roi:
            mock_line_instance = Mock()
            mock_line_instance.handles = [{}, {}, {'type': None}]
            mock_line_instance.sigRegionChanged = Mock()
            mock_line_instance.sigRegionChanged.connect = Mock()
            mock_line_instance.setZValue = Mock()
            mock_line_roi.return_value = mock_line_instance
            
            with patch.object(roi_handler, 'update_roi_markers'):
                roi_handler.redraw_roi()
                
                # Should create LineROI with correct endpoints
                mock_line_roi.assert_called_once()
                call_args = mock_line_roi.call_args[0]
                assert call_args[0] == [10, 15]
                assert call_args[1] == [50, 60]
    
    def test_redraw_roi_removes_old_line(self, roi_handler, mock_parent):
        """Test redraw_roi removes existing LineROI before creating new one."""
        old_line = Mock()
        mock_parent.line_roi = old_line
        mock_parent.x1 = 10
        mock_parent.y1 = 15
        mock_parent.x2 = 50
        mock_parent.y2 = 60
        
        mock_viewbox = Mock()
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        with patch('frasta.gui.dialogs.profile_viewer.roi_handler.pg.LineROI'):
            with patch.object(roi_handler, 'update_roi_markers'):
                roi_handler.redraw_roi()
                
                # Should remove old line
                mock_viewbox.removeItem.assert_called()
    
    def test_update_roi_markers_clears_old_markers(self, roi_handler, mock_parent):
        """Test update_roi_markers removes old markers before adding new ones."""
        old_marker1 = Mock()
        old_marker2 = Mock()
        mock_parent.roi_endpoint_markers = [old_marker1, old_marker2]
        
        mock_viewbox = Mock()
        mock_parent.image_view.getView.return_value = mock_viewbox
        
        # Create mock line ROI with handles
        mock_handle0 = Mock()
        mock_handle0.pos = Mock(return_value=Mock(x=lambda: 10, y=lambda: 15))
        mock_handle1 = Mock()
        mock_handle1.pos = Mock(return_value=Mock(x=lambda: 50, y=lambda: 60))
        
        mock_line_roi = Mock()
        mock_line_roi.getHandles = Mock(return_value=[mock_handle0, mock_handle1])
        mock_line_roi.mapToParent = Mock(side_effect=lambda pos: pos)
        mock_parent.line_roi = mock_line_roi
        
        with patch('frasta.gui.dialogs.profile_viewer.roi_handler.pg.ScatterPlotItem'):
            with patch('frasta.gui.dialogs.profile_viewer.roi_handler.pg.TextItem'):
                roi_handler.update_roi_markers()
                
                # Should remove old markers
                assert mock_viewbox.removeItem.call_count >= 2
