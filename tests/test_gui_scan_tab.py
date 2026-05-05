"""Tests for frasta.gui.scan_tab modules.

This module tests the scan tab components:
- HistogramManager: Histogram display and threshold controls
- InteractiveHandler: Mouse interaction modes (zero point, tilt, seed points)
- TransformOperations: Geometric transformations (flip, rotate, invert)
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from frasta.gui.scan_tab.histogram_manager import HistogramManager
from frasta.gui.scan_tab.interactive_handler import InteractiveHandler
from frasta.gui.scan_tab.scan_tab import ScanTab
from frasta.gui.scan_tab.transform_operations import TransformOperations
from frasta.gui.widgets import HistogramViewBox


# ============================================================================
# HistogramManager Tests
# ============================================================================

class TestHistogramManager:
    """Test suite for HistogramManager."""
    
    @pytest.fixture
    def mock_hist_widget(self):
        """Create mock PlotWidget for histogram."""
        widget = Mock(spec=pg.PlotWidget)
        widget.clear = Mock()
        widget.plot = Mock(return_value=Mock())
        widget.addItem = Mock()
        widget.setXRange = Mock()
        view_box = Mock()
        view_box.set_data_bounds = Mock()
        view_box.viewRange = Mock(return_value=([10.0, 20.0], [0.0, 1.0]))
        view_box._clamp_x_range = Mock(side_effect=lambda x0, x1: (x0, x1))
        widget.getViewBox = Mock(return_value=view_box)
        return widget
    
    @pytest.fixture
    def mock_update_callback(self):
        """Create mock callback function."""
        return Mock()
    
    @pytest.fixture
    def hist_manager(self, mock_hist_widget, mock_update_callback):
        """Create HistogramManager instance."""
        return HistogramManager(mock_hist_widget, mock_update_callback)
    
    def test_initialization(self, hist_manager, mock_hist_widget):
        """Test HistogramManager initializes correctly."""
        assert hist_manager.hist_widget == mock_hist_widget
        assert hist_manager.hist_plot is None
        assert hist_manager.hist_bars is None
        assert hist_manager.hist_min_line is None
        assert hist_manager.hist_max_line is None
        assert hist_manager._updating_histogram is False
        assert hist_manager.hide_below_range is True
        assert hist_manager.hide_above_range is True
    
    def test_update_histogram_basic(self, hist_manager):
        """Test update_histogram creates histogram display."""
        grid = np.arange(100, dtype=float).reshape(10, 10)
        
        with patch('frasta.gui.scan_tab.histogram_manager.ResponsiveInfiniteLine'):
            hist_manager.update_histogram(grid, colormap_name="Metrology")
            
            # Should clear and create new plot
            hist_manager.hist_widget.clear.assert_called_once()
            hist_manager.hist_widget.plot.assert_called_once()
            assert hist_manager.hist_bars is not None
            assert "pen" in hist_manager.hist_widget.plot.call_args.kwargs
    
    def test_update_histogram_with_nan(self, hist_manager):
        """Test update_histogram handles NaN values."""
        grid = np.ones((10, 10), dtype=float)
        grid[3:5, 4:6] = np.nan
        
        with patch('frasta.gui.scan_tab.histogram_manager.ResponsiveInfiniteLine'):
            hist_manager.update_histogram(grid)
            
            # Should process successfully
            hist_manager.hist_widget.clear.assert_called_once()

    def test_update_histogram_preserves_visible_x_range(self, hist_manager):
        """Histogram redraw should keep the current horizontal zoom window."""
        grid = np.arange(100, dtype=float).reshape(10, 10)

        with patch('frasta.gui.scan_tab.histogram_manager.ResponsiveInfiniteLine'):
            hist_manager.update_histogram(grid, colormap_name="Metrology")
            hist_manager.hist_min_line.value.return_value = 10.0
            hist_manager.hist_max_line.value.return_value = 90.0
            hist_manager.hist_widget.setXRange.reset_mock()
            hist_manager.update_histogram(grid, colormap_name="viridis")

        hist_manager.hist_widget.setXRange.assert_called()
        args = hist_manager.hist_widget.setXRange.call_args.args
        assert args[0] == 10.0
        assert args[1] == 20.0
    
    def test_update_histogram_all_nan(self, hist_manager):
        """Test update_histogram handles all-NaN data."""
        grid = np.full((10, 10), np.nan, dtype=float)
        
        hist_manager.update_histogram(grid)
        
        # Should clear but not create plot
        hist_manager.hist_widget.clear.assert_called_once()
    
    def test_update_histogram_none_grid(self, hist_manager):
        """Test update_histogram handles None input."""
        hist_manager.update_histogram(None)
        
        # Should not crash
        assert hist_manager.hist_plot is None
    
    def test_update_histogram_preserves_thresholds(self, hist_manager):
        """Test update_histogram tries to preserve threshold values."""
        grid1 = np.arange(100, dtype=float).reshape(10, 10)
        grid2 = np.arange(100, 200, dtype=float).reshape(10, 10)
        
        # Create mock threshold lines with values
        mock_min_line = Mock()
        mock_min_line.value = Mock(return_value=20.0)
        mock_max_line = Mock()
        mock_max_line.value = Mock(return_value=80.0)
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        
        with patch('frasta.gui.scan_tab.histogram_manager.ResponsiveInfiniteLine'):
            hist_manager.update_histogram(grid2)
            
            # Should attempt to use old values if they fit new range
            assert hist_manager.hist_widget.clear.called
    
    def test_update_histogram_inverted_data(self, hist_manager):
        """Test update_histogram handles inverted data flag."""
        grid = np.arange(100, dtype=float).reshape(10, 10)
        
        # Create mock lines
        mock_min_line = Mock()
        mock_min_line.value = Mock(return_value=20.0)
        mock_max_line = Mock()
        mock_max_line.value = Mock(return_value=80.0)
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        
        with patch('frasta.gui.scan_tab.histogram_manager.ResponsiveInfiniteLine'):
            hist_manager.update_histogram(grid, was_data_negated=True)
            
            # Should handle inversion in threshold logic
            assert hist_manager.hist_widget.clear.called
    
    def test_get_threshold_range(self, hist_manager):
        """Test get_threshold_range returns current values."""
        mock_min_line = Mock()
        mock_min_line.value = Mock(return_value=10.0)
        mock_max_line = Mock()
        mock_max_line.value = Mock(return_value=90.0)
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        
        vmin, vmax = hist_manager.get_threshold_range()
        
        assert vmin == 10.0
        assert vmax == 90.0
    
    def test_get_threshold_range_swapped(self, hist_manager):
        """Test get_threshold_range returns sorted values."""
        mock_min_line = Mock()
        mock_min_line.value = Mock(return_value=90.0)  # Swapped
        mock_max_line = Mock()
        mock_max_line.value = Mock(return_value=10.0)
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        
        vmin, vmax = hist_manager.get_threshold_range()
        
        # Should return in correct order
        assert vmin == 10.0
        assert vmax == 90.0
    
    def test_get_threshold_range_no_lines(self, hist_manager):
        """Test get_threshold_range returns None when lines don't exist."""
        vmin, vmax = hist_manager.get_threshold_range()
        
        assert vmin is None
        assert vmax is None
    
    def test_set_threshold_values(self, hist_manager):
        """Test set_threshold_values updates line positions."""
        mock_min_line = Mock()
        mock_max_line = Mock()
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        hist_manager._data_min = 0.0
        hist_manager._data_max = 100.0
        
        hist_manager.set_threshold_values(15.0, 85.0)
        
        mock_min_line.setValue.assert_called_once_with(15.0)
        mock_max_line.setValue.assert_called_once_with(85.0)

    def test_set_threshold_values_clamps_to_data_range(self, hist_manager):
        """Threshold updates should stay inside the histogram data range."""
        mock_min_line = Mock()
        mock_max_line = Mock()

        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        hist_manager._data_min = 10.0
        hist_manager._data_max = 90.0

        hist_manager.set_threshold_values(-5.0, 150.0)

        mock_min_line.setValue.assert_called_once_with(10.0)
        mock_max_line.setValue.assert_called_once_with(90.0)

    def test_get_data_range_returns_histogram_bounds(self, hist_manager):
        """Histogram manager should expose current data bounds."""
        hist_manager._data_min = -2.5
        hist_manager._data_max = 7.5

        assert hist_manager.get_data_range() == (-2.5, 7.5)

    def test_set_out_of_range_visibility_updates_flags(self, hist_manager):
        """Histogram manager should track below/above visibility separately."""
        hist_manager.set_out_of_range_visibility(False, True)

        assert hist_manager.hide_below_range is False
        assert hist_manager.hide_above_range is True
    
    def test_on_threshold_changed_blocks_during_update(self, hist_manager, mock_update_callback):
        """Test _on_threshold_changed blocks callbacks during histogram setup."""
        hist_manager._updating_histogram = True
        
        hist_manager._on_threshold_changed(50.0)
        
        # Should not call update callback
        mock_update_callback.assert_not_called()
    
    def test_on_threshold_changed_calls_callback(self, hist_manager, mock_update_callback):
        """Test _on_threshold_changed triggers callback when not blocked."""
        mock_min_line = Mock()
        mock_min_line.value = Mock(return_value=20.0)
        mock_max_line = Mock()
        mock_max_line.value = Mock(return_value=80.0)
        
        hist_manager.hist_min_line = mock_min_line
        hist_manager.hist_max_line = mock_max_line
        hist_manager._updating_histogram = False
        
        hist_manager._on_threshold_changed(50.0)
        
        # Should call update callback with sorted range
        mock_update_callback.assert_called_once_with(20.0, 80.0)


# ============================================================================
# InteractiveHandler Tests
# ============================================================================

class TestInteractiveHandler:
    """Test suite for InteractiveHandler."""
    
    @pytest.fixture
    def mock_parent_tab(self):
        """Create mock ScanTab."""
        tab = Mock()
        tab.grid = np.arange(100, dtype=float).reshape(10, 10)
        tab.image_view = Mock()
        tab.histogram_manager = Mock()
        tab.histogram_manager.get_threshold_range = Mock(return_value=(10.0, 90.0))
        tab.histogram_manager.update_histogram = Mock()
        tab.histogram_manager.set_threshold_values = Mock()
        tab.update_image = Mock()
        tab.update_histogram = Mock()
        tab.physical_to_indices = Mock(side_effect=lambda x, y: (int(x), int(y)))
        tab.indices_to_physical = Mock(side_effect=lambda x, y: (float(x), float(y)))
        return tab
    
    @pytest.fixture
    def interactive_handler(self, mock_parent_tab):
        """Create InteractiveHandler instance."""
        return InteractiveHandler(mock_parent_tab)
    
    def test_initialization(self, interactive_handler, mock_parent_tab):
        """Test InteractiveHandler initializes correctly."""
        assert interactive_handler.parent_tab == mock_parent_tab
        assert interactive_handler.zero_point_mode is False
        assert interactive_handler.tilt_mode is False
        assert interactive_handler.seed_points == []
        assert interactive_handler.zero_window_size == 15
        assert interactive_handler.zero_sigma == 2.0
    
    def test_set_zero_point_mode(self, interactive_handler):
        """Test set_zero_point_mode enables mode."""
        interactive_handler.set_zero_point_mode()
        
        assert interactive_handler.zero_point_mode is True
    
    def test_set_tilt_mode(self, interactive_handler):
        """Test set_tilt_mode enables mode."""
        interactive_handler.set_tilt_mode()
        
        assert interactive_handler.tilt_mode is True
    
    def test_handle_mouse_click_out_of_bounds(self, interactive_handler, mock_parent_tab):
        """Test handle_mouse_click ignores out-of-bounds clicks."""
        mock_event = Mock()
        mock_event.scenePos = Mock(return_value=Mock())
        
        mock_view = Mock()
        mock_view.mapSceneToView = Mock(return_value=Mock(x=lambda: 100, y=lambda: 100))
        mock_parent_tab.image_view.getView = Mock(return_value=mock_view)
        
        # Should not crash with out-of-bounds coordinates
        interactive_handler.handle_mouse_click(mock_event)
    
    def test_handle_zero_point_click_with_nan(self, interactive_handler, mock_parent_tab):
        """Test _handle_zero_point_click shows warning for NaN value."""
        mock_parent_tab.grid[5, 5] = np.nan
        
        with patch.object(QtWidgets.QMessageBox, 'warning'):
            with patch.object(interactive_handler, '_get_zero_point_value', return_value=np.nan):
                interactive_handler._handle_zero_point_click(5, 5)
                
                QtWidgets.QMessageBox.warning.assert_called_once()
                assert interactive_handler.zero_point_mode is False
    
    def test_handle_zero_point_click_valid_value(self, interactive_handler, mock_parent_tab):
        """Test _handle_zero_point_click adjusts grid."""
        original_grid = mock_parent_tab.grid.copy()
        
        with patch.object(interactive_handler, '_get_zero_point_value', return_value=50.0):
            interactive_handler._handle_zero_point_click(5, 5)
            
            # Grid should be shifted by -50.0
            assert not np.array_equal(mock_parent_tab.grid, original_grid)
            mock_parent_tab.update_image.assert_called_once()
            mock_parent_tab.update_histogram.assert_called_once()
            assert interactive_handler.zero_point_mode is False
    
    def test_handle_tilt_click_fits_plane(self, interactive_handler, mock_parent_tab):
        """Test _handle_tilt_click attempts plane fitting."""
        with patch('frasta.gui.scan_tab.interactive_handler.fit_plane_local_median_filter',
                   return_value=(0.1, 0.2, 5.0)):
            interactive_handler._handle_tilt_click(5, 5)
            
            # Should update grid and image
            mock_parent_tab.update_image.assert_called_once()
            mock_parent_tab.update_histogram.assert_called_once()
            assert interactive_handler.tilt_mode is False
    
    def test_handle_tilt_click_handles_error(self, interactive_handler, mock_parent_tab):
        """Test _handle_tilt_click handles fitting errors."""
        with patch('frasta.gui.scan_tab.interactive_handler.fit_plane_local_median_filter',
                   side_effect=ValueError("Fitting failed")):
            with patch.object(QtWidgets.QMessageBox, 'warning'):
                interactive_handler._handle_tilt_click(5, 5)
                
                QtWidgets.QMessageBox.warning.assert_called_once()
    
    def test_handle_seed_point_click_adds_point(self, interactive_handler, mock_parent_tab):
        """Test _handle_seed_point_click adds seed point."""
        mock_view_box = Mock()
        mock_view_box.addItem = Mock()
        
        with patch('frasta.gui.scan_tab.interactive_handler.pg.ScatterPlotItem'):
            interactive_handler._handle_seed_point_click(5, 5, mock_view_box)
            
            assert len(interactive_handler.seed_points) == 1
            assert interactive_handler.seed_points[0] == (5, 5)
            mock_view_box.addItem.assert_called_once()
    
    def test_get_zero_point_value_returns_float(self, interactive_handler, mock_parent_tab):
        """Test _get_zero_point_value calculates robust value."""
        value = interactive_handler._get_zero_point_value(5, 5)
        
        assert isinstance(value, (float, np.floating))
        assert not np.isnan(value)


# ============================================================================
# TransformOperations Tests
# ============================================================================

class TestTransformOperations:
    """Test suite for TransformOperations."""
    
    def test_flip_scan_up_down(self):
        """Test flip_scan with UD direction."""
        grid = np.arange(20, dtype=float).reshape(4, 5)
        
        result = TransformOperations.flip_scan(grid, direction='UD')
        
        assert result.shape == grid.shape
        # First row should become last row
        assert np.array_equal(result[0, :], grid[-1, :])
        assert np.array_equal(result[-1, :], grid[0, :])
    
    def test_flip_scan_left_right(self):
        """Test flip_scan with LR direction."""
        grid = np.arange(20, dtype=float).reshape(4, 5)
        
        result = TransformOperations.flip_scan(grid, direction='LR')
        
        assert result.shape == grid.shape
        # First column should become last column
        assert np.array_equal(result[:, 0], grid[:, -1])
        assert np.array_equal(result[:, -1], grid[:, 0])
    
    def test_flip_scan_none_grid(self):
        """Test flip_scan handles None input."""
        result = TransformOperations.flip_scan(None, direction='UD')
        
        assert result is None
    
    def test_flip_scan_preserves_nan(self):
        """Test flip_scan preserves NaN values."""
        grid = np.ones((4, 5), dtype=float)
        grid[1, 2] = np.nan
        
        result = TransformOperations.flip_scan(grid, direction='UD')
        
        # NaN count should be preserved
        assert np.isnan(result).sum() == np.isnan(grid).sum()
    
    def test_rotate_90_basic(self):
        """Test rotate_90 rotates counter-clockwise."""
        grid = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=float)
        
        result = TransformOperations.rotate_90(grid)
        
        # Shape should swap dimensions
        assert result.shape == (3, 3)
        # Check one corner value
        assert result[0, 0] == 3  # Top-left becomes top-right value
    
    def test_rotate_90_rectangular(self):
        """Test rotate_90 with rectangular grid."""
        grid = np.arange(12, dtype=float).reshape(3, 4)
        
        result = TransformOperations.rotate_90(grid)
        
        # Dimensions should swap
        assert result.shape == (4, 3)
    
    def test_rotate_90_none_grid(self):
        """Test rotate_90 handles None input."""
        result = TransformOperations.rotate_90(None)
        
        assert result is None
    
    def test_rotate_90_preserves_nan(self):
        """Test rotate_90 preserves NaN values."""
        grid = np.ones((4, 5), dtype=float)
        grid[1, 2] = np.nan
        grid[3, 4] = np.nan
        
        result = TransformOperations.rotate_90(grid)
        
        # NaN count should be preserved
        assert np.isnan(result).sum() == 2
    
    def test_invert_z_basic(self):
        """Test invert_z negates values."""
        grid = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=float)
        
        result = TransformOperations.invert_z(grid)
        
        assert result.shape == grid.shape
        assert np.array_equal(result, -grid)
    
    def test_invert_z_with_negative(self):
        """Test invert_z handles negative values."""
        grid = np.array([[-5, 0, 5],
                        [-10, 1, 10]], dtype=float)
        
        result = TransformOperations.invert_z(grid)
        
        assert result[0, 0] == 5
        assert result[0, 1] == 0
        assert result[0, 2] == -5
    
    def test_invert_z_none_grid(self):
        """Test invert_z handles None input."""
        result = TransformOperations.invert_z(None)
        
        assert result is None
    
    def test_invert_z_preserves_nan(self):
        """Test invert_z preserves NaN values."""
        grid = np.array([[1, np.nan, 3],
                        [4, 5, np.nan]], dtype=float)
        
        result = TransformOperations.invert_z(grid)
        
        # NaN should remain NaN
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 2])
        # Other values should be negated
        assert result[0, 0] == -1
        assert result[0, 2] == -3
    
    def test_delete_unmasked_basic(self):
        """Test delete_unmasked sets values outside mask to NaN."""
        grid = np.ones((5, 5), dtype=float)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True  # Keep only center
        
        result = TransformOperations.delete_unmasked(grid, mask)
        
        # Only center should remain
        assert not np.isnan(result[2, 2])
        assert result[2, 2] == 1.0
        # All others should be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[4, 4])
    
    def test_delete_unmasked_all_masked(self):
        """Test delete_unmasked with all True mask."""
        grid = np.ones((3, 3), dtype=float) * 5
        mask = np.ones((3, 3), dtype=bool)
        
        result = TransformOperations.delete_unmasked(grid, mask)
        
        # All values should remain
        assert not np.any(np.isnan(result))
        assert np.all(result == 5.0)
    
    def test_delete_unmasked_no_mask(self):
        """Test delete_unmasked with all False mask."""
        grid = np.ones((3, 3), dtype=float)
        mask = np.zeros((3, 3), dtype=bool)
        
        result = TransformOperations.delete_unmasked(grid, mask)
        
        # All values should be NaN
        assert np.all(np.isnan(result))
    
    def test_delete_unmasked_none_grid(self):
        """Test delete_unmasked handles None input."""
        mask = np.ones((3, 3), dtype=bool)
        
        result = TransformOperations.delete_unmasked(None, mask)
        
        assert result is None
    
    def test_delete_unmasked_preserves_existing_nan(self):
        """Test delete_unmasked preserves existing NaN values."""
        grid = np.ones((3, 3), dtype=float)
        grid[0, 0] = np.nan  # Existing NaN
        mask = np.ones((3, 3), dtype=bool)
        
        result = TransformOperations.delete_unmasked(grid, mask)
        
        # Existing NaN should remain
        assert np.isnan(result[0, 0])
        # Other values should be preserved
        assert not np.isnan(result[1, 1])


class TestScanTabColormap:
    """Test suite for ScanTab colormap selection helpers."""

    @pytest.fixture
    def scan_tab(self, qapp):
        """Create ScanTab widget."""
        tab = ScanTab()
        try:
            yield tab
        finally:
            tab.deleteLater()

    def test_set_colormap_gray_disables_lookup(self, scan_tab):
        """Gray mode should disable the color lookup table."""
        scan_tab.update_image = Mock()
        scan_tab.update_histogram = Mock()

        scan_tab.set_colormap("Gray")

        assert scan_tab.is_colormap is False
        assert scan_tab.current_colormap is None
        assert scan_tab.get_colormap_name() == "Gray"
        scan_tab.update_histogram.assert_called_once()
        scan_tab.update_image.assert_called_once()

    def test_set_colormap_named_enables_lookup(self, scan_tab):
        """Named colormap should enable lookup-table rendering."""
        scan_tab.update_image = Mock()
        scan_tab.update_histogram = Mock()

        scan_tab.set_colormap("Metrology")

        assert scan_tab.is_colormap is True
        assert scan_tab.current_colormap == "metrology"
        assert scan_tab.get_colormap_name() == "Metrology"
        scan_tab.update_histogram.assert_called_once()
        scan_tab.update_image.assert_called_once()

    def test_histogram_uses_custom_view_box(self, scan_tab):
        """Histogram widget should use the wheel-zoom-capable view box."""
        assert isinstance(scan_tab.hist_widget.getViewBox(), HistogramViewBox)

    def test_histogram_view_box_stores_bounds(self, scan_tab):
        """Histogram view box should expose clamped data bounds."""
        view_box = scan_tab.hist_widget.getViewBox()
        view_box.set_data_bounds(-5.0, 10.0)
        assert view_box._data_x_min == -5.0
        assert view_box._data_x_max == 10.0

    def test_histogram_controls_exist(self, scan_tab):
        """Scan tab should expose manual histogram controls."""
        assert isinstance(scan_tab.range_min_spin, QtWidgets.QDoubleSpinBox)
        assert isinstance(scan_tab.range_max_spin, QtWidgets.QDoubleSpinBox)
        assert isinstance(scan_tab.hide_below_range_checkbox, QtWidgets.QCheckBox)
        assert isinstance(scan_tab.hide_above_range_checkbox, QtWidgets.QCheckBox)
        assert scan_tab.hide_below_range_checkbox.isChecked() is True
        assert scan_tab.hide_above_range_checkbox.isChecked() is True

    def test_scan_view_background_is_not_black(self, scan_tab):
        """Background should contrast with clipped low-end grayscale values."""
        color = scan_tab.image_view.ui.graphicsView.backgroundBrush().color()
        assert (color.red(), color.green(), color.blue()) == (34, 34, 34)

    def test_set_surface_syncs_manual_histogram_controls(self, scan_tab):
        """Loading a surface should update the manual threshold controls."""
        data = np.arange(9, dtype=float).reshape(3, 3)
        surface = Mock()
        surface.height = data
        surface.xi = np.array([0.0, 1.0, 2.0], dtype=float)
        surface.yi = np.array([0.0, 1.0, 2.0], dtype=float)
        surface.dx = 1.0
        surface.dy = 1.0
        surface.vmin = 2.0
        surface.vmax = 6.0

        scan_tab.set_surface(surface)

        assert scan_tab.range_min_spin.value() == pytest.approx(2.0)
        assert scan_tab.range_max_spin.value() == pytest.approx(6.0)

    def test_manual_threshold_edit_updates_histogram_manager(self, scan_tab):
        """Changing manual threshold controls should push values to the manager."""
        scan_tab.histogram_manager.set_threshold_values = Mock()
        scan_tab.histogram_manager.get_threshold_range = Mock(return_value=(1.5, 8.5))
        scan_tab._updating_threshold_controls = False

        scan_tab.range_min_spin.setValue(1.5)
        scan_tab.range_max_spin.setValue(8.5)
        scan_tab._on_manual_threshold_changed(0.0)

        assert scan_tab.histogram_manager.set_threshold_values.call_args_list[-1] == call(1.5, 8.5)

    def test_out_of_range_toggles_update_mode(self, scan_tab):
        """Toggles should switch masking separately below and above the range."""
        scan_tab.histogram_manager.set_out_of_range_visibility = Mock()
        scan_tab.update_image = Mock()

        scan_tab.hide_below_range_checkbox.setChecked(False)
        scan_tab.hide_above_range_checkbox.setChecked(True)
        scan_tab._on_out_of_range_visibility_toggled(False)

        assert scan_tab.hide_below_range is False
        assert scan_tab.hide_above_range is True
        scan_tab.histogram_manager.set_out_of_range_visibility.assert_called_with(False, True)
        assert scan_tab.update_image.call_count >= 1

    def test_build_export_image_returns_qimage(self, scan_tab):
        """Export image builder should return a non-empty raster."""
        scan_tab.grid = np.arange(12, dtype=float).reshape(3, 4)
        scan_tab.xi = np.arange(4, dtype=float)
        scan_tab.yi = np.arange(3, dtype=float)
        scan_tab.dx = 1.0
        scan_tab.dy = 1.0
        scan_tab.histogram_manager.get_threshold_range = Mock(return_value=(0.0, 11.0))

        image = scan_tab.build_export_image(source="full", transparent_background=True)

        assert isinstance(image, QtGui.QImage)
        assert image.width() > 0
        assert image.height() > 0

    def test_build_export_colorbar_returns_requested_size(self, scan_tab):
        """Colorbar builder should respect explicit output dimensions."""
        scan_tab.grid = np.linspace(0.0, 1.0, 12, dtype=float).reshape(3, 4)
        scan_tab.xi = np.arange(4, dtype=float)
        scan_tab.yi = np.arange(3, dtype=float)
        scan_tab.dx = 1.0
        scan_tab.dy = 1.0
        scan_tab.histogram_manager.get_threshold_range = Mock(return_value=(0.0, 1.0))
        scan_tab.is_colormap = True
        scan_tab.current_colormap = "metrology"

        image = scan_tab.build_export_colorbar(
            source="full",
            width=180,
            height=640,
            transparent_background=True,
            include_histogram=True,
        )

        assert isinstance(image, QtGui.QImage)
        assert image.width() == 180
        assert image.height() == 640
