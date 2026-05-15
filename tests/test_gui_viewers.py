"""Tests for the active 3D-viewer components."""

from unittest.mock import Mock

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5 import QtGui

from frasta.gui.docks.frasta_profile_dock import FrastaProfileDock
from frasta.gui.docks.frasta_binary_dock import FrastaBinaryDock
from frasta.core import SurfaceOrientation
from frasta.gui.orientation import (
    build_image_rect,
    grid_to_image_data,
    index_to_3d_world,
    indices_to_physical,
    physical_to_indices,
    points_to_3d_world,
)
from frasta.gui.main_window.main_window import MainWindow
from frasta.gui.viewers.surface_3d_viewer import ColormapManager, Point3DViewer
from frasta.gui.viewers.surface_3d_viewer.point_cloud_gl_widget import PointCloudGLWidget


class TestOrientationHelpers:
    """Test the shared GUI orientation adapter."""

    def test_grid_to_image_data_uses_shared_transpose(self):
        """2D image export and viewers should receive one consistent orientation."""
        grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        image = grid_to_image_data(grid, orientation=SurfaceOrientation.DEFAULT)

        np.testing.assert_array_equal(image, np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float))

    def test_physical_and_index_conversions_roundtrip(self):
        """Physical coordinates should round-trip through the shared adapter."""
        x_phys, y_phys = indices_to_physical(
            3, 4, dx=2.0, dy=5.0, x0=10.0, y0=20.0, orientation=SurfaceOrientation.DEFAULT
        )
        col, row = physical_to_indices(
            x_phys, y_phys, dx=2.0, dy=5.0, x0=10.0, y0=20.0, orientation=SurfaceOrientation.DEFAULT
        )

        assert (x_phys, y_phys) == (16.0, 40.0)
        assert (col, row) == (3, 4)

    def test_3d_world_conversion_matches_current_visual_convention(self):
        """3D world coordinates should share one axis convention everywhere."""
        assert index_to_3d_world(2, 3, 7.5, dx=4.0, dy=6.0, orientation=SurfaceOrientation.DEFAULT) == (8.0, -18.0, 7.5)

        points = points_to_3d_world(
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([0.0, 3.0], dtype=np.float32),
            np.array([1.0, 7.5], dtype=np.float32),
            dx=4.0,
            dy=6.0,
            orientation=SurfaceOrientation.DEFAULT,
        )
        np.testing.assert_allclose(points, np.array([[0.0, 0.0, 1.0], [8.0, -18.0, 7.5]], dtype=np.float32))

    def test_build_image_rect_uses_shared_pixel_center_convention(self):
        """Physical image rect should stay centered on pixel centers."""
        rect = build_image_rect((3, 4), dx=2.0, dy=5.0, x0=10.0, y0=20.0, orientation=SurfaceOrientation.DEFAULT)

        assert rect.x() == pytest.approx(9.0)
        assert rect.y() == pytest.approx(17.5)
        assert rect.width() == pytest.approx(8.0)
        assert rect.height() == pytest.approx(15.0)


class TestColormapManager:
    """Test suite for ColormapManager."""

    @pytest.fixture
    def colormap_manager(self):
        """Create ColormapManager instance."""
        return ColormapManager()

    def test_initialization(self, colormap_manager):
        """Test ColormapManager initializes with correct defaults."""
        assert colormap_manager.colormap_ref == "Metrology"
        assert colormap_manager.colormap_adj == "Metrology"
        assert colormap_manager.range_linked is False
        assert colormap_manager.range_ref_auto is True
        assert colormap_manager.range_adj_auto is True
        assert colormap_manager.range_ref == (None, None)
        assert colormap_manager.range_adj == (None, None)

    def test_set_widgets(self, colormap_manager):
        """Test set_widgets stores widget references."""
        mock_widgets = [Mock() for _ in range(7)]

        colormap_manager.set_widgets(*mock_widgets)

        assert colormap_manager.spin_lo_ref == mock_widgets[0]
        assert colormap_manager.spin_hi_ref == mock_widgets[1]
        assert colormap_manager.chk_link == mock_widgets[6]

    def test_set_data_cache(self, colormap_manager):
        """Test set_data_cache stores cached data."""
        ref_data = (np.array([1, 2]), np.array([3, 4]), np.array([[5, 6]]))
        adj_data = (np.array([7, 8]), np.array([9, 10]), np.array([[11, 12]]))

        colormap_manager.set_data_cache(ref_data, adj_data)

        assert colormap_manager._ref_last == ref_data
        assert colormap_manager._adj_last == adj_data

    def test_compute_auto_lo_hi_basic(self, colormap_manager):
        """Test compute_auto_lo_hi calculates range from finite data."""
        values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        lo, hi = colormap_manager.compute_auto_lo_hi(values)

        assert lo < hi
        assert lo >= 1.0
        assert hi <= 9.0

    def test_compute_auto_lo_hi_with_nan(self, colormap_manager):
        """Test compute_auto_lo_hi ignores NaN values."""
        values = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]], dtype=float)

        lo, hi = colormap_manager.compute_auto_lo_hi(values)

        assert not np.isnan(lo)
        assert not np.isnan(hi)
        assert lo < hi

    def test_compute_auto_lo_hi_all_nan(self, colormap_manager):
        """Test compute_auto_lo_hi handles all-NaN data."""
        values = np.full((3, 3), np.nan, dtype=float)

        lo, hi = colormap_manager.compute_auto_lo_hi(values)

        assert lo == 0.0
        assert hi == 1.0

    def test_compute_auto_lo_hi_constant_data(self, colormap_manager):
        """Test compute_auto_lo_hi creates a fallback span for constant data."""
        values = np.full((3, 3), 5.0, dtype=float)

        lo, hi = colormap_manager.compute_auto_lo_hi(values)

        assert lo < 5.0
        assert hi > 5.0

    def test_get_lo_hi_for_ref_manual_mode(self, colormap_manager):
        """Test get_lo_hi_for returns manual ref range when set."""
        values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        colormap_manager.range_ref_auto = False
        colormap_manager.range_ref = (2.0, 8.0)

        lo, hi = colormap_manager.get_lo_hi_for("ref", values)

        assert lo == 2.0
        assert hi == 8.0

    def test_get_lo_hi_for_adj_linked_mode(self, colormap_manager):
        """Test adjusted range follows the reference range when linked."""
        values = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
        colormap_manager.range_linked = True
        colormap_manager.range_ref_auto = False
        colormap_manager.range_ref = (2.0, 8.0)

        lo, hi = colormap_manager.get_lo_hi_for("adj", values)

        assert lo == 2.0
        assert hi == 8.0

    def test_get_lo_hi_for_adj_not_linked(self, colormap_manager):
        """Test adjusted range uses its own manual settings when unlinked."""
        values = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
        colormap_manager.range_linked = False
        colormap_manager.range_adj_auto = False
        colormap_manager.range_adj = (20.0, 80.0)

        lo, hi = colormap_manager.get_lo_hi_for("adj", values)

        assert lo == 20.0
        assert hi == 80.0


class TestPointCloudGLWidget:
    """Test lightweight state changes on the OpenGL widget wrapper."""

    def test_profile_plane_color_roundtrip(self, qapp):
        """Plane color setter should preserve RGBA values."""
        widget = PointCloudGLWidget()

        color = QtGui.QColor.fromRgbF(0.1, 0.2, 0.3, 0.4)
        widget.set_profile_plane_color(color)
        stored = widget.get_profile_plane_color()

        assert stored.isValid()
        assert stored.redF() == pytest.approx(0.1, abs=1e-3)
        assert stored.greenF() == pytest.approx(0.2, abs=1e-3)
        assert stored.blueF() == pytest.approx(0.3, abs=1e-3)
        assert stored.alphaF() == pytest.approx(0.4, abs=1e-3)
        widget.deleteLater()


class TestPoint3DViewer:
    """Test non-OpenGL helper logic on the 3D viewer widget."""

    def test_profile_plane_z_limits_keep_minimum_visual_height(self, qapp):
        """Flat profiles should still produce a visibly tall section plane."""
        viewer = Point3DViewer()
        viewer._ref_grid = np.full((10, 1001), 5.0, dtype=np.float32)
        viewer._pixel_size_x = 1.0
        viewer._pixel_size_y = 1.0

        z_min, z_max = viewer._compute_profile_plane_z_limits(np.array([5.0, 5.0], dtype=np.float32))

        assert (z_max - z_min) == pytest.approx(5.0, abs=1e-6)
        viewer.deleteLater()

    def test_profile_plane_maximum_mode_uses_full_scene_height(self, qapp):
        """Maximum mode should span the global Z range of loaded surfaces."""
        viewer = Point3DViewer()
        viewer._ref_grid = np.array([[0.0, 10.0]], dtype=np.float32)
        viewer._adj_grid = np.array([[20.0, 30.0]], dtype=np.float32)
        viewer._separation = 5.0
        viewer._profile_plane_height_mode = "maximum"

        z_min, z_max = viewer._compute_profile_plane_z_limits(np.array([9.0, 9.0], dtype=np.float32))

        assert z_min == pytest.approx(-1.75, abs=1e-6)
        assert z_max == pytest.approx(36.75, abs=1e-6)
        viewer.deleteLater()

    def test_profile_plane_manual_mode_interpolates_to_full_scene_height(self, qapp):
        """Manual mode should interpolate between dynamic and full-scene heights."""
        viewer = Point3DViewer()
        viewer._ref_grid = np.full((10, 1001), 5.0, dtype=np.float32)
        viewer._adj_grid = np.array([[0.0, 20.0]], dtype=np.float32)
        viewer._profile_plane_height_mode = "manual"
        viewer.slider_plane_height.setValue(100)

        z_min, z_max = viewer._compute_profile_plane_z_limits(np.array([5.0, 5.0], dtype=np.float32))

        assert (z_max - z_min) == pytest.approx(22.0, abs=1e-6)
        assert 5.0 == pytest.approx(0.5 * (z_min + z_max), abs=1e-6)
        viewer.deleteLater()


class TestFrastaProfileDock:
    """Test FRASTA profile-dock view state updates."""

    def test_set_profiles_clears_no_data_title(self, qapp):
        """Loading profiles should remove the placeholder no-data title."""
        dock = FrastaProfileDock()
        dock._plot.setTitle("No data")

        positions = np.array([0.0, 1.0, 2.0], dtype=float)
        profiles = [
            ("Ref", np.array([1.0, 2.0, 3.0], dtype=float), pg.mkPen("g", width=2)),
            ("Adj", np.array([1.5, 2.5, 3.5], dtype=float), pg.mkPen("b", width=2)),
        ]

        dock.set_profiles(positions, profiles, np.array([0.5, 0.5, 0.5], dtype=float))

        assert dock._plot.plotItem.titleLabel.text == ""
        dock.deleteLater()


class TestFrastaBinaryDock:
    """Test FRASTA binary-dock coordinate conventions."""

    def test_binary_dock_uses_same_y_orientation_as_other_2d_views(self, qapp):
        """Binary dock should keep direct row-to-view mapping on the Y axis."""
        dock = FrastaBinaryDock()
        dock.set_data(np.arange(12, dtype=float).reshape(3, 4))

        assert dock._image_view.getView().state["yInverted"] is True
        assert dock._numpy_to_view(0, 0) == (0.0, 0.0)
        assert dock._numpy_to_view(2, 3) == (3.0, 2.0)
        assert dock._view_to_numpy(0.0, 0.0) == (0, 0)
        assert dock._view_to_numpy(3.0, 2.0) == (2, 3)
        dock.deleteLater()


class TestMainWindowHelpers:
    """Test helper logic on the main window without full UI setup."""

    def test_hide_empty_frasta_docks_hides_only_empty_docks(self):
        """Startup helper should hide docks that have no loaded data."""
        empty_binary = Mock()
        empty_binary._diff_map = None
        empty_profile = Mock()
        empty_profile._positions = None
        loaded_binary = Mock()
        loaded_binary._diff_map = np.ones((2, 2), dtype=float)
        loaded_profile = Mock()
        loaded_profile._positions = np.array([0.0, 1.0], dtype=float)

        main_window = Mock()
        main_window.frasta_controller = Mock(
            binary_dock=empty_binary,
            profile_dock=empty_profile,
        )
        MainWindow._hide_empty_frasta_docks(main_window)
        empty_binary.hide.assert_called_once()
        empty_profile.hide.assert_called_once()

        other_window = Mock()
        other_window.frasta_controller = Mock(
            binary_dock=loaded_binary,
            profile_dock=loaded_profile,
        )
        MainWindow._hide_empty_frasta_docks(other_window)
        loaded_binary.hide.assert_not_called()
        loaded_profile.hide.assert_not_called()
