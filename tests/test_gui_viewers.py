"""Tests for frasta.gui.viewers.grid_3d_viewer modules.

This module tests the 3D viewer components:
- LODManager: Level-of-detail management
- ColormapManager: Colormap and range control
- SurfaceRenderer: Surface rendering and geometry
- ProfileManager: Profile lines and cross-sections
- CameraController: Camera positioning
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from PyQt5 import QtCore, QtGui
import pyqtgraph.opengl as gl

from frasta.gui.viewers.grid_3d_viewer import (
    LODManager, ColormapManager, SurfaceRenderer, 
    ProfileManager, CameraController
)
from frasta.gui.viewers.grid_3d_viewer.camera_controller import (
    SCAN_VIEW_AZIMUTH_DEGREES,
    SCAN_VIEW_ELEVATION_DEGREES,
)
from frasta.gui.viewers.lod_surface import LODSurface


# ============================================================================
# LODManager Tests
# ============================================================================

class TestLODManager:
    """Test suite for LODManager."""
    
    @pytest.fixture
    def mock_view(self):
        """Create mock GLViewWidget."""
        view = Mock(spec=gl.GLViewWidget)
        view.addItem = Mock()
        view.removeItem = Mock()
        return view
    
    @pytest.fixture
    def lod_manager(self, mock_view):
        """Create LODManager instance with mocked timer."""
        with patch('frasta.gui.viewers.grid_3d_viewer.lod_manager.QtCore.QTimer'):
            manager = LODManager(mock_view)
            return manager
    
    def test_initialization(self, lod_manager, mock_view):
        """Test LODManager initializes with correct defaults."""
        assert lod_manager.view == mock_view
        assert lod_manager.lod_steps == (1, 2, 4, 8, 16, 32)
        assert lod_manager.lod_target_px == 1.8
        assert lod_manager.lod_hysteresis == 0.3
        assert lod_manager._lod['ref'] is None
        assert lod_manager._lod['adj'] is None
    
    def test_ensure_lod_creates_lod_surface(self, lod_manager):
        """Test ensure_lod creates LODSurface when needed."""
        with patch('frasta.gui.viewers.grid_3d_viewer.lod_manager.LODSurface') as mock_lod:
            mock_lod_instance = Mock()
            mock_lod.return_value = mock_lod_instance
            
            result = lod_manager.ensure_lod('ref')
            
            assert result == mock_lod_instance
            mock_lod.assert_called_once()
            mock_lod_instance.set_lod_params.assert_called_once()
    
    def test_ensure_lod_returns_existing(self, lod_manager):
        """Test ensure_lod returns existing LODSurface."""
        mock_lod = Mock()
        lod_manager._lod['ref'] = mock_lod
        
        result = lod_manager.ensure_lod('ref')
        
        assert result == mock_lod
    
    def test_get_lod_returns_correct_surface(self, lod_manager):
        """Test get_lod returns correct LODSurface."""
        mock_ref = Mock()
        mock_adj = Mock()
        lod_manager._lod['ref'] = mock_ref
        lod_manager._lod['adj'] = mock_adj
        
        assert lod_manager.get_lod('ref') == mock_ref
        assert lod_manager.get_lod('adj') == mock_adj
        # Note: get_lod treats any non-'ref' key as 'adj'
        assert lod_manager.get_lod('other') == mock_adj
    
    def test_set_lod_params_updates_values(self, lod_manager):
        """Test set_lod_params updates parameters."""
        lod_manager.set_lod_params(
            target_px=2.5,
            hysteresis=0.5,
            base_cell=1.0
        )
        
        assert lod_manager.lod_target_px == 2.5
        assert lod_manager.lod_hysteresis == 0.5
        assert lod_manager.lod_base_cell == 1.0
    
    def test_set_lod_params_updates_existing_surfaces(self, lod_manager):
        """Test set_lod_params propagates to existing LOD surfaces."""
        mock_lod_ref = Mock()
        mock_lod_adj = Mock()
        lod_manager._lod['ref'] = mock_lod_ref
        lod_manager._lod['adj'] = mock_lod_adj
        
        lod_manager.set_lod_params(target_px=3.0)
        
        mock_lod_ref.set_lod_params.assert_called_once()
        mock_lod_adj.set_lod_params.assert_called_once()
    
    def test_destroy_lod_removes_surface(self, lod_manager):
        """Test destroy_lod removes LODSurface."""
        mock_lod = Mock()
        lod_manager._lod['ref'] = mock_lod
        
        lod_manager.destroy_lod('ref')
        
        mock_lod.destroy.assert_called_once()
        assert lod_manager._lod['ref'] is None
    
    def test_update_lod_tick_handles_errors(self, lod_manager):
        """Test _update_lod_tick handles exceptions gracefully."""
        mock_lod = Mock()
        mock_lod.update_lod.side_effect = Exception("Test error")
        lod_manager._lod['ref'] = mock_lod
        
        # Should not raise exception
        lod_manager._update_lod_tick()


class TestLODSurface:
    """Test suite for direct LOD surface mesh generation."""

    @pytest.fixture
    def mock_view(self):
        """Create mock GLViewWidget."""
        view = Mock(spec=gl.GLViewWidget)
        view.addItem = Mock()
        view.removeItem = Mock()
        view.opts = {'distance': 100.0, 'fov': 60.0}
        view.height = Mock(return_value=600)
        return view

    def test_build_item_uses_unlit_shader(self, mock_view):
        """LOD surface rendering should not depend on directional lighting."""
        with patch('frasta.gui.viewers.lod_surface.QtCore.QTimer'):
            lod = LODSurface(mock_view)
        lod.data = (
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        with patch('frasta.gui.viewers.lod_surface.gl.GLMeshItem') as mock_item:
            lod._build_item_for_step(1)

        assert mock_item.call_args.kwargs['shader'] is None


# ============================================================================
# ColormapManager Tests
# ============================================================================

class TestColormapManager:
    """Test suite for ColormapManager."""
    
    @pytest.fixture
    def colormap_manager(self):
        """Create ColormapManager instance."""
        return ColormapManager()
    
    def test_initialization(self, colormap_manager):
        """Test ColormapManager initializes with correct defaults."""
        assert colormap_manager.colormap_ref == 'Metrology'
        assert colormap_manager.colormap_adj == 'Metrology'
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
        """Test compute_auto_lo_hi calculates range from data."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        lo, hi = colormap_manager.compute_auto_lo_hi(Z)
        
        assert lo < hi
        assert lo >= 1.0
        assert hi <= 9.0
    
    def test_compute_auto_lo_hi_with_nan(self, colormap_manager):
        """Test compute_auto_lo_hi handles NaN values."""
        Z = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]], dtype=float)
        
        lo, hi = colormap_manager.compute_auto_lo_hi(Z)
        
        assert not np.isnan(lo)
        assert not np.isnan(hi)
        assert lo < hi
    
    def test_compute_auto_lo_hi_all_nan(self, colormap_manager):
        """Test compute_auto_lo_hi handles all-NaN data."""
        Z = np.full((3, 3), np.nan, dtype=float)
        
        lo, hi = colormap_manager.compute_auto_lo_hi(Z)
        
        assert lo == 0.0
        assert hi == 1.0
    
    def test_compute_auto_lo_hi_constant_data(self, colormap_manager):
        """Test compute_auto_lo_hi handles constant values."""
        Z = np.full((3, 3), 5.0, dtype=float)
        
        lo, hi = colormap_manager.compute_auto_lo_hi(Z)
        
        # Should create some range around the value
        assert lo < 5.0
        assert hi > 5.0
        assert hi > lo
    
    def test_get_lo_hi_for_ref_auto_mode(self, colormap_manager):
        """Test get_lo_hi_for returns auto-calculated range for ref."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        colormap_manager.range_ref_auto = True
        
        lo, hi = colormap_manager.get_lo_hi_for('ref', Z)
        
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi
    
    def test_get_lo_hi_for_ref_manual_mode(self, colormap_manager):
        """Test get_lo_hi_for returns manual range for ref."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        colormap_manager.range_ref_auto = False
        colormap_manager.range_ref = (2.0, 8.0)
        
        lo, hi = colormap_manager.get_lo_hi_for('ref', Z)
        
        assert lo == 2.0
        assert hi == 8.0
    
    def test_get_lo_hi_for_adj_linked_mode(self, colormap_manager):
        """Test get_lo_hi_for uses ref range when linked."""
        Z_ref = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        Z_adj = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
        
        colormap_manager.range_linked = True
        colormap_manager.range_ref_auto = False
        colormap_manager.range_ref = (2.0, 8.0)
        
        lo, hi = colormap_manager.get_lo_hi_for('adj', Z_adj)
        
        # Should use ref range when linked
        assert lo == 2.0
        assert hi == 8.0
    
    def test_get_lo_hi_for_adj_not_linked(self, colormap_manager):
        """Test get_lo_hi_for uses adj range when not linked."""
        Z = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
        
        colormap_manager.range_linked = False
        colormap_manager.range_adj_auto = False
        colormap_manager.range_adj = (20.0, 80.0)
        
        lo, hi = colormap_manager.get_lo_hi_for('adj', Z)
        
        assert lo == 20.0
        assert hi == 80.0


# ============================================================================
# SurfaceRenderer Tests
# ============================================================================

class TestSurfaceRenderer:
    """Test suite for SurfaceRenderer."""
    
    @pytest.fixture
    def mock_view(self):
        """Create mock GLViewWidget."""
        view = Mock(spec=gl.GLViewWidget)
        view.addItem = Mock()
        view.removeItem = Mock()
        return view
    
    @pytest.fixture
    def mock_lod_manager(self):
        """Create mock LODManager."""
        manager = Mock()
        manager.ensure_lod = Mock(return_value=Mock())
        manager.get_lod = Mock(return_value=Mock())
        return manager
    
    @pytest.fixture
    def mock_colormap_manager(self):
        """Create mock ColormapManager."""
        manager = Mock()
        manager.get_lo_hi_for = Mock(return_value=(0.0, 10.0))
        return manager
    
    @pytest.fixture
    def surface_renderer(self, mock_view, mock_lod_manager, mock_colormap_manager):
        """Create SurfaceRenderer instance."""
        return SurfaceRenderer(mock_view, mock_lod_manager, mock_colormap_manager)
    
    def test_initialization(self, surface_renderer, mock_view):
        """Test SurfaceRenderer initializes correctly."""
        assert surface_renderer.view == mock_view
        assert surface_renderer.surface_ref_item is None
        assert surface_renderer.surface_adj_item is None
    
    def test_prepare_reference_surface_basic(self, surface_renderer):
        """Test prepare_reference_surface processes grid correctly."""
        grid = np.arange(100, dtype=float).reshape(10, 10)
        
        xs, ys, Z, xs_idx, ys_idx = surface_renderer.prepare_reference_surface(
            grid, max_points=10, dx=1.0, dy=1.0
        )
        
        assert len(xs) > 0
        assert len(ys) > 0
        assert Z.shape[0] == len(ys)
        assert Z.shape[1] == len(xs)
        assert Z.dtype == np.float32

    def test_prepare_reference_surface_preserves_scan_orientation(self, surface_renderer):
        """Test that 3D preparation keeps the input scan column order."""
        grid = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=float)

        xs, ys, Z, _, _ = surface_renderer.prepare_reference_surface(
            grid, max_points=10, dx=2.0, dy=3.0
        )

        np.testing.assert_array_equal(xs, np.array([0.0, 2.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(ys, np.array([0.0, -3.0], dtype=np.float32))
        np.testing.assert_array_equal(Z, grid.astype(np.float32))
    
    def test_prepare_reference_surface_downsamples(self, surface_renderer):
        """Test prepare_reference_surface downsamples large grids."""
        grid = np.arange(10000, dtype=float).reshape(100, 100)
        
        xs, ys, Z, xs_idx, ys_idx = surface_renderer.prepare_reference_surface(
            grid, max_points=20, dx=1.0, dy=1.0
        )
        
        # Should downsample to ~20 points or less
        assert len(xs) <= 20
        assert len(ys) <= 20
    
    def test_prepare_reference_surface_preserves_nan(self, surface_renderer):
        """Test prepare_reference_surface preserves NaN values."""
        grid = np.ones((10, 10), dtype=float)
        grid[3:5, 4:6] = np.nan
        
        xs, ys, Z, xs_idx, ys_idx = surface_renderer.prepare_reference_surface(
            grid, max_points=10, dx=1.0, dy=1.0
        )
        
        # Should have NaN values
        assert np.any(np.isnan(Z))
    
    def test_prepare_reference_surface_clips_outliers(self, surface_renderer):
        """Test prepare_reference_surface clips extreme values."""
        grid = np.ones((10, 10), dtype=float) * 100
        grid[5, 5] = 1e10  # Extreme outlier
        
        xs, ys, Z, xs_idx, ys_idx = surface_renderer.prepare_reference_surface(
            grid, max_points=10, clip_abs=1e6, dx=1.0, dy=1.0
        )
        
        # Outlier should be converted to NaN
        # (need to check if the corresponding position has NaN)
        assert np.any(np.isnan(Z))
    
    def test_prepare_reference_surface_with_custom_origin(self, surface_renderer):
        """Test prepare_reference_surface handles custom origin."""
        grid = np.ones((10, 10), dtype=float)
        
        xs, ys, Z, xs_idx, ys_idx = surface_renderer.prepare_reference_surface(
            grid, max_points=10, dx=2.0, dy=3.0, x0=10.0, y0=20.0
        )
        
        # First coordinates should include origin
        assert xs[0] == 10.0
        assert ys[0] == -20.0
    
    def test_prepare_adjusted_surface(self, surface_renderer):
        """Test prepare_adjusted_surface creates matching surface."""
        grid = np.arange(100, dtype=float).reshape(10, 10)
        ys_idx = np.arange(10)
        xs_idx = np.arange(10)
        Z_ref = grid.copy()
        
        Z_adj = surface_renderer.prepare_adjusted_surface(
            grid, ys_idx, xs_idx, separation=5.0, Z_ref=Z_ref
        )
        
        assert Z_adj.shape == Z_ref.shape
        assert Z_adj.dtype == np.float32
        # Values should be shifted by separation
        valid_mask = ~np.isnan(Z_adj)
        if valid_mask.any():
            assert np.mean(Z_adj[valid_mask]) > np.mean(Z_ref[valid_mask])
    
    def test_prepare_adjusted_surface_none_grid(self, surface_renderer):
        """Test prepare_adjusted_surface handles None input."""
        Z_ref = np.ones((5, 5), dtype=float)
        ys_idx = np.arange(5)
        xs_idx = np.arange(5)
        
        Z_adj = surface_renderer.prepare_adjusted_surface(
            None, ys_idx, xs_idx, separation=5.0, Z_ref=Z_ref
        )
        
        # Should return NaN array of same shape
        assert Z_adj.shape == Z_ref.shape
        assert np.all(np.isnan(Z_adj))

    def test_make_voxel_mesh_uses_unlit_shader(self, surface_renderer):
        """Mesh-mode rendering should not depend on directional lighting."""
        grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        with patch('frasta.gui.viewers.grid_3d_viewer.surface_renderer.gl.GLMeshItem') as mock_item:
            surface_renderer.make_voxel_mesh(grid)

        assert mock_item.call_args.kwargs['shader'] is None


class TestSurfaceTriangulationOrientation:
    """Test suite for 3D mesh winding after scan-oriented axis mapping."""

    def test_lod_surface_faces_point_upward_for_flat_scan(self):
        """LOD triangulation should keep positive Z normals for a flat surface."""
        xs = np.array([0.0, 1.0], dtype=np.float32)
        ys = np.array([0.0, -1.0], dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        vertices = np.c_[X.ravel(), Y.ravel(), np.zeros(4, dtype=np.float32)]

        idx = np.arange(4, dtype=np.uint32).reshape(2, 2)
        face_a = vertices[[idx[0, 0], idx[1, 0], idx[1, 1]]]
        face_b = vertices[[idx[0, 0], idx[1, 1], idx[0, 1]]]

        normal_a = np.cross(face_a[1] - face_a[0], face_a[2] - face_a[0])
        normal_b = np.cross(face_b[1] - face_b[0], face_b[2] - face_b[0])

        assert normal_a[2] > 0.0
        assert normal_b[2] > 0.0

    def test_mesh_surface_faces_point_upward_for_flat_scan(self):
        """Mesh-mode triangulation should keep positive Z normals for a flat surface."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ], dtype=np.float32)

        face_a = verts[[0, 2, 3]]
        face_b = verts[[0, 3, 1]]

        normal_a = np.cross(face_a[1] - face_a[0], face_a[2] - face_a[0])
        normal_b = np.cross(face_b[1] - face_b[0], face_b[2] - face_b[0])

        assert normal_a[2] > 0.0
        assert normal_b[2] > 0.0


# ============================================================================
# ProfileManager Tests
# ============================================================================

class TestProfileManager:
    """Test suite for ProfileManager."""
    
    @pytest.fixture
    def mock_view(self):
        """Create mock GLViewWidget."""
        view = Mock(spec=gl.GLViewWidget)
        view.addItem = Mock()
        view.removeItem = Mock()
        return view
    
    @pytest.fixture
    def profile_manager(self, mock_view):
        """Create ProfileManager instance."""
        return ProfileManager(mock_view)
    
    def test_initialization(self, profile_manager, mock_view):
        """Test ProfileManager initializes correctly."""
        assert profile_manager.view == mock_view
        assert profile_manager.ref_profile_line_item is None
        assert profile_manager.adj_profile_line_item is None
        assert profile_manager.cross_plane_item is None
    
    def test_add_profile_and_plane_basic(self, profile_manager):
        """Test add_profile_and_plane creates visualization items."""
        ref_grid = np.random.randn(50, 50) + 100
        adj_grid = np.random.randn(50, 50) + 105
        line_points = [[10, 10], [40, 40]]
        
        with patch.object(profile_manager, 'add_cross_section_plane', return_value=Mock()):
            with patch.object(profile_manager, 'add_profile_line_segments'):
                profile_manager.add_profile_and_plane(
                    ref_grid, adj_grid, line_points,
                    separation=5.0, z_min=90.0, z_max=120.0,
                    pixel_size_x=1.0, pixel_size_y=1.0
                )
                
                # Should call helper methods
                profile_manager.add_cross_section_plane.assert_called_once()
                assert profile_manager.add_profile_line_segments.call_count == 2

    def test_add_profile_and_plane_uses_scan_oriented_y_axis(self, profile_manager):
        """Test profile geometry uses the same Y direction as the 2D scan view."""
        ref_grid = np.random.randn(20, 20) + 100
        line_points = [[2, 3], [10, 12]]

        with patch.object(profile_manager, 'add_cross_section_plane', return_value=Mock()) as mock_plane:
            with patch.object(profile_manager, 'add_profile_line_segments'):
                profile_manager.add_profile_and_plane(
                    ref_grid, None, line_points,
                    separation=0.0, z_min=90.0, z_max=120.0,
                    pixel_size_x=2.0, pixel_size_y=5.0
                )

        pts = mock_plane.call_args.args[0]
        np.testing.assert_array_equal(pts[:, 0], np.array([4.0, 20.0], dtype=np.float32))
        np.testing.assert_array_equal(pts[:, 1], np.array([-15.0, -60.0], dtype=np.float32))
    
    def test_add_profile_and_plane_out_of_bounds(self, profile_manager):
        """Test add_profile_and_plane handles out-of-bounds points."""
        ref_grid = np.random.randn(10, 10) + 100
        adj_grid = np.random.randn(10, 10) + 105
        line_points = [[100, 100], [200, 200]]  # Out of bounds
        
        with patch.object(profile_manager, 'add_cross_section_plane', return_value=Mock()):
            with patch.object(profile_manager, 'add_profile_line_segments'):
                # Should handle gracefully (no crash)
                profile_manager.add_profile_and_plane(
                    ref_grid, adj_grid, line_points,
                    separation=5.0, z_min=90.0, z_max=120.0,
                    pixel_size_x=1.0, pixel_size_y=1.0
                )
    
    def test_add_profile_and_plane_with_none_adjusted(self, profile_manager):
        """Test add_profile_and_plane works with only reference grid."""
        ref_grid = np.random.randn(50, 50) + 100
        line_points = [[10, 10], [40, 40]]
        
        with patch.object(profile_manager, 'add_cross_section_plane', return_value=Mock()):
            with patch.object(profile_manager, 'add_profile_line_segments'):
                profile_manager.add_profile_and_plane(
                    ref_grid, None, line_points,
                    separation=5.0, z_min=90.0, z_max=120.0,
                    pixel_size_x=1.0, pixel_size_y=1.0
                )
                
                # Should call for ref only
                assert profile_manager.add_profile_line_segments.call_count == 1


# ============================================================================
# CameraController Tests
# ============================================================================

class TestCameraController:
    """Test suite for CameraController."""
    
    @pytest.fixture
    def mock_view(self):
        """Create mock GLViewWidget."""
        view = Mock(spec=gl.GLViewWidget)
        view.setCameraPosition = Mock()
        return view
    
    @pytest.fixture
    def camera_controller(self, mock_view):
        """Create CameraController instance."""
        return CameraController(mock_view)
    
    def test_initialization(self, camera_controller, mock_view):
        """Test CameraController initializes correctly."""
        assert camera_controller.view == mock_view
    
    def test_center_camera_basic(self, camera_controller, mock_view):
        """Test center_camera calculates position correctly."""
        xs = np.array([0, 10, 20])
        ys = np.array([0, 15, 30])
        Z_ref = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        Z_adj = None
        line_points = None
        
        camera_controller.center_camera(
            xs, ys, Z_ref, Z_adj, line_points,
            pixel_size_x=1.0, pixel_size_y=1.0
        )
        
        # Should call setCameraPosition with calculated values
        mock_view.setCameraPosition.assert_called_once()
        call_args = mock_view.setCameraPosition.call_args
        assert 'pos' in call_args[1]
        assert 'distance' in call_args[1]
        assert call_args[1]['azimuth'] == SCAN_VIEW_AZIMUTH_DEGREES
        assert call_args[1]['elevation'] == SCAN_VIEW_ELEVATION_DEGREES
    
    def test_center_camera_with_adjusted(self, camera_controller, mock_view):
        """Test center_camera includes adjusted grid in calculation."""
        xs = np.array([0, 10, 20])
        ys = np.array([0, 15, 30])
        Z_ref = np.ones((3, 3), dtype=float) * 5.0
        Z_adj = np.ones((3, 3), dtype=float) * 15.0
        line_points = None
        
        camera_controller.center_camera(
            xs, ys, Z_ref, Z_adj, line_points,
            pixel_size_x=1.0, pixel_size_y=1.0
        )
        
        mock_view.setCameraPosition.assert_called_once()
    
    def test_center_camera_with_line_points(self, camera_controller, mock_view):
        """Test center_camera includes profile line in calculation."""
        xs = np.array([0, 10, 20])
        ys = np.array([0, 15, 30])
        Z_ref = np.ones((3, 3), dtype=float) * 5.0
        Z_adj = None
        line_points = [[5, 5], [15, 15]]
        
        camera_controller.center_camera(
            xs, ys, Z_ref, Z_adj, line_points,
            pixel_size_x=1.0, pixel_size_y=1.0
        )
        
        mock_view.setCameraPosition.assert_called_once()
    
    def test_center_camera_calculates_distance(self, camera_controller, mock_view):
        """Test center_camera calculates appropriate distance."""
        xs = np.array([0, 100])
        ys = np.array([0, 100])
        Z_ref = np.array([[0, 0], [0, 0]], dtype=float)
        Z_adj = None
        line_points = None
        
        camera_controller.center_camera(
            xs, ys, Z_ref, Z_adj, line_points,
            pixel_size_x=1.0, pixel_size_y=1.0
        )
        
        # Distance should be proportional to scene size
        call_args = mock_view.setCameraPosition.call_args
        distance = call_args[1]['distance']
        assert distance > 100  # Should be larger than max dimension
