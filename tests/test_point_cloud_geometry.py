"""Tests for experimental point cloud geometry helpers."""

import numpy as np

from frasta.gui.viewers.surface_3d_viewer.point_cloud_geometry import (
    build_colormap_lut,
    build_point_geometry_from_grid,
    build_mesh_geometry_from_grid,
    build_point_positions_from_grid,
    build_point_cloud_from_grid,
    compute_progressive_stride_schedule,
    compute_stride_for_point_budget,
    compute_bounds,
)


class TestBuildPointPositionsFromGrid:
    """Test point-position extraction for GPU-side color mapping."""

    def test_returns_positions_without_color_buffer(self):
        """Point extraction should preserve scan orientation and heights."""
        grid = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

        positions = build_point_positions_from_grid(grid, dx=2.0, dy=5.0)

        np.testing.assert_allclose(
            positions,
            np.array(
                [
                    [0.0, -0.0, 1.0],
                    [0.0, -5.0, 3.0],
                    [2.0, -5.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )


class TestBuildPointGeometryFromGrid:
    """Test point-geometry extraction with shading normals."""

    def test_flat_grid_produces_upward_normals(self):
        """A flat sampled grid should emit normals aligned with +Z."""
        grid = np.full((3, 3), 5.0, dtype=np.float32)

        positions, normals = build_point_geometry_from_grid(grid)

        assert positions.shape == normals.shape == (9, 3)
        np.testing.assert_allclose(normals, np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (9, 1)))

    def test_sloped_grid_produces_tilted_normals(self):
        """A monotonic slope should yield non-vertical point normals."""
        grid = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        _positions, normals = build_point_geometry_from_grid(grid, dx=1.0, dy=1.0)

        assert np.all(normals[:, 2] > 0.0)
        assert np.any(np.abs(normals[:, 0]) > 1e-3)


class TestBuildPointCloudFromGrid:
    """Test conversion from regular grids to point-cloud buffers."""

    def test_skips_invalid_cells_and_applies_scan_orientation(self):
        """Valid cells should become points and Y should follow scan orientation."""
        grid = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

        positions, colors = build_point_cloud_from_grid(grid, dx=2.0, dy=5.0, colormap=None)

        assert positions.shape == (3, 3)
        assert colors.shape == (3, 4)
        np.testing.assert_allclose(
            positions,
            np.array(
                [
                    [0.0, -0.0, 1.0],
                    [0.0, -5.0, 3.0],
                    [2.0, -5.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_stride_decimates_regular_grid(self):
        """Stride should decimate the regular grid before point extraction."""
        grid = np.arange(16, dtype=np.float32).reshape(4, 4)

        positions, _ = build_point_cloud_from_grid(grid, stride=2, colormap=None)

        assert len(positions) == 4
        np.testing.assert_allclose(
            positions[:, :2],
            np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, -2.0],
                    [2.0, -2.0],
                ],
                dtype=np.float32,
            ),
        )


class TestBuildMeshGeometryFromGrid:
    """Test mesh extraction for the shaded experimental backend."""

    def test_builds_two_triangles_for_one_valid_quad(self):
        """A fully valid 2x2 grid should produce one quad mesh."""
        grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        positions, normals, indices = build_mesh_geometry_from_grid(grid)

        assert positions.shape == (4, 3)
        assert normals.shape == (4, 3)
        assert indices.shape == (2, 3)

    def test_skips_quads_with_missing_vertices(self):
        """NaN vertices should prevent triangle generation across the gap."""
        grid = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

        positions, normals, indices = build_mesh_geometry_from_grid(grid)

        assert positions.shape == (3, 3)
        assert normals.shape == (3, 3)
        assert indices.shape == (0, 3)


class TestComputeBounds:
    """Test axis-aligned point-cloud bounds calculation."""

    def test_returns_fallback_bounds_for_empty_inputs(self):
        """Empty point sets should use a stable fallback bounding box."""
        mins, maxs = compute_bounds(None, np.empty((0, 3), dtype=np.float32))

        np.testing.assert_allclose(mins, np.array([-1.0, -1.0, -1.0], dtype=np.float32))
        np.testing.assert_allclose(maxs, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def test_combines_multiple_point_sets(self):
        """Bounds should cover all provided point sets."""
        ref = np.array([[0.0, -1.0, 2.0], [3.0, -2.0, 5.0]], dtype=np.float32)
        adj = np.array([[-4.0, 7.0, -6.0]], dtype=np.float32)

        mins, maxs = compute_bounds(ref, adj)

        np.testing.assert_allclose(mins, np.array([-4.0, -2.0, -6.0], dtype=np.float32))
        np.testing.assert_allclose(maxs, np.array([3.0, 7.0, 5.0], dtype=np.float32))


class TestComputeProgressiveStrideSchedule:
    """Test progressive stride selection for staged point-cloud loading."""

    def test_small_grid_uses_full_resolution_immediately(self):
        """Small grids should skip progressive decimation."""
        schedule = compute_progressive_stride_schedule((100, 200), cloud_count=1)

        assert schedule == [1]

    def test_large_two_cloud_view_starts_coarse_and_ends_at_full_resolution(self):
        """Large comparison views should begin with a coarser stride."""
        schedule = compute_progressive_stride_schedule((4000, 4000), cloud_count=2)

        assert schedule[0] > 1
        assert schedule[-1] == 1
        assert schedule == sorted(schedule, reverse=True)

    def test_schedule_can_stop_at_configured_min_stride(self):
        """Automatic refinement should respect a configured lower stride bound."""
        schedule = compute_progressive_stride_schedule(
            (4000, 4000),
            cloud_count=2,
            min_stride=4,
        )

        assert schedule[-1] == 4
        assert all(step >= 4 for step in schedule)


class TestComputeStrideForPointBudget:
    """Test hard stride limits derived from 3D geometry budgets."""

    def test_small_grid_keeps_full_resolution(self):
        """Small grids should not be decimated when already under budget."""
        stride = compute_stride_for_point_budget((200, 300), target_points=100_000)

        assert stride == 1

    def test_large_grid_returns_stride_needed_for_budget(self):
        """Huge grids should be clamped to a coarser but bounded stride."""
        stride = compute_stride_for_point_budget((11_000, 11_000), target_points=250_000)

        assert stride >= 22


class TestBuildColormapLut:
    """Test compact LUT creation for GPU color mapping."""

    def test_none_colormap_returns_constant_rgba_table(self):
        """The neutral LUT should have the requested size and constant values."""
        lut = build_colormap_lut(None, size=8)

        assert lut.shape == (8, 4)
        assert lut.dtype == np.uint8
        np.testing.assert_array_equal(lut[0], lut[-1])
