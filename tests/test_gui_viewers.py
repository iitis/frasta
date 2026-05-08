"""Tests for the active 3D-viewer colormap manager."""

from unittest.mock import Mock

import numpy as np
import pytest

from frasta.gui.viewers.surface_3d_viewer import ColormapManager


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
