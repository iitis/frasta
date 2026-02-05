import numpy as np
import pytest
from src.profileViewer import ProfileViewer

@pytest.fixture
def profile_viewer(qapp):
    viewer = ProfileViewer()
    # Przygotuj przykładowe dane
    grid1 = np.arange(100, dtype=float).reshape(10, 10)
    grid2 = np.arange(100, 200, dtype=float).reshape(10, 10)
    viewer.set_data(grid1, grid2, 1.0, 1.0, 1.0, 1.0)
    yield viewer
    viewer.close()

def test_set_data_sets_attributes(profile_viewer):
    assert profile_viewer.reference_grid.shape == (10, 10)
    assert profile_viewer.adjusted_grid.shape == (10, 10)
    assert profile_viewer.reference_grid_smooth.shape == (10, 10)
    assert profile_viewer.adjusted_grid_smooth.shape == (10, 10)
    assert profile_viewer.valid_mask.shape == (10, 10)
    assert profile_viewer.adjusted_grid_corrected.shape == (10, 10)

def test_update_plot_returns_shape(profile_viewer):
    shape = profile_viewer.update_plot()
    assert isinstance(shape, tuple)
    assert len(shape) == 2

def test_resize_image_view(profile_viewer):
    profile_viewer.resize_image_view((10, 20))
    # Sprawdź, czy rozmiar widgetu został ustawiony
    w = profile_viewer.image_view.width()
    h = profile_viewer.image_view.height()
    assert w > 0 and h > 0

def test_update_profile_from_roi(profile_viewer):
    # Ustaw ROI na przekątną
    profile_viewer.x1, profile_viewer.y1 = 0, 0
    profile_viewer.x2, profile_viewer.y2 = 9, 9
    profile_viewer.redraw_roi()
    profile_viewer.update_profile_from_roi()
    assert hasattr(profile_viewer, "positions_line")
    assert hasattr(profile_viewer, "reference_profile")
    assert hasattr(profile_viewer, "adjusted_profile")
    assert len(profile_viewer.positions_line) == len(profile_viewer.reference_profile)

def test_fit_profile(profile_viewer):
    x = np.linspace(0, 1, 10)
    y = 2 * x + 1
    slope, angle, reg = profile_viewer._fit_profile(x, y)
    assert np.isclose(slope, 0.002)
    assert hasattr(reg, "coef_")
    assert hasattr(reg, "intercept_")

def test_draw_diff_and_angle_text(profile_viewer):
    # Sprawdź, czy nie rzuca wyjątku
    profile_viewer._draw_diff_and_angle_text(1.0, 45.0, 30.0, 15.0)

def test_clear_fit_lines_and_marker(profile_viewer):
    # Sprawdź, czy nie rzuca wyjątku
    profile_viewer._clear_fit_lines_and_marker()
