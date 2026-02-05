import numpy as np
import pytest
from src.scanTab import ScanTab

@pytest.fixture
def scantab(qapp):
    tab = ScanTab()
    # Większa siatka, by mieć >10 punktów w oknie
    tab.grid = np.arange(100, dtype=float).reshape(10, 10)
    tab.xi = np.arange(10)
    tab.yi = np.arange(10)
    tab.px_x = 1.0
    tab.px_y = 1.0
    return tab

def test_fit_plane_to_grid_returns_tuple(scantab):
    a, b, c = scantab.fit_plane_to_grid(scantab.grid, 5, 5, s=2)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)

def test_fit_plane_to_grid_robust_returns_tuple(scantab):
    a, b, c = scantab.fit_plane_to_grid_robust(scantab.grid, 5, 5, s=2)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)

def test_fit_plane_to_grid_median_filter_returns_tuple(scantab):
    a, b, c = scantab.fit_plane_to_grid_median_filter(scantab.grid, 5, 5, s=2)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)


def test_get_zero_point_value_mean_and_median(scantab):
    # Test mean (brak outlierów)
    val = scantab.get_zero_point_value(1, 1)
    assert isinstance(val, float)
    # Test median fallback (wszystko NaN)
    scantab.grid[:] = np.nan
    val = scantab.get_zero_point_value(1, 1)
    assert np.isnan(val)

def test_delete_unmasked_sets_nan(scantab):
    mask = np.zeros_like(scantab.grid, dtype=bool)
    mask[0, 0] = True
    scantab.delete_unmasked(mask)
    assert np.isnan(scantab.grid[1, 1])
    assert not np.isnan(scantab.grid[0, 0])

def test_grid_to_mesh_vectorized_shape(scantab):
    v, f = scantab.grid_to_mesh_vectorized(scantab.grid)
    assert v.shape[1] == 3
    assert f.shape[1] == 3

def test_flip_scan_changes_grid(scantab):
    original = scantab.grid.copy()
    scantab.flip_scan()
    assert not np.allclose(scantab.grid, original)

# def test_fit_plane_to_grid_returns_tuple(scantab):
#     a, b, c = scantab.fit_plane_to_grid(scantab.grid, 1, 1, s=1)
#     assert isinstance(a, float)
#     assert isinstance(b, float)
#     assert isinstance(c, float)

# def test_fit_plane_to_grid_robust_returns_tuple(scantab):
#     a, b, c = scantab.fit_plane_to_grid_robust(scantab.grid, 1, 1, s=1)
#     assert isinstance(a, float)
#     assert isinstance(b, float)
#     assert isinstance(c, float)

# def test_fit_plane_to_grid_median_filter_returns_tuple(scantab):
#     a, b, c = scantab.fit_plane_to_grid_median_filter(scantab.grid, 1, 1, s=1)
#     assert isinstance(a, float)
#     assert isinstance(b, float)
#     assert isinstance(c, float)

def test_set_data_sets_attributes(scantab):
    grid = np.ones((2, 2))
    xi = np.array([0, 1])
    yi = np.array([0, 1])
    px_x = 1.0
    px_y = 1.0
    scantab.set_data(grid, xi, yi, px_x, px_y)
    assert np.allclose(scantab.grid, grid)
    assert np.allclose(scantab.xi, xi)
    assert np.allclose(scantab.yi, yi)
    assert scantab.px_x == px_x
    assert scantab.px_y == px_y

def test_toggle_colormap_switches_flag(scantab):
    initial = scantab.is_colormap
    scantab.toggle_colormap()
    assert scantab.is_colormap != initial
