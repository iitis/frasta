import pytest
import numpy as np
from PyQt5 import QtWidgets
from src.grid3DViewer import Grid3DViewer
from src.lodSurface import LODSurface

@pytest.fixture
def viewer():
    return Grid3DViewer()

def test_init_controls_creates_controls(viewer):
    assert hasattr(viewer, 'controls_panel')
    assert hasattr(viewer, 'checkbox_ref')
    assert hasattr(viewer, 'checkbox_adj')
    assert hasattr(viewer, 'checkbox_line')
    assert hasattr(viewer, 'checkbox_plane')

def test_compute_auto_lo_hi_returns_percentiles(viewer):
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    lo, hi = viewer._compute_auto_lo_hi(arr)
    assert lo < hi
    assert isinstance(lo, float)
    assert isinstance(hi, float)

def test_update_range_widgets_ref(viewer):
    viewer._update_range_widgets('ref', 1.0, 2.0, auto=True)
    assert viewer.spin_lo_ref.value() == 1.0
    assert viewer.spin_hi_ref.value() == 2.0
    assert viewer.chk_auto_ref.isChecked()
    assert not viewer.spin_lo_ref.isEnabled()
    assert not viewer.spin_hi_ref.isEnabled()

def test_update_range_widgets_adj(viewer):
    viewer._update_range_widgets('adj', 3.0, 4.0, auto=False)
    assert viewer.spin_lo_adj.value() == 3.0
    assert viewer.spin_hi_adj.value() == 4.0
    assert not viewer.chk_auto_adj.isChecked()
    assert viewer.spin_lo_adj.isEnabled()
    assert viewer.spin_hi_adj.isEnabled()

def test_get_lo_hi_for_auto(viewer):
    arr = np.array([[1, 2], [3, 4]])
    viewer.range_ref_auto = True
    lo, hi = viewer._get_lo_hi_for('ref', arr)
    assert lo < hi

def test_get_lo_hi_for_manual(viewer):
    arr = np.array([[1, 2], [3, 4]])
    viewer.range_ref_auto = False
    viewer.range_ref = (1.5, 3.5)
    lo, hi = viewer._get_lo_hi_for('ref', arr)
    assert lo == 1.5
    assert hi == 3.5

def test_prepare_reference_surface_downsamples_and_masks(viewer):
    arr = np.random.rand(20, 20)
    arr[0, 0] = np.nan
    arr[1, 1] = 1e7
    xs, ys, Z, xs_idx, ys_idx = viewer._prepare_reference_surface(arr, max_points=10)
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert np.isnan(Z[0, 0]) or np.isnan(Z[1, 1])

def test_remove_existing_items_resets_items(viewer):
    viewer.surface_ref_item = object()
    viewer.surface_adj_item = object()
    viewer.ref_profile_line_item = object()
    viewer.adj_profile_line_item = object()
    viewer.cross_plane_item = object()
    viewer._lod['ref'] = None
    viewer._lod['adj'] = None
    viewer.remove_existing_items()
    assert viewer.surface_ref_item is None
    assert viewer.surface_adj_item is None
    assert viewer.ref_profile_line_item is None
    assert viewer.adj_profile_line_item is None
    assert viewer.cross_plane_item is None

def test_update_data_with_adjusted_grid(viewer):
    arr = np.random.rand(10, 10)
    adj = np.random.rand(10, 10)
    viewer.update_data(arr, adj)
    assert viewer._ref_last is not None
    assert viewer._adj_last is not None

def test_prepare_adjusted_surface_with_valid_grid(viewer):
    arr = np.random.rand(10, 10)
    ys_idx = np.arange(0, 10, 2)
    xs_idx = np.arange(0, 10, 2)
    Z_ref = np.random.rand(len(ys_idx), len(xs_idx))
    separation = 1.5
    Z_adj = viewer._prepare_adjusted_surface(arr, ys_idx, xs_idx, separation, Z_ref)
    assert Z_adj.shape == Z_ref.shape
    expected = arr[np.ix_(ys_idx, xs_idx)].astype(np.float32) + separation
    np.testing.assert_allclose(Z_adj, expected, equal_nan=True)

def test_prepare_adjusted_surface_with_nan_and_outliers(viewer):
    arr = np.random.rand(10, 10)
    arr[0, 0] = np.nan
    arr[1, 1] = 1e7
    ys_idx = np.arange(0, 10, 2)
    xs_idx = np.arange(0, 10, 2)
    Z_ref = np.random.rand(len(ys_idx), len(xs_idx))
    Z_adj = viewer._prepare_adjusted_surface(arr, ys_idx, xs_idx, 0, Z_ref)
    assert np.isnan(Z_adj[0, 0]) or np.isnan(Z_adj[1, 1])

def test_prepare_adjusted_surface_with_none_grid(viewer):
    ys_idx = np.arange(0, 5)
    xs_idx = np.arange(0, 5)
    Z_ref = np.random.rand(5, 5)
    Z_adj = viewer._prepare_adjusted_surface(None, ys_idx, xs_idx, 0, Z_ref)
    assert np.all(np.isnan(Z_adj))
    assert Z_adj.shape == Z_ref.shape

def test_prepare_adjusted_surface_with_custom_clip_abs(viewer):
    arr = np.ones((10, 10)) * 100
    arr[2, 2] = 1e5
    ys_idx = np.arange(0, 10, 2)
    xs_idx = np.arange(0, 10, 2)
    Z_ref = np.random.rand(len(ys_idx), len(xs_idx))
    Z_adj = viewer._prepare_adjusted_surface(arr, ys_idx, xs_idx, 0, Z_ref, clip_abs=50)
    assert np.all(np.isnan(Z_adj))

# --- Uzupełnione testy dla metod GUI i mesh ---
def test_set_controls_visible(viewer):
    viewer.set_controls_visible(True)
    assert viewer.show_controls is True
    viewer.set_controls_visible(False)
    assert viewer.show_controls is False

def test_toggle_surface_ref_lodsurface(viewer, mocker):
    lod_mock = mocker.Mock(spec=LODSurface)
    viewer.surface_ref_item = lod_mock
    viewer.toggle_surface_ref(True)
    lod_mock.set_visible.assert_called_with(True)
    viewer.toggle_surface_ref(False)
    lod_mock.set_visible.assert_called_with(False)

def test_toggle_surface_ref_qtobject(viewer, mocker):
    qt_mock = mocker.Mock()
    viewer.surface_ref_item = qt_mock
    viewer.toggle_surface_ref(True)
    qt_mock.setVisible.assert_called_with(True)
    viewer.toggle_surface_ref(False)
    qt_mock.setVisible.assert_called_with(False)    

def test_toggle_surface_adj_lodsurface(viewer, mocker):
    lod_mock = mocker.Mock(spec=LODSurface)
    viewer.surface_adj_item = lod_mock
    viewer.toggle_surface_adj(True)
    lod_mock.set_visible.assert_called_with(True)
    viewer.toggle_surface_adj(False)
    lod_mock.set_visible.assert_called_with(False)

def test_toggle_surface_adj_qtobject(viewer, mocker):
    qt_mock = mocker.Mock()
    viewer.surface_adj_item = qt_mock
    viewer.toggle_surface_adj(True)
    qt_mock.setVisible.assert_called_with(True)
    viewer.toggle_surface_adj(False)
    qt_mock.setVisible.assert_called_with(False)    

def test_toggle_profile_line_and_cross_plane(viewer, mocker):
    item_mock = mocker.Mock()
    viewer.ref_profile_line_item = item_mock
    viewer.adj_profile_line_item = item_mock
    viewer.cross_plane_item = item_mock
    viewer.toggle_profile_line(True)
    item_mock.setVisible.assert_called_with(True)
    viewer.toggle_profile_line(False)
    item_mock.setVisible.assert_called_with(False)
    viewer.toggle_cross_plane(True)
    item_mock.setVisible.assert_called_with(True)
    viewer.toggle_cross_plane(False)
    item_mock.setVisible.assert_called_with(False)

def test_add_cross_section_plane(viewer):
    pts = np.array([[0, 0], [1, 1]])
    z_min, z_max = 0, 10
    mesh = viewer.add_cross_section_plane(pts, z_min, z_max)
    assert mesh is not None

def test_create_verts_grid_and_normals(viewer):
    Z = np.random.rand(3, 3)
    xs = np.arange(3)
    ys = np.arange(3)
    verts_grid = viewer.create_verts_grid(Z, xs, ys, 3, 3)
    assert verts_grid.shape == (4, 4, 3)
    normals = viewer.calculate_normals(verts_grid, Z, 3, 3)
    assert normals.shape == (16, 3)

def test_make_voxel_mesh(viewer):
    Z = np.random.rand(3, 3)
    mesh = viewer.make_voxel_mesh(Z)
    assert mesh is not None

# --- Testy mesh/surface/wireframe ---
def test_place_surface_mesh_mode_creates_mesh_and_hides_lod(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.random.rand(5, 5)
    viewer.view = mocker.Mock()
    viewer.make_voxel_mesh = mocker.Mock(return_value="mesh_item")
    viewer._lod = {'ref': mocker.Mock(), 'adj': mocker.Mock()}
    viewer._place_surface('surface_ref_item', xs, ys, Z, 'mesh', (1, 0, 0, 1), None)
    viewer.make_voxel_mesh.assert_called_once_with(Z, xs=xs, ys=ys, color=(1, 0, 0, 1))
    viewer.view.addItem.assert_called_once_with("mesh_item")
    viewer._lod['ref'].set_visible.assert_called_once_with(False)
    assert getattr(viewer, 'surface_ref_item') == "mesh_item"

def test_place_surface_mesh_mode_with_colormap(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.random.rand(5, 5)
    viewer.view = mocker.Mock()
    viewer.make_voxel_mesh = mocker.Mock(return_value="mesh_item")
    viewer._lod = {'ref': mocker.Mock(), 'adj': mocker.Mock()}
    viewer._place_surface('surface_ref_item', xs, ys, Z, 'mesh', (1, 0, 0, 1), 'viridis')
    viewer.make_voxel_mesh.assert_called_once_with(Z, xs=xs, ys=ys, colormap='viridis')
    viewer.view.addItem.assert_called_once_with("mesh_item")
    viewer._lod['ref'].set_visible.assert_called_once_with(False)
    assert getattr(viewer, 'surface_ref_item') == "mesh_item"

def test_place_surface_surface_mode_sets_lod_and_updates_style(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.random.rand(5, 5)
    lod_mock = mocker.Mock()
    viewer._ensure_lod = mocker.Mock(return_value=lod_mock)
    viewer._get_lo_hi_for = mocker.Mock(return_value=(0.0, 1.0))
    viewer._lod = {'ref': lod_mock, 'adj': mocker.Mock()}
    viewer._place_surface('surface_ref_item', xs, ys, Z, 'surface', (1, 0, 0, 1), 'viridis')
    lod_mock.set_visible.assert_called_once_with(True)
    lod_mock.set_data.assert_called_once_with(xs, ys, Z)
    lod_mock.update_style.assert_called_once_with(mode='surface', colormap='viridis', base_color=(1, 0, 0, 1), lo=0.0, hi=1.0)
    assert getattr(viewer, 'surface_ref_item') == lod_mock

def test_place_surface_returns_if_shape_invalid(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.random.rand(4, 4)  # Invalid shape
    viewer._lod = {'ref': mocker.Mock(), 'adj': mocker.Mock()}
    result = viewer._place_surface('surface_ref_item', xs, ys, Z, 'surface', (1, 0, 0, 1), None)
    assert result is None

def test_place_surface_returns_if_all_nan(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.full((5, 5), np.nan)
    viewer._lod = {'ref': mocker.Mock(), 'adj': mocker.Mock()}
    result = viewer._place_surface('surface_ref_item', xs, ys, Z, 'surface', (1, 0, 0, 1), None)
    assert result is None

def test_place_surface_mesh_removes_old_item(viewer, mocker):
    xs = np.arange(5)
    ys = np.arange(5)
    Z = np.random.rand(5, 5)
    viewer.view = mocker.Mock()
    viewer.make_voxel_mesh = mocker.Mock(return_value="mesh_item")
    viewer._lod = {'ref': mocker.Mock(), 'adj': mocker.Mock()}
    old_item = mocker.Mock()
    setattr(viewer, 'surface_ref_item', old_item)
    viewer._place_surface('surface_ref_item', xs, ys, Z, 'mesh', (1, 0, 0, 1), None)
    viewer.view.removeItem.assert_called_once_with(old_item)
    viewer.view.addItem.assert_called_once_with("mesh_item")

