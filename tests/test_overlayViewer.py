import numpy as np
import pytest
from src.overlayViewer import OverlayViewer
from src.gridData import GridData

@pytest.fixture
def overlay_viewer(qapp):
    # Przygotuj przykładowe dane
    grid = np.arange(100, dtype=float).reshape(10, 10)
    xi = np.arange(10)
    yi = np.arange(10)
    px_x = 1.0
    px_y = 1.0
    data1 = GridData(grid=grid, xi=xi, yi=yi, px_x=px_x, px_y=px_y, vmin=None, vmax=None)
    data2 = GridData(grid=grid + 1, xi=xi, yi=yi, px_x=px_x, px_y=px_y, vmin=None, vmax=None)
    viewer = OverlayViewer(data1, data2)
    yield viewer
    viewer.close()
    # QtWidgets.QApplication.processEvents()  # opcjonalnie

def test_overlay_viewer_init(overlay_viewer):
    assert overlay_viewer.scan1.shape == (10, 10)
    assert overlay_viewer.scan2.shape == (10, 10)
    assert overlay_viewer.img1.image.shape == (10, 10)
    assert overlay_viewer.img2.image.shape == (10, 10)

def test_toggle_visibility(overlay_viewer):
    overlay_viewer.checkbox_blink.setChecked(False)
    overlay_viewer.toggleVisibility(0)
    assert not overlay_viewer.img2.isVisible()
    overlay_viewer.toggleVisibility(2)
    assert overlay_viewer.img2.isVisible()

def test_toggle_transparency(overlay_viewer):
    overlay_viewer.toggleTransparency(2)
    assert overlay_viewer.img2.opacity() == 0.5
    overlay_viewer.toggleTransparency(0)
    assert overlay_viewer.img2.opacity() == 1.0

def test_toggle_blinking(overlay_viewer):
    overlay_viewer.toggleBlinking(2)
    assert overlay_viewer.blink_timer.isActive()
    overlay_viewer.toggleBlinking(0)
    assert not overlay_viewer.blink_timer.isActive()

def test_update_transform_changes_label(overlay_viewer):
    overlay_viewer.slider_tx.setValue(10)
    overlay_viewer.slider_ty.setValue(20)
    overlay_viewer.slider_angle.setValue(30)
    overlay_viewer.updateTransform()
    assert "X: 10" in overlay_viewer.label_tx.text()
    assert "Y: 20" in overlay_viewer.label_ty.text()
    assert "Angle: 3.0" in overlay_viewer.label_angle.text()

def test_apply_overlay_mask_sets_nan(overlay_viewer):
    overlay_viewer.vmin1 = 50
    overlay_viewer.vmax1 = 60
    overlay_viewer.vmin2 = 51
    overlay_viewer.vmax2 = 61
    overlay_viewer.apply_overlay_mask()
    masked1 = overlay_viewer.img1.image
    masked2 = overlay_viewer.img2.image
    assert np.isnan(masked1[0, 0])
    assert np.isnan(masked2[0, 0])
