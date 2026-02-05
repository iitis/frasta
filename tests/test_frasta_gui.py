import numpy as np
import pytest
from PyQt5 import QtWidgets
from src.frasta_gui import MainWindow
import types
import sys
# --- Pomocnicze klasy ---
class DummyTab(QtWidgets.QWidget):
    def __init__(self, grid=None):
        super().__init__()
        self.grid = grid if grid is not None else np.ones((5,5))
    def delete_unmasked(self, mask): self.mask = mask
    def set_data(self, *args): self.called = True

class DummyROI:
    def __init__(self, visible=True, x=0, y=0, w=1, h=1):
        self._visible = visible
        self._x = x
        self._y = y
        self._w = w
        self._h = h
    def isVisible(self): return self._visible
    def pos(self):
        class P:
            def __init__(self, x, y):
                self._x = x
                self._y = y
            def x(self): return self._x
            def y(self): return self._y
        return P(self._x, self._y)
    def size(self): return (self._w, self._h)

class DummyView:
    def __init__(self):
        self.items = []
    def addItem(self, item): 
        if item not in self.items: self.items.append(item)
    def removeItem(self, item): 
        if item in self.items: self.items.remove(item)
    def allChildren(self): 
        return list(self.items)

class DummyImageView:
    def __init__(self, view): self._v = view
    def getView(self): return self._v

class DummyScanTab(QtWidgets.QWidget):
    def __init__(self, grid_shape=(100,100)):
        super().__init__()
        self.grid = np.ones(grid_shape)
        self.image_view = DummyImageView(DummyView())

class DummyROIBase:
    def __init__(self, pos, size, **kw):
        self._visible = True
        self._pos = pos
        self._size = size
        self._z = 0
    def isVisible(self): return self._visible
    def setVisible(self, v): self._visible = v
    def show(self): self._visible = True
    def setZValue(self, z): self._z = z
    def pos(self):
        return types.SimpleNamespace(x=lambda: self._pos[0], y=lambda: self._pos[1])
    def size(self): return self._size

# --- Fixture ---
@pytest.fixture
def mainwindow(qapp):
    win = MainWindow()
    yield win
    win.close()
    QtWidgets.QApplication.processEvents()

def test_show_circle_roi_creates_and_toggles(monkeypatch, mainwindow):
    # wstaw dwie zakładki żeby można było przenosić ROI
    t1, t2 = DummyScanTab(), DummyScanTab()
    mainwindow.tabs.addTab(t1, "t1")
    mainwindow.tabs.addTab(t2, "t2")
    mainwindow.tabs.setCurrentIndex(0)

    # podmień pyqtgraph ROI na stub
    class DummyCircle(DummyROIBase): pass
    monkeypatch.setitem(sys.modules, 'pyqtgraph', types.SimpleNamespace(
        CircleROI=DummyCircle, RectROI=None, mkPen=lambda *a, **k: None
    ))

    # pierwszy toggle -> ROI utworzone i widoczne
    mainwindow.show_circle_roi()
    assert mainwindow.shared_circle_roi is not None
    assert mainwindow.shared_circle_roi.isVisible()
    assert mainwindow.shared_circle_roi in t1.image_view.getView().allChildren()

    # drugi toggle -> ukrycie
    mainwindow.show_circle_roi()
    assert mainwindow.shared_circle_roi.isVisible() is False

def test_show_rectangle_hides_circle(monkeypatch, mainwindow):
    t = DummyScanTab()
    mainwindow.tabs.addTab(t, "t")
    mainwindow.tabs.setCurrentIndex(0)

    class DummyCircle(DummyROIBase): pass
    class DummyRect(DummyROIBase): pass
    monkeypatch.setitem(sys.modules, 'pyqtgraph', types.SimpleNamespace(
        CircleROI=DummyCircle, RectROI=DummyRect, mkPen=lambda *a, **k: None
    ))

    mainwindow.show_circle_roi()          # koło widoczne
    assert mainwindow.shared_circle_roi.isVisible()
    mainwindow.show_rectangle_roi()       # prostokąt pokazany, koło ukryte
    assert mainwindow.shared_rectangle_roi.isVisible()
    assert mainwindow.shared_circle_roi.isVisible() is False

def test_move_roi_to_current_tab(monkeypatch, mainwindow):
    t1, t2 = DummyScanTab(), DummyScanTab()
    mainwindow.tabs.addTab(t1, "t1")
    mainwindow.tabs.addTab(t2, "t2")

    class DummyRect(DummyROIBase): pass
    monkeypatch.setitem(sys.modules, 'pyqtgraph', types.SimpleNamespace(
        RectROI=DummyRect, CircleROI=None, mkPen=lambda *a, **k: None
    ))
    # pokaż prostokąt na t1
    mainwindow.tabs.setCurrentIndex(0)
    mainwindow.show_rectangle_roi()
    assert mainwindow.shared_rectangle_roi in t1.image_view.getView().allChildren()

    # przełącz na t2 -> ROI powinno zostać przeniesione
    mainwindow.move_roi_to_current_tab(1)
    assert mainwindow.shared_rectangle_roi in t2.image_view.getView().allChildren()
    assert mainwindow.shared_rectangle_roi not in t1.image_view.getView().allChildren()


# --- Testy maskowania ---
def test_create_circle_mask(mainwindow):
    mask = mainwindow.create_circle_mask((10, 10), (5, 5), 3)
    assert mask.shape == (10, 10)
    assert mask[5, 5]

def test_create_rectangle_mask(mainwindow):
    mask = mainwindow.create_rectangle_mask((10, 10), (5, 5), 4, 4)
    assert mask.shape == (10, 10)
    assert mask[5, 5]

@pytest.mark.parametrize("circle,rect,expected", [
    (DummyROI(True, 2, 3, 4, 4), None, True),
    (None, DummyROI(True, 1, 2, 6, 8), True),
    (DummyROI(False), None, False),
    (None, DummyROI(False), False),
    (None, None, False)
])
def test_create_mask_variants(mainwindow, circle, rect, expected):
    mainwindow.shared_circle_roi = circle
    mainwindow.shared_rectangle_roi = rect
    mask = mainwindow.create_mask(10, 10)
    if expected:
        assert mask.shape == (10, 10)
    else:
        assert mask is None

# --- Testy recent files ---
def test_add_to_recent_files(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.csv")
    with open(test_file, "w") as f:
        f.write("x;y;z\n1;2;3\n")
    mainwindow.add_to_recent_files(test_file)
    assert mainwindow.recent_files[0] == test_file

def test_update_recent_files_menu(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.csv")
    mainwindow.recent_files = [test_file]
    mainwindow.update_recent_files_menu()
    assert mainwindow.recent_menu.actions()[0].text() == test_file

def test_update_recent_files_menu_empty(mainwindow):
    mainwindow.recent_files = []
    mainwindow.update_recent_files_menu()
    actions = mainwindow.recent_menu.actions()
    assert actions[0].text() == "No recent files"
    assert not actions[0].isEnabled()

def test_load_recent_files(mainwindow):
    mainwindow.recent_files = []
    mainwindow.settings.setValue("recentFiles", ["a", "b"])
    mainwindow.load_recent_files()
    assert mainwindow.recent_files == ["a", "b"]

def test_open_file_from_recent_not_exists(mainwindow, monkeypatch, tmp_path):
    test_file = str(tmp_path / "not_exists.csv")
    mainwindow.recent_files = [test_file]
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    mainwindow.open_file_from_recent(test_file)
    assert test_file not in mainwindow.recent_files

# --- Testy zakładek ---
def test_close_tab(mainwindow):
    tab = DummyTab()
    mainwindow.tabs.addTab(tab, "dummy")
    idx = mainwindow.tabs.indexOf(tab)
    mainwindow.close_tab(idx)
    assert mainwindow.tabs.count() == 0

def test_current_tab(mainwindow):
    tab = DummyTab()
    mainwindow.tabs.addTab(tab, "dummy")
    mainwindow.tabs.setCurrentWidget(tab)
    assert mainwindow.current_tab() == tab

# --- Testy dialogów ---
def test_show_about_dialog(monkeypatch, mainwindow):
    called = {}
    class DummyDialog:
        def __init__(self, parent=None): called['init'] = True
        def exec_(self): called['exec'] = True
    monkeypatch.setattr('src.frasta_gui.AboutDialog', DummyDialog)
    mainwindow.show_about_dialog()
    assert called == {'init': True, 'exec': True}

# --- Testy metod bez zakładek ---
@pytest.mark.parametrize("method", [
    "save_single_scan", "save_multiple_scans", "fill_holes", "flipUD_scan",
    "flipLR_scan", "scan_rot90", "invert_scan", "set_zero_point_mode",
    "set_tilt_mode", "toggle_colormap_current_tab", "repair_grid",
    "compare_scans", "start_profile_analysis"
])
def test_methods_no_tab(mainwindow, method, monkeypatch):
    mainwindow.tabs.clear()
    if "save" in method or "compare" in method or "profile" in method:
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    getattr(mainwindow, method)()

def test_save_tabs_no_tabs(mainwindow):
    mainwindow.save_tabs(tabs=None)

def test_save_tabs_none(mainwindow, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    mainwindow.save_tabs(None)

def test_create_tab_and_load_unknown_format(mainwindow, tmp_path):
    test_file = str(tmp_path / "test.unknown")
    with open(test_file, "w") as f:
        f.write("dummy")
    mainwindow.create_tab_and_load(test_file)

# --- Testy ładowania/zapisu plików ---
def test_load_npz(monkeypatch, mainwindow, tmp_path):
    monkeypatch.setattr('src.frasta_gui.ScanTab', DummyTab)
    arr = np.array([[1,2],[3,4]])
    fname = str(tmp_path / "test.npz")
    np.savez(fname, frasta_info="grid_data", frasta_cnt=1, name_00="tab", grid_00=arr, xi_00=arr, yi_00=arr, px_00=1, py_00=1)
    assert mainwindow.load_npz(fname) is True

def test_load_npz_wrong(monkeypatch, mainwindow, tmp_path):
    fname = str(tmp_path / "test.npz")
    np.savez(fname, dummy=1)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    assert mainwindow.load_npz(fname) is False

def test_load_h5(monkeypatch, mainwindow, tmp_path):
    import h5py
    fname = str(tmp_path / "test.h5")
    with h5py.File(fname, "w") as f:
        f.attrs['frasta_info'] = "grid_data"
        f.attrs['frasta_cnt'] = 1
        g = f.create_group("tab_00")
        g.create_dataset("name", data=np.bytes_("tab"))
        arr = np.array([[1,2],[3,4]])
        g.create_dataset("grid", data=arr)
        g.create_dataset("xi", data=arr)
        g.create_dataset("yi", data=arr)
        g.create_dataset("px_x", data=[1])
        g.create_dataset("px_y", data=[1])
    monkeypatch.setattr('src.frasta_gui.ScanTab', DummyTab)
    assert mainwindow.load_h5(fname) is True

def test_load_h5_wrong(monkeypatch, mainwindow, tmp_path):
    import h5py
    fname = str(tmp_path / "test.h5")
    with h5py.File(fname, "w") as f:
        pass
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    assert mainwindow.load_h5(fname) is False

def test_save_tabs(monkeypatch, mainwindow, tmp_path):
    fname = str(tmp_path / "out.npz")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (fname, "NPZ (*.npz)"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    mainwindow.save_tabs([("tab", DummyTab())])

def test_save_tabs_h5(monkeypatch, mainwindow, tmp_path):
    fname = str(tmp_path / "out.h5")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (fname, "HDF5 (*.h5)"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    mainwindow.save_tabs([("tab", DummyTab())])

def test_save_tabs_exception(monkeypatch, mainwindow):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", "NPZ (*.npz)"))
    mainwindow.save_tabs([("tab", None)])

# --- Testy funkcji maskujących na zakładce ---
def test_view3d(monkeypatch, mainwindow):
    tab = DummyTab(np.array([[1]]))
    mainwindow.tabs.addTab(tab, "dummy")
    mainwindow.tabs.setCurrentWidget(tab)
    monkeypatch.setattr('src.frasta_gui.show_3d_viewer', lambda *a, **k: setattr(mainwindow, "_3d_called", True))
    mainwindow.view3d()
    assert hasattr(mainwindow, "_3d_called")

def test_apply_roi_mask(mainwindow):
    tab = DummyTab()
    mainwindow.tabs.addTab(tab, "dummy")
    mainwindow.tabs.setCurrentIndex(0)
    mainwindow.shared_circle_roi = None
    mainwindow.shared_rectangle_roi = None
    mainwindow.apply_roi_mask(True)

def test_del_inside_outside_mask(mainwindow):
    tab = DummyTab()
    mainwindow.tabs.addTab(tab, "dummy")
    mainwindow.tabs.setCurrentIndex(0)
    mainwindow.shared_circle_roi = None
    mainwindow.shared_rectangle_roi = None
    mainwindow.del_inside_mask()
    mainwindow.del_outside_mask()

def test_create_mask_none_when_no_roi(mainwindow):
    mainwindow.shared_circle_roi = None
    mainwindow.shared_rectangle_roi = None
    assert mainwindow.create_mask(10, 20) is None

def test_circle_mask_float_center(monkeypatch, mainwindow):
    class R(DummyROIBase): pass
    r = R(pos=(4.3, 5.7), size=(6.0, 6.0))
    r.setVisible(True)
    mainwindow.shared_circle_roi = r
    mainwindow.shared_rectangle_roi = None
    m = mainwindow.create_mask(12, 12)
    assert m.shape == (12, 12)
    assert m.any()

def test_recent_files_de_dupe_and_order(mainwindow, tmp_path):
    f1 = str(tmp_path / "a.csv")
    f2 = str(tmp_path / "b.csv")
    for p in (f1, f2): open(p, "w").close()
    mainwindow.add_to_recent_files(f1)
    mainwindow.add_to_recent_files(f2)
    mainwindow.add_to_recent_files(f1)  # f1 na początek
    assert mainwindow.recent_files[:2] == [f1, f2]

def test_create_tab_and_load_csv(monkeypatch, mainwindow, tmp_path):
    f = str(tmp_path / "d.csv"); open(f, "w").close()
    called = {}
    monkeypatch.setattr('src.frasta_gui.MainWindow.load_csv', lambda self, fname, tab: called.setdefault('f', fname))
    mainwindow.create_tab_and_load(f)
    assert called.get('f') == f
    assert mainwindow.recent_files[0] == f

def test_resource_path_meipass(monkeypatch):
    import src.frasta_gui as fg
    monkeypatch.setattr(fg.sys, "_MEIPASS", "/tmp/meipass", raising=False)
    p = fg.resource_path("icons/x.png")
    assert p.endswith("icons/x.png") and "/tmp/meipass" in p

def test_actions_wired(mainwindow):
    for key in ["open","save_scan","save_multi","fill","repair","flipUD","flipLR","rot90","inverse","zero","tilt","colormap","view3d","compare","profile","about","exit"]:
        assert key in mainwindow.actions
        assert isinstance(mainwindow.actions[key], QtWidgets.QAction)
