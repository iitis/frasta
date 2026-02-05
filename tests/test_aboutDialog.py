import pytest
from PyQt5 import QtWidgets
from src.aboutDialog import AboutDialog

@pytest.fixture
def dlg(qapp):
    return AboutDialog()

def test_about_dialog_properties(dlg):
    assert dlg.windowTitle() == "About"
    assert dlg.minimumWidth() == 350
    assert dlg.isModal() is True

def test_about_dialog_layout_and_widgets(dlg):
    layout = dlg.layout()
    assert isinstance(layout, QtWidgets.QVBoxLayout)
    assert layout.count() == 2

    label = layout.itemAt(0).widget()
    assert isinstance(label, QtWidgets.QLabel)
    assert label.openExternalLinks() is True
    assert "FRASTA - converter" in label.text()
    assert "Author: Dariusz Pojda" in label.text()
    assert "Version: 1.0.0" in label.text()
    assert "https://icons8.com/icons/set/eyedropper" in label.text()
    assert "<a href='https://icons8.com/icons/set/eyedropper'>" in label.text()

    btn = layout.itemAt(1).widget()
    assert isinstance(btn, QtWidgets.QPushButton)
    assert btn.text() == "OK"

def test_about_dialog_ok_button_accepts_dialog(dlg):
    btn = dlg.layout().itemAt(1).widget()
    btn.click()
    assert dlg.result() == QtWidgets.QDialog.Accepted

