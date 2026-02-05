import sys
import pytest
from PyQt5.QtWidgets import QApplication
from unittest.mock import patch

@pytest.fixture(scope="function", autouse=True)
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture(autouse=True, scope="session")
def mock_qmessagebox():
    with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_warning, \
         patch("PyQt5.QtWidgets.QMessageBox.critical") as mock_critical, \
         patch("PyQt5.QtWidgets.QMessageBox.information") as mock_info, \
         patch("PyQt5.QtWidgets.QMessageBox.question", return_value=0) as mock_question:
        yield

