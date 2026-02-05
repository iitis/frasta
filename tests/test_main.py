import sys
import pytest
from unittest import mock
import main

@pytest.fixture
def mock_qapplication():
    with mock.patch("main.QtWidgets.QApplication") as mock_app:
        yield mock_app

@pytest.fixture
def mock_mainwindow():
    with mock.patch("main.MainWindow") as mock_win:
        yield mock_win

def test_run_calls_set_logger(monkeypatch, mock_qapplication, mock_mainwindow):
    set_logger_called = False

    def fake_set_logger():
        nonlocal set_logger_called
        set_logger_called = True

    monkeypatch.setattr(main, "set_logger", fake_set_logger)
    monkeypatch.setattr(sys, "exit", lambda code=0: None)
    mock_instance = mock_qapplication.return_value
    mock_instance.exec_.return_value = 0

    main.run()

    assert set_logger_called
    mock_qapplication.assert_called_once_with(sys.argv)
    mock_mainwindow.assert_called_once()
    mock_mainwindow.return_value.show.assert_called_once()
    mock_instance.exec_.assert_called_once()

def test_run_exits(monkeypatch, mock_qapplication, mock_mainwindow):
    exit_called = {}

    def fake_exit(code=0):
        exit_called['called'] = True
        exit_called['code'] = code

    monkeypatch.setattr(main, "set_logger", lambda: None)
    monkeypatch.setattr(sys, "exit", fake_exit)
    mock_instance = mock_qapplication.return_value
    mock_instance.exec_.return_value = 123

    main.run()

    assert exit_called['called']
    assert exit_called['code'] == 123