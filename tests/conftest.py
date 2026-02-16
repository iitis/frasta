"""Pytest configuration and fixtures for FRASTA tests."""

import sys
import pytest
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from unittest.mock import patch


@pytest.fixture(scope="session")
def qapp():
    """Provide QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True, scope="session")
def mock_qmessagebox():
    """Mock Qt message boxes to prevent GUI dialogs during tests."""
    with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_warning, \
         patch("PyQt5.QtWidgets.QMessageBox.critical") as mock_critical, \
         patch("PyQt5.QtWidgets.QMessageBox.information") as mock_info, \
         patch("PyQt5.QtWidgets.QMessageBox.question", return_value=0) as mock_question:
        yield


@pytest.fixture
def sample_grid():
    """Create a simple 10x10 grid with some NaN values."""
    grid = np.arange(100, dtype=float).reshape(10, 10)
    grid[2:4, 3:5] = np.nan  # Create holes
    grid[7, 8] = np.nan
    return grid


@pytest.fixture
def sample_csv_content():
    """Generate sample CSV content for testing."""
    lines = ["X [um];Y [um];Z [nm]"]
    for i in range(10):
        for j in range(10):
            x = i * 0.5
            y = j * 0.5
            z = (i + j) * 100
            lines.append(f"{x:.3f};{y:.3f};{z:.1f}")
    return "\n".join(lines)


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_content):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(sample_csv_content, encoding='utf-8')
    return str(csv_file)


@pytest.fixture
def temp_npz_file(tmp_path, sample_grid):
    """Create a temporary NPZ file with sample grid data."""
    npz_file = tmp_path / "test_data.npz"
    xi = np.linspace(0, 9, 10)
    yi = np.linspace(0, 9, 10)
    
    # Use the save format expected by frasta
    save_dict = {
        'frasta_info': 1,
        'frasta_cnt': 1,
        'name_00': 'test_scan',
        'grid_00': sample_grid,
        'xi_00': xi,
        'yi_00': yi,
        'px_00': 1.0,
        'py_00': 1.0
    }
    np.savez(str(npz_file), **save_dict)
    return str(npz_file)


@pytest.fixture
def temp_h5_file(tmp_path, sample_grid):
    """Create a temporary HDF5 file with sample grid data."""
    import h5py
    h5_file = tmp_path / "test_data.h5"
    with h5py.File(str(h5_file), "w") as f:
        f.attrs['frasta_info'] = 1
        f.attrs['frasta_cnt'] = 1
        
        grp = f.create_group("tab_00")
        grp.create_dataset("name", data=np.bytes_("test_scan"))
        grp.create_dataset("grid", data=sample_grid)
        grp.create_dataset("xi", data=np.linspace(0, 9, 10))
        grp.create_dataset("yi", data=np.linspace(0, 9, 10))
        grp.create_dataset("px_x", data=1.0)
        grp.create_dataset("px_y", data=1.0)
    return str(h5_file)

