"""Tests for frasta.gui.workers module (background loading workers)."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtCore import QObject
from frasta.gui.workers import GridWorker, ProfileWorker
from frasta.core import Surface


class TestGridWorker:
    """Test suite for GridWorker (CSV loader)."""
    
    def test_grid_worker_initialization(self):
        """Test that GridWorker initializes with correct parameters."""
        worker = GridWorker("test.csv", units_xy='nm', units_z='um')
        
        assert worker.fname == "test.csv"
        assert worker.units_xy == 'nm'
        assert worker.units_z == 'um'
    
    def test_grid_worker_is_qobject(self):
        """Test that GridWorker is a QObject."""
        worker = GridWorker("test.csv")
        assert isinstance(worker, QObject)
    
    @patch('frasta.gui.workers.csv_loader_worker.load_csv_data')
    def test_grid_worker_process_calls_loader(self, mock_load, temp_csv_file):
        """Test that process method calls load_csv_data."""
        # Setup mock return value - Surface object
        mock_grid = np.array([[1, 2], [3, 4]])
        mock_surface = Surface(
            height=mock_grid,
            dx=1.0,
            dy=1.0,
            x0=0.0,
            y0=0.0
        )
        mock_load.return_value = mock_surface
        
        worker = GridWorker(temp_csv_file, units_xy='um', units_z='um')
        
        # Mock signals
        worker.progress = Mock()
        worker.finished = Mock()
        
        # Execute
        worker.process()
        
        # Verify load_csv_data was called
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs['units_xy'] == 'um'
        assert call_kwargs['units_z'] == 'um'
    
    @patch('frasta.gui.workers.csv_loader_worker.load_csv_data')
    def test_grid_worker_emits_finished_signal(self, mock_load, temp_csv_file):
        """Test that worker emits finished signal with Surface object."""
        # Setup mock return value - Surface object
        mock_grid = np.array([[1, 2], [3, 4]])
        mock_surface = Surface(
            height=mock_grid,
            dx=1.5,
            dy=2.0,
            x0=0.0,
            y0=0.0
        )
        mock_load.return_value = mock_surface
        
        worker = GridWorker(temp_csv_file)
        
        # Create a mock for finished signal
        finished_spy = Mock()
        worker.finished.connect(finished_spy)
        
        # Execute
        worker.process()
        
        # Verify finished signal was emitted with Surface object
        finished_spy.assert_called_once()
        args = finished_spy.call_args[0]
        assert len(args) == 1
        assert isinstance(args[0], Surface)
        assert args[0].dx == 1.5
        assert args[0].dy == 2.0
    
    @patch('frasta.gui.workers.csv_loader_worker.load_csv_data')
    def test_grid_worker_progress_callback(self, mock_load, temp_csv_file):
        """Test that progress callback is wired correctly."""
        def mock_load_with_progress(*args, **kwargs):
            # Simulate progress updates
            callback = kwargs.get('progress_callback')
            if callback:
                callback(25)
                callback(50)
                callback(100)
            # Return Surface object
            return Surface(
                height=np.array([[1]]),
                dx=1.0,
                dy=1.0,
                x0=0.0,
                y0=0.0
            )
        
        mock_load.side_effect = mock_load_with_progress
        
        worker = GridWorker(temp_csv_file)
        progress_spy = Mock()
        worker.progress.connect(progress_spy)
        
        # Execute
        worker.process()
        
        # Verify progress was emitted
        assert progress_spy.call_count >= 1


class TestProfileWorker:
    """Test suite for ProfileWorker (HDF5 loader)."""
    
    def test_profile_worker_initialization(self):
        """Test that ProfileWorker initializes with correct parameters."""
        worker = ProfileWorker("test.h5", sigma=2.0)
        
        assert worker.filepath == "test.h5"
        assert worker.sigma == 2.0
    
    @patch('frasta.gui.workers.profile_loader_worker.h5py.File')
    def test_profile_worker_loads_h5_data(self, mock_h5):
        """Test that ProfileWorker loads data from HDF5 file."""
        # Setup mock HDF5 file
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        
        ref_grid = np.array([[1, 2], [3, 4]], dtype=float)
        adj_grid = np.array([[2, 3], [4, 5]], dtype=float)
        
        mock_file.__getitem__.side_effect = lambda key: {
            "scan1": ref_grid,
            "scan2": adj_grid
        }[key]
        
        mock_h5.return_value = mock_file
        
        worker = ProfileWorker("test.h5", sigma=1.0)
        
        # Mock signals  
        finished_spy = Mock()
        worker.finished.connect(finished_spy)
        
        # Execute
        worker.run()
        
        # Verify finished signal was emitted with data
        finished_spy.assert_called_once()
        result = finished_spy.call_args[0][0]
        
        assert 'reference_grid' in result
        assert 'adjusted_grid' in result
        assert 'valid_mask' in result
    
    @patch('frasta.gui.workers.profile_loader_worker.h5py.File')
    def test_profile_worker_computes_offset_correction(self, mock_h5):
        """Test that ProfileWorker computes offset correction."""
        # Setup mock with known offset
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        
        ref_grid = np.ones((3, 3), dtype=float) * 10.0
        adj_grid = np.ones((3, 3), dtype=float) * 8.0  # Offset by -2
        
        mock_file.__getitem__.side_effect = lambda key: {
            "scan1": ref_grid,
            "scan2": adj_grid
        }[key]
        
        mock_h5.return_value = mock_file
        
        worker = ProfileWorker("test.h5", sigma=0)
        finished_spy = Mock()
        worker.finished.connect(finished_spy)
        
        # Execute
        worker.run()
        
        result = finished_spy.call_args[0][0]
        
        # Check that corrected grid has offset applied
        assert 'adjusted_grid_corrected' in result
        corrected = result['adjusted_grid_corrected']
        
        # Corrected grid should be close to reference
        assert np.allclose(corrected, ref_grid, rtol=0.01)
    
    @patch('frasta.gui.workers.profile_loader_worker.h5py.File')
    def test_profile_worker_handles_errors(self, mock_h5):
        """Test that ProfileWorker emits error signal on failure."""
        # Make h5py.File raise an exception
        mock_h5.side_effect = Exception("File not found")
        
        worker = ProfileWorker("nonexistent.h5", sigma=1.0)
        
        error_spy = Mock()
        worker.error.connect(error_spy)
        
        # Execute
        worker.run()
        
        # Verify error signal was emitted
        error_spy.assert_called_once()
        error_msg = error_spy.call_args[0][0]
        assert "File not found" in error_msg
    
    @patch('frasta.gui.workers.profile_loader_worker.h5py.File')
    def test_profile_worker_creates_valid_mask(self, mock_h5):
        """Test that ProfileWorker creates proper valid mask."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        
        # Grids with some NaN values
        ref_grid = np.array([[1, 2, np.nan], [4, 5, 6]], dtype=float)
        adj_grid = np.array([[2, np.nan, 4], [5, 6, 7]], dtype=float)
        
        mock_file.__getitem__.side_effect = lambda key: {
            "scan1": ref_grid,
            "scan2": adj_grid
        }[key]
        
        mock_h5.return_value = mock_file
        
        worker = ProfileWorker("test.h5", sigma=0)
        finished_spy = Mock()
        worker.finished.connect(finished_spy)
        
        worker.run()
        
        result = finished_spy.call_args[0][0]
        valid_mask = result['valid_mask']
        
        # Only [0,0], [1,0], [1,1], [1,2] should be valid
        assert valid_mask[0, 0] == True
        assert valid_mask[0, 1] == False  # adj has NaN
        assert valid_mask[0, 2] == False  # ref has NaN
        assert valid_mask[1, 0] == True
        assert valid_mask[1, 1] == True
        assert valid_mask[1, 2] == True
