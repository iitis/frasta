"""HDF5 profile data loader worker for background processing.

Provides ProfileWorker class for loading and preprocessing scan data
from HDF5 files in a separate thread without blocking the GUI.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import h5py


class ProfileWorker(QThread):
    """Background worker for loading and preprocessing scan data from HDF5.
    
    Loads reference and adjusted grids from an HDF5 file, applies optional
    smoothing, and computes offset correction in a separate thread.
    
    Signals:
        finished (dict): Emitted with processed data dictionary.
        error (str): Emitted if an error occurs during processing.
    """
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, sigma):
        """Initialize the profile worker.
        
        Args:
            filepath (str): Path to HDF5 file containing scan data.
            sigma (float): Gaussian smoothing parameter (currently unused).
        """
        super().__init__()
        self.filepath = filepath
        self.sigma = sigma

    def run(self):
        """Load and process scan data from HDF5 file.
        
        Loads both scans, computes offset correction, and emits results.
        Emits error signal if loading fails.
        """
        try:
            with h5py.File(self.filepath, "r") as f:
                reference_grid = f["scan1"][:]
                adjusted_grid = f["scan2"][:]
            
            # reference_grid_smooth = gaussian_filter(reference_grid, sigma=self.sigma)
            # adjusted_grid_smooth = gaussian_filter(adjusted_grid, sigma=self.sigma)
            reference_grid_smooth = reference_grid
            adjusted_grid_smooth = adjusted_grid
            
            valid_mask = ~np.isnan(reference_grid_smooth) & ~np.isnan(adjusted_grid_smooth)
            offset_correction = np.nanmean(reference_grid_smooth - adjusted_grid_smooth)
            adjusted_grid_corrected = adjusted_grid_smooth + offset_correction
            # Done, return everything in dict
            result = {
                "reference_grid": reference_grid,
                "adjusted_grid": adjusted_grid,
                "reference_grid_smooth": reference_grid_smooth,
                "adjusted_grid_smooth": adjusted_grid_smooth,
                "valid_mask": valid_mask,
                "adjusted_grid_corrected": adjusted_grid_corrected,
            }
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + '\n' + traceback.format_exc())
