"""CSV data loader worker for background processing.

Provides GridWorker class for loading and gridding CSV scan data
in a separate thread without blocking the GUI.
"""

from PyQt5 import QtCore
from ...io import load_csv_data


class GridWorker(QtCore.QObject):
    """Background worker for loading and gridding CSV scan data.
    
    Uses load_csv_data from io module to process files in background thread.
    
    Signals:
        progress (int): Progress percentage (0-100).
        finished (object, object, object, float, float): Emitted when processing completes.
    """
    
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object, object, object, float, float)

    def __init__(self, fname, units_xy='um', units_z='um'):
        """Initialize the CSV loader worker.
        
        Args:
            fname (str): Path to CSV file to load.
            units_xy (str): Units for X/Y coordinates ('um', 'nm', 'mm', 'm').
            units_z (str): Units for Z values ('um', 'nm', 'mm', 'm').
        """
        super().__init__()
        self.fname = fname
        self.units_xy = units_xy
        self.units_z = units_z

    @QtCore.pyqtSlot()
    def process(self):
        """Process the CSV file using io.load_csv_data.
        
        Emits progress signals during loading and finished signal with results.
        Results: (grid, xi, yi, pixel_size_x, pixel_size_y)
        """
        def progress_callback(value):
            self.progress.emit(value)
        
        grid, xi, yi, px_x, px_y = load_csv_data(
            self.fname,
            units_xy=self.units_xy,
            units_z=self.units_z,
            progress_callback=progress_callback
        )
        self.finished.emit(grid, xi, yi, px_x, px_y)
