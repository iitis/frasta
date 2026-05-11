"""Data structure for storing gridded scan data.

This module provides the GridData class for encapsulating 2D scan data along with
its metadata, including coordinate arrays, pixel sizes, and value ranges.
"""

class GridData:
    """Container for 2D gridded scan data with associated metadata.
    
    Stores a 2D array of scan data along with coordinate arrays, pixel sizes,
    and optional value ranges for display purposes.

    Notes:
        ``Surface`` is the primary data container used by the current
        application code. ``GridData`` is kept only for backward compatibility
        with older helper code and should not be used for new features.
    
    Attributes:
        grid (np.ndarray): 2D array containing scan height/depth values.
        xi (np.ndarray): 1D array of x-coordinates.
        yi (np.ndarray): 1D array of y-coordinates.
        px_x (float): Pixel size in x-direction.
        px_y (float): Pixel size in y-direction.
        vmin (float, optional): Minimum display value.
        vmax (float, optional): Maximum display value.
    """
    
    def __init__(self, grid, xi, yi, px_x, px_y, vmin=None, vmax=None):
        """Initialize GridData with scan data and metadata.
        
        Args:
            grid (np.ndarray): 2D array of scan values.
            xi (np.ndarray): 1D array of x-coordinates.
            yi (np.ndarray): 1D array of y-coordinates.
            px_x (float): Pixel size in x-direction.
            px_y (float): Pixel size in y-direction.
            vmin (float, optional): Minimum display value. Defaults to None.
            vmax (float, optional): Maximum display value. Defaults to None.
        """
        self.grid = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        self.vmin = vmin
        self.vmax = vmax

    def crop(self, h, w):
        """Returns a new GridData cropped to the specified dimensions.
        
        Args:
            h (int): Height (number of rows) to crop to.
            w (int): Width (number of columns) to crop to.
            
        Returns:
            GridData: New GridData instance with cropped data.
        """
        return GridData(
            self.grid[:h, :w],
            self.xi[:w],
            self.yi[:h],
            self.px_x,
            self.px_y,
            self.vmin,
            self.vmax
        )

    def copy(self):
        """Creates a deep copy of the GridData object.
        
        Returns:
            GridData: New GridData instance with copied arrays and metadata.
        """
        return GridData(
            self.grid.copy(),
            self.xi.copy(),
            self.yi.copy(),
            self.px_x,
            self.px_y,
            self.vmin,
            self.vmax
        )
