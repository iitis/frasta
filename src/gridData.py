class GridData:
    def __init__(self, grid, xi, yi, px_x, px_y, vmin=None, vmax=None):
        self.grid = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        self.vmin = vmin
        self.vmax = vmax

    def crop(self, h, w):
        """Zwraca nowy GridData przycięty do wymiarów h x w"""
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
        """Pełna kopia"""
        return GridData(
            self.grid.copy(),
            self.xi.copy(),
            self.yi.copy(),
            self.px_x,
            self.px_y,
            self.vmin,
            self.vmax
        )
