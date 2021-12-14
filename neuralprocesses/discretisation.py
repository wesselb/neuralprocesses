import math

import lab as B

__all__ = ["AbstractDiscretisation", "Discretisation1d"]


class AbstractDiscretisation:
    pass


class Discretisation1d(AbstractDiscretisation):
    def __init__(self, points_per_unit, multiple, margin):
        self.points_per_unit = points_per_unit
        self.resolution = 1 / self.points_per_unit
        self.multiple = multiple
        self.margin = margin

    def __call__(self, *args):
        args = [arg for arg in args if B.length(arg) > 0]
        grid_min = min([B.to_numpy(B.min(x)) for x in args]) - self.margin
        grid_max = max([B.to_numpy(B.max(x)) for x in args]) + self.margin

        # Account for snapping to the grid (below).
        grid_min -= self.resolution
        grid_max += self.resolution

        # Ensure that the multiple is respected. Add one point to account for the end.
        n_raw = (grid_max - grid_min) / self.resolution + 1
        n = math.ceil(n_raw / self.multiple) * self.multiple

        # Nicely shift the grid to account for the extra points.
        grid_start = grid_min - (n - n_raw) * self.resolution / 2

        # Snap to the nearest grid point. We accounted for this above.
        grid_start = round(grid_start / self.resolution) * self.resolution

        # Produce the grid.
        batch = B.shape(args[0], 0)
        return B.tile(
            B.expand_dims(
                B.linspace(
                    B.dtype(args[0]),
                    grid_start,
                    grid_start + (n - 1) * self.resolution,
                    n,
                ),
                axis=0,
                times=2,
            ),
            batch,
            1,
            1,
        )
