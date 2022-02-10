import lab as B
from lab.shape import Dimension
from plum import List

from . import _dispatch
from .parallel import Parallel
from .util import register_module

__all__ = ["AbstractDiscretisation", "Discretisation"]


class AbstractDiscretisation:
    pass


@register_module
class Discretisation(AbstractDiscretisation):
    def __init__(self, points_per_unit, multiple=1, margin=0.1, dim=None):
        self.points_per_unit = points_per_unit
        self.resolution = 1 / self.points_per_unit
        self.multiple = multiple
        self.margin = margin
        self.dim = dim

    def discretise(self, *args, margin):
        grid_min = B.min(B.stack(*[B.min(x) for x in args if x is not None]))
        grid_max = B.max(B.stack(*[B.max(x) for x in args if x is not None]))

        # Add margin.
        grid_min = grid_min - margin
        grid_max = grid_max + margin

        # Account for snapping to the grid (below).
        grid_min = grid_min - self.resolution
        grid_max = grid_max + self.resolution

        # Ensure that the multiple is respected. Add one point to account for the end.
        n_raw = (grid_max - grid_min) / self.resolution + 1
        n = B.ceil(n_raw / self.multiple) * self.multiple

        # Nicely shift the grid to account for the extra points.
        grid_start = grid_min - (n - n_raw) * self.resolution / 2

        # Snap to the nearest grid point. We accounted for this above.
        grid_start = B.round(grid_start / self.resolution) * self.resolution

        # Produce the grid.
        batch = B.shape(args[0], 0)
        return B.tile(
            B.expand_dims(
                B.linspace(
                    B.dtype(args[0]),
                    grid_start,
                    grid_start + (n - 1) * self.resolution,
                    # Tell LAB that it can be interpreted as an integer.
                    Dimension(B.cast(B.dtype_int(n), n)),
                ),
                axis=0,
                times=2,
            ),
            batch,
            1,
            1,
        )

    def __call__(self, *args, margin=None, **kw_args):
        if margin is None:
            margin = self.margin
        coords = _split_coordinates(Parallel(*args), dim=self.dim)
        discs = tuple(self.discretise(*cs, margin=margin) for cs in coords)
        return discs[0] if len(discs) == 1 else discs


@_dispatch
def _split_coordinates(x: B.Numeric, dim=None) -> List[List[B.Numeric]]:
    # Cast with `int` so we can safely pass it to `range` below!
    dim = dim or int(B.shape(x, 1))
    return [[x[:, i, :]] for i in range(dim)]


@_dispatch
def _split_coordinates(x: Parallel, dim=None) -> List[List[B.Numeric]]:
    all_coords = zip(*(_split_coordinates(xi, dim=dim) for xi in x))
    return [sum(coords, []) for coords in all_coords]


@_dispatch
def _split_coordinates(x: tuple, dim=None) -> List[List[B.Numeric]]:
    return [[xi] for xi in x]
