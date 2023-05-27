from typing import List, Optional

import lab as B
from lab.shape import Dimension

from . import _dispatch
from .aggregate import AggregateInput
from .augment import AugmentedInput
from .parallel import Parallel
from .util import batch, is_nonempty, register_module

__all__ = ["Discretisation"]


@register_module
class Discretisation:
    """Discretisation.

    Args:
        points_per_unit (float): Density of the discretisation.
        multiple (int, optional): Always produce a discretisation which is a multiple
            of this number. Defaults to `1`.
        margin (float, optional): Leave this much space around the most extremal points.
            Defaults to `0.1`.
        dim (int, optional): Dimensionality of the inputs.

    Attributes:
        resolution (float): Resolution of the discretisation. Equal to the inverse of
            `points_per_unit`.
        multiple (int): Always produce a discretisation which is a multiple of this
            number.
        margin (float): Leave this much space around the most extremal points.
        dim (int): Dimensionality of the inputs.
    """

    def __init__(self, points_per_unit, multiple=1, margin=0.1, dim=None):
        self.points_per_unit = points_per_unit
        self.resolution = 1 / self.points_per_unit
        self.multiple = multiple
        self.margin = margin
        self.dim = dim

    def discretise_1d(self, *args, margin):
        """Perform the discretisation for one-dimensional inputs.

        Args:
            *args (input): One-dimensional inputs.
            margin (float): Leave this much space around the most extremal points.

        Returns:
            tensor: Discretisation.
        """
        # Filter global and empty inputs.
        args = [x for x in args if x is not None and is_nonempty(x)]
        grid_min = B.min(B.stack(*[B.min(x) for x in args]))
        grid_max = B.max(B.stack(*[B.max(x) for x in args]))

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
        b = batch(args[0], 2)
        with B.on_device(args[0]):
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
                    times=len(b) + 1,
                ),
                *b,
                1,
                1,
            )

    def __call__(self, *args, margin=None, **kw_args):
        """Perform the discretisation for multi-dimensional inputs.

        Args:
            *args (input): Multi-dimensional inputs.
            margin (float, optional): Leave this much space around the most extremal
                points. Defaults to `self.margin`.

        Returns:
            input: Discretisation.
        """
        if margin is None:
            margin = self.margin
        coords = _split_coordinates(Parallel(*args), dim=self.dim)
        discs = tuple(self.discretise_1d(*cs, margin=margin) for cs in coords)
        return discs[0] if len(discs) == 1 else discs


@_dispatch
def _split_coordinates(
    x: B.Numeric, dim: Optional[int] = None
) -> List[List[B.Numeric]]:
    # Cast with `int` so we can safely pass it to `range` below!
    dim = dim or int(B.shape(x, -2))
    return [[x[..., i : i + 1, :]] for i in range(dim)]


@_dispatch
def _split_coordinates(x: Parallel, dim: Optional[int] = None) -> List[List[B.Numeric]]:
    all_coords = zip(*(_split_coordinates(xi, dim=dim) for xi in x))
    return [sum(coords, []) for coords in all_coords]


@_dispatch
def _split_coordinates(x: tuple, dim: Optional[int] = None) -> List[List[B.Numeric]]:
    return [[xi] for xi in x]


@_dispatch
def _split_coordinates(
    x: AugmentedInput, dim: Optional[int] = None
) -> List[List[B.Numeric]]:
    return _split_coordinates(x.x, dim=dim)


@_dispatch
def _split_coordinates(
    x: AggregateInput, dim: Optional[int] = None
) -> List[List[B.Numeric]]:
    # Can treat it like a parallel of inputs. However, be sure to remove the indices.
    return _split_coordinates(Parallel(*(xi for xi, i in x)), dim=dim)
