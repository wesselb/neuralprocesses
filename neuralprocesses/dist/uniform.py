import lab as B
from lab.shape import Dimension
from plum import Tuple
import torch

from .dist import AbstractDistribution
from .. import _dispatch

__all__ = ["UniformContinuous", "UniformDiscrete", "Grid"]


class UniformContinuous(AbstractDistribution):
    """A uniform continuous distribution.

    Also takes in tuples of its arguments.

    Args:
        lower (float): Lower bound.
        upper (float): Upper bound.

    Attributes:
        lowers (vector): Lower bounds.
        uppers (vector): Upper bounds.
    """

    @_dispatch
    def __init__(self, lower: B.Number, upper: B.Number):
        self.__init__((lower, upper))

    @_dispatch
    def __init__(self, *bounds: Tuple[B.Number, B.Number]):
        lowers, uppers = zip(*bounds)
        self.lowers = B.stack(*lowers)
        self.uppers = B.stack(*uppers)

    def sample(self, state, dtype, *shape):
        lowers = B.to_active_device(B.cast(dtype, self.lowers))
        uppers = B.to_active_device(B.cast(dtype, self.uppers))
        # Wrap everything in `Dimension`s to make dispatch work.
        shape = (Dimension(d) for d in shape)
        state, rand = B.rand(state, dtype, *shape, B.shape(lowers, 0))
        return state, lowers + (uppers - lowers) * rand


class UniformDiscrete(AbstractDistribution):
    """A uniform discrete distribution.

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.

    Attributes:
        lower (int): Lower bound.
        upper (int): Upper bound.
    """

    @_dispatch
    def __init__(self, lower: B.Int, upper: B.Int):
        self.lower = lower
        self.upper = upper

    def sample(self, state, dtype, *shape):
        return B.randint(state, dtype, lower=self.lower, upper=self.upper + 1)


class Grid(AbstractDistribution):
    """A uniform continuous distribution.

    Also takes in tuples of its arguments.

    Args:
        lower (float): Lower bound.
        upper (float): Upper bound.

    Attributes:
        lowers (vector): Lower bounds.
        uppers (vector): Upper bounds.
    """

    @_dispatch
    def __init__(self, lower: B.Number, upper: B.Number):
        self.__init__((lower, upper))

    @_dispatch
    def __init__(self, *bounds: Tuple[B.Number, B.Number]):
        lowers, uppers = zip(*bounds)
        self.lowers = B.stack(*lowers)
        self.uppers = B.stack(*uppers)

    def sample(self, state, dtype, *shape):
        # TODO: this is pretty broken for most uses probably, but I cannot figure
        # out how to shape the tensor correct just using B. and B.cast won't
        # conver the datatype
        batch_size = shape[0]
        n = shape[1]
        lowers = B.to_active_device(B.cast(dtype, self.lowers))
        uppers = B.to_active_device(B.cast(dtype, self.uppers))
        grid = B.linspace(lowers[0].item(), uppers[0].item(), n.item())
        # repeat the grid for each batch
        grids = [grid for _ in range(batch_size)]
        tg = torch.Tensor(grids).reshape(batch_size, n.item(), 1)
        # tg = B.to_active_device(B.cast(dtype, tg))
        # B.cast(dtype, grids)
        # Wrap everything in `Dimension`s to make dispatch work.
        shape = (Dimension(d) for d in shape)
        state, rand = B.rand(state, dtype, *shape, B.shape(lowers, 0))
        return state, tg
