from typing import Tuple

import lab as B
from lab.shape import Dimension

from .. import _dispatch
from .dist import AbstractDistribution

__all__ = ["UniformContinuous", "UniformDiscrete"]


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

    def __str__(self):
        return f"UniformContinuous({self.lower}, {self.upper})"

    def __repr__(self):
        return f"UniformContinuous({self.lower!r}, {self.uppers!r})"

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
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

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        return B.randint(state, dtype, lower=self.lower, upper=self.upper + 1, *shape)

    def __str__(self):
        return f"UniformDiscrete({self.lower}, {self.upper})"

    def __repr__(self):
        return f"UniformDiscrete({self.lower!r}, {self.uppers!r})"
