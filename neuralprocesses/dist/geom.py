import lab as B
import numpy as np

from .dist import AbstractDistribution
from .. import _dispatch

__all__ = ["TruncatedGeometric"]


class TruncatedGeometric(AbstractDistribution):
    """A truncated geometric distribution.

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        factor (float): Probability of the lower bound divided by the probability of
            the upper bound.

    Attributes:
        lower (int): Lower bound.
        upper (int): Upper bound.
        factor (float): Probability of the lower bound divided by the probability of
            the upper bound.
    """

    @_dispatch
    def __init__(self, lower: B.Int, upper: B.Int, factor: B.Number):
        self.lower = lower
        self.upper = upper
        self.factor = factor

    def sample(self, state, dtype, *shape):
        dtype_float = B.promote_dtypes(dtype, np.float16)
        realisations = B.range(dtype, self.lower, self.upper)
        lam = B.cast(dtype_float, B.log(self.factor) / (self.upper - self.lower))
        lam = B.to_active_device(lam)
        probs = B.exp(-lam * B.cast(dtype_float, realisations))
        return B.choice(state, realisations, *shape, p=probs)
