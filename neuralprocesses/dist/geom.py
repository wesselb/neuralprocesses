from typing import Union

import lab as B
import numpy as np

from .. import _dispatch
from .dist import AbstractDistribution

__all__ = ["TruncatedGeometric"]


class TruncatedGeometric(AbstractDistribution):
    """A truncated geometric distribution.

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        factor (float): Probability of the lower bound divided by the probability of
            `factor_at`.
        factor_at (int): Upper bound for `factor`. Defaults to `upper`.

    Attributes:
        lower (int): Lower bound.
        upper (int): Upper bound.
        factor (float): Probability of the lower bound divided by the probability of
            `factor_at`.
        factor_at (int): Upper bound for `factor`.
    """

    @_dispatch
    def __init__(
        self,
        lower: B.Int,
        upper: B.Int,
        factor: B.Number,
        factor_at: Union[B.Number, None] = None,
    ):
        self.lower = lower
        self.upper = upper
        self.factor = factor
        self.factor_at = upper if factor_at is None else factor_at

    def sample(self, state, dtype, *shape):
        dtype_float = B.promote_dtypes(dtype, np.float16)
        realisations = B.range(dtype, self.lower, self.upper + 1)
        if self.upper > self.lower:
            lam = B.log(self.factor) / (self.factor_at - self.lower)
            lam = B.cast(dtype_float, B.to_active_device(lam))
            probs = B.exp(-lam * B.cast(dtype_float, realisations))
        else:
            probs = B.to_active_device(B.ones(dtype_float, 1))
        return B.choice(state, realisations, *shape, p=probs)

    def __str__(self):
        return (
            f"TruncatedGeometric("
            f"{self.lower}, {self.upper}, {self.factor}, {self.factor_at}"
            f")"
        )

    def __repr__(self):
        return (
            f"TruncatedGeometric("
            f"{self.lower!r}, {self.uppers!r}, {self.factor!r}, {self.factor_at!r}"
            f")"
        )
