import lab as B
from plum import Dispatcher

from .dist import MultiOutputNormal

__all__ = ["HeterogeneousGaussianLikelihood"]

_dispatch = Dispatcher()


class HeterogeneousGaussianLikelihood:
    """Gaussian likelihood with heterogeneous noise."""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "HeterogeneousGaussianLikelihood()"

    def __call__(self, z):
        i = B.shape(z, 1)
        if i % 2 != 0:
            raise ValueError("Must give an even number of channels.")
        i = i // 2
        return MultiOutputNormal.diagonal(z[:, :i, :], B.softplus(z[:, i:, :]))


class LowRankGaussianLikelihood:
    """Gaussian likelihood with low-rank noise.

    Args:
        rank (int): Rank of the low-rank part of the noise variance.

    Attributes:
        rank (int): Rank of the low-rank part of the noise variance.
    """

    @_dispatch
    def __init__(self, rank: int):
        self.rank = rank

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"LowRankGaussianLikelihood({self.rank}])"

    def __call__(self, z):
        i = B.shape(z, 1)
        if i <= self.rank or (i - self.rank) % 2 != 0:
            raise ValueError(
                "After substracting the rank, the number of channels must be even."
            )
        return MultiOutputNormal.lowrank(
            z[:, :i, :],
            z[:, i : 2 * i, :],
            z[:, 2 * i :, :],
        )
