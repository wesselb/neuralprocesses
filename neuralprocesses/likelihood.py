import lab as B
from plum import Dispatcher

from .dist import MultiOutputNormal

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]

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
        dim_y = i // 2
        return MultiOutputNormal.diagonal(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y:, :]),
        )


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
        if i % (2 + self.rank) != 0:
            raise ValueError("Must provide `(2 + rank) * dim_y` channels.")
        dim_y = i // (2 + self.rank)
        return MultiOutputNormal.lowrank(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y : 2 * dim_y, :]),
            z[:, 2 * dim_y :, :] / B.sqrt(self.rank),
        )
