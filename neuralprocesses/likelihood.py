import lab as B
from plum import Dispatcher

from .dist import MultiOutputNormal
from .util import register_module

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]

_dispatch = Dispatcher()


@register_module
class HeterogeneousGaussianLikelihood:
    """Gaussian likelihood with heterogeneous noise."""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "HeterogeneousGaussianLikelihood()"

    def __call__(self, z):
        dim_y = B.shape(z, 1) // 2
        return MultiOutputNormal.diagonal(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y:, :]),
        )


@register_module
class LowRankGaussianLikelihood:
    """Gaussian likelihood with low-rank noise.

    Args:
        rank (int): Rank of the low-rank part of the noise variance.

    Attributes:
        rank (int): Rank of the low-rank part of the noise variance.
    """

    @_dispatch
    def __init__(self, rank: B.Int):
        self.rank = rank

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"LowRankGaussianLikelihood({self.rank})"

    def __call__(self, z):
        dim_y = B.shape(z, 1) // (2 + self.rank)
        return MultiOutputNormal.lowrank(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y : 2 * dim_y, :]),
            z[:, 2 * dim_y :, :] / B.sqrt(self.rank * dim_y),
        )
