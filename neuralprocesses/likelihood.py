import lab as B
from plum import Dispatcher, Union

from .dist import MultiOutputNormal
from .util import register_module
from .parallel import Parallel

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

    def __init__(self, rank):
        self.rank = rank

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"LowRankGaussianLikelihood(rank={self.rank})"

    @_dispatch
    def __call__(self, z: B.Numeric, *, middle=None):
        dim_y = B.shape(z, 1) // (2 + self.rank)
        dim_inner = B.shape(z, 1) - 2 * dim_y
        return MultiOutputNormal.lowrank(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y : 2 * dim_y, :]),
            # If everything were independent, we should divide by `B.sqrt(dim_inner)`
            # to keep the variance constant. However, we really don't want the variance
            # to be blowing up. Therefore, assume that everything is perfectly
            # correlated, and divide by `dim_inner` instead.
            z[:, 2 * dim_y :, :] / dim_inner,
            middle,
        )

    @_dispatch
    def __call__(self, z: Parallel):
        # Unpack middle and transform it appropriately.
        z, middle = z
        num_factors = int(B.sqrt(B.shape(middle, -2)))
        # Make it square.
        middle = B.reshape(middle, *B.shape(middle)[:-2], num_factors, num_factors)
        # Make it positive definite.
        middle = B.matmul(middle, middle, tr_b=True) / num_factors
        return self(z, middle=middle)
