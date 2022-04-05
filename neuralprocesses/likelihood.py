import lab as B
from plum import Dispatcher, Union

from .dist import MultiOutputNormal
from .util import register_module
from .parallel import Parallel

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]

_dispatch = Dispatcher()


@register_module
class HeterogeneousGaussianLikelihood:
    """Gaussian likelihood with heterogeneous noise.

    Args:
        epsilon (float, optional): Smallest allowable variance. Defaults to `1e-6`.

    Args:
        epsilon (float): Smallest allowable variance.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "HeterogeneousGaussianLikelihood()"

    def __call__(self, z):
        dim_y = B.shape(z, 1) // 2
        return MultiOutputNormal.diagonal(
            z[:, :dim_y, :],
            self.epsilon + B.softplus(z[:, dim_y:, :]),
        )


@register_module
class LowRankGaussianLikelihood:
    """Gaussian likelihood with low-rank noise.

    Args:
        rank (int): Rank of the low-rank part of the noise variance.
        epsilon (float, optional): Smallest allowable variance. Defaults to `1e-6`.

    Attributes:
        rank (int): Rank of the low-rank part of the noise variance.
        epsilon (float): Smallest allowable diagonal variance.
    """

    @_dispatch
    def __init__(self, rank, epsilon: float = 1e-6):
        self.rank = rank
        self.epsilon = epsilon

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
            self.epsilon + B.softplus(z[:, dim_y : 2 * dim_y, :]),
            z[:, 2 * dim_y :, :] / B.sqrt(dim_inner),
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
