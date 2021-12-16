import lab as B
from plum import Dispatcher, Union

from .dist import MultiOutputNormal
from .util import register_module

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]

_dispatch = Dispatcher()


@register_module
class HeterogeneousGaussianLikelihood:
    """Gaussian likelihood with heterogeneous noise.

    Args:
        noise (scalar, optional): Initialisation for the homogeneous part of the noise.
            Defaults to `0.1`.
        dtype (dtype, optional): Data type.

    Attributes:
        log_noise (scalar): Logarithm of the homogeneous part of the noise.
    """

    def __init__(self, noise=0.1, dtype=None):
        self.log_noise = self.nn.Parameter(B.log(noise), dtype=dtype)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f"HeterogeneousGaussianLikelihood("
            f"noise={B.exp(self.log_noise)}, "
            f"dtype={B.dtype(self.log_noise)}"
            f")"
        )

    def __call__(self, z):
        dim_y = B.shape(z, 1) // 2
        return MultiOutputNormal.diagonal(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y:, :]) + B.exp(self.log_noise),
        )


@register_module
class LowRankGaussianLikelihood:
    """Gaussian likelihood with low-rank noise.

    Args:
        rank (int): Rank of the low-rank part of the noise variance.
        noise (scalar, optional): Initialisation for the homogeneous part of the noise.
            Defaults to `0.1`.
        dtype (dtype, optional): Data type.

    Attributes:
        rank (int): Rank of the low-rank part of the noise variance.
        log_noise (scalar): Logarithm of the homogeneous part of the noise.
    """

    def __init__(self, rank, noise=0.1, dtype=None):
        self.rank = rank
        self.log_noise = self.nn.Parameter(B.log(noise), dtype=dtype)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f"LowRankGaussianLikelihood("
            f"rank={self.rank}"
            f"noise={B.exp(self.log_noise)}"
            f"dtype={B.dtype(self.log_noise)}"
            f")"
        )

    def __call__(self, z):
        dim_y = B.shape(z, 1) // (2 + self.rank)
        return MultiOutputNormal.lowrank(
            z[:, :dim_y, :],
            B.softplus(z[:, dim_y : 2 * dim_y, :]) + B.exp(self.log_noise),
            z[:, 2 * dim_y :, :] / B.sqrt(self.rank * dim_y),
        )
