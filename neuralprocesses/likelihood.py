import lab as B

from . import _dispatch
from .dist import MultiOutputNormal
from .util import register_module

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]


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


@_dispatch
def code(
    coder: HeterogeneousGaussianLikelihood,
    xz,
    z: B.Numeric,
    x,
    *,
    noiseless=False,
    **kw_args,
):
    dim_y = B.shape(z, 1) // 2
    noise = coder.epsilon + B.softplus(z[:, dim_y:, :])
    if noiseless:
        with B.on_device(var):
            noise = B.zeros(var)
    return xz, MultiOutputNormal.diagonal(z[:, :dim_y, :], noise)


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
def code(
    coder: LowRankGaussianLikelihood,
    xz,
    z: B.Numeric,
    x,
    *,
    noiseless=False,
    **kw_args,
):
    dim_y = B.shape(z, 1) // (2 + coder.rank)
    dim_inner = B.shape(z, 1) - 2 * dim_y
    noise = coder.epsilon + B.softplus(z[:, dim_y : 2 * dim_y, :])
    if noiseless:
        with B.on_device(noise):
            noise = B.zeros(noise)
    return xz, MultiOutputNormal.lowrank(
        z[:, :dim_y, :],
        noise,
        z[:, 2 * dim_y :, :] / B.sqrt(dim_inner),
    )
